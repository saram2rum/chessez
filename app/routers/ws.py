import asyncio
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect


logger = logging.getLogger(__name__)

from app.core.config import get_settings
from app.core.redis import redis_client
from app.repositories.room_repository import RoomRepository
from app.schemas.room import (
    RoomJoinPayload,
    RoomScopedUserPayload,
    RoomSettingsUpdatePayload,
    UserIdentityPayload,
)
from app.routers.ai import engine_service
from app.services.connection_manager import ConnectionManager
from app.services.eval_queue_service import EvalQueueService
from app.services.game_service import GameService
from app.services.live_position_eval import eval_fen_for_personal
from app.services.room_service import RoomService
from app.services.timer_service import TimerService


router = APIRouter(tags=["ws"])
settings = get_settings()
room_repository = RoomRepository(redis_client)
timer_service = TimerService()
room_service = RoomService(
    repository=room_repository,
    default_time_limit_sec=settings.default_time_limit_sec,
    default_increment_sec=settings.default_increment_sec,
    max_active_rooms=settings.max_active_rooms,
    waiting_room_ttl_sec=settings.waiting_room_ttl_sec,
    finished_room_ttl_sec=settings.finished_room_ttl_sec,
    playing_room_disconnect_ttl_sec=settings.playing_room_disconnect_ttl_sec,
    abandonment_grace_ms=settings.abandonment_grace_ms,
)
game_service = GameService(repository=room_repository, timer_service=timer_service)
connection_manager = ConnectionManager(max_active_connections=settings.max_active_connections)


async def _broadcast_eval_result(
    room_id: str, fen: str, move_sequence: int, eval_payload: dict[str, Any],
) -> None:
    msg = _build_position_eval_message(room_id, {
        "fen": fen,
        "moveSequence": move_sequence,
        "eval": eval_payload,
    })
    await connection_manager.broadcast(room_id, msg)


eval_queue_service = EvalQueueService(
    budget_ms=settings.eval_queue_budget_ms,
    engine_evaluate_fn=engine_service.evaluate,
    broadcast_fn=_broadcast_eval_result,
    enable_complexity_bucket=settings.enable_complexity_bucket,
    enable_queue_supersede=settings.enable_queue_supersede,
    baseline_fixed_ms=settings.baseline_fixed_ms,
)

clock_task: asyncio.Task | None = None
cleanup_task: asyncio.Task | None = None


def _notify_connection_topology(room_id: str | None) -> None:
    if not room_id:
        return
    room_service.after_connection_topology_change(
        room_id,
        timer_service.now_ms(),
        connection_manager,
    )


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    current_room: str | None = None
    await websocket.accept()
    try:
        while True:
            raw_message = await websocket.receive_json()
            event_name = _get_event_name(raw_message)
            room_id = raw_message.get("roomId")
            payload = raw_message.get("payload") or {}
            user_id = raw_message.get("userId") or payload.get("userId")
            nickname = raw_message.get("nickname") or payload.get("nickname") or "Guest"

            if event_name == "ping":
                await _send_message(websocket, {"event": "pong"})
                continue

            if not user_id:
                raise ValueError("userId is required.")

            if event_name == "room.create":
                validated = UserIdentityPayload(userId=user_id, nickname=nickname)
                room = room_service.create_room(user_id=validated.userId, nickname=validated.nickname)
                room_id = room["roomId"]
                await connection_manager.connect(websocket, room_id, validated.userId)
                current_room = room_id
                _notify_connection_topology(room_id)
                await _send_message(websocket, _build_room_created_message(room_id))
                continue

            if not room_id:
                raise ValueError("roomId is required.")

            validated_scope = RoomScopedUserPayload(userId=user_id, roomId=room_id)
            room_id = validated_scope.roomId
            user_id = validated_scope.userId

            if current_room != room_id:
                old_room = current_room
                connection_manager.disconnect(websocket, old_room)
                _notify_connection_topology(old_room)
                became_nonempty = await connection_manager.connect(websocket, room_id, user_id)
            else:
                became_nonempty = False
            current_room = room_id
            _notify_connection_topology(room_id)

            if event_name == "room.join":
                validated = RoomJoinPayload(userId=user_id, nickname=nickname, roomId=room_id)
                room = room_service.join_room(
                    room_id=validated.roomId,
                    user_id=validated.userId,
                    nickname=validated.nickname,
                )
                await _broadcast_room_state(room, room_id)
                continue

            if event_name == "room.sync":
                room = room_service.sync_room(room_id=room_id)
                await _send_sync_messages(websocket, room)
                continue

            if event_name == "room.leave":
                room = room_service.leave_room(room_id=room_id, user_id=user_id)
                if room is None:
                    await connection_manager.broadcast(
                        room_id,
                        _build_player_left_message("Host has left the room."),
                    )
                    connection_manager.disconnect(websocket, room_id)
                else:
                    connection_manager.disconnect(websocket, room_id)
                    await _broadcast_room_state(room, room_id)
                _notify_connection_topology(room_id)
                continue

            if event_name == "room.settings.update":
                validated = RoomSettingsUpdatePayload(
                    userId=user_id,
                    roomId=room_id,
                    settings=payload,
                )
                room = room_service.update_settings(
                    room_id=validated.roomId,
                    user_id=validated.userId,
                    settings=validated.settings.model_dump(),
                )
                await _broadcast_room_state(room, room_id)
                continue

            if event_name == "room.return":
                room = room_service.return_to_lobby(room_id=room_id, user_id=user_id)
                await _broadcast_room_state(room, room_id)
                continue

            if event_name == "game.start":
                room = room_service.start_game(
                    room_id=room_id,
                    user_id=user_id,
                    now_ms=timer_service.now_ms(),
                )
                start_message = _build_game_started_message(room)
                await connection_manager.broadcast(room_id, start_message)
                _schedule_broadcast_position_eval(
                    room_id,
                    str(room["fen"]),
                    int(room["moveSequence"]),
                )
                continue

            if event_name == "game.move":
                move_payload = payload.get("move") or {
                    "source": payload.get("source") or payload.get("from"),
                    "target": payload.get("target") or payload.get("to"),
                    "promotion": payload.get("promotion"),
                }
                result = game_service.apply_move(
                    room_id=room_id,
                    user_id=user_id,
                    move_data=move_payload,
                )
                await connection_manager.broadcast(
                    room_id,
                    _build_move_applied_message(result, room_id),
                )
                _schedule_broadcast_position_eval(
                    room_id,
                    str(result["fen"]),
                    int(result["moveSequence"]),
                )
                continue

            if event_name == "game.resign":
                result = game_service.resign_game(room_id=room_id, user_id=user_id)
                await connection_manager.broadcast(
                    room_id,
                    _build_game_over_message(result, room_id),
                )
                _schedule_broadcast_position_eval(
                    room_id,
                    str(result["fen"]),
                    int(result["moveSequence"]),
                )
                continue

            raise ValueError(f"Unsupported event: {event_name}")
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, current_room)
        _notify_connection_topology(current_room)
    except Exception as exc:
        await _send_message(websocket, _build_error_message(str(exc)))
        connection_manager.disconnect(websocket, current_room)
        _notify_connection_topology(current_room)


@router.on_event("startup")
async def startup_clock_monitor() -> None:
    global cleanup_task, clock_task
    if clock_task is None or clock_task.done():
        clock_task = asyncio.create_task(_clock_monitor_loop())
    if cleanup_task is None or cleanup_task.done():
        cleanup_task = asyncio.create_task(_room_cleanup_loop())
    eval_queue_service.start()


@router.on_event("shutdown")
async def shutdown_clock_monitor() -> None:
    global cleanup_task, clock_task
    await eval_queue_service.stop()
    if clock_task:
        clock_task.cancel()
        try:
            await clock_task
        except asyncio.CancelledError:
            pass
        clock_task = None
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        cleanup_task = None


def _build_position_eval_message(room_id: str, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": "game.eval",
        "roomId": room_id,
        "payload": {
            "fen": body["fen"],
            "moveSequence": body["moveSequence"],
            "eval": body.get("eval"),
        },
    }


def _schedule_broadcast_position_eval(room_id: str, fen: str, move_sequence: int) -> None:
    eval_queue_service.enqueue(room_id, fen, move_sequence)


def _schedule_personal_position_eval(
    websocket: WebSocket,
    room_id: str,
    fen: str,
    move_sequence: int,
) -> None:
    async def _run() -> None:
        try:
            eval_result = await eval_fen_for_personal(str(fen))
            msg = _build_position_eval_message(room_id, {
                "fen": fen,
                "moveSequence": move_sequence,
                "eval": eval_result,
            })
            await _send_message(websocket, msg)
        except Exception:
            logger.exception("Personal position eval failed room=%s", room_id)

    asyncio.create_task(_run())


def _get_event_name(message: dict[str, Any]) -> str:
    event = message.get("event")
    if not event or not isinstance(event, str):
        raise ValueError("event field is required.")
    return event


def _build_room_created_message(room_id: str) -> dict[str, Any]:
    return {"event": "room.created", "roomId": room_id, "payload": {"roomId": room_id}}


def _build_room_state_message(room: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": "room.state",
        "roomId": room["roomId"],
        "payload": room_service.build_room_state(room),
    }


def _build_game_started_message(room: dict[str, Any]) -> dict[str, Any]:
    server_now_ms = room.get("serverNowMs", room.get("lastMoveAt", 0))
    payload = {
        "roomId": room["roomId"],
        "fen": room["fen"],
        "hostColor": room["hostColor"],
        "guestColor": room["guestColor"],
        "turn": room["turn"],
        "moveSequence": room["moveSequence"],
        "whiteTimeMs": room["whiteTimeMs"],
        "blackTimeMs": room["blackTimeMs"],
        "serverNowMs": server_now_ms,
    }
    return {"event": "game.started", "roomId": room["roomId"], "payload": payload}


def _build_move_applied_message(result: dict[str, Any], room_id: str) -> dict[str, Any]:
    return {"event": "game.move_applied", "roomId": room_id, "payload": result}


def _build_error_message(message: str) -> dict[str, Any]:
    return {"event": "error", "payload": {"code": "BAD_REQUEST", "message": message}}


def _build_player_left_message(message: str) -> dict[str, Any]:
    return {"event": "player.left", "payload": {"message": message}}


async def _broadcast_room_state(room: dict[str, Any], room_id: str) -> None:
    await connection_manager.broadcast(room_id, _build_room_state_message(room))


def _build_game_over_message(result: dict[str, Any], room_id: str) -> dict[str, Any]:
    return {"event": "game.over", "roomId": room_id, "payload": result}


async def _send_message(websocket: WebSocket, message: dict[str, Any]) -> None:
    await connection_manager.send_personal_message(websocket, message)


async def _send_sync_messages(websocket: WebSocket, room: dict[str, Any]) -> None:
    await _send_message(websocket, _build_room_state_message(room))
    if room["status"] in {"playing", "finished"}:
        snapshot = game_service.build_live_game_snapshot(room["roomId"], now_ms=timer_service.now_ms())
        message = (
            _build_game_started_message(snapshot["payload"])
            if snapshot["type"] == "started"
            else _build_game_over_message(snapshot["payload"], room["roomId"])
        )
        await _send_message(websocket, message)
        inner = message.get("payload") or {}
        if isinstance(inner.get("fen"), str) and inner.get("moveSequence") is not None:
            _schedule_personal_position_eval(
                websocket,
                room["roomId"],
                str(inner["fen"]),
                int(inner["moveSequence"]),
            )


async def _clock_monitor_loop() -> None:
    try:
        while True:
            await asyncio.sleep(0.25)
            room_ids = list(connection_manager.active_connections.keys())
            if not room_ids:
                continue

            now_ms = timer_service.now_ms()
            for room_id in room_ids:
                result = game_service.check_timeout(room_id, now_ms=now_ms)
                if not result:
                    result = game_service.check_abandonment(room_id, now_ms, connection_manager)
                if not result:
                    continue
                await connection_manager.broadcast(
                    room_id,
                    _build_game_over_message(result, room_id),
                )
                _schedule_broadcast_position_eval(
                    room_id,
                    str(result["fen"]),
                    int(result["moveSequence"]),
                )
    except asyncio.CancelledError:
        raise


async def _room_cleanup_loop() -> None:
    try:
        while True:
            await asyncio.sleep(60)
            deleted_room_ids = room_service.cleanup_stale_rooms(
                now_ms=timer_service.now_ms(),
                room_connection_count=connection_manager.room_connection_count,
            )
            for room_id in deleted_room_ids:
                stale_connections = list(connection_manager.active_connections.get(room_id, []))
                for ws in stale_connections:
                    connection_manager.disconnect(ws, room_id)
    except asyncio.CancelledError:
        raise
