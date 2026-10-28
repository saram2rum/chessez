import random
import time
from typing import Any, Callable

import chess

from app.repositories.room_repository import RoomRepository
from app.services.connection_manager import ConnectionManager
from app.schemas.room import RoomPlayer, RoomSettings, RoomStatePayload
from app.utils.room_id import generate_room_id


class RoomService:
    def __init__(
        self,
        repository: RoomRepository,
        default_time_limit_sec: int,
        default_increment_sec: int,
        max_active_rooms: int,
        waiting_room_ttl_sec: int,
        finished_room_ttl_sec: int,
        playing_room_disconnect_ttl_sec: int = 60,
        abandonment_grace_ms: int = 60_000,
    ) -> None:
        self.repository = repository
        self.default_time_limit_sec = default_time_limit_sec
        self.default_increment_sec = default_increment_sec
        self.max_active_rooms = max_active_rooms
        self.waiting_room_ttl_ms = waiting_room_ttl_sec * 1000
        self.finished_room_ttl_ms = finished_room_ttl_sec * 1000
        self.playing_room_disconnect_ttl_ms = playing_room_disconnect_ttl_sec * 1000
        self.abandonment_grace_ms = abandonment_grace_ms

    def create_room(self, user_id: str, nickname: str) -> dict[str, Any]:
        existing_room_id = self.repository.get_user_room(user_id)
        if existing_room_id:
            existing_room = self.repository.get_room(existing_room_id)
            if existing_room:
                return existing_room

        if self.repository.count_rooms({"waiting", "playing"}) >= self.max_active_rooms:
            raise RuntimeError("Too many active rooms. Please try again later.")

        room_id = self._generate_unique_room_id()
        now_ms = self._now_ms()
        room = {
            "roomId": room_id,
            "status": "waiting",
            "hostUserId": user_id,
            "guestUserId": None,
            "hostNickname": nickname,
            "guestNickname": None,
            "hostColor": None,
            "guestColor": None,
            "turn": "WHITE",
            "fen": chess.STARTING_FEN,
            "moveSequence": 0,
            "whiteTimeMs": self.default_time_limit_sec * 1000,
            "blackTimeMs": self.default_time_limit_sec * 1000,
            "lastMoveAt": 0,
            "lastActivityAt": now_ms,
            "finishedAt": 0,
            "allDisconnectedAt": 0,
            "abandonDeadlineMs": 0,
            "abandonForUserId": None,
            "winner": None,
            "endReason": None,
            "settings": RoomSettings(
                startingSide="RANDOM",
                timeLimitSec=self.default_time_limit_sec,
                incrementSec=self.default_increment_sec,
            ).model_dump(),
        }
        self.repository.save_room(room)
        self.repository.bind_user_to_room(user_id, room_id)
        return room

    def join_room(self, room_id: str, user_id: str, nickname: str) -> dict[str, Any]:
        room = self._get_room_or_raise(room_id)

        if user_id == room["hostUserId"]:
            room["hostNickname"] = nickname
        elif user_id == room.get("guestUserId"):
            room["guestNickname"] = nickname
        elif room.get("guestUserId") is None:
            room["guestUserId"] = user_id
            room["guestNickname"] = nickname
        else:
            raise ValueError("Room is full.")

        room["lastActivityAt"] = self._now_ms()
        self.repository.save_room(room)
        self.repository.bind_user_to_room(user_id, room_id)
        return room

    def sync_room(self, room_id: str) -> dict[str, Any]:
        room = self._get_room_or_raise(room_id)
        room["lastActivityAt"] = self._now_ms()
        self.repository.save_room(room)
        return room

    def update_settings(self, room_id: str, user_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        room = self._get_room_or_raise(room_id)
        if room["hostUserId"] != user_id:
            raise PermissionError("Only the host can update room settings.")
        if room["status"] != "waiting":
            raise ValueError("Room settings can only be changed before the game starts.")

        current_settings = RoomSettings(**room["settings"])
        merged = current_settings.model_copy(update=settings)
        room["settings"] = merged.model_dump()
        room["lastActivityAt"] = self._now_ms()
        self.repository.save_room(room)
        return room

    def start_game(self, room_id: str, user_id: str, now_ms: int) -> dict[str, Any]:
        room = self._get_room_or_raise(room_id)
        if room["hostUserId"] != user_id:
            raise PermissionError("Only the host can start the game.")
        if not room.get("guestUserId"):
            raise ValueError("You need an opponent to start the game.")

        settings = RoomSettings(**room["settings"])
        host_color, guest_color = self._resolve_colors(settings.startingSide)
        time_ms = settings.timeLimitSec * 1000

        room["status"] = "playing"
        room["fen"] = chess.STARTING_FEN
        room["turn"] = "WHITE"
        room["moveSequence"] = 0
        room["hostColor"] = host_color
        room["guestColor"] = guest_color
        room["whiteTimeMs"] = time_ms
        room["blackTimeMs"] = time_ms
        room["lastMoveAt"] = now_ms
        room["lastActivityAt"] = now_ms
        room["finishedAt"] = 0
        room["allDisconnectedAt"] = 0
        room["abandonDeadlineMs"] = 0
        room["abandonForUserId"] = None
        room["winner"] = None
        room["endReason"] = None

        self.repository.save_room(room)
        return room

    def after_connection_topology_change(
        self,
        room_id: str,
        now_ms: int,
        connection_manager: ConnectionManager,
    ) -> None:
        """WS 연결 수 변화 반영: 전원 끊김, 부분 끊김 유기 타이머, 복귀 시 유기 해제."""
        room = self.repository.get_room(room_id)
        if not room:
            return

        if room.get("status") != "playing":
            if int(room.get("abandonDeadlineMs") or 0) or room.get("abandonForUserId"):
                room["abandonDeadlineMs"] = 0
                room["abandonForUserId"] = None
                self.repository.save_room(room)
            return

        total = connection_manager.room_connection_count(room_id)
        if total == 0:
            room["allDisconnectedAt"] = now_ms
            room["abandonDeadlineMs"] = 0
            room["abandonForUserId"] = None
            room["lastActivityAt"] = now_ms
            self.repository.save_room(room)
            return

        room["allDisconnectedAt"] = 0
        host_id = room["hostUserId"]
        guest_id = room.get("guestUserId")
        if not guest_id:
            room["abandonDeadlineMs"] = 0
            room["abandonForUserId"] = None
            self.repository.save_room(room)
            return

        hc = connection_manager.user_connection_count(room_id, host_id)
        gc = connection_manager.user_connection_count(room_id, guest_id)

        if hc == 0 and gc > 0:
            room["abandonForUserId"] = host_id
            room["abandonDeadlineMs"] = now_ms + self.abandonment_grace_ms
        elif gc == 0 and hc > 0:
            room["abandonForUserId"] = guest_id
            room["abandonDeadlineMs"] = now_ms + self.abandonment_grace_ms
        else:
            room["abandonDeadlineMs"] = 0
            room["abandonForUserId"] = None

        room["lastActivityAt"] = now_ms
        self.repository.save_room(room)

    def leave_room(self, room_id: str, user_id: str) -> dict[str, Any] | None:
        room = self._get_room_or_raise(room_id)

        if user_id == room["hostUserId"]:
            if room.get("guestUserId"):
                self.repository.unbind_user_room(room["guestUserId"])
            self.repository.unbind_user_room(room["hostUserId"])
            self.repository.delete_room(room_id)
            return None

        if user_id == room.get("guestUserId"):
            room["guestUserId"] = None
            room["guestNickname"] = None
            room["guestColor"] = None
            room["status"] = "waiting"
            room["fen"] = chess.STARTING_FEN
            room["turn"] = "WHITE"
            room["moveSequence"] = 0
            room["winner"] = None
            room["endReason"] = None
            settings = RoomSettings(**room["settings"])
            room["whiteTimeMs"] = settings.timeLimitSec * 1000
            room["blackTimeMs"] = settings.timeLimitSec * 1000
            room["lastMoveAt"] = 0
            room["lastActivityAt"] = self._now_ms()
            room["finishedAt"] = 0
            room["allDisconnectedAt"] = 0
            room["abandonDeadlineMs"] = 0
            room["abandonForUserId"] = None
            self.repository.save_room(room)
            self.repository.unbind_user_room(user_id)
            return room

        raise PermissionError("You are not part of this room.")

    def return_to_lobby(self, room_id: str, user_id: str) -> dict[str, Any]:
        room = self._get_room_or_raise(room_id)
        if user_id not in {room["hostUserId"], room.get("guestUserId")}:
            raise PermissionError("You are not part of this room.")

        settings = RoomSettings(**room["settings"])
        room["status"] = "waiting"
        room["fen"] = chess.STARTING_FEN
        room["turn"] = "WHITE"
        room["moveSequence"] = 0
        room["winner"] = None
        room["endReason"] = None
        room["hostColor"] = None
        room["guestColor"] = None
        room["whiteTimeMs"] = settings.timeLimitSec * 1000
        room["blackTimeMs"] = settings.timeLimitSec * 1000
        room["lastMoveAt"] = 0
        room["lastActivityAt"] = self._now_ms()
        room["finishedAt"] = 0
        room["allDisconnectedAt"] = 0
        room["abandonDeadlineMs"] = 0
        room["abandonForUserId"] = None
        self.repository.save_room(room)
        return room

    def cleanup_stale_rooms(self, now_ms: int, room_connection_count: Callable[[str], int]) -> list[str]:
        deleted_room_ids: list[str] = []
        for room in self.repository.list_rooms():
            room_id = room["roomId"]
            if room_connection_count(room_id) > 0:
                continue

            status = room.get("status")
            last_activity_at = int(room.get("lastActivityAt") or 0)
            finished_at = int(room.get("finishedAt") or 0)

            if status == "finished" and finished_at and now_ms - finished_at >= self.finished_room_ttl_ms:
                self._delete_room(room)
                deleted_room_ids.append(room_id)
                continue

            if (
                status == "waiting"
                and room.get("guestUserId") is None
                and last_activity_at
                and now_ms - last_activity_at >= self.waiting_room_ttl_ms
            ):
                self._delete_room(room)
                deleted_room_ids.append(room_id)
                continue

            if status == "playing":
                all_disconnected_at = int(room.get("allDisconnectedAt") or 0)
                if (
                    all_disconnected_at
                    and now_ms - all_disconnected_at >= self.playing_room_disconnect_ttl_ms
                ):
                    self._delete_room(room)
                    deleted_room_ids.append(room_id)

        return deleted_room_ids

    def build_room_state(self, room: dict[str, Any]) -> dict[str, Any]:
        payload = RoomStatePayload(
            roomId=room["roomId"],
            status=room["status"],
            host=RoomPlayer(userId=room["hostUserId"], nickname=room["hostNickname"]),
            guest=RoomPlayer(userId=room.get("guestUserId"), nickname=room.get("guestNickname")),
            settings=RoomSettings(**room["settings"]),
        )
        return payload.model_dump()

    def _generate_unique_room_id(self) -> str:
        for _ in range(20):
            room_id = generate_room_id()
            if not self.repository.room_exists(room_id):
                return room_id
        raise RuntimeError("Could not generate a unique room id.")

    def _get_room_or_raise(self, room_id: str) -> dict[str, Any]:
        room = self.repository.get_room(room_id)
        if not room:
            raise ValueError("Room not found.")
        return room

    def _delete_room(self, room: dict[str, Any]) -> None:
        if room.get("hostUserId"):
            self.repository.unbind_user_room(room["hostUserId"])
        if room.get("guestUserId"):
            self.repository.unbind_user_room(room["guestUserId"])
        self.repository.delete_room(room["roomId"])

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _resolve_colors(starting_side: str) -> tuple[str, str]:
        if starting_side == "WHITE":
            return "WHITE", "BLACK"
        if starting_side == "BLACK":
            return "BLACK", "WHITE"
        return random.choice([("WHITE", "BLACK"), ("BLACK", "WHITE")])
