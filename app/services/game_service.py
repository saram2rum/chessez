from typing import Any

import chess

from app.repositories.room_repository import RoomRepository
from app.services.connection_manager import ConnectionManager
from app.schemas.game import GameOverPayload, MoveAppliedPayload, MovePayload
from app.services.timer_service import TimerService
from app.utils.board_snapshot import board_to_snapshot


class GameService:
    def __init__(self, repository: RoomRepository, timer_service: TimerService) -> None:
        self.repository = repository
        self.timer_service = timer_service

    def apply_move(self, room_id: str, user_id: str, move_data: dict[str, Any]) -> dict[str, Any]:
        room = self._get_room_or_raise(room_id)
        if room["status"] != "playing":
            raise ValueError("Game has not started.")

        mover_color = self._resolve_user_color(room, user_id)
        if mover_color != room["turn"]:
            raise PermissionError("It is not your turn.")

        move_payload = MovePayload(**move_data)
        promotion = self._normalize_promotion(move_payload.promotion)
        move_uci = f"{move_payload.source}{move_payload.target}{promotion}"

        board = chess.Board(room["fen"])
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError as exc:
            raise ValueError("Invalid move format.") from exc

        now_ms = self.timer_service.now_ms()
        white_time, black_time, timeout_color = self.timer_service.apply_elapsed(room, now_ms)
        if timeout_color:
            game_over_payload = self._finalize_timeout(room, board, white_time, black_time, now_ms)
            return MoveAppliedPayload(
                fen=game_over_payload["fen"],
                move=move_payload,
                nextTurn=game_over_payload["nextTurn"],
                moveSequence=game_over_payload["moveSequence"],
                whiteTimeMs=game_over_payload["whiteTimeMs"],
                blackTimeMs=game_over_payload["blackTimeMs"],
                isCheck=game_over_payload["isCheck"],
                isGameOver=True,
                winner=game_over_payload["winner"],
                endReason=game_over_payload["endReason"],
                boardSnapshot=game_over_payload["boardSnapshot"],
                serverNowMs=game_over_payload["serverNowMs"],
            ).model_dump()

        if move not in board.legal_moves:
            raise ValueError("Illegal move.")

        board.push(move)
        white_time, black_time = self.timer_service.apply_increment(
            white_time,
            black_time,
            mover_color,
            int(room["settings"].get("incrementSec", 0)),
        )

        next_turn = "WHITE" if board.turn == chess.WHITE else "BLACK"
        is_game_over = board.is_game_over()
        winner = None
        end_reason = None

        if is_game_over:
            outcome = board.outcome()
            if outcome and outcome.winner == chess.WHITE:
                winner = "WHITE"
            elif outcome and outcome.winner == chess.BLACK:
                winner = "BLACK"
            else:
                winner = "DRAW"

            if board.is_checkmate():
                end_reason = "Checkmate"
            elif board.is_stalemate():
                end_reason = "Stalemate"
            elif board.is_repetition():
                end_reason = "Threefold Repetition"
            elif board.is_insufficient_material():
                end_reason = "Insufficient Material"
            else:
                end_reason = "Game Over"

        room["fen"] = board.fen()
        room["turn"] = next_turn
        room["moveSequence"] = int(room.get("moveSequence") or 0) + 1
        room["whiteTimeMs"] = white_time
        room["blackTimeMs"] = black_time
        room["lastMoveAt"] = now_ms
        room["lastActivityAt"] = now_ms
        room["winner"] = winner
        room["endReason"] = end_reason
        room["status"] = "finished" if is_game_over else "playing"
        room["finishedAt"] = now_ms if is_game_over else 0
        self.repository.save_room(room)

        return MoveAppliedPayload(
            fen=room["fen"],
            move=move_payload,
            nextTurn=next_turn,
            moveSequence=room["moveSequence"],
            whiteTimeMs=white_time,
            blackTimeMs=black_time,
            isCheck=board.is_check(),
            isGameOver=is_game_over,
            winner=winner,
            endReason=end_reason,
            boardSnapshot=board_to_snapshot(board),
            serverNowMs=now_ms,
        ).model_dump()

    def resign_game(self, room_id: str, user_id: str) -> dict[str, Any]:
        room = self._get_room_or_raise(room_id)
        if room["status"] != "playing":
            raise ValueError("Game has not started.")

        resigning_color = self._resolve_user_color(room, user_id)
        winner = "BLACK" if resigning_color == "WHITE" else "WHITE"
        board = chess.Board(room["fen"])
        now_ms = self.timer_service.now_ms()
        white_time, black_time, _ = self.timer_service.apply_elapsed(room, now_ms)

        room["status"] = "finished"
        room["winner"] = winner
        room["endReason"] = "Resignation"
        room["whiteTimeMs"] = white_time
        room["blackTimeMs"] = black_time
        room["lastMoveAt"] = now_ms
        room["lastActivityAt"] = now_ms
        room["finishedAt"] = now_ms
        self.repository.save_room(room)

        return GameOverPayload(
            fen=room["fen"],
            nextTurn=room["turn"],
            moveSequence=room["moveSequence"],
            whiteTimeMs=white_time,
            blackTimeMs=black_time,
            isCheck=board.is_check(),
            winner=winner,
            endReason="Resignation",
            boardSnapshot=board_to_snapshot(board),
            serverNowMs=now_ms,
        ).model_dump()

    def check_timeout(self, room_id: str, now_ms: int | None = None) -> dict[str, Any] | None:
        room = self.repository.get_room(room_id)
        if not room or room["status"] != "playing":
            return None

        now_ms = now_ms if now_ms is not None else self.timer_service.now_ms()
        board = chess.Board(room["fen"])
        white_time, black_time, timeout_color = self.timer_service.apply_elapsed(room, now_ms)
        if not timeout_color:
            return None

        return self._finalize_timeout(room, board, white_time, black_time, now_ms)

    def check_abandonment(
        self,
        room_id: str,
        now_ms: int,
        connection_manager: ConnectionManager,
    ) -> dict[str, Any] | None:
        room = self.repository.get_room(room_id)
        if not room or room["status"] != "playing":
            return None

        deadline = int(room.get("abandonDeadlineMs") or 0)
        abandon_uid = room.get("abandonForUserId")
        if not deadline or not abandon_uid or now_ms < deadline:
            return None

        if connection_manager.user_connection_count(room_id, abandon_uid) > 0:
            room["abandonDeadlineMs"] = 0
            room["abandonForUserId"] = None
            self.repository.save_room(room)
            return None

        host_id = room["hostUserId"]
        guest_id = room.get("guestUserId")
        if not guest_id:
            return None

        other_id = guest_id if abandon_uid == host_id else host_id
        if connection_manager.user_connection_count(room_id, other_id) == 0:
            return None

        return self._finalize_abandonment(room, abandon_uid, now_ms)

    def build_live_game_snapshot(self, room_id: str, now_ms: int | None = None) -> dict[str, Any]:
        room = self._get_room_or_raise(room_id)
        now_ms = now_ms if now_ms is not None else self.timer_service.now_ms()

        if room["status"] == "playing":
            white_time, black_time, timeout_color = self.timer_service.apply_elapsed(room, now_ms)
            board = chess.Board(room["fen"])
            if timeout_color:
                return self._finalize_timeout(room, board, white_time, black_time, now_ms)

            return {
                "type": "started",
                "payload": {
                    "roomId": room["roomId"],
                    "fen": room["fen"],
                    "hostColor": room["hostColor"],
                    "guestColor": room["guestColor"],
                    "turn": room["turn"],
                    "moveSequence": room["moveSequence"],
                    "whiteTimeMs": white_time,
                    "blackTimeMs": black_time,
                    "serverNowMs": now_ms,
                },
            }

        board = chess.Board(room["fen"])
        return {
            "type": "over",
            "payload": GameOverPayload(
                fen=room["fen"],
                nextTurn=room["turn"],
                moveSequence=room["moveSequence"],
                whiteTimeMs=room["whiteTimeMs"],
                blackTimeMs=room["blackTimeMs"],
                isCheck=board.is_check(),
                winner=room["winner"],
                endReason=room["endReason"],
                boardSnapshot=board_to_snapshot(board),
                serverNowMs=now_ms,
            ).model_dump(),
        }

    def _get_room_or_raise(self, room_id: str) -> dict[str, Any]:
        room = self.repository.get_room(room_id)
        if not room:
            raise ValueError("Room not found.")
        return room

    @staticmethod
    def _resolve_user_color(room: dict[str, Any], user_id: str) -> str:
        if user_id == room["hostUserId"]:
            return room["hostColor"]
        if user_id == room.get("guestUserId"):
            return room["guestColor"]
        raise PermissionError("You are not part of this room.")

    @staticmethod
    def _normalize_promotion(promotion: str | None) -> str:
        if not promotion:
            return ""
        lowered = promotion.lower()
        if lowered in {"q", "r", "b", "n"}:
            return lowered
        return lowered[0]

    def _finalize_abandonment(
        self,
        room: dict[str, Any],
        abandoning_user_id: str,
        now_ms: int,
    ) -> dict[str, Any]:
        loser_color = self._resolve_user_color(room, abandoning_user_id)
        winner = "BLACK" if loser_color == "WHITE" else "WHITE"
        board = chess.Board(room["fen"])
        white_time, black_time, _ = self.timer_service.apply_elapsed(room, now_ms)
        room["status"] = "finished"
        room["winner"] = winner
        room["endReason"] = "Abandonment"
        room["whiteTimeMs"] = white_time
        room["blackTimeMs"] = black_time
        room["lastMoveAt"] = now_ms
        room["lastActivityAt"] = now_ms
        room["finishedAt"] = now_ms
        room["abandonDeadlineMs"] = 0
        room["abandonForUserId"] = None
        self.repository.save_room(room)
        return GameOverPayload(
            fen=room["fen"],
            nextTurn=room["turn"],
            moveSequence=room["moveSequence"],
            whiteTimeMs=white_time,
            blackTimeMs=black_time,
            isCheck=board.is_check(),
            winner=winner,
            endReason="Abandonment",
            boardSnapshot=board_to_snapshot(board),
            serverNowMs=now_ms,
        ).model_dump()

    def _finalize_timeout(
        self,
        room: dict[str, Any],
        board: chess.Board,
        white_time: int,
        black_time: int,
        now_ms: int,
    ) -> dict[str, Any]:
        timeout_color = "WHITE" if white_time == 0 else "BLACK"
        winner = "BLACK" if timeout_color == "WHITE" else "WHITE"
        room["status"] = "finished"
        room["winner"] = winner
        room["endReason"] = "Time Out"
        room["whiteTimeMs"] = white_time
        room["blackTimeMs"] = black_time
        room["lastMoveAt"] = now_ms
        room["lastActivityAt"] = now_ms
        room["finishedAt"] = now_ms
        self.repository.save_room(room)
        return GameOverPayload(
            fen=room["fen"],
            nextTurn=room["turn"],
            moveSequence=room["moveSequence"],
            whiteTimeMs=white_time,
            blackTimeMs=black_time,
            isCheck=board.is_check(),
            winner=winner,
            endReason="Time Out",
            boardSnapshot=board_to_snapshot(board),
            serverNowMs=now_ms,
        ).model_dump()
