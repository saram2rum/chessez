import json
from typing import Any

import redis


class RoomRepository:
    def __init__(self, client: redis.Redis):
        self.client = client

    @staticmethod
    def room_key(room_id: str) -> str:
        return f"room:{room_id}"

    @staticmethod
    def user_room_key(user_id: str) -> str:
        return f"user_room:{user_id}"

    def room_exists(self, room_id: str) -> bool:
        return bool(self.client.exists(self.room_key(room_id)))

    def get_room(self, room_id: str) -> dict[str, Any] | None:
        raw = self.client.hgetall(self.room_key(room_id))
        if not raw:
            return None
        return self._deserialize_room(raw)

    def save_room(self, room: dict[str, Any]) -> None:
        payload = self._serialize_room(room)
        self.client.hset(self.room_key(room["roomId"]), mapping=payload)

    def delete_room(self, room_id: str) -> None:
        self.client.delete(self.room_key(room_id))

    def list_rooms(self) -> list[dict[str, Any]]:
        rooms: list[dict[str, Any]] = []
        for key in self.client.scan_iter(match="room:*"):
            raw = self.client.hgetall(key)
            if raw:
                rooms.append(self._deserialize_room(raw))
        return rooms

    def count_rooms(self, statuses: set[str] | None = None) -> int:
        if statuses is None:
            return len(self.list_rooms())
        return sum(1 for room in self.list_rooms() if room.get("status") in statuses)

    def bind_user_to_room(self, user_id: str, room_id: str) -> None:
        self.client.set(self.user_room_key(user_id), room_id)

    def get_user_room(self, user_id: str) -> str | None:
        return self.client.get(self.user_room_key(user_id))

    def unbind_user_room(self, user_id: str) -> None:
        self.client.delete(self.user_room_key(user_id))

    def _serialize_room(self, room: dict[str, Any]) -> dict[str, str]:
        payload: dict[str, str] = {}
        for key, value in room.items():
            if key == "settings":
                payload[key] = json.dumps(value)
            elif value is None:
                payload[key] = ""
            else:
                payload[key] = str(value)
        return payload

    def _deserialize_room(self, raw: dict[str, str]) -> dict[str, Any]:
        int_fields = {
            "moveSequence",
            "whiteTimeMs",
            "blackTimeMs",
            "lastMoveAt",
            "lastActivityAt",
            "finishedAt",
            "allDisconnectedAt",
            "abandonDeadlineMs",
        }
        room: dict[str, Any] = {}
        for key, value in raw.items():
            if key == "settings":
                room[key] = json.loads(value) if value else {}
            elif key in int_fields:
                room[key] = int(value or 0)
            else:
                room[key] = value or None
        room.setdefault("settings", {})
        room.setdefault("guestUserId", None)
        room.setdefault("guestNickname", None)
        room.setdefault("hostColor", None)
        room.setdefault("guestColor", None)
        room.setdefault("winner", None)
        room.setdefault("endReason", None)
        room.setdefault("lastActivityAt", 0)
        room.setdefault("finishedAt", 0)
        room.setdefault("allDisconnectedAt", 0)
        room.setdefault("abandonDeadlineMs", 0)
        room.setdefault("abandonForUserId", None)
        return room
