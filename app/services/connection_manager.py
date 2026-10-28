from fastapi import WebSocket


class ConnectionManager:
    def __init__(self, max_active_connections: int | None = None) -> None:
        self.active_connections: dict[str, list[WebSocket]] = {}
        self.max_active_connections = max_active_connections
        self._ws_registry: dict[int, tuple[str, str]] = {}
        self._room_user_counts: dict[str, dict[str, int]] = {}

    async def connect(self, websocket: WebSocket, room_id: str, user_id: str) -> bool:
        if (
            websocket not in self.active_connections.get(room_id, [])
            and self.max_active_connections is not None
        ):
            if self.connection_count() >= self.max_active_connections:
                raise RuntimeError("Server is full. Please try again later.")

        self.active_connections.setdefault(room_id, [])
        was_empty = len(self.active_connections[room_id]) == 0

        if websocket not in self.active_connections[room_id]:
            self.active_connections[room_id].append(websocket)
            self._ws_registry[id(websocket)] = (room_id, user_id)
            self._room_user_counts.setdefault(room_id, {})
            self._room_user_counts[room_id][user_id] = self._room_user_counts[room_id].get(user_id, 0) + 1

        return was_empty and len(self.active_connections[room_id]) > 0

    def disconnect(self, websocket: WebSocket, room_id: str | None) -> tuple[str, str] | None:
        key = id(websocket)
        meta = self._ws_registry.pop(key, None)
        actual_room: str | None
        user_id: str

        if meta:
            actual_room, user_id = meta
        elif room_id and room_id in self.active_connections:
            actual_room = room_id
            user_id = ""
        else:
            return None

        if actual_room not in self.active_connections:
            return meta

        if websocket in self.active_connections[actual_room]:
            self.active_connections[actual_room].remove(websocket)
            if meta and user_id:
                counts = self._room_user_counts.get(actual_room, {})
                if user_id in counts:
                    counts[user_id] -= 1
                    if counts[user_id] <= 0:
                        del counts[user_id]
                if not counts and actual_room in self._room_user_counts:
                    del self._room_user_counts[actual_room]

            if not self.active_connections[actual_room]:
                del self.active_connections[actual_room]

        return meta

    async def send_personal_message(self, websocket: WebSocket, message: dict) -> None:
        await websocket.send_json(message)

    async def broadcast(self, room_id: str, message: dict) -> None:
        for websocket in list(self.active_connections.get(room_id, [])):
            await websocket.send_json(message)

    def connection_count(self) -> int:
        return sum(len(connections) for connections in self.active_connections.values())

    def room_connection_count(self, room_id: str) -> int:
        return len(self.active_connections.get(room_id, []))

    def user_connection_count(self, room_id: str, user_id: str) -> int:
        return self._room_user_counts.get(room_id, {}).get(user_id, 0)
