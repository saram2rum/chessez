"""브라우저 세션 복구: Redis의 user_room 바인딩 기준 활성 방 조회."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.redis import redis_client
from app.repositories.room_repository import RoomRepository
from app.schemas.room import MAX_USER_ID_LENGTH


router = APIRouter(prefix="/api/session", tags=["session"])
room_repository = RoomRepository(redis_client)


class ActiveRoomResponse(BaseModel):
    roomId: str | None = Field(default=None, description="현재 유저가 속한 방 코드, 없으면 null.")


@router.get("/active-room", response_model=ActiveRoomResponse)
def get_active_room(
    userId: str = Query(..., min_length=1, max_length=MAX_USER_ID_LENGTH),
) -> ActiveRoomResponse:
    normalized = str(userId).strip()[:MAX_USER_ID_LENGTH]
    if not normalized or any(c in "\r\n\t" for c in normalized):
        raise HTTPException(status_code=400, detail="Invalid userId.")

    bound = room_repository.get_user_room(normalized)
    if not bound:
        return ActiveRoomResponse(roomId=None)

    room = room_repository.get_room(bound)
    if not room:
        room_repository.unbind_user_room(normalized)
        return ActiveRoomResponse(roomId=None)

    return ActiveRoomResponse(roomId=bound)
