from __future__ import annotations

from enum import Enum
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


ROOM_ID_LENGTH = 5
MAX_USER_ID_LENGTH = 64
MAX_NICKNAME_LENGTH = 12
ROOM_ID_REGEX = re.compile(r"^[A-Z0-9]{5}$")


def sanitize_nickname(value: str) -> str:
    cleaned = "".join(char for char in str(value) if char.isprintable() and char not in "\r\n\t")
    return cleaned.strip()


class StartingSide(str, Enum):
    WHITE = "WHITE"
    BLACK = "BLACK"
    RANDOM = "RANDOM"


class RoomSettings(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    startingSide: StartingSide = StartingSide.RANDOM
    timeLimitSec: int = Field(default=600, ge=60, le=3600)
    incrementSec: int = Field(default=0, ge=0, le=60)


class RoomPlayer(BaseModel):
    userId: str | None = None
    nickname: str | None = None


class RoomStatePayload(BaseModel):
    roomId: str
    status: str
    host: RoomPlayer
    guest: RoomPlayer
    settings: RoomSettings


class UserIdentityPayload(BaseModel):
    userId: str = Field(min_length=1, max_length=MAX_USER_ID_LENGTH)
    nickname: str = Field(min_length=1, max_length=MAX_NICKNAME_LENGTH)

    @field_validator("userId")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("userId is required.")
        if any(char in "\r\n\t" for char in normalized):
            raise ValueError("userId contains invalid whitespace.")
        return normalized[:MAX_USER_ID_LENGTH]

    @field_validator("nickname")
    @classmethod
    def validate_nickname(cls, value: str) -> str:
        normalized = sanitize_nickname(value)
        if not normalized:
            raise ValueError("Nickname is required.")
        return normalized[:MAX_NICKNAME_LENGTH]


class RoomScopedUserPayload(BaseModel):
    userId: str = Field(min_length=1, max_length=MAX_USER_ID_LENGTH)
    roomId: str = Field(min_length=ROOM_ID_LENGTH, max_length=ROOM_ID_LENGTH)

    @field_validator("userId")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("userId is required.")
        if any(char in "\r\n\t" for char in normalized):
            raise ValueError("userId contains invalid whitespace.")
        return normalized[:MAX_USER_ID_LENGTH]

    @field_validator("roomId")
    @classmethod
    def validate_room_id(cls, value: str) -> str:
        normalized = str(value).strip().upper()
        if not ROOM_ID_REGEX.fullmatch(normalized):
            raise ValueError("roomId must be 5 uppercase letters or digits.")
        return normalized


class RoomJoinPayload(UserIdentityPayload):
    roomId: str = Field(min_length=ROOM_ID_LENGTH, max_length=ROOM_ID_LENGTH)

    @field_validator("roomId")
    @classmethod
    def validate_room_id(cls, value: str) -> str:
        normalized = str(value).strip().upper()
        if not ROOM_ID_REGEX.fullmatch(normalized):
            raise ValueError("roomId must be 5 uppercase letters or digits.")
        return normalized


class RoomSettingsUpdatePayload(RoomScopedUserPayload):
    settings: RoomSettings
