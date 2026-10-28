from pydantic import BaseModel, Field, field_validator


class MovePayload(BaseModel):
    source: str = Field(min_length=2, max_length=2)
    target: str = Field(min_length=2, max_length=2)
    promotion: str | None = None

    @field_validator("source", "target")
    @classmethod
    def validate_square(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if len(normalized) != 2 or normalized[0] not in "abcdefgh" or normalized[1] not in "12345678":
            raise ValueError("Invalid board square.")
        return normalized

    @field_validator("promotion")
    @classmethod
    def validate_promotion(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = str(value).strip().lower()
        if normalized not in {"q", "r", "b", "n", "queen", "rook", "bishop", "knight"}:
            raise ValueError("Invalid promotion piece.")
        return normalized


class GameStartedPayload(BaseModel):
    roomId: str
    fen: str
    hostColor: str
    guestColor: str
    turn: str
    moveSequence: int
    whiteTimeMs: int
    blackTimeMs: int
    serverNowMs: int


class MoveAppliedPayload(BaseModel):
    fen: str
    move: MovePayload
    nextTurn: str
    moveSequence: int
    whiteTimeMs: int
    blackTimeMs: int
    isCheck: bool
    isGameOver: bool
    winner: str | None = None
    endReason: str | None = None
    boardSnapshot: dict[str, str]
    serverNowMs: int


class GameOverPayload(BaseModel):
    fen: str
    nextTurn: str
    moveSequence: int
    whiteTimeMs: int
    blackTimeMs: int
    isCheck: bool
    isGameOver: bool = True
    winner: str
    endReason: str
    boardSnapshot: dict[str, str]
    serverNowMs: int
