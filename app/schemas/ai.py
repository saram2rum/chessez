from typing import Optional

from pydantic import BaseModel, Field


class AiStatusResponse(BaseModel):
    available: bool
    activeRequests: int
    maxConcurrent: int


class AiMoveRequest(BaseModel):
    fen: str
    skillLevel: int = Field(default=10, ge=0, le=20)
    moveTimeMs: int = Field(default=500, ge=50, le=30_000)


class AiMoveResponse(BaseModel):
    bestMove: str
    fen: str


class AiEvaluateRequest(BaseModel):
    fen: str
    searchTimeMs: int = Field(default=500, ge=50, le=30_000)
    isTestMode: bool = False
    searchNodes: Optional[int] = Field(default=None, ge=1, le=100_000_000)


class AiEvaluateResponse(BaseModel):
    type: str
    value: int
    fen: str
    nodes: Optional[int] = None
    nps: Optional[int] = None
    depth: Optional[int] = None
