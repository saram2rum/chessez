from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.schemas.ai import (
    AiEvaluateRequest,
    AiEvaluateResponse,
    AiMoveRequest,
    AiMoveResponse,
    AiStatusResponse,
)
from app.services.engine_service import EngineService


router = APIRouter(prefix="/api/ai", tags=["ai"])
settings = get_settings()
engine_service = EngineService(
    engine_path=settings.stockfish_path,
    max_concurrent=settings.ai_max_concurrent,
    engine_threads=settings.ai_engine_threads,
    engine_hash_mb=settings.ai_engine_hash_mb,
    move_max_time_ms=settings.ai_move_max_time_ms,
    eval_min_search_ms=settings.ai_eval_min_search_ms,
    eval_max_search_ms=settings.ai_eval_max_search_ms,
)


@router.get("/status", response_model=AiStatusResponse)
def get_ai_status() -> AiStatusResponse:
    return AiStatusResponse(**engine_service.status())


@router.post("/move", response_model=AiMoveResponse)
def get_ai_move(request: AiMoveRequest) -> AiMoveResponse:
    try:
        best_move = engine_service.get_best_move(
            fen=request.fen,
            skill_level=request.skillLevel,
            move_time_ms=request.moveTimeMs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AiMoveResponse(bestMove=best_move, fen=request.fen)


@router.post("/evaluate", response_model=AiEvaluateResponse)
def evaluate_position(request: AiEvaluateRequest) -> AiEvaluateResponse:
    try:
        evaluation = engine_service.evaluate(
            fen=request.fen,
            search_time_ms=request.searchTimeMs,
            is_test_mode=request.isTestMode,
            search_nodes=request.searchNodes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AiEvaluateResponse(**evaluation)
