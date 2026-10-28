"""온라인 대국: 국면당 서버에서 한 번만 Stockfish 평가 후 WS 페이로드에 실음 (SSOT).

브로드캐스트 평가는 EvalQueueService가 담당하며,
이 모듈은 개인 sync 평가(_schedule_personal_position_eval)용으로 유지한다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.config import get_settings
from app.routers.ai import engine_service

logger = logging.getLogger(__name__)


async def eval_fen_for_personal(fen: str) -> dict[str, Any] | None:
    """개인(sync) 평가용. 고정 검색 시간 기반."""
    if not fen or not str(fen).strip():
        return None
    settings = get_settings()
    search_ms = max(
        1,
        min(int(settings.game_online_eval_search_ms), settings.ai_eval_max_search_ms),
    )
    try:
        raw = await asyncio.to_thread(engine_service.evaluate, fen, search_ms)
    except Exception as exc:
        logger.warning("Personal position eval failed: %s", exc)
        return None
    try:
        ev_type = raw.get("type")
        if ev_type not in ("cp", "mate"):
            return None
        return {"type": ev_type, "value": int(raw["value"])}
    except (KeyError, TypeError, ValueError):
        logger.warning("Personal position eval malformed response: %s", raw)
        return None
