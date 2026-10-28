"""평가 요청 큐: 복잡도 기반 입장 제어 + 같은 방 Last-Move-Wins + 노드 기반 엔진 실행."""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.utils.position_complexity import (
    EVAL_NODES_BY_BUCKET,
    EVAL_TIME_BY_BUCKET,
    _NODES_PER_MS,
    evaluate_position_complexity,
)

logger = logging.getLogger(__name__)


@dataclass
class QueueItem:
    room_id: str
    fen: str
    move_sequence: int
    estimated_ms: int
    search_nodes: int
    bucket: str


class EvalQueueService:
    """비동기 평가 큐.

    * enqueue()로 요청을 받아 복잡도 분석 → 입장 제어 → 큐 삽입
    * 소비자 루프가 큐에서 꺼내 노드 기반 엔진 호출 → 브로드캐스트
    * 같은 room_id의 이전 요청은 자동 교체(Last-Move-Wins)
    """

    def __init__(
        self,
        budget_ms: int,
        engine_evaluate_fn: Callable[..., dict[str, Any]],
        broadcast_fn: Callable[[str, str, int, dict[str, Any]], Awaitable[None]],
        *,
        enable_complexity_bucket: bool = True,
        enable_queue_supersede: bool = True,
        baseline_fixed_ms: int = 288,
    ) -> None:
        self._budget_ms = max(1, budget_ms)
        self._engine_evaluate_fn = engine_evaluate_fn
        self._broadcast_fn = broadcast_fn

        self._enable_bucket = enable_complexity_bucket
        self._enable_supersede = enable_queue_supersede
        self._baseline_fixed_ms = max(1, baseline_fixed_ms)
        self._baseline_fixed_nodes = self._baseline_fixed_ms * _NODES_PER_MS

        self._queue: OrderedDict[str, QueueItem] = OrderedDict()
        self._queue_total_ms: float = 0
        self._seq = itertools.count()

        self._current_item: QueueItem | None = None
        self._current_start: float = 0

        self._wake: asyncio.Event = asyncio.Event()
        self._consumer_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(self._consumer_loop())
            bucket_label = "ON" if self._enable_bucket else "OFF"
            supersede_label = "ON" if self._enable_supersede else "OFF"
            logger.info(
                "[Ablation] Bucket: %s, Supersede: %s, budget=%dms, baseline_fixed=%dms",
                bucket_label,
                supersede_label,
                self._budget_ms,
                self._baseline_fixed_ms,
            )

    async def stop(self) -> None:
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None
        self._queue.clear()
        self._queue_total_ms = 0
        self._current_item = None
        logger.info("EvalQueueService stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, room_id: str, fen: str, move_sequence: int) -> bool:
        """입장 제어 후 큐에 삽입. 드랍되면 False 반환."""
        if self._enable_bucket:
            try:
                complexity = evaluate_position_complexity(fen)
            except (ValueError, Exception) as exc:
                logger.warning("Complexity analysis failed room=%s: %s", room_id, exc)
                return False

            if complexity.skip_eval:
                eval_payload = {"type": "cp", "value": complexity.skip_value}
                asyncio.create_task(
                    self._broadcast_fn(room_id, fen, move_sequence, eval_payload)
                )
                logger.debug(
                    "Eval skipped room=%s reason=%s value=%d",
                    room_id,
                    complexity.features.reason,
                    complexity.skip_value,
                )
                return True

            bucket = complexity.bucket
            estimated_ms = EVAL_TIME_BY_BUCKET[bucket]
            search_nodes = EVAL_NODES_BY_BUCKET[bucket]
        else:
            bucket = "FIXED"
            estimated_ms = self._baseline_fixed_ms
            search_nodes = self._baseline_fixed_nodes

        if self._enable_supersede and room_id in self._queue:
            old = self._queue.pop(room_id)
            self._queue_total_ms -= old.estimated_ms
            logger.debug(
                "Eval superseded room=%s old_seq=%d new_seq=%d",
                room_id,
                old.move_sequence,
                move_sequence,
            )

        total = self._remaining_current_ms() + self._queue_total_ms + estimated_ms
        if total > self._budget_ms:
            logger.info(
                "Eval dropped room=%s bucket=%s total=%.0fms budget=%dms",
                room_id,
                bucket,
                total,
                self._budget_ms,
            )
            return False

        key = room_id if self._enable_supersede else f"{room_id}:{next(self._seq)}"

        self._queue[key] = QueueItem(
            room_id=room_id,
            fen=fen,
            move_sequence=move_sequence,
            estimated_ms=estimated_ms,
            search_nodes=search_nodes,
            bucket=bucket,
        )
        self._queue_total_ms += estimated_ms
        self._wake.set()
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _remaining_current_ms(self) -> float:
        if self._current_item is None:
            return 0
        elapsed = (time.monotonic() - self._current_start) * 1000
        return max(0, self._current_item.estimated_ms - elapsed)

    async def _consumer_loop(self) -> None:
        try:
            while True:
                if not self._queue:
                    self._wake.clear()
                    await self._wake.wait()

                if not self._queue:
                    continue

                _key, item = self._queue.popitem(last=False)
                self._queue_total_ms -= item.estimated_ms

                self._current_item = item
                self._current_start = time.monotonic()

                try:
                    raw = await asyncio.to_thread(
                        self._engine_evaluate_fn,
                        item.fen,
                        0,
                        search_nodes=item.search_nodes,
                    )
                except Exception as exc:
                    logger.warning(
                        "Eval engine failed room=%s bucket=%s: %s",
                        item.room_id,
                        item.bucket,
                        exc,
                    )
                    raw = None
                finally:
                    actual_ms = (time.monotonic() - self._current_start) * 1000
                    self._current_item = None

                if raw is not None:
                    logger.debug(
                        "Eval done room=%s bucket=%s nodes=%d estimated=%dms actual=%.0fms",
                        item.room_id,
                        item.bucket,
                        item.search_nodes,
                        item.estimated_ms,
                        actual_ms,
                    )
                    ev_type = raw.get("type")
                    if ev_type in ("cp", "mate"):
                        eval_payload = {"type": ev_type, "value": int(raw["value"])}
                        try:
                            await self._broadcast_fn(
                                item.room_id,
                                item.fen,
                                item.move_sequence,
                                eval_payload,
                            )
                        except Exception:
                            logger.exception(
                                "Eval broadcast failed room=%s", item.room_id
                            )
        except asyncio.CancelledError:
            raise
