from __future__ import annotations

import os
import threading
from contextlib import suppress
from typing import Callable, TypeVar

import chess
import chess.engine


EngineResult = TypeVar("EngineResult")


class EngineService:
    def __init__(
        self,
        engine_path: str,
        max_concurrent: int,
        engine_threads: int = 2,
        engine_hash_mb: int = 0,
        move_max_time_ms: int = 800,
        eval_min_search_ms: int = 50,
        eval_max_search_ms: int = 2000,
    ) -> None:
        self.engine_path = engine_path
        self.max_concurrent = max(1, max_concurrent)
        self.engine_threads = max(1, engine_threads)
        self.engine_hash_mb = max(0, engine_hash_mb)
        self.move_max_time_ms = max(1, move_max_time_ms)
        self.eval_min_search_ms = max(1, eval_min_search_ms)
        self.eval_max_search_ms = max(self.eval_min_search_ms, eval_max_search_ms)

        self._active_requests = 0
        self._state_lock = threading.Lock()
        self._engine_lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(value=self.max_concurrent)
        self._engine: chess.engine.SimpleEngine | None = None

        self._test_engine: chess.engine.SimpleEngine | None = None
        self._test_engine_lock = threading.Lock()

    def initialize(self) -> None:
        with self._engine_lock:
            self._ensure_engine_locked()

    def close(self) -> None:
        with self._engine_lock:
            self._close_engine_locked()
        with self._test_engine_lock:
            self._close_test_engine_locked()

    def status(self) -> dict[str, int | bool]:
        with self._state_lock:
            active_requests = self._active_requests
        return {
            "available": active_requests < self.max_concurrent,
            "activeRequests": active_requests,
            "maxConcurrent": self.max_concurrent,
        }

    def get_best_move(self, fen: str, skill_level: int, move_time_ms: int) -> str:
        board = self._build_board(fen)
        capped_move_time_ms = max(1, min(int(move_time_ms or self.move_max_time_ms), self.move_max_time_ms))
        bounded_skill_level = max(0, min(20, int(skill_level)))

        def operation(engine: chess.engine.SimpleEngine) -> str:
            try:
                engine.configure({"Skill Level": bounded_skill_level})
            except chess.engine.EngineError:
                pass
            result = engine.play(board, chess.engine.Limit(time=capped_move_time_ms / 1000))
            if result.move is None:
                raise ValueError("No legal moves available.")
            return result.move.uci()

        return self._run_request(operation)

    def evaluate(
        self,
        fen: str,
        search_time_ms: int,
        *,
        is_test_mode: bool = False,
        search_nodes: int | None = None,
    ) -> dict[str, str | int]:
        board = self._build_board(fen)

        if is_test_mode:
            return self._evaluate_deterministic(board, fen, search_time_ms, search_nodes)

        if search_nodes is not None:
            limit = chess.engine.Limit(nodes=max(1, search_nodes))
        else:
            capped_search_time_ms = max(
                self.eval_min_search_ms,
                min(int(search_time_ms or self.eval_max_search_ms), self.eval_max_search_ms),
            )
            limit = chess.engine.Limit(time=capped_search_time_ms / 1000)

        def operation(engine: chess.engine.SimpleEngine) -> dict[str, str | int]:
            info = engine.analyse(board, limit)
            return self._score_to_dict(info, fen)

        return self._run_request(operation)

    def _evaluate_deterministic(
        self,
        board: chess.Board,
        fen: str,
        search_time_ms: int,
        search_nodes: int | None,
    ) -> dict[str, str | int]:
        with self._test_engine_lock:
            engine = self._ensure_test_engine_locked()
            try:
                engine._run(engine.protocol._ucinewgame())
            except Exception:
                self._close_test_engine_locked()
                engine = self._ensure_test_engine_locked()

            if search_nodes is not None:
                limit = chess.engine.Limit(nodes=max(1, search_nodes))
            else:
                capped = max(
                    self.eval_min_search_ms,
                    min(int(search_time_ms or self.eval_max_search_ms), self.eval_max_search_ms),
                )
                limit = chess.engine.Limit(time=capped / 1000)

            try:
                info = engine.analyse(board, limit)
            except chess.engine.EngineTerminatedError:
                self._close_test_engine_locked()
                engine = self._ensure_test_engine_locked()
                info = engine.analyse(board, limit)

            result = self._score_to_dict(info, fen)
            if "nodes" in info:
                result["nodes"] = int(info["nodes"])
            if "nps" in info:
                result["nps"] = int(info["nps"])
            if "depth" in info:
                result["depth"] = int(info["depth"])
            return result

    def _ensure_test_engine_locked(self) -> chess.engine.SimpleEngine:
        if self._test_engine is not None:
            return self._test_engine

        resolved_path = self._resolve_engine_path()
        try:
            engine = chess.engine.SimpleEngine.popen_uci(resolved_path)
        except (FileNotFoundError, OSError) as exc:
            raise RuntimeError(f"Stockfish not found at '{resolved_path}'.") from exc

        try:
            engine.configure({"Threads": 1, "Hash": 16})
        except chess.engine.EngineError:
            pass

        self._test_engine = engine
        return engine

    def _close_test_engine_locked(self) -> None:
        if self._test_engine is None:
            return
        with suppress(Exception):
            self._test_engine.quit()
        self._test_engine = None

    @staticmethod
    def _score_to_dict(info: dict, fen: str) -> dict[str, str | int]:
        score = info["score"].white()
        if score.is_mate():
            return {"type": "mate", "value": score.mate() or 0, "fen": fen}
        return {"type": "cp", "value": int(score.score() or 0), "fen": fen}

    def _run_request(self, operation: Callable[[chess.engine.SimpleEngine], EngineResult]) -> EngineResult:
        self._acquire_slot()
        try:
            return self._run_with_engine(operation)
        finally:
            self._release_slot()

    def _run_with_engine(self, operation: Callable[[chess.engine.SimpleEngine], EngineResult]) -> EngineResult:
        last_error: Exception | None = None
        for _ in range(2):
            with self._engine_lock:
                engine = self._ensure_engine_locked()
                try:
                    return operation(engine)
                except chess.engine.EngineTerminatedError as exc:
                    last_error = exc
                    self._close_engine_locked()
                    continue
        raise RuntimeError("Stockfish engine terminated unexpectedly.") from last_error

    def _build_board(self, fen: str) -> chess.Board:
        try:
            board = chess.Board(fen)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid FEN") from exc
        if not board.is_valid():
            raise ValueError("Invalid FEN")
        return board

    def _ensure_engine_locked(self) -> chess.engine.SimpleEngine:
        if self._engine is not None:
            return self._engine

        resolved_path = self._resolve_engine_path()
        try:
            engine = chess.engine.SimpleEngine.popen_uci(resolved_path)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Stockfish not found at '{resolved_path}'.") from exc
        except OSError as exc:
            raise RuntimeError(f"Could not start Stockfish from '{resolved_path}'.") from exc

        options = {"Threads": self.engine_threads}
        if self.engine_hash_mb > 0:
            options["Hash"] = self.engine_hash_mb
        try:
            engine.configure(options)
        except chess.engine.EngineError:
            # Some environments expose fewer tunables; keep the engine usable.
            pass

        self._engine = engine
        return engine

    def _resolve_engine_path(self) -> str:
        fallback_paths = [
            self.engine_path,
            "/opt/homebrew/bin/stockfish",
            "/usr/local/bin/stockfish",
            "/usr/games/stockfish",
            "/usr/bin/stockfish",
        ]
        seen: set[str] = set()
        for path in fallback_paths:
            if not path or path in seen:
                continue
            seen.add(path)
            if os.path.isfile(path):
                return path
        return self.engine_path

    def _close_engine_locked(self) -> None:
        if self._engine is None:
            return
        with suppress(Exception):
            self._engine.quit()
        self._engine = None

    def _acquire_slot(self) -> None:
        acquired = self._semaphore.acquire(blocking=False)
        if not acquired:
            raise RuntimeError("AI server is busy.")
        with self._state_lock:
            self._active_requests += 1

    def _release_slot(self) -> None:
        with self._state_lock:
            self._active_requests -= 1
        self._semaphore.release()
