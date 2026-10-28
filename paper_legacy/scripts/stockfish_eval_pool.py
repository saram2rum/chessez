"""
mixed_workload_pool.csv 의 FEN을 Stockfish depth 20으로 일괄 분석.

컬럼 채우기:
  ground_truth  : Elo 승률 공식  1 / (1 + 10^(-cp/400))          [0-1]
  eval_value    : Stockfish 센티폰 (메이트 = ±10000)             [정수]
  new_win_rate  : Lichess 변환 공식  100 / (1 + exp(-0.00368208·cp)) [0-100]

병렬 구조:
  ProcessPoolExecutor(max_workers=4)
  워커마다 Stockfish 1개 (Threads=1, Hash=64MB)
  → M4 퍼포먼스 코어 4개 포화

체크포인트:
  mixed_workload_checkpoint.csv 에 200개마다 저장
  재실행 시 완료된 FEN은 스킵

Usage:
  python3 scripts/stockfish_eval_pool.py
  python3 scripts/stockfish_eval_pool.py --depth 18 --workers 6
"""

from __future__ import annotations

import argparse
import atexit
import csv
import math
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

import chess
import chess.engine

# stdout 버퍼링 비활성화 (백그라운드 실행 시 로그가 즉시 보이도록)
sys.stdout.reconfigure(line_buffering=True)

# ── 경로 ──────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parent.parent
CSV_PATH         = ROOT / "mixed_workload_pool.csv"
CHECKPOINT_PATH  = ROOT / "mixed_workload_checkpoint.csv"
STOCKFISH_PATH   = "/opt/homebrew/bin/stockfish"

# ── 기본값 ─────────────────────────────────────────────────────────────────────
DEFAULT_DEPTH   = 20
DEFAULT_WORKERS = 4
HASH_MB         = 64     # 워커당 Stockfish Hash (MB)
TIME_CAP_S      = 60.0   # 포지션당 최대 분석 시간 (초) — 무한루프 방지
MATE_CP         = 10_000 # 메이트 포지션을 대체할 cp 값

# ── 워커 전역 엔진 ─────────────────────────────────────────────────────────────
_engine: chess.engine.SimpleEngine | None = None
_depth: int = DEFAULT_DEPTH


def _init_worker(stockfish_path: str, depth: int, hash_mb: int) -> None:
    """각 워커 프로세스 초기화: Stockfish 인스턴스 생성."""
    global _engine, _depth
    _depth = depth
    _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    _engine.configure({"Threads": 1, "Hash": hash_mb})
    atexit.register(_engine.close)


# ── 분석 함수 (워커에서 실행) ───────────────────────────────────────────────────
def _analyze(fen: str) -> dict:
    """FEN 하나를 분석해 결과 dict 반환."""
    global _engine, _depth
    try:
        board = chess.Board(fen)
        info  = _engine.analyse(
            board,
            chess.engine.Limit(depth=_depth, time=TIME_CAP_S),
        )
        pov = info["score"].white()  # 항상 백 기준 점수

        if pov.is_mate():
            m  = pov.mate()          # 양수 = 백이 이김, 음수 = 흑이 이김
            cp = MATE_CP if m > 0 else -MATE_CP
        else:
            cp = pov.score()
            if cp is None:
                return {"fen": fen, "error": "score=None"}

        # ground_truth : 표준 Elo 승률 (0~1, 백 기준)
        ground_truth = 1.0 / (1.0 + 10.0 ** (-cp / 400.0))

        # new_win_rate : Lichess 변환 공식 (0~100, 백 기준)
        new_win_rate = 100.0 / (1.0 + math.exp(-0.00368208 * cp))

        return {
            "fen":          fen,
            "ground_truth": ground_truth,
            "eval_value":   cp,
            "new_win_rate": new_win_rate,
        }

    except Exception as exc:
        return {"fen": fen, "error": str(exc)}


# ── CSV 쓰기 ───────────────────────────────────────────────────────────────────
def _write_csv(path: Path, ordered_fens: list[str], results: dict[str, dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "ground_truth", "eval_value", "new_win_rate"])
        for fen in ordered_fens:
            r = results.get(fen, {})
            writer.writerow([
                fen,
                r.get("ground_truth", ""),
                r.get("eval_value",   ""),
                r.get("new_win_rate", ""),
            ])


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main(depth: int, workers: int) -> int:
    # ── FEN 목록 읽기 ─────────────────────────────────────────────────────────
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ordered_fens: list[str] = [r["fen"] for r in rows]
    total = len(ordered_fens)

    print(f"{'='*60}")
    print(f"  Stockfish 병렬 평가  |  depth {depth}  |  workers {workers}")
    print(f"  FEN {total}개  |  Stockfish: {STOCKFISH_PATH}")
    print(f"{'='*60}\n")

    # ── 체크포인트 로드 ───────────────────────────────────────────────────────
    done: dict[str, dict] = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ev = r.get("eval_value", "")
                if ev not in ("", None):
                    done[r["fen"]] = {
                        "fen":          r["fen"],
                        "ground_truth": float(r["ground_truth"]),
                        "eval_value":   int(float(r["eval_value"])),
                        "new_win_rate": float(r["new_win_rate"]),
                    }
        if done:
            print(f"  체크포인트 로드: {len(done):,}개 기완료 (스킵)\n")

    pending = [f for f in ordered_fens if f not in done]
    print(f"  분석 대상: {len(pending):,}개  |  이미 완료: {len(done):,}개\n")

    if not pending:
        print("  모든 FEN 이미 분석 완료. CSV 재저장합니다.")
        _write_csv(CSV_PATH, ordered_fens, done)
        return 0

    # ── 병렬 분석 ─────────────────────────────────────────────────────────────
    # ProcessPoolExecutor 대신 mp.Pool + imap_unordered 사용
    # → 결과 큐 포화(deadlock) 문제 없이 스트리밍으로 결과 수신 가능
    results: dict[str, dict] = dict(done)
    completed_new = 0
    errors: list[str] = []
    start_ts = time.time()

    ctx = mp.get_context("spawn")  # macOS fork 안전 문제 회피
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(STOCKFISH_PATH, depth, HASH_MB),
    ) as pool:
        for res in pool.imap_unordered(_analyze, pending, chunksize=1):
            fen = res["fen"]
            completed_new += 1
            total_done = len(done) + completed_new

            if "error" in res:
                print(f"  [오류] {fen[:55]}  → {res['error']}", file=sys.stderr)
                errors.append(fen)
                results[fen] = {"fen": fen, "ground_truth": "", "eval_value": "", "new_win_rate": ""}
            else:
                results[fen] = res

            # ── 진행상황 로그 (100개마다 + 마지막) ──────────────────────────
            if completed_new % 100 == 0 or total_done == total:
                elapsed = time.time() - start_ts
                rate    = completed_new / elapsed if elapsed > 0 else 0
                eta     = (len(pending) - completed_new) / rate if rate > 0 else 0
                pct     = total_done / total * 100
                ev_str  = ""
                if "error" not in res:
                    ev_str = f"  eval={res['eval_value']:+6d}cp  wr={res['new_win_rate']:5.1f}%"
                print(
                    f"  [{total_done:4d}/{total}] {pct:5.1f}%"
                    f"  경과 {elapsed:6.1f}s"
                    f"  {rate:5.1f} pos/s"
                    f"  ETA {eta:5.0f}s"
                    f"{ev_str}"
                )

            # ── 체크포인트 저장 (200개마다) ───────────────────────────────────
            if completed_new % 200 == 0:
                _write_csv(CHECKPOINT_PATH, ordered_fens, results)
                print(f"  [체크포인트] {CHECKPOINT_PATH.name} 저장 ({total_done}개)")

    # ── 최종 저장 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  분석 완료: {total}개  |  오류: {len(errors)}개")
    _write_csv(CSV_PATH, ordered_fens, results)
    print(f"  저장: {CSV_PATH}")

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print(f"  체크포인트 삭제 완료")

    # ── eval_value 통계 ────────────────────────────────────────────────────────
    vals: list[float] = []
    for fen in ordered_fens:
        r = results.get(fen, {})
        ev = r.get("eval_value", "")
        if ev not in ("", None):
            vals.append(float(ev))

    if vals:
        print(f"\n  ── eval_value 통계 ──────────────────────────────")
        print(f"     분석 성공:  {len(vals):,} / {total:,}")
        print(f"     평균:       {statistics.mean(vals):+.1f} cp")
        print(f"     중앙값:     {statistics.median(vals):+.1f} cp")
        print(f"     표준편차:   {statistics.stdev(vals):.1f} cp")
        print(f"     범위:       {min(vals):+.0f} ~ {max(vals):+.0f} cp")
        balanced = sum(1 for v in vals if abs(v) <= 200)
        print(f"     균형 포지션 (±200cp): {balanced:,}개 ({balanced/len(vals)*100:.1f}%)")

    if errors:
        print(f"\n  ── 오류 FEN ({len(errors)}개) ─────────────────────────")
        for fen in errors[:10]:
            print(f"     {fen}")
        if len(errors) > 10:
            print(f"     ... 외 {len(errors)-10}개")

    print(f"\n  최종 FEN 개수: {len(vals):,}")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stockfish 병렬 FEN 평가")
    parser.add_argument("--depth",   type=int, default=DEFAULT_DEPTH,
                        help=f"탐색 깊이 (기본값: {DEFAULT_DEPTH})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"병렬 워커 수 (기본값: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    raise SystemExit(main(args.depth, args.workers))
