#!/usr/bin/env python3
"""로컬 EngineService 직접 호출 기반 searchtime sweep raw 측정.

KL/gwmae 같은 metric은 raw에서 사후 계산하므로 여기서는 pred_win_pct, gt_win_pct, cp만
저장한다. 결과는 searchtime_raw_local/searchtime_{ms}ms_raw.csv 형태.

Usage:
    python3 scripts/measure_searchtime_local.py            # 전체 FEN
    python3 scripts/measure_searchtime_local.py --sample 100
    python3 scripts/measure_searchtime_local.py --no-resume
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from pathlib import Path

from chess import Board
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.engine_service import EngineService  # noqa: E402

POOL_CSV = PROJECT_ROOT / "mixed_workload_pool.csv"
OUTPUT_RAW_DIR = PROJECT_ROOT / "searchtime_raw_local"
CHECKPOINT_PATH = PROJECT_ROOT / "searchtime_measure_local_checkpoint.json"

SEARCHTIMES_MS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

# ms → nodes 변환에 사용하는 NPS. EC2 1-thread 측정값(360k)을 유지해서
# 이전 실험들과 'searchTimeMs ↔ searchNodes' 의미가 통일되게 한다.
# 로컬 stockfish 자체의 처리 속도와는 무관하다.
NPS_FOR_MS_TO_NODES = 360_000
LICHESS_CP_SCALE = 0.00368208
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def ms_to_nodes(ms: int) -> int:
    return max(1, int(round(ms * NPS_FOR_MS_TO_NODES / 1000)))


def lichess_cp_to_win_percent(cp: float) -> float:
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-LICHESS_CP_SCALE * float(cp))) - 1.0)


def eval_to_win_percent_lichess(data: dict) -> float | None:
    if not data or "type" not in data or "value" not in data:
        return None
    raw_value = data["value"]
    value = int(raw_value) if isinstance(raw_value, (int, float)) else raw_value
    if data["type"] == "cp":
        return lichess_cp_to_win_percent(float(value))
    if data["type"] == "mate":
        return 100.0 if int(value) > 0 else 0.0
    return None


def _resolve_gt_column(header: list[str]) -> int:
    header = [h.strip() for h in header]
    if "new_win_rate" in header:
        return header.index("new_win_rate")
    if len(header) >= 4:
        log.warning("헤더에 new_win_rate 없음 → 4번째 열을 정답으로 사용합니다.")
        return 3
    raise ValueError("mixed_workload_pool.csv에 new_win_rate 열이 없습니다.")


def _resolve_fen_column(header: list[str]) -> int:
    if "fen" in header:
        return header.index("fen")
    return 0


def load_pool(sample: int | None, exclude_game_over: bool = True) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    excluded = 0
    with POOL_CSV.open(encoding="utf-8-sig", newline="") as fp:
        reader = csv.reader(fp)
        header = next(reader)
        header = [h.strip() for h in header]
        fen_idx = _resolve_fen_column(header)
        gt_idx = _resolve_gt_column(header)
        for row in reader:
            if not row or len(row) <= max(fen_idx, gt_idx):
                continue
            fen = row[fen_idx].strip()
            if not fen:
                continue
            try:
                gt = float(row[gt_idx].strip())
            except (ValueError, IndexError):
                continue
            if exclude_game_over:
                try:
                    if Board(fen).is_game_over():
                        excluded += 1
                        continue
                except Exception:
                    excluded += 1
                    continue
            rows.append((fen, gt))
    if excluded:
        log.info("게임 종료 국면 %d개 제외", excluded)
    if sample:
        random.seed(42)
        rows = random.sample(rows, min(sample, len(rows)))
    return rows


def load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        return {"completed_st": [], "counts": {}}
    try:
        with CHECKPOINT_PATH.open() as fp:
            raw = json.load(fp)
    except Exception:
        return {"completed_st": [], "counts": {}}
    completed = [int(t) for t in raw.get("completed_st", [])]
    counts = {int(k): int(v) for k, v in raw.get("counts", {}).items()}
    return {"completed_st": completed, "counts": counts}


def save_checkpoint(data: dict) -> None:
    with CHECKPOINT_PATH.open("w") as fp:
        json.dump(data, fp, indent=2)


def run(svc: EngineService, pool: list[tuple[str, float]], resume: bool) -> None:
    OUTPUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = load_checkpoint() if resume else {"completed_st": [], "counts": {}}
    completed_st = set(int(t) for t in ckpt.get("completed_st", []))
    counts = dict(ckpt.get("counts", {}))

    total_calls = len(pool) * len(SEARCHTIMES_MS)
    done_count = sum(counts.get(st, 0) for st in SEARCHTIMES_MS if st in completed_st)
    # in-progress st 도 done에 반영
    for st in SEARCHTIMES_MS:
        if st in completed_st:
            continue
        done_count += counts.get(st, 0)

    fieldnames = [
        "fen",
        "searchTimeMs",
        "searchNodes",
        "cp",
        "pred_win_pct_0_100",
        "ground_truth_win_pct_0_100",
    ]

    with logging_redirect_tqdm(), tqdm(total=total_calls, desc="sweep raw", unit="call") as pbar:
        pbar.update(min(done_count, total_calls))
        for st in SEARCHTIMES_MS:
            if st in completed_st:
                continue
            output_path = OUTPUT_RAW_DIR / f"searchtime_{st}ms_raw.csv"
            start_index = counts.get(st, 0)
            mode = "a" if start_index > 0 and output_path.exists() else "w"
            file = output_path.open(mode, encoding="utf-8", newline="")
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            nodes = ms_to_nodes(st)
            try:
                for index in range(start_index, len(pool)):
                    fen, gt = pool[index]
                    try:
                        data = svc.evaluate(
                            fen=fen,
                            search_time_ms=st,
                            is_test_mode=True,
                            search_nodes=nodes,
                        )
                        cp_val = int(data["value"]) if data.get("type") == "cp" else None
                        pred = eval_to_win_percent_lichess(data)
                        if pred is not None:
                            writer.writerow({
                                "fen": fen,
                                "searchTimeMs": st,
                                "searchNodes": nodes,
                                "cp": cp_val if cp_val is not None else "",
                                "pred_win_pct_0_100": f"{pred:.6f}",
                                "ground_truth_win_pct_0_100": f"{gt:.6f}",
                            })
                    except Exception as exc:
                        log.warning("FEN %s nodes=%d 실패: %s", fen[:40], nodes, exc)
                    counts[st] = index + 1
                    pbar.update(1)
                    if (index + 1) % 200 == 0:
                        file.flush()
                        save_checkpoint({
                            "completed_st": sorted(completed_st),
                            "counts": counts,
                        })
            finally:
                file.close()
            completed_st.add(st)
            save_checkpoint({
                "completed_st": sorted(completed_st),
                "counts": counts,
            })


def main() -> int:
    parser = argparse.ArgumentParser(
        description="로컬 EngineService로 sweep 50~500ms raw 측정 (사후분석용)"
    )
    parser.add_argument("--sample", type=int, default=None, help="FEN 샘플 수 (기본: 전체)")
    parser.add_argument("--no-resume", action="store_true", help="체크포인트 무시")
    args = parser.parse_args()

    if not POOL_CSV.exists():
        log.error("mixed_workload_pool.csv 없음: %s", POOL_CSV)
        return 1

    try:
        pool = load_pool(args.sample)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    log.info("local stockfish: %s (testMode, Threads=1, Hash=16)", STOCKFISH_PATH)
    log.info(
        "FEN %d개 | searchtime %d종 (%s) | NPS=%d 변환",
        len(pool),
        len(SEARCHTIMES_MS),
        SEARCHTIMES_MS,
        NPS_FOR_MS_TO_NODES,
    )

    svc = EngineService(engine_path=STOCKFISH_PATH, max_concurrent=1)
    try:
        run(svc, pool, resume=not args.no_resume)
    finally:
        svc.close()
    log.info("raw CSV: %s", OUTPUT_RAW_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
