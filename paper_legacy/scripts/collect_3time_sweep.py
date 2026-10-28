#!/usr/bin/env python3
"""
FEN별 3-시간 sweep 데이터 수집 (100ms, 250ms, 400ms).

grid_search_complexity.py 에서 오프라인 그리드 서치에 사용할
per-FEN 예측값을 수집한다.

Usage:
    python3 scripts/collect_3time_sweep.py
    python3 scripts/collect_3time_sweep.py --no-resume
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

import requests
from chess import Board
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POOL_CSV = PROJECT_ROOT / "mixed_workload_pool.csv"
OUTPUT_CSV = PROJECT_ROOT / "sweep_3times_raw.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "sweep_3times_checkpoint.json"

API_BASE_URL = "https://chessez.com"
SEARCH_TIMES = [100, 250, 400]
REQUEST_DELAY_SEC = 0.1
LICHESS_CP_SCALE = 0.00368208

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def lichess_cp_to_win_percent(cp: float) -> float:
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-LICHESS_CP_SCALE * cp)) - 1.0)


def eval_to_win_percent(data: dict) -> float | None:
    if not data or "type" not in data or "value" not in data:
        return None
    if data["type"] == "cp":
        return lichess_cp_to_win_percent(float(data["value"]))
    if data["type"] == "mate":
        return 100.0 if int(data["value"]) > 0 else 0.0
    return None


def load_pool() -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    excluded = 0
    with open(POOL_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        fen_idx = header.index("fen")
        gt_idx = header.index("new_win_rate")
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
    return rows


def load_checkpoint() -> dict[str, dict]:
    if not CHECKPOINT_PATH.exists():
        return {}
    try:
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_checkpoint(results: dict[str, dict]) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f)


def main() -> int:
    parser = argparse.ArgumentParser(description="FEN별 100/250/400ms 예측 수집")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    pool = load_pool()
    log.info("FEN %d개 로드", len(pool))

    endpoint = f"{API_BASE_URL.rstrip('/')}/api/ai/evaluate"
    results: dict[str, dict] = {} if args.no_resume else load_checkpoint()

    remaining = [(i, fen, gt) for i, (fen, gt) in enumerate(pool) if fen not in results]
    log.info("수집 대상 %d / 전체 %d (API: %s)", len(remaining), len(pool), endpoint)

    if not remaining:
        log.info("이미 모든 FEN 수집 완료")
    else:
        done = len(pool) - len(remaining)
        with tqdm(total=len(pool), desc="3-time sweep", initial=done, unit="fen") as pbar:
            for seq, (i, fen, gt) in enumerate(remaining):
                preds: dict[int, float] = {}
                ok = True
                for st in SEARCH_TIMES:
                    try:
                        resp = requests.post(
                            endpoint,
                            json={"fen": fen, "searchTimeMs": st},
                            timeout=30,
                        )
                        resp.raise_for_status()
                        pred = eval_to_win_percent(resp.json())
                        if pred is None:
                            ok = False
                            break
                        preds[st] = pred
                    except Exception as exc:
                        log.warning("FEN %s st=%d 실패: %s", fen[:40], st, exc)
                        ok = False
                        break
                    time.sleep(REQUEST_DELAY_SEC)

                if ok:
                    results[fen] = {
                        "gt": gt,
                        "p100": preds[100],
                        "p250": preds[250],
                        "p400": preds[400],
                    }

                pbar.update(1)
                if (seq + 1) % 50 == 0:
                    save_checkpoint(results)

        save_checkpoint(results)

    fieldnames = ["fen", "ground_truth", "pred_100ms", "pred_250ms", "pred_400ms"]
    written = 0
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fen, _gt in pool:
            if fen not in results:
                continue
            r = results[fen]
            writer.writerow({
                "fen": fen,
                "ground_truth": f"{r['gt']:.6f}",
                "pred_100ms": f"{r['p100']:.6f}",
                "pred_250ms": f"{r['p250']:.6f}",
                "pred_400ms": f"{r['p400']:.6f}",
            })
            written += 1

    log.info("CSV 저장 완료: %s (%d FENs)", OUTPUT_CSV, written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
