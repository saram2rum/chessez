#!/usr/bin/env python3
"""
기존 sweep_3times_raw.csv 에 50ms, 750ms 예측값을 추가 수집한다.
결과: sweep_5times_raw.csv (fen, gt, pred_50ms, pred_100ms, pred_250ms, pred_400ms, pred_750ms)
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
EXISTING_CSV = PROJECT_ROOT / "sweep_3times_raw.csv"
OUTPUT_CSV = PROJECT_ROOT / "sweep_5times_raw.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "sweep_extended_checkpoint.json"

API_BASE_URL = "https://chessez.com"
NEW_TIMES = [50, 750]
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


def load_existing() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(EXISTING_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fen = row["fen"].strip()
            rows[fen] = {
                "gt": float(row["ground_truth"]),
                "p100": float(row["pred_100ms"]),
                "p250": float(row["pred_250ms"]),
                "p400": float(row["pred_400ms"]),
            }
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    existing = load_existing()
    fens = list(existing.keys())
    log.info("기존 sweep: %d FENs, 추가 수집: %s ms", len(fens), NEW_TIMES)

    endpoint = f"{API_BASE_URL.rstrip('/')}/api/ai/evaluate"
    results: dict[str, dict] = {} if args.no_resume else load_checkpoint()

    remaining = [fen for fen in fens if fen not in results]
    log.info("수집 대상 %d / 전체 %d", len(remaining), len(fens))

    if remaining:
        done = len(fens) - len(remaining)
        with tqdm(total=len(fens), desc="50+750ms sweep", initial=done, unit="fen") as pbar:
            for seq, fen in enumerate(remaining):
                preds: dict[int, float] = {}
                ok = True
                for st in NEW_TIMES:
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
                    results[fen] = {"p50": preds[50], "p750": preds[750]}

                pbar.update(1)
                if (seq + 1) % 50 == 0:
                    save_checkpoint(results)

        save_checkpoint(results)

    fieldnames = [
        "fen", "ground_truth",
        "pred_50ms", "pred_100ms", "pred_250ms", "pred_400ms", "pred_750ms",
    ]
    written = 0
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fen in fens:
            if fen not in results:
                continue
            ex = existing[fen]
            nw = results[fen]
            writer.writerow({
                "fen": fen,
                "ground_truth": f"{ex['gt']:.6f}",
                "pred_50ms": f"{nw['p50']:.6f}",
                "pred_100ms": f"{ex['p100']:.6f}",
                "pred_250ms": f"{ex['p250']:.6f}",
                "pred_400ms": f"{ex['p400']:.6f}",
                "pred_750ms": f"{nw['p750']:.6f}",
            })
            written += 1

    log.info("CSV 저장: %s (%d FENs)", OUTPUT_CSV, written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
