#!/usr/bin/env python3
"""
Oracle 버킷 실험: delta 기반 33% 분위 S/M/L 라벨로 차등 탐색 시간 배분 후 평균 베르누이 KL 측정.

oracle_bucket_labels.csv (FEN별 S/M/L 라벨)를 읽어 각 버킷에 맞는 ms를 배분하고
FastAPI /api/ai/evaluate 를 호출한다.

S=100ms / M=250ms / L=400ms (균등 분포 가정 시 평균 250ms = uniform 기준과 동일 예산)

Usage:
    python3 scripts/run_oracle_experiment.py
    python3 scripts/run_oracle_experiment.py --no-resume
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
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

API_BASE_URL = "https://chessez.com"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.bernoulli_kl import bernoulli_kl_nats  # noqa: E402

ORACLE_LABELS_CSV = PROJECT_ROOT / "oracle_bucket_labels.csv"
OUTPUT_JSON = PROJECT_ROOT / "searchtime_kl_results_oracle.json"
OUTPUT_RAW_DIR = PROJECT_ROOT / "searchtime_raw_fastapi"
OUTPUT_RAW_CSV = OUTPUT_RAW_DIR / "searchtime_oracle_raw_kl.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "searchtime_experiment_checkpoint_oracle_kl.json"

ORACLE_EVAL_TIME_BY_BUCKET = {"S": 100, "M": 250, "L": 400}

EC2_NPS_1THREAD = 360_000
REQUEST_DELAY_SEC = 0.1
LICHESS_CP_SCALE = 0.00368208


def ms_to_nodes(ms: int) -> int:
    return max(1, int(round(ms * EC2_NPS_1THREAD / 1000)))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def lichess_cp_to_win_percent(cp: float) -> float:
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-LICHESS_CP_SCALE * float(cp))) - 1.0)


def eval_to_win_percent_lichess(data: dict) -> float | None:
    if not data or "type" not in data or "value" not in data:
        return None
    eval_type = data["type"]
    raw_value = data["value"]
    value = int(raw_value) if isinstance(raw_value, (int, float)) else raw_value
    if eval_type == "cp":
        return lichess_cp_to_win_percent(float(value))
    if eval_type == "mate":
        return 100.0 if int(value) > 0 else 0.0
    return None


def load_oracle_pool() -> list[tuple[str, str]]:
    """(fen, oracle_bucket) 리스트 반환."""
    rows = []
    with open(ORACLE_LABELS_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            fen = row["fen"].strip()
            bucket = row["oracle_bucket"].strip()
            if fen and bucket in ORACLE_EVAL_TIME_BY_BUCKET:
                rows.append((fen, bucket))
    return rows


def load_pool_ground_truth() -> dict[str, float]:
    """mixed_workload_pool.csv에서 FEN → new_win_rate 매핑."""
    pool_csv = PROJECT_ROOT / "mixed_workload_pool.csv"
    gt_map: dict[str, float] = {}
    with open(pool_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            fen = row.get("fen", "").strip()
            try:
                gt_map[fen] = float(row["new_win_rate"].strip())
            except (KeyError, ValueError):
                continue
    return gt_map


def load_checkpoint() -> list[float]:
    if not CHECKPOINT_PATH.exists():
        return []
    try:
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            return [float(x) for x in json.load(f).get("kls", [])]
    except Exception:
        return []


def save_checkpoint(kls: list[float]) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"kls": kls}, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Oracle 버킷 차등 탐색 시간 평균 베르누이 KL 실험")
    parser.add_argument("--no-resume", action="store_true", help="체크포인트 무시하고 처음부터")
    args = parser.parse_args()

    if not ORACLE_LABELS_CSV.exists():
        log.error("oracle_bucket_labels.csv 없음: %s", ORACLE_LABELS_CSV)
        return 1

    resume = not args.no_resume
    if not resume:
        for p in (CHECKPOINT_PATH, OUTPUT_RAW_CSV):
            if p.exists():
                p.unlink()

    oracle_pool = load_oracle_pool()
    gt_map = load_pool_ground_truth()

    bucket_counts = {"S": 0, "M": 0, "L": 0}
    for _, b in oracle_pool:
        bucket_counts[b] += 1
    log.info("oracle 라벨 로드: 총 %d개 | S=%d M=%d L=%d", len(oracle_pool), *bucket_counts.values())
    log.info("배분 ms: S=%d M=%d L=%d", *[ORACLE_EVAL_TIME_BY_BUCKET[b] for b in "SML"])
    log.info("evaluate API: %s/api/ai/evaluate", API_BASE_URL)

    kls = load_checkpoint() if resume else []
    start_index = len(kls)

    raw_fieldnames = ["fen", "oracle_bucket", "searchTimeMs", "searchNodes",
                      "pred_win_pct_0_100", "ground_truth_win_pct_0_100", "kl_nats"]
    OUTPUT_RAW_DIR.mkdir(exist_ok=True)
    file_mode = "a" if resume and start_index > 0 else "w"
    write_header = file_mode == "w" or not OUTPUT_RAW_CSV.exists()

    endpoint = f"{API_BASE_URL.rstrip('/')}/api/ai/evaluate"

    with open(OUTPUT_RAW_CSV, file_mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fieldnames)
        if write_header:
            writer.writeheader()

        with logging_redirect_tqdm():
            with tqdm(total=len(oracle_pool), desc="oracle 실험", unit="call") as progress:
                progress.update(start_index)
                for index in range(start_index, len(oracle_pool)):
                    fen, bucket = oracle_pool[index]
                    gt = gt_map.get(fen)
                    if gt is None:
                        log.warning("ground_truth 없음: %s", fen[:40])
                        progress.update(1)
                        continue

                    st = ORACLE_EVAL_TIME_BY_BUCKET[bucket]
                    nodes = ms_to_nodes(st)

                    try:
                        response = requests.post(
                            endpoint,
                            json={"fen": fen, "searchNodes": nodes, "isTestMode": True},
                            timeout=30,
                        )
                        response.raise_for_status()
                        data = response.json()
                        pred = eval_to_win_percent_lichess(data)
                        if pred is not None:
                            kl = bernoulli_kl_nats(pred, gt)
                            kls.append(kl)
                            writer.writerow({
                                "fen": fen,
                                "oracle_bucket": bucket,
                                "searchTimeMs": st,
                                "searchNodes": nodes,
                                "pred_win_pct_0_100": f"{pred:.6f}",
                                "ground_truth_win_pct_0_100": f"{gt:.6f}",
                                "kl_nats": f"{kl:.6f}",
                            })
                    except Exception as exc:
                        log.warning("FEN %s bucket=%s 실패: %s", fen[:40], bucket, exc)

                    progress.update(1)
                    if REQUEST_DELAY_SEC:
                        time.sleep(REQUEST_DELAY_SEC)
                    if (index + 1) % 100 == 0:
                        f.flush()
                        save_checkpoint(kls)

    save_checkpoint(kls)
    mean_kl = sum(kls) / len(kls) if kls else float("nan")
    kl_by_bucket: dict[str, list[float]] = {}
    with open(OUTPUT_RAW_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            b = row["oracle_bucket"]
            kl_v = float(row["kl_nats"])
            kl_by_bucket.setdefault(b, []).append(kl_v)

    payload = {
        "mode": "oracle",
        "evalTimeByBucket": ORACLE_EVAL_TIME_BY_BUCKET,
        "bucketCounts": bucket_counts,
        "mean_kl_nats": mean_kl,
        "n": len(kls),
        "mean_kl_by_bucket": {
            b: {"mean_kl_nats": sum(v) / len(v), "n": len(v)}
            for b, v in kl_by_bucket.items()
        },
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log.info("KL 결과(JSON): %s", OUTPUT_JSON)
    kl_text = f"{mean_kl:.6f}" if math.isfinite(mean_kl) else "nan"
    print(f"\n=== oracle policy mean KL = {kl_text} nats (n={len(kls)}) ===")
    for b in "SML":
        v = kl_by_bucket.get(b, [])
        if v:
            print(f"  {b} ({ORACLE_EVAL_TIME_BY_BUCKET[b]}ms): mean KL = {sum(v)/len(v):.6f} nats (n={len(v)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
