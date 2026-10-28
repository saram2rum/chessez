#!/usr/bin/env python3
"""
L 버킷 최적 searchTimeMs 탐색.

S/M 포지션은 한 번만 API 호출, L 포지션만 여러 시간으로 반복 호출하여
각 L 시간별 전체 평균 베르누이 KL(nats)을 비교한다.

Usage:
    python3 scripts/sweep_l_time.py
    python3 scripts/sweep_l_time.py --l-times 400 600 800 1000
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.bernoulli_kl import bernoulli_kl_nats
from app.utils.position_complexity import evaluate_position_complexity

API_BASE_URL = "https://chessez.com"
POOL_CSV = PROJECT_ROOT / "mixed_workload_pool.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "sweep_l_checkpoints"
REQUEST_DELAY_SEC = 0.1
LICHESS_CP_SCALE = 0.00368208

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def lichess_cp_to_win_percent(cp: float) -> float:
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-LICHESS_CP_SCALE * float(cp))) - 1.0)


def eval_to_win_percent(data: dict) -> float | None:
    if not data or "type" not in data or "value" not in data:
        return None
    if data["type"] == "cp":
        return lichess_cp_to_win_percent(float(data["value"]))
    if data["type"] == "mate":
        return 100.0 if int(data["value"]) > 0 else 0.0
    return None


def _resolve_columns(header: list[str]) -> tuple[int, int]:
    header = [h.strip() for h in header]
    fen_idx = header.index("fen") if "fen" in header else 0
    gt_idx = header.index("new_win_rate") if "new_win_rate" in header else 3
    return fen_idx, gt_idx


def load_pool() -> list[tuple[str, float]]:
    rows = []
    excluded = 0
    with open(POOL_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        fen_idx, gt_idx = _resolve_columns(next(reader))
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


def call_api(endpoint: str, fen: str, search_time_ms: int) -> float | None:
    st = max(50, min(30_000, search_time_ms))
    resp = requests.post(endpoint, json={"fen": fen, "searchTimeMs": st}, timeout=30)
    resp.raise_for_status()
    return eval_to_win_percent(resp.json())


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="L 버킷 최적 searchTimeMs 탐색")
    parser.add_argument(
        "--l-times",
        nargs="+",
        type=int,
        default=list(range(400, 1050, 50)),
        help="테스트할 L 시간 목록 (ms)",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    pool = load_pool()
    endpoint = f"{API_BASE_URL.rstrip('/')}/api/ai/evaluate"
    log.info("FEN %d개 로드", len(pool))

    m_fens, l_fens = [], []
    for fen, gt in pool:
        result = evaluate_position_complexity(fen)
        bucket = result.bucket
        if bucket == "L":
            l_fens.append((fen, gt))
        else:
            m_fens.append((fen, gt))

    log.info("M(+S)=%d, L=%d", len(m_fens), len(l_fens))

    ckpt_path = CHECKPOINT_DIR / "sweep_l_kl.json"
    ckpt = {} if args.no_resume else load_checkpoint(ckpt_path)

    # --- Phase 1: M 결과 (한 번만) ---
    sm_kls = ckpt.get("sm_kls")
    if sm_kls is None:
        log.info("Phase 1: M(%d) 포지션 API 호출", len(m_fens))
        sm_kls = []
        sm_items = [(fen, gt, 250) for fen, gt in m_fens]
        for i, (fen, gt, st) in enumerate(tqdm(sm_items, desc="M", unit="call")):
            try:
                pred = call_api(endpoint, fen, st)
                if pred is not None:
                    sm_kls.append(bernoulli_kl_nats(pred, gt))
            except Exception as exc:
                log.warning("FEN %s st=%d 실패: %s", fen[:40], st, exc)
            if REQUEST_DELAY_SEC:
                time.sleep(REQUEST_DELAY_SEC)
            if (i + 1) % 100 == 0:
                ckpt["sm_kls"] = sm_kls
                save_checkpoint(ckpt_path, ckpt)
        ckpt["sm_kls"] = sm_kls
        save_checkpoint(ckpt_path, ckpt)
    else:
        log.info("Phase 1: M 캐시 사용 (%d건)", len(sm_kls))

    sm_kl_sum = sum(sm_kls)
    sm_count = len(sm_kls)

    # --- Phase 2: L 포지션을 각 시간별로 호출 ---
    l_results = ckpt.get("l_results", {})
    l_times_todo = [t for t in args.l_times if str(t) not in l_results]

    if l_times_todo:
        total_calls = len(l_fens) * len(l_times_todo)
        log.info(
            "Phase 2: L %d포지션 × %d시간 = %d API 호출",
            len(l_fens), len(l_times_todo), total_calls,
        )
        with tqdm(total=total_calls, desc="L sweep", unit="call") as pbar:
            for l_time in l_times_todo:
                kls = []
                for fen, gt in l_fens:
                    try:
                        pred = call_api(endpoint, fen, l_time)
                        if pred is not None:
                            kls.append(bernoulli_kl_nats(pred, gt))
                    except Exception as exc:
                        log.warning("FEN %s st=%d 실패: %s", fen[:40], l_time, exc)
                    pbar.update(1)
                    if REQUEST_DELAY_SEC:
                        time.sleep(REQUEST_DELAY_SEC)
                l_results[str(l_time)] = kls
                ckpt["l_results"] = l_results
                save_checkpoint(ckpt_path, ckpt)
    else:
        log.info("Phase 2: 모든 L 시간 캐시 사용")

    # --- 결과 출력 ---
    print(f"\n{'='*60}")
    print(f"M={len(m_fens)}(250ms), L={len(l_fens)}  (단위: nats)")
    print(f"{'='*60}")
    print(f"{'L time':>8s}  {'mean KL':>12s}  {'avg ms':>8s}  {'L mean KL':>12s}")
    print(f"{'-'*8}  {'-'*12}  {'-'*8}  {'-'*12}")

    best_kl = float("inf")
    best_l_time = 0
    for l_time in sorted(args.l_times):
        key = str(l_time)
        if key not in l_results:
            continue
        l_kls = l_results[key]
        total_kl = sm_kl_sum + sum(l_kls)
        total_count = sm_count + len(l_kls)
        mean_kl = total_kl / total_count if total_count else float("nan")

        l_mean_kl = sum(l_kls) / len(l_kls) if l_kls else float("nan")
        avg_ms = (len(m_fens) * 250 + len(l_fens) * l_time) / len(pool)

        marker = ""
        if mean_kl < best_kl:
            best_kl = mean_kl
            best_l_time = l_time
            marker = " ★"

        print(f"{l_time:>7d}ms  {mean_kl:>12.6f}  {avg_ms:>7.1f}ms  {l_mean_kl:>12.6f}{marker}")

    print(f"\n최적 L = {best_l_time}ms → mean KL = {best_kl:.6f} nats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
