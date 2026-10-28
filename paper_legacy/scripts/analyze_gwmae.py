#!/usr/bin/env python3
"""sweep raw → σ × 정책 × 천장 분석.

searchtime_raw_local/searchtime_{ms}ms_raw.csv 들을 읽어
FEN × searchTimeMs 격자를 만들고, 각 σ 값에 대해:
  - uniform_t 정책 (각 t)
  - complexity bucket 정책 (S/M/L → 100/250/400)
  - weighted_uniform 정책 (baseline ms)
  - 라그랑주 천장 (평균 시간 ≤ baseline 제약 하 최저 gwmae)
의 가중 평균 MAE를 계산하고, 천장과 정책 차이의 95% paired bootstrap CI를 산출.

Usage:
    python3 scripts/analyze_gwmae.py
    python3 scripts/analyze_gwmae.py --sigmas 0.05 0.10 0.15 0.20 0.25
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.gaussian_weighted_mae import gaussian_weight_array  # noqa: E402
from app.utils.position_complexity import (  # noqa: E402
    EVAL_TIME_BY_BUCKET,
    evaluate_position_complexity,
)

DEFAULT_SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.25]
RAW_DIR = PROJECT_ROOT / "searchtime_raw_local"
SEARCHTIMES_MS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
DEFAULT_OUTPUT = PROJECT_ROOT / "gwmae_analysis.json"


def load_raw_grid(
    raw_dir: Path, search_times: list[int]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """returns (fens, pred_mat[N×T], gt_vec[N])"""
    pred_by_st: dict[int, dict[str, float]] = {}
    gt_map: dict[str, float] = {}
    for st in search_times:
        path = raw_dir / f"searchtime_{st}ms_raw.csv"
        if not path.exists():
            raise FileNotFoundError(f"raw 파일 누락: {path}")
        pred_map: dict[str, float] = {}
        with path.open() as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                fen = row["fen"]
                pred_map[fen] = float(row["pred_win_pct_0_100"])
                gt_map.setdefault(fen, float(row["ground_truth_win_pct_0_100"]))
        pred_by_st[st] = pred_map

    fens_common = set(gt_map.keys())
    for st in search_times:
        fens_common &= set(pred_by_st[st].keys())
    fens = sorted(fens_common)
    if not fens:
        raise RuntimeError("공통 FEN 없음")
    N, T = len(fens), len(search_times)
    pred_mat = np.zeros((N, T), dtype=np.float64)
    gt_vec = np.array([gt_map[f] for f in fens], dtype=np.float64)
    for j, st in enumerate(search_times):
        col = pred_by_st[st]
        pred_mat[:, j] = [col[f] for f in fens]
    return fens, pred_mat, gt_vec


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total_w = float(weights.sum())
    if total_w <= 0:
        return float("nan")
    return float((values * weights).sum() / total_w)


def policy_err(err_mat: np.ndarray, time_idx: np.ndarray) -> np.ndarray:
    return err_mat[np.arange(err_mat.shape[0]), time_idx]


def ceiling_lagrangian(
    err_mat: np.ndarray,
    weights: np.ndarray,
    times_ms: np.ndarray,
    target_mean_ms: float,
) -> tuple[np.ndarray, float]:
    """평균 시간 ≤ target_mean_ms 제약 하 라그랑주 최적 분배.

    각 FEN i에서 cost_i(j) = w_i * err_ij + alpha * t_j 를 최소화.
    alpha 를 이분탐색해서 평균 시간이 target 에 맞을 때까지.
    반환: (time_idx[N], 실제 평균 시간)
    """
    N, T = err_mat.shape

    def assign(alpha: float) -> tuple[np.ndarray, float]:
        cost = weights[:, None] * err_mat + alpha * times_ms[None, :]
        j_choice = np.argmin(cost, axis=1)
        mean_time = float(times_ms[j_choice].mean())
        return j_choice, mean_time

    # alpha = 0 이면 시간 무시하고 cost 최소 → 평균 시간 큰 시간 위주.
    # alpha 클수록 짧은 시간 위주 → 평균 시간 ↓
    lo, hi = 0.0, 1.0
    # hi 가 target 이하 mean_time 만들 때까지 확장
    for _ in range(40):
        _, mean_time_hi = assign(hi)
        if mean_time_hi <= target_mean_ms:
            break
        hi *= 4.0
    # 이분탐색
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        _, mean_time = assign(mid)
        if mean_time > target_mean_ms:
            lo = mid
        else:
            hi = mid
    j_choice, mean_time = assign(hi)
    return j_choice, mean_time


def paired_bootstrap_ci(
    err_a: np.ndarray,
    err_b: np.ndarray,
    weights: np.ndarray,
    n_boot: int,
    seed: int = 42,
    chunk: int = 1000,
) -> tuple[float, float]:
    """mean_gwmae(A) - mean_gwmae(B) 의 95% paired bootstrap CI. 청크로 메모리 제한."""
    rng = np.random.default_rng(seed)
    N = len(err_a)
    diffs = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = rng.integers(0, N, size=(size, N))
        w_b = weights[idx]
        sum_w = w_b.sum(axis=1)
        a = (err_a[idx] * w_b).sum(axis=1) / sum_w
        b = (err_b[idx] * w_b).sum(axis=1) / sum_w
        diffs[done:done + size] = a - b
        done += size
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def baseline_weighted_uniform_ms(fens: list[str]) -> float:
    counts = {"S": 0, "M": 0, "L": 0}
    for fen in fens:
        bucket = evaluate_position_complexity(fen).bucket
        counts[bucket] += 1
    total = len(fens)
    weighted = sum(counts[b] * EVAL_TIME_BY_BUCKET[b] for b in counts)
    return weighted / total


def closest_time_idx(t: float, times: list[int]) -> int:
    return int(np.argmin([abs(x - t) for x in times]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    fens, pred_mat, gt_vec = load_raw_grid(Path(args.raw_dir), SEARCHTIMES_MS)
    times_arr = np.array(SEARCHTIMES_MS, dtype=np.float64)
    N = len(fens)
    print(f"FEN {N}개 | sweep times: {SEARCHTIMES_MS}")

    # complexity 정책: FEN별 버킷 → 시간 → sweep 시간들 중 가장 가까운 인덱스
    cx_idx = np.array([
        closest_time_idx(EVAL_TIME_BY_BUCKET[evaluate_position_complexity(f).bucket], SEARCHTIMES_MS)
        for f in fens
    ])
    baseline_ms = baseline_weighted_uniform_ms(fens)
    wu_idx_scalar = closest_time_idx(baseline_ms, SEARCHTIMES_MS)
    print(f"baseline (weighted_uniform target) = {baseline_ms:.2f}ms → sweep에서 {SEARCHTIMES_MS[wu_idx_scalar]}ms로 근사")
    cx_time_dist = {int(t): int((np.array(SEARCHTIMES_MS)[cx_idx] == t).sum()) for t in SEARCHTIMES_MS}
    print(f"complexity time histogram: {{ {', '.join(f'{t}: {v}' for t, v in cx_time_dist.items() if v > 0)} }}")

    results = {
        "n_fens": N,
        "search_times_ms": SEARCHTIMES_MS,
        "baseline_ms": baseline_ms,
        "wu_ms_used": SEARCHTIMES_MS[wu_idx_scalar],
        "complexity_time_histogram": cx_time_dist,
        "sigmas": {},
    }

    err_mat_full = np.abs(pred_mat - gt_vec[:, None])

    for sigma in args.sigmas:
        w = gaussian_weight_array(gt_vec, sigma)

        # uniform per t
        uniform_means: dict[int, float] = {}
        for j, st in enumerate(SEARCHTIMES_MS):
            uniform_means[int(st)] = weighted_mean(err_mat_full[:, j], w)

        # complexity
        err_cx = policy_err(err_mat_full, cx_idx)
        cx_mean = weighted_mean(err_cx, w)

        # weighted_uniform (가까운 sweep 시간으로 근사)
        wu_idx_arr = np.full(N, wu_idx_scalar, dtype=np.int64)
        err_wu = policy_err(err_mat_full, wu_idx_arr)
        wu_mean = weighted_mean(err_wu, w)

        # ceiling
        ceil_idx, ceil_mean_time = ceiling_lagrangian(err_mat_full, w, times_arr, baseline_ms)
        err_ceil = policy_err(err_mat_full, ceil_idx)
        ceil_mean = weighted_mean(err_ceil, w)
        ceil_hist = {int(t): int((times_arr[ceil_idx] == t).sum()) for t in SEARCHTIMES_MS}

        # paired bootstrap CI: A - B
        ci_wu_minus_ceil = paired_bootstrap_ci(err_wu, err_ceil, w, args.n_boot)
        ci_cx_minus_ceil = paired_bootstrap_ci(err_cx, err_ceil, w, args.n_boot)
        ci_cx_minus_wu = paired_bootstrap_ci(err_cx, err_wu, w, args.n_boot)

        key = f"{sigma:.2f}"
        results["sigmas"][key] = {
            "uniform_per_t": uniform_means,
            "complexity": cx_mean,
            "weighted_uniform": wu_mean,
            "ceiling": ceil_mean,
            "ceiling_mean_time_ms": ceil_mean_time,
            "ceiling_time_histogram": ceil_hist,
            "diffs": {
                "wu_minus_ceiling": wu_mean - ceil_mean,
                "cx_minus_ceiling": cx_mean - ceil_mean,
                "cx_minus_wu": cx_mean - wu_mean,
            },
            "ci95": {
                "wu_minus_ceiling": list(ci_wu_minus_ceil),
                "cx_minus_ceiling": list(ci_cx_minus_ceil),
                "cx_minus_wu": list(ci_cx_minus_wu),
            },
        }

        print(f"\n=== sigma = {sigma:.2f} ===")
        u_str = ", ".join(f"{t}: {uniform_means[t]:.4f}" for t in SEARCHTIMES_MS)
        print(f"  uniform_t:   {{ {u_str} }}")
        print(f"  complexity        = {cx_mean:.4f}")
        print(f"  weighted_uniform  = {wu_mean:.4f}  (used {SEARCHTIMES_MS[wu_idx_scalar]}ms)")
        print(f"  ceiling           = {ceil_mean:.4f}  (mean_t={ceil_mean_time:.1f}ms)")
        print(f"  ceiling histogram: {{ {', '.join(f'{t}: {v}' for t, v in ceil_hist.items() if v > 0)} }}")
        print(f"  diff wu - ceiling = {wu_mean - ceil_mean:+.4f}  CI95 [{ci_wu_minus_ceil[0]:+.4f}, {ci_wu_minus_ceil[1]:+.4f}]")
        print(f"  diff cx - ceiling = {cx_mean - ceil_mean:+.4f}  CI95 [{ci_cx_minus_ceil[0]:+.4f}, {ci_cx_minus_ceil[1]:+.4f}]")
        print(f"  diff cx - wu      = {cx_mean - wu_mean:+.4f}  CI95 [{ci_cx_minus_wu[0]:+.4f}, {ci_cx_minus_wu[1]:+.4f}]")

    with open(args.output, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n결과 저장: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
