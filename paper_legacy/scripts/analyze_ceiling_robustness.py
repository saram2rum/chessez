#!/usr/bin/env python3
"""사후 라그랑주 천장의 noise-fitting 여부 검증.

1. half-sample 안정성: 무작위 50:50 fold N회 반복 → 두 fold 천장의 분포.
   - 표준편차/평균차이가 정책 차이(0.5~0.7)에 비해 크면 천장이 sample-dependent (noise fit).
2. random-time null: 시간을 무작위 균등 배정한 정책의 평균 gwmae.
   - null 과 천장 차이가 작으면 "임의 분배가 거의 천장 수준" → 천장의 의미 약함.
3. complexity 신호: 라그랑주 best_time × complexity bucket 분포 + chi-square 독립성.
   - bucket 별 best_t 분포가 비슷 → complexity 신호로 best_t 예측 불가.

Usage:
    python3 scripts/analyze_ceiling_robustness.py
    python3 scripts/analyze_ceiling_robustness.py --sigma 0.10 --n-iter 200
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
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

RAW_DIR = PROJECT_ROOT / "searchtime_raw_local"
SEARCHTIMES_MS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]


def load_raw_grid(raw_dir: Path, search_times: list[int]):
    pred_by_st: dict[int, dict[str, float]] = {}
    gt_map: dict[str, float] = {}
    for st in search_times:
        path = raw_dir / f"searchtime_{st}ms_raw.csv"
        pred_map: dict[str, float] = {}
        with path.open() as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                fen = row["fen"]
                pred_map[fen] = float(row["pred_win_pct_0_100"])
                gt_map.setdefault(fen, float(row["ground_truth_win_pct_0_100"]))
        pred_by_st[st] = pred_map
    fens_common = set(gt_map)
    for st in search_times:
        fens_common &= set(pred_by_st[st])
    fens = sorted(fens_common)
    N, T = len(fens), len(search_times)
    pred_mat = np.zeros((N, T), dtype=np.float64)
    gt_vec = np.array([gt_map[f] for f in fens], dtype=np.float64)
    for j, st in enumerate(search_times):
        col = pred_by_st[st]
        pred_mat[:, j] = [col[f] for f in fens]
    return fens, pred_mat, gt_vec


def ceiling_lagrangian(err_mat, weights, times_ms, target_mean_ms):
    def assign(alpha):
        cost = weights[:, None] * err_mat + alpha * times_ms[None, :]
        j = np.argmin(cost, axis=1)
        return j, float(times_ms[j].mean())
    lo, hi = 0.0, 1.0
    for _ in range(40):
        _, mt = assign(hi)
        if mt <= target_mean_ms:
            break
        hi *= 4.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        _, mt = assign(mid)
        if mt > target_mean_ms:
            lo = mid
        else:
            hi = mid
    j, mt = assign(hi)
    return j, mt


def weighted_mean(values, weights):
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    return float((values * weights).sum() / total)


def half_sample_stability(err_mat, weights, times_arr, target_ms, n_iter, seed):
    rng = np.random.default_rng(seed)
    N = err_mat.shape[0]
    half = N // 2
    a_vals, b_vals, diffs = [], [], []
    for _ in range(n_iter):
        perm = rng.permutation(N)
        idx_a, idx_b = perm[:half], perm[half:2 * half]
        ja, _ = ceiling_lagrangian(err_mat[idx_a], weights[idx_a], times_arr, target_ms)
        ca = weighted_mean(err_mat[idx_a][np.arange(half), ja], weights[idx_a])
        jb, _ = ceiling_lagrangian(err_mat[idx_b], weights[idx_b], times_arr, target_ms)
        cb = weighted_mean(err_mat[idx_b][np.arange(half), jb], weights[idx_b])
        a_vals.append(ca)
        b_vals.append(cb)
        diffs.append(abs(ca - cb))
    return np.array(a_vals), np.array(b_vals), np.array(diffs)


def random_time_null(err_mat, weights, times_ms, target_ms, n_iter, seed):
    """시간 분포가 target_ms 평균이 되도록 무작위 배정한 정책의 평균 gwmae 분포."""
    rng = np.random.default_rng(seed)
    N, T = err_mat.shape
    # target_ms 평균이 되도록 시간 선택 — 그냥 균등 무작위로 하고 target_ms와 평균 차이 작은 결과만 채택
    # 더 간단: 각 FEN에 동일 확률로 시간 t 배정 후 평균 시간이 target에 가깝게 가중치 매핑
    # 여기서는 그냥 균등 무작위 (평균 = mean(times_ms))
    null_vals = []
    for _ in range(n_iter):
        j = rng.integers(0, T, size=N)
        err = err_mat[np.arange(N), j]
        null_vals.append(weighted_mean(err, weights))
    return np.array(null_vals)


def best_time_by_bucket(fens, ceil_idx, times):
    by_bucket = {"S": Counter(), "M": Counter(), "L": Counter()}
    for fen, j in zip(fens, ceil_idx):
        bucket = evaluate_position_complexity(fen).bucket
        by_bucket[bucket][times[j]] += 1
    return by_bucket


def chi_square_independence(table: np.ndarray) -> tuple[float, int, float]:
    """numpy로 chi-square 독립성 검정. (chi2, dof, p_approx)
    p는 chi2 분포 cdf 근사. 간단히 chi2 / dof 비율로 출력 (정확 p는 scipy 없어 생략).
    """
    table = np.asarray(table, dtype=np.float64)
    row_sum = table.sum(axis=1, keepdims=True)
    col_sum = table.sum(axis=0, keepdims=True)
    total = table.sum()
    expected = row_sum * col_sum / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.where(expected > 0, (table - expected) ** 2 / expected, 0).sum()
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    # scipy 없이 chi2 critical value 근사: dof=18 (3 bucket × 10 time → dof=18)에서
    # alpha=0.05 critical ≈ 28.87 (chi2 table). 비교용으로 chi2/dof만 보고.
    return float(chi2), int(dof), float(chi2 / dof if dof > 0 else 0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fens, pred_mat, gt_vec = load_raw_grid(Path(args.raw_dir), SEARCHTIMES_MS)
    N = len(fens)
    times_arr = np.array(SEARCHTIMES_MS, dtype=np.float64)

    counts = {"S": 0, "M": 0, "L": 0}
    for f in fens:
        counts[evaluate_position_complexity(f).bucket] += 1
    baseline_ms = sum(counts[b] * EVAL_TIME_BY_BUCKET[b] for b in counts) / N
    print(f"FEN {N}개 | sigma={args.sigma} | baseline_ms={baseline_ms:.2f} | bucket {counts}")

    err_mat = np.abs(pred_mat - gt_vec[:, None])
    w = gaussian_weight_array(gt_vec, args.sigma)

    # full ceiling (참고용)
    ceil_idx_full, mean_t_full = ceiling_lagrangian(err_mat, w, times_arr, baseline_ms)
    ceil_full = weighted_mean(err_mat[np.arange(N), ceil_idx_full], w)
    wu_idx = int(np.argmin([abs(t - baseline_ms) for t in SEARCHTIMES_MS]))
    wu_mean = weighted_mean(err_mat[:, wu_idx], w)
    print(f"\n전체 천장 = {ceil_full:.4f} (mean_t={mean_t_full:.1f}ms)")
    print(f"weighted_uniform({SEARCHTIMES_MS[wu_idx]}ms) = {wu_mean:.4f}")

    # 1. half-sample 안정성
    print(f"\n=== 1) half-sample 안정성 ({args.n_iter}회 반복) ===")
    a, b, diffs = half_sample_stability(err_mat, w, times_arr, baseline_ms, args.n_iter, args.seed)
    print(f"  fold A 천장: mean={a.mean():.4f}, std={a.std():.4f}, range=[{a.min():.4f}, {a.max():.4f}]")
    print(f"  fold B 천장: mean={b.mean():.4f}, std={b.std():.4f}, range=[{b.min():.4f}, {b.max():.4f}]")
    print(f"  |A−B|  diff: mean={diffs.mean():.4f}, max={diffs.max():.4f}")
    print(f"  → 두 fold 천장 평균과 전체 천장({ceil_full:.4f})이 비슷하면 sample-robust")

    # 2. random-time null
    print(f"\n=== 2) random-time null (시간 균등 무작위 배정) ===")
    null_vals = random_time_null(err_mat, w, times_arr, baseline_ms, args.n_iter, args.seed)
    print(f"  null 평균 gwmae: mean={null_vals.mean():.4f}, std={null_vals.std():.4f}")
    print(f"  → 천장({ceil_full:.4f}) << null({null_vals.mean():.4f}) ⇒ 라그랑주가 sample 패턴을 진짜로 잡고 있다는 신호")
    print(f"  → 천장 ≈ null 이면 random과 다를 바 없음 (천장 의미 약함)")

    # 3. complexity 신호 분석
    print(f"\n=== 3) 라그랑주 best_time × complexity bucket ===")
    by_bucket = best_time_by_bucket(fens, ceil_idx_full, SEARCHTIMES_MS)
    header = f"{'bucket':8}{'n':>6}  " + "".join(f"{t:>7}" for t in SEARCHTIMES_MS)
    print(header)
    print("-" * len(header))
    for bucket in ("S", "M", "L"):
        n = sum(by_bucket[bucket].values())
        if n == 0:
            continue
        row = f"{bucket:8}{n:>6}  "
        for t in SEARCHTIMES_MS:
            pct = by_bucket[bucket].get(t, 0) / n * 100
            row += f"{pct:>6.1f}%"
        print(row)
    # 전체 분포 (참고)
    total_row = f"{'TOTAL':8}{N:>6}  "
    for t in SEARCHTIMES_MS:
        c = sum(by_bucket[b].get(t, 0) for b in ("S", "M", "L"))
        total_row += f"{c/N*100:>6.1f}%"
    print(total_row)
    # chi-square
    table = np.array([
        [by_bucket[b].get(t, 0) for t in SEARCHTIMES_MS]
        for b in ("S", "M", "L")
        if sum(by_bucket[b].values()) > 0
    ])
    if table.shape[0] >= 2 and table.shape[1] >= 2:
        chi2, dof, ratio = chi_square_independence(table)
        # critical values (alpha=0.05): dof=9 → 16.92, dof=18 → 28.87
        print(f"\n  chi-square(독립성): chi2={chi2:.2f}, dof={dof}, chi2/dof={ratio:.2f}")
        if dof == 18:
            crit = 28.87
        elif dof == 9:
            crit = 16.92
        else:
            crit = None
        if crit is not None:
            verdict = "독립 아님 (신호 있음)" if chi2 > crit else "독립 (신호 없음)"
            print(f"  vs critical({dof}, α=0.05) = {crit:.2f} → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
