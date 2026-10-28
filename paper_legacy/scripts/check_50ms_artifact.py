#!/usr/bin/env python3
"""best_t=50ms = 19% 의 정체 분석.

두 가설 검증:
  H1: argmin tie-breaking artifact (동률 시 가장 작은 인덱스 = 50ms)
  H2: 50ms 가 진짜로 GT (depth 20) 와 자주 일치하는 실질 신호

방법:
  1. raw argmin (현재): 50ms 가 19%
  2. tolerance + 랜덤 tie-breaking: 동률 안에서 랜덤 선택
     - tolerance 0 (정확 동률만): 50ms 비중 어떻게 변하나?
     - tolerance 0.01, 0.1, 1.0 %p
  3. 50ms 가 wide tie 안에 들어간 비율 측정

H1 이 맞으면 → tolerance 적용 시 50ms 가 다른 시간과 거의 같아짐
H2 가 맞으면 → tolerance 적용 후에도 50ms 가 여전히 가장 많음
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "searchtime_raw_local"
SEARCHTIMES_MS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]


def load_raw_grid(raw_dir, search_times):
    pred_by_st, gt_map = {}, {}
    for st in search_times:
        path = raw_dir / f"searchtime_{st}ms_raw.csv"
        pmap = {}
        with path.open() as fp:
            for row in csv.DictReader(fp):
                f = row["fen"]
                pmap[f] = float(row["pred_win_pct_0_100"])
                gt_map.setdefault(f, float(row["ground_truth_win_pct_0_100"]))
        pred_by_st[st] = pmap
    fens_common = set(gt_map)
    for st in search_times:
        fens_common &= set(pred_by_st[st])
    fens = sorted(fens_common)
    N, T = len(fens), len(search_times)
    pred_mat = np.zeros((N, T))
    gt_vec = np.array([gt_map[f] for f in fens])
    for j, st in enumerate(search_times):
        pmap = pred_by_st[st]
        pred_mat[:, j] = [pmap[f] for f in fens]
    return fens, pred_mat, gt_vec


def best_t_with_tolerance(err_mat, tol, rng_seed=42):
    """tolerance 안에서 랜덤 tie-breaking 으로 best_t 결정."""
    rng = np.random.default_rng(rng_seed)
    N, T = err_mat.shape
    min_err = err_mat.min(axis=1, keepdims=True)
    tie_mask = err_mat <= min_err + tol
    random_scores = rng.random((N, T))
    random_scores[~tie_mask] = np.inf
    return np.argmin(random_scores, axis=1)


def print_distribution(label, idx_arr, N):
    counter = Counter(int(SEARCHTIMES_MS[i]) for i in idx_arr)
    print(f"\n[{label}]")
    print(f"  {'time':>5}  {'count':>6}  {'%':>6}")
    for t in SEARCHTIMES_MS:
        c = counter.get(t, 0)
        print(f"  {t:>5}  {c:>6}  {c/N*100:>5.1f}%")


def main():
    fens, pred_mat, gt_vec = load_raw_grid(RAW_DIR, SEARCHTIMES_MS)
    N = len(fens)
    err_mat = np.abs(pred_mat - gt_vec[:, None])

    print(f"FEN {N}개\n")
    print(f"{'='*70}")
    print("가설 H1: argmin tie-breaking artifact 인가?")
    print(f"{'='*70}")

    # raw argmin (현재 방식)
    idx_raw = np.argmin(err_mat, axis=1)
    print_distribution("raw argmin (현재) — 동률 시 가장 작은 인덱스 = 50ms", idx_raw, N)

    # tolerance 0 (정확 동률만 랜덤화)
    idx_tol0 = best_t_with_tolerance(err_mat, 0.0, rng_seed=42)
    print_distribution("tolerance 0 + 랜덤 tie — 정확한 동률만 랜덤화", idx_tol0, N)

    # tolerance 다양
    for tol in [0.01, 0.05, 0.1, 0.5, 1.0]:
        idx = best_t_with_tolerance(err_mat, tol, rng_seed=42)
        print_distribution(f"tolerance {tol}%p + 랜덤 tie", idx, N)

    # 동률 통계
    print(f"\n{'='*70}")
    print("동률 통계 — 각 FEN 의 best 안에 몇 개 시간이 동률에 들어있나")
    print(f"{'='*70}")
    min_err = err_mat.min(axis=1, keepdims=True)
    for tol in [0.0, 0.01, 0.1, 1.0]:
        tie_count = (err_mat <= min_err + tol).sum(axis=1)
        print(f"\ntolerance {tol}%p:")
        print(f"  동률 인 시간 수의 분포:")
        for n in [1, 2, 3, 5, 10]:
            if n == 1:
                count = int((tie_count == 1).sum())
                print(f"    정확히 1개 (동률 없음): {count} ({count/N*100:.1f}%)")
            elif n == 10:
                count = int((tie_count >= 10).sum())
                print(f"    모든 시간 동률 (10개): {count} ({count/N*100:.1f}%)")
            else:
                count = int((tie_count == n).sum())
                print(f"    정확히 {n}개: {count} ({count/N*100:.1f}%)")
        print(f"  평균 동률 시간 수: {tie_count.mean():.2f}")

    # 50ms 에 한정해서 분석
    print(f"\n{'='*70}")
    print("50ms 만의 특이성 — 50ms 가 GT 에 더 자주 가까운 본질적 이유 있는가?")
    print(f"{'='*70}")

    # 각 시간이 best_t 가 된 FEN 들의 오차 평균 (raw argmin)
    print(f"\n[raw argmin 으로 각 시간이 best 인 FEN 들의 평균 오차]")
    for j, t in enumerate(SEARCHTIMES_MS):
        mask = idx_raw == j
        if mask.sum() > 0:
            avg_err = err_mat[mask, j].mean()
            print(f"  best_t={t}ms ({int(mask.sum())} FEN): 평균 오차 = {avg_err:.4f}")

    # 핵심: 50ms 가 tolerance 적용 후에도 우월한가
    print(f"\n{'='*70}")
    print("결론")
    print(f"{'='*70}")
    pct_raw = (idx_raw == 0).sum() / N * 100
    pct_tol0 = (idx_tol0 == 0).sum() / N * 100
    idx_tol1 = best_t_with_tolerance(err_mat, 1.0, rng_seed=42)
    pct_tol1 = (idx_tol1 == 0).sum() / N * 100

    print(f"  50ms 비중:")
    print(f"    raw argmin (동률 → 50ms): {pct_raw:.1f}%")
    print(f"    tolerance 0 + 랜덤 동률:  {pct_tol0:.1f}%")
    print(f"    tolerance 1%p + 랜덤 동률: {pct_tol1:.1f}%")
    print(f"  균등 분포 기대값: 10.0%")

    print()
    if pct_tol0 < pct_raw * 0.7:
        print("  → tolerance 0 만으로도 50ms 비중 크게 감소 → 정확 동률 artifact 비중 큼")
    elif pct_tol0 > pct_raw * 0.9:
        print("  → 정확 동률만으로는 잘 안 떨어짐 → 50ms 가 진짜로 GT 와 단독 일치하는 경우 많음")
    if pct_tol1 < 12:
        print("  → tolerance 1%p 적용 시 거의 균등 → '50ms 우월' 은 사실상 가까운 동률들의 모음")


if __name__ == "__main__":
    sys.exit(main())
