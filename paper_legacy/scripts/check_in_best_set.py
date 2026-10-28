#!/usr/bin/env python3
"""각 시간이 'best set' (최소 오차와 같은 시간들의 집합) 에 포함되는 FEN 비중.

랜덤 처리 없이 결정적 카운트. 한 FEN 이 여러 시간에서 동률이면 그 시간들 모두에 +1.
합은 100% 초과 가능.
"""
from __future__ import annotations

import csv
import sys
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


def main():
    fens, pred_mat, gt_vec = load_raw_grid(RAW_DIR, SEARCHTIMES_MS)
    N = len(fens)
    err_mat = np.abs(pred_mat - gt_vec[:, None])

    print(f"FEN {N}개\n")

    # 각 FEN 의 min err
    min_err = err_mat.min(axis=1, keepdims=True)

    # 각 시간이 min 과 정확히 일치하는 FEN 카운트
    in_best_set = (err_mat == min_err)  # shape (N, T), bool

    print("="*70)
    print("각 시간이 '최소 오차에 포함되는' FEN 비중 (동률 모두 카운트)")
    print("="*70)
    print(f"{'time':>6}  {'FEN 수':>8}  {'비중':>8}")
    for j, t in enumerate(SEARCHTIMES_MS):
        c = int(in_best_set[:, j].sum())
        print(f"  {t:>3}  {c:>8}  {c/N*100:>7.2f}%")
    total = int(in_best_set.sum())
    print(f"\n  총합: {total} (= 평균 {total/N:.2f}개 시간/FEN 동률)")

    # 각 FEN 의 동률 시간 개수 분포
    tie_count_per_fen = in_best_set.sum(axis=1)
    print(f"\nFEN 의 동률 시간 개수 분포:")
    print(f"  1개 (단독 best): {int((tie_count_per_fen == 1).sum())} ({int((tie_count_per_fen == 1).sum())/N*100:.1f}%)")
    print(f"  2개: {int((tie_count_per_fen == 2).sum())} ({int((tie_count_per_fen == 2).sum())/N*100:.1f}%)")
    print(f"  3개: {int((tie_count_per_fen == 3).sum())} ({int((tie_count_per_fen == 3).sum())/N*100:.1f}%)")
    print(f"  4~9개: {int(((tie_count_per_fen >= 4) & (tie_count_per_fen <= 9)).sum())}")
    print(f"  10개 (모두 동률): {int((tie_count_per_fen == 10).sum())} ({int((tie_count_per_fen == 10).sum())/N*100:.1f}%)")

    # 단독 best 만 카운트 (한 FEN 에서 그 시간만 min)
    print(f"\n{'='*70}")
    print("각 시간이 '단독 best' (다른 시간 모두 더 큰 오차) 인 FEN 비중")
    print(f"{'='*70}")
    unique_best = in_best_set & (tie_count_per_fen[:, None] == 1)
    print(f"{'time':>6}  {'FEN 수':>8}  {'비중':>8}")
    for j, t in enumerate(SEARCHTIMES_MS):
        c = int(unique_best[:, j].sum())
        print(f"  {t:>3}  {c:>8}  {c/N*100:>7.2f}%")
    print(f"\n  총합: {int(unique_best.sum())} (= 동률 없는 FEN 1361개)")


if __name__ == "__main__":
    sys.exit(main())
