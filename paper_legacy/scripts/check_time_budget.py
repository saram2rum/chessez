#!/usr/bin/env python3
"""정책 mean_t=265.7ms 의 진짜 공정 검증 — wu 를 시간별로 측정.

만약 wu(300) 이 정책(265.7) 보다 좋다면, 정책의 우월은 단순 시간 예산 우위.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import chess
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.gaussian_weighted_mae import (  # noqa: E402
    bernoulli_variance_weight_array,
    gaussian_weight_array,
)
from app.utils.position_complexity import _extract_features  # noqa: E402

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


def weighted_mean(values, weights):
    sw = float(weights.sum())
    if sw <= 0:
        return float("nan")
    return float((values * weights).sum() / sw)


def main():
    fens, pred_mat, gt_vec = load_raw_grid(RAW_DIR, SEARCHTIMES_MS)
    N = len(fens)
    err_mat = np.abs(pred_mat - gt_vec[:, None])

    w_mae = np.ones(N, dtype=np.float64)
    w_gw = gaussian_weight_array(gt_vec, 0.10)
    w_bv = bernoulli_variance_weight_array(gt_vec)

    print(f"FEN {N}개\n")
    print("=== 시간별 uniform_t 성능 (모든 위치 같은 t) ===")
    print(f"{'time':>5}  {'MAE':>8}  {'gwmae':>8}  {'Bern':>8}")
    print("-" * 40)
    for j, t in enumerate(SEARCHTIMES_MS):
        m = weighted_mean(err_mat[:, j], w_mae)
        g = weighted_mean(err_mat[:, j], w_gw)
        b = weighted_mean(err_mat[:, j], w_bv)
        print(f"  {t:>3}  {m:8.4f}  {g:8.4f}  {b:8.4f}")

    # 선형 보간으로 wu(265.7) 추정
    target_t = 265.7
    j_lo = max(j for j, t in enumerate(SEARCHTIMES_MS) if t <= target_t)
    j_hi = j_lo + 1
    t_lo, t_hi = SEARCHTIMES_MS[j_lo], SEARCHTIMES_MS[j_hi]
    alpha = (target_t - t_lo) / (t_hi - t_lo)
    print(f"\n=== wu({target_t}ms) 선형 보간 (between {t_lo}ms and {t_hi}ms, α={alpha:.3f}) ===")
    for name, w in [("MAE", w_mae), ("gwmae", w_gw), ("Bern", w_bv)]:
        v_lo = weighted_mean(err_mat[:, j_lo], w)
        v_hi = weighted_mean(err_mat[:, j_hi], w)
        v_int = v_lo + alpha * (v_hi - v_lo)
        print(f"  {name}: wu({t_lo})={v_lo:.4f}, wu({t_hi})={v_hi:.4f}, 보간 wu({target_t})={v_int:.4f}")

    # 정책 imb≥6, f≥3, S=100/M=250/L=400 재현해서 비교
    print("\n=== 정책 imb≥6, f≥3, S=100/M=250/L=400 vs wu(265.7 보간) ===")
    imb = np.zeros(N)
    minfor = np.zeros(N)
    for i, f in enumerate(fens):
        b = chess.Board(f)
        feat = _extract_features(b)
        imb[i] = feat.material_imbalance
        minfor[i] = min(feat.white_forcing, feat.black_forcing)
    bucket = np.full(N, "M", dtype="<U1")
    bucket[imb >= 6] = "S"
    is_L = (minfor >= 3) & (bucket != "S")
    bucket[is_L] = "L"
    time_map = {"S": 100, "M": 250, "L": 400}
    policy_ms = np.array([time_map[b] for b in bucket], dtype=np.float64)
    t2i = {t: i for i, t in enumerate(SEARCHTIMES_MS)}
    policy_idx = np.array([t2i[int(t)] for t in policy_ms])
    err_policy = err_mat[np.arange(N), policy_idx]
    print(f"정책 평균 시간: {policy_ms.mean():.2f}ms (격자 정확)")

    for name, w in [("MAE", w_mae), ("gwmae", w_gw), ("Bern", w_bv)]:
        p_m = weighted_mean(err_policy, w)
        v_lo = weighted_mean(err_mat[:, j_lo], w)
        v_hi = weighted_mean(err_mat[:, j_hi], w)
        v_int = v_lo + alpha * (v_hi - v_lo)
        # 보간 wu 와 비교
        d_vs_interp = p_m - v_int
        rel_vs_interp = -d_vs_interp / v_int * 100
        # wu(250) 도 같이
        wu250 = weighted_mean(err_mat[:, t2i[250]], w)
        d_vs_250 = p_m - wu250
        rel_vs_250 = -d_vs_250 / wu250 * 100
        # wu(300) 도
        wu300 = weighted_mean(err_mat[:, t2i[300]], w)
        d_vs_300 = p_m - wu300
        rel_vs_300 = -d_vs_300 / wu300 * 100
        print(f"\n  {name}: policy = {p_m:.4f}")
        print(f"    vs wu(250) = {wu250:.4f} : diff = {d_vs_250:+.4f} ({rel_vs_250:+.2f}%)")
        print(f"    vs wu(266) 보간 = {v_int:.4f} : diff = {d_vs_interp:+.4f} ({rel_vs_interp:+.2f}%)")
        print(f"    vs wu(300) = {wu300:.4f} : diff = {d_vs_300:+.4f} ({rel_vs_300:+.2f}%)")


if __name__ == "__main__":
    sys.exit(main())
