#!/usr/bin/env python3
"""극단 일방적 국면만 S(50ms), 나머지 다 250ms 인 단순 정책 시뮬.

여러 imbalance 컷오프에 대해 정책 비교:
  - thresholds: [3, 5, 9, 12, 15, 18, 25]
  - 각각 cutoff 이상 → 50ms, 미만 → 250ms
대상 메트릭: MAE, gwmae σ=0.10
비교 대상: wu(250ms 균등)
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

from app.utils.gaussian_weighted_mae import gaussian_weight_array  # noqa: E402
from app.utils.position_complexity import _extract_features  # noqa: E402

RAW_DIR = PROJECT_ROOT / "searchtime_raw_local"
SEARCHTIMES_MS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
THRESHOLDS = [3, 5, 9, 12, 15, 18, 25]
SIGMA = 0.10
N_BOOT = 10_000


def load_raw_grid(raw_dir, search_times):
    pred_by_st = {}
    gt_map = {}
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
        col = pred_by_st[st]
        pred_mat[:, j] = [col[f] for f in fens]
    return fens, pred_mat, gt_vec


def imbalance_of(fen):
    return _extract_features(chess.Board(fen)).material_imbalance


def boot_ci_uniform(a, b, n_boot, seed=42, chunk=1000):
    rng = np.random.default_rng(seed)
    N = len(a)
    diffs = np.empty(n_boot)
    done = 0
    while done < n_boot:
        sz = min(chunk, n_boot - done)
        idx = rng.integers(0, N, size=(sz, N))
        diffs[done:done+sz] = a[idx].mean(axis=1) - b[idx].mean(axis=1)
        done += sz
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def boot_ci_weighted(a, b, w, n_boot, seed=42, chunk=1000):
    rng = np.random.default_rng(seed)
    N = len(a)
    diffs = np.empty(n_boot)
    done = 0
    while done < n_boot:
        sz = min(chunk, n_boot - done)
        idx = rng.integers(0, N, size=(sz, N))
        wb = w[idx]
        sw = wb.sum(axis=1)
        aa = (a[idx] * wb).sum(axis=1) / sw
        bb = (b[idx] * wb).sum(axis=1) / sw
        diffs[done:done+sz] = aa - bb
        done += sz
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    fens, pred_mat, gt_vec = load_raw_grid(RAW_DIR, SEARCHTIMES_MS)
    N = len(fens)
    err_mat = np.abs(pred_mat - gt_vec[:, None])
    imbalances = np.array([imbalance_of(f) for f in fens])
    t2i = {st: i for i, st in enumerate(SEARCHTIMES_MS)}
    err_wu = err_mat[:, t2i[250]]
    w = gaussian_weight_array(gt_vec, SIGMA)
    mae_wu = float(err_wu.mean())
    gw_wu = float((err_wu * w).sum() / w.sum())

    print(f"FEN {N}개 | wu(250ms): MAE={mae_wu:.4f}, gwmae(σ={SIGMA})={gw_wu:.4f}")
    print(f"\n{'cutoff':>7}{'n_S':>6}{'%S':>7}  {'avg_t':>8}{'MAE':>10}{'ΔMAE':>10}{'MAE CI95':>22}  {'gwmae':>10}{'Δgwmae':>10}{'gwmae CI95':>22}")
    print("-" * 130)

    for cutoff in THRESHOLDS:
        mask_s = imbalances >= cutoff
        n_s = int(mask_s.sum())
        if n_s == 0:
            continue
        policy_idx = np.where(mask_s, t2i[50], t2i[250])
        err_policy = err_mat[np.arange(N), policy_idx]
        policy_ms = np.where(mask_s, 50, 250)
        avg_t = float(policy_ms.mean())

        mae_p = float(err_policy.mean())
        gw_p = float((err_policy * w).sum() / w.sum())
        d_mae = mae_p - mae_wu
        d_gw = gw_p - gw_wu
        ci_mae = boot_ci_uniform(err_policy, err_wu, N_BOOT)
        ci_gw = boot_ci_weighted(err_policy, err_wu, w, N_BOOT)
        print(
            f"{cutoff:>7}{n_s:>6}{n_s/N*100:>6.2f}%  {avg_t:>7.2f}ms"
            f"{mae_p:>10.4f}{d_mae:>+10.4f}  [{ci_mae[0]:+.4f},{ci_mae[1]:+.4f}]"
            f"  {gw_p:>10.4f}{d_gw:>+10.4f}  [{ci_gw[0]:+.4f},{ci_gw[1]:+.4f}]"
        )

    print("\n해석: Δ<0 이면 정책이 wu 보다 좋음. CI 가 0 안 포함하면 통계적 유의.")


if __name__ == "__main__":
    sys.exit(main())
