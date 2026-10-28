#!/usr/bin/env python3
"""각 feature 의 Spearman ρ + p-value + ANOVA F + F 의 p-value 정확 계산."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import chess
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def main():
    fens, pred_mat, gt_vec = load_raw_grid(RAW_DIR, SEARCHTIMES_MS)
    N = len(fens)
    err_mat = np.abs(pred_mat - gt_vec[:, None])
    times_arr = np.array(SEARCHTIMES_MS, dtype=np.float64)

    best_t_idx = np.argmin(err_mat, axis=1)
    best_t_ms = times_arr[best_t_idx]

    feats = {
        "imbalance": [], "capture_count": [], "legal_moves": [],
        "total_material": [], "forcing_total": [], "is_check": [], "is_endgame": [],
    }
    for fen in fens:
        board = chess.Board(fen)
        f = _extract_features(board)
        feats["imbalance"].append(f.material_imbalance)
        feats["capture_count"].append(f.capture_count)
        feats["legal_moves"].append(f.legal_moves)
        feats["total_material"].append(f.total_material)
        feats["forcing_total"].append(f.white_forcing + f.black_forcing)
        feats["is_check"].append(int(f.is_check))
        feats["is_endgame"].append(int(f.is_endgame))
    feats = {k: np.asarray(v) for k, v in feats.items()}

    print(f"N = {N}\n")
    print(f"{'feature':<20}{'Spearman ρ':>12}{'p-value':>14}{'ANOVA F':>12}{'F p-value':>14}{'유의?':>10}")
    print("-" * 82)
    bonferroni_alpha = 0.05 / 7

    for fname, vals in feats.items():
        rho, p_rho = stats.spearmanr(vals, best_t_ms)
        # ANOVA F: feature ~ best_t (10 groups)
        groups = [vals[best_t_idx == j] for j in range(len(SEARCHTIMES_MS))]
        groups = [g for g in groups if len(g) > 0]
        f_stat, p_f = stats.f_oneway(*groups)
        sig = "유의" if p_rho < 0.05 else "무의"
        bonf = "(Bonf 통과)" if p_rho < bonferroni_alpha else ""
        print(f"{fname:<20}{rho:>+12.4f}{p_rho:>14.4f}{f_stat:>12.2f}{p_f:>14.4f}  {sig:>6} {bonf}")

    print(f"\nBonferroni 보정 임계값 (7개 동시 검정): α = 0.05/7 ≈ {bonferroni_alpha:.4f}")


if __name__ == "__main__":
    sys.exit(main())
