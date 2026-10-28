#!/usr/bin/env python3
"""사용자 지정 정책 v2: imbalance + forcing 결합 매핑.

매핑:
  imbalance ≥ 9  → 50ms
  imbalance 7~8  → 100ms
  imbalance 5~6  → 150ms
  imbalance 4    → 200ms
  imbalance 3    → 250ms
  imbalance 2    → 300ms
  imbalance 0~1  → forcing 기반:
                   forcing == 0  → 300ms
                   forcing == 1  → 350ms
                   forcing == 2  → 400ms
                   forcing == 3  → 450ms
                   forcing ≥ 4   → 500ms
  (forcing = min(white_forcing, black_forcing))
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

from app.utils.position_complexity import _extract_features  # noqa: E402

RAW_DIR = PROJECT_ROOT / "searchtime_raw_local"
SEARCHTIMES_MS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
N_BOOT = 10_000


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


def assign_time(imb, forcing):
    if imb >= 9:
        return 50
    elif imb in (7, 8):
        return 100
    elif imb in (5, 6):
        return 150
    elif imb in (3, 4):
        return 200
    else:  # imb <= 2
        if forcing == 0:
            return 250
        elif forcing == 1:
            return 300
        elif forcing == 2:
            return 350
        elif forcing == 3:
            return 400
        elif forcing == 4:
            return 450
        else:  # forcing >= 5
            return 500


def boot_ci(a, b, n_boot, seed=42, chunk=1000):
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


def main():
    fens, pred_mat, gt_vec = load_raw_grid(RAW_DIR, SEARCHTIMES_MS)
    N = len(fens)
    err_mat = np.abs(pred_mat - gt_vec[:, None])
    t2i = {st: i for i, st in enumerate(SEARCHTIMES_MS)}

    print(f"FEN {N}개 — feature 추출 중...")
    imb_values = []
    forcing_values = []
    for fen in fens:
        b = chess.Board(fen)
        f = _extract_features(b)
        imb_values.append(f.material_imbalance)
        forcing_values.append(min(f.white_forcing, f.black_forcing))
    imb_arr = np.array(imb_values, dtype=np.int64)
    forcing_arr = np.array(forcing_values, dtype=np.int64)

    # 정책 시간 할당
    policy_ms = np.array([assign_time(int(imb_arr[i]), int(forcing_arr[i])) for i in range(N)])
    policy_idx = np.array([t2i[int(t)] for t in policy_ms])
    err_policy = err_mat[np.arange(N), policy_idx]

    mean_t = float(policy_ms.mean())
    mae_policy = float(err_policy.mean())

    # 정책 시간별 분포
    print(f"\n정책 시간 분포:")
    print(f"{'시간(ms)':>10}  {'FEN 수':>8}  {'비중':>8}")
    for t in SEARCHTIMES_MS:
        mask = policy_ms == t
        n = int(mask.sum())
        if n > 0:
            print(f"  {t:>8}  {n:>8}  {n/N*100:>6.1f}%")

    # imbalance 0~1 의 forcing 분포 (확인용)
    print(f"\nimbalance 0~1 FEN 들의 forcing 분포:")
    mask_low_imb = (imb_arr <= 1)
    n_low = int(mask_low_imb.sum())
    print(f"  총 {n_low} FEN ({n_low/N*100:.1f}%)")
    for f_val in range(5):
        if f_val == 4:
            m = mask_low_imb & (forcing_arr >= 4)
            label = "≥4"
        else:
            m = mask_low_imb & (forcing_arr == f_val)
            label = str(f_val)
        n = int(m.sum())
        t = assign_time(0, f_val) if f_val < 4 else assign_time(0, 4)
        print(f"  forcing {label}: {n} FEN → {t}ms")

    print(f"\n정책 평균 시간: {mean_t:.2f}ms")
    print(f"정책 MAE: {mae_policy:.4f}")

    # 비교
    print(f"\n=== 다양한 baseline 과 비교 ===")
    for baseline_t in SEARCHTIMES_MS:
        wu_idx = t2i[baseline_t]
        err_wu = err_mat[:, wu_idx]
        mae_wu = float(err_wu.mean())
        diff = mae_policy - mae_wu
        ci = boot_ci(err_policy, err_wu, N_BOOT)
        if ci[1] < 0:
            v = "정책 우월"
        elif ci[0] > 0:
            v = "wu 우월"
        else:
            v = "유의차 없음"
        rel = -diff / mae_wu * 100
        marker = "←★" if baseline_t == 250 else "    "
        print(f"  wu({baseline_t}ms)  MAE={mae_wu:.4f}  diff={diff:+.4f}  "
              f"CI95=[{ci[0]:+.4f}, {ci[1]:+.4f}]  rel={rel:+.2f}%  {v}  {marker}")

    # 보간 wu (공정 비교)
    print(f"\n=== 보간 wu({mean_t:.0f}ms) — 공정 비교 ===")
    j_lo = max((j for j, t in enumerate(SEARCHTIMES_MS) if t <= mean_t), default=0)
    j_hi = min(len(SEARCHTIMES_MS) - 1, j_lo + 1)
    t_lo, t_hi = SEARCHTIMES_MS[j_lo], SEARCHTIMES_MS[j_hi]
    if t_lo == t_hi or t_hi == t_lo:
        v_interp = float(err_mat[:, j_lo].mean())
    else:
        alpha = (mean_t - t_lo) / (t_hi - t_lo)
        v_lo = float(err_mat[:, j_lo].mean())
        v_hi = float(err_mat[:, j_hi].mean())
        v_interp = v_lo + alpha * (v_hi - v_lo)
    diff_interp = mae_policy - v_interp
    rel_interp = -diff_interp / v_interp * 100
    print(f"  보간 wu({mean_t:.0f}ms) MAE = {v_interp:.4f}")
    print(f"  정책 MAE = {mae_policy:.4f}")
    print(f"  diff = {diff_interp:+.4f} ({rel_interp:+.2f}%)")
    if diff_interp < 0:
        print(f"  → 같은 시간 예산에서 정책 우월 ({rel_interp:+.2f}%)")
    elif diff_interp > 0:
        print(f"  → 같은 시간 예산에서 wu 우월 ({rel_interp:+.2f}%)")


if __name__ == "__main__":
    sys.exit(main())
