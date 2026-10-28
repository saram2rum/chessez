#!/usr/bin/env python3
"""사용자 지정 정책 두 옵션 동시 비교.

옵션 1: 이전 매핑 + forcing 시간 50ms 씩 축소
  imb ≥ 9: 50ms,  7~8: 100ms,  5~6: 150ms,  4: 200ms,  3: 250ms,  2: 300ms
  imb 0~1: forcing 0→250, 1→300, 2→350, 3→400, 4+→450

옵션 2: imb 5+ 짧은 시간 + forcing 6단계
  imb ≥ 5: 50ms,  4: 100ms,  3: 150ms,  2: 200ms
  imb 0~1: forcing 0→250, 1→300, 2→350, 3→400, 4→450, 5+→500
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


def assign_time_opt1(imb, forcing):
    """옵션 1: 이전 매핑 + forcing 시간 50ms 씩 줄임."""
    if imb >= 9:
        return 50
    elif imb in (7, 8):
        return 100
    elif imb in (5, 6):
        return 150
    elif imb == 4:
        return 200
    elif imb == 3:
        return 250
    elif imb == 2:
        return 300
    else:  # imb 0 or 1
        if forcing == 0:
            return 250
        elif forcing == 1:
            return 300
        elif forcing == 2:
            return 350
        elif forcing == 3:
            return 400
        else:  # forcing >= 4
            return 450


def assign_time_opt2(imb, forcing):
    """옵션 2: imb 5+ 짧은 시간 + forcing 6단계 (5+→500)."""
    if imb >= 5:
        return 50
    elif imb == 4:
        return 100
    elif imb == 3:
        return 150
    elif imb == 2:
        return 200
    else:  # imb 0 or 1
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


def evaluate_policy(name, assign_fn, fens, imb_arr, forcing_arr, err_mat, t2i, N):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    policy_ms = np.array([assign_fn(int(imb_arr[i]), int(forcing_arr[i])) for i in range(N)])
    policy_idx = np.array([t2i[int(t)] for t in policy_ms])
    err_policy = err_mat[np.arange(N), policy_idx]

    mean_t = float(policy_ms.mean())
    mae_policy = float(err_policy.mean())

    print(f"\n시간 분포:")
    print(f"{'시간(ms)':>10}  {'FEN 수':>8}  {'비중':>8}")
    for t in SEARCHTIMES_MS:
        mask = policy_ms == t
        n = int(mask.sum())
        if n > 0:
            print(f"  {t:>8}  {n:>8}  {n/N*100:>6.1f}%")

    print(f"\n평균 시간: {mean_t:.2f}ms,  MAE: {mae_policy:.4f}")

    # vs wu(250)
    err_wu250 = err_mat[:, t2i[250]]
    mae_wu250 = float(err_wu250.mean())
    diff_250 = mae_policy - mae_wu250
    ci_250 = boot_ci(err_policy, err_wu250, N_BOOT)
    if ci_250[1] < 0:
        v_250 = "정책 우월"
    elif ci_250[0] > 0:
        v_250 = "wu 우월"
    else:
        v_250 = "유의차 없음"
    rel_250 = -diff_250 / mae_wu250 * 100

    print(f"\n[vs wu(250) = 1.8029]")
    print(f"  diff: {diff_250:+.4f},  CI95: [{ci_250[0]:+.4f}, {ci_250[1]:+.4f}],  "
          f"rel: {rel_250:+.2f}%,  {v_250}")

    # 보간 wu (공정 비교)
    j_lo = max((j for j, t in enumerate(SEARCHTIMES_MS) if t <= mean_t), default=0)
    j_hi = min(len(SEARCHTIMES_MS) - 1, j_lo + 1)
    t_lo, t_hi = SEARCHTIMES_MS[j_lo], SEARCHTIMES_MS[j_hi]
    if t_lo == t_hi:
        v_interp = float(err_mat[:, j_lo].mean())
    else:
        alpha = (mean_t - t_lo) / (t_hi - t_lo)
        v_lo = float(err_mat[:, j_lo].mean())
        v_hi = float(err_mat[:, j_hi].mean())
        v_interp = v_lo + alpha * (v_hi - v_lo)
    diff_interp = mae_policy - v_interp
    rel_interp = -diff_interp / v_interp * 100
    print(f"\n[vs wu({mean_t:.0f}ms 보간) = {v_interp:.4f}  공정 비교]")
    print(f"  diff: {diff_interp:+.4f},  rel: {rel_interp:+.2f}%")
    if diff_interp < -0.001:
        print(f"  → 같은 시간 예산에서 정책 우월")
    elif diff_interp > 0.001:
        print(f"  → 같은 시간 예산에서 wu 우월")
    else:
        print(f"  → 공정 비교 사실상 동등")

    return {
        "name": name,
        "mean_t": mean_t,
        "mae": mae_policy,
        "diff_vs_wu250": diff_250,
        "ci_vs_wu250": ci_250,
        "verdict_vs_wu250": v_250,
        "rel_vs_wu250": rel_250,
        "mae_wu_interp": v_interp,
        "diff_vs_wu_fair": diff_interp,
        "rel_vs_wu_fair": rel_interp,
    }


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

    res1 = evaluate_policy(
        "옵션 1: 이전 매핑 + forcing 50ms 씩 축소",
        assign_time_opt1, fens, imb_arr, forcing_arr, err_mat, t2i, N
    )
    res2 = evaluate_policy(
        "옵션 2: imb 5+ 50ms + forcing 6단계 (5+ → 500)",
        assign_time_opt2, fens, imb_arr, forcing_arr, err_mat, t2i, N
    )

    # 비교 요약
    print(f"\n{'='*70}")
    print(f"  요약 비교")
    print(f"{'='*70}")
    print(f"{'옵션':<6}  {'mean_t':>7}  {'MAE':>7}  {'vs wu(250)':>11}  {'CI95':>20}  "
          f"{'판정':>10}  {'vs wu_fair':>11}")
    for r in [res1, res2]:
        ci = r["ci_vs_wu250"]
        print(f"  {r['name'][:5]:<6}  {r['mean_t']:7.1f}  {r['mae']:7.4f}  "
              f"{r['diff_vs_wu250']:+11.4f}  "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]  {r['verdict_vs_wu250']:>10}  "
              f"{r['rel_vs_wu_fair']:+10.2f}%")


if __name__ == "__main__":
    sys.exit(main())
