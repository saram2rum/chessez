#!/usr/bin/env python3
"""cp50_abs (50ms 평가의 |cp|) 기반 quintile 정책 vs wu(250).

설계: 모든 FEN 을 우선 50ms 로 평가하고, |cp50| 의 크기에 따라
  cp50_abs 큰 순서대로 quintile → [50, 150, 250, 350, 450]ms 매핑
  (큰 |cp50| = 이미 결판난 평가 → 짧은 시간으로 종료)
평균 총 시간 = 250ms 정확. wu(250 단일 평가)와 동일 예산.

비교: MAE + gwmae σ ∈ {0.05~0.25}.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.gaussian_weighted_mae import gaussian_weight_array  # noqa: E402

RAW_DIR = PROJECT_ROOT / "searchtime_raw_local"
SEARCHTIMES_MS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.25]
POLICY_TIMES = [50, 150, 250, 350, 450]  # quintile 평균 250
N_BOOT = 10_000


def load_raw_grid(raw_dir, search_times):
    pred_by_st, gt_map = {}, {}
    cp_by_st = {}
    # mate 평가는 cp 가 빈 칸 → 극단값(10000)으로 치환해서 |cp| 최대로.
    MATE_CP_SENTINEL = 10_000.0
    for st in search_times:
        path = raw_dir / f"searchtime_{st}ms_raw.csv"
        pmap, cmap = {}, {}
        with path.open() as fp:
            for row in csv.DictReader(fp):
                f = row["fen"]
                pred = float(row["pred_win_pct_0_100"])
                pmap[f] = pred
                cs = row.get("cp", "")
                if cs:
                    cmap[f] = float(cs)
                else:
                    # mate: pred_win_pct 가 100 또는 0 → 부호로 ±sentinel
                    cmap[f] = MATE_CP_SENTINEL if pred >= 50 else -MATE_CP_SENTINEL
                gt_map.setdefault(f, float(row["ground_truth_win_pct_0_100"]))
        pred_by_st[st] = pmap
        cp_by_st[st] = cmap
    fens_common = set(gt_map)
    for st in search_times:
        fens_common &= set(pred_by_st[st])
    fens = sorted(fens_common)
    N, T = len(fens), len(search_times)
    pred_mat = np.zeros((N, T))
    cp_mat = np.zeros((N, T))
    gt_vec = np.array([gt_map[f] for f in fens])
    for j, st in enumerate(search_times):
        pmap, cmap = pred_by_st[st], cp_by_st[st]
        pred_mat[:, j] = [pmap[f] for f in fens]
        cp_mat[:, j] = [cmap[f] for f in fens]
    return fens, pred_mat, cp_mat, gt_vec


def assign_quintile_desc(values, times):
    """values 큰 순서로 5분할 → times[0]~times[4]."""
    N = len(values)
    rng = np.random.default_rng(0)
    shuffle_idx = rng.permutation(N)
    order_in_shuffle = np.argsort(-values[shuffle_idx], kind="stable")
    final_order = shuffle_idx[order_in_shuffle]
    assigned = np.zeros(N, dtype=np.int64)
    bucket_size = N // 5
    rem = N % 5
    start = 0
    for i, t in enumerate(times):
        sz = bucket_size + (1 if i < rem else 0)
        end = start + sz
        assigned[final_order[start:end]] = t
        start = end
    return assigned


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
    fens, pred_mat, cp_mat, gt_vec = load_raw_grid(RAW_DIR, SEARCHTIMES_MS)
    N = len(fens)
    err_mat = np.abs(pred_mat - gt_vec[:, None])
    t2i = {st: i for i, st in enumerate(SEARCHTIMES_MS)}

    cp50_abs = np.abs(cp_mat[:, t2i[50]])
    print(f"FEN {N}개 | |cp50| min={cp50_abs.min():.0f}, max={cp50_abs.max():.0f}, mean={cp50_abs.mean():.1f}")

    policy_ms = assign_quintile_desc(cp50_abs, POLICY_TIMES)
    print(f"\npolicy 평균 시간: {policy_ms.mean():.2f}ms\n")
    print("quintile 그룹별:")
    for t in POLICY_TIMES:
        m = policy_ms == t
        n_t = int(m.sum())
        cp_in = cp50_abs[m]
        gt_in = gt_vec[m]
        print(f"  {t:>3}ms: n={n_t} | |cp50| range [{cp_in.min():.0f}, {cp_in.max():.0f}] mean={cp_in.mean():.1f} | gt win% mean={gt_in.mean():.2f}")

    policy_idx = np.array([t2i[int(t)] for t in policy_ms])
    err_policy = err_mat[np.arange(N), policy_idx]
    err_wu = err_mat[:, t2i[250]]

    mae_p = float(err_policy.mean())
    mae_wu = float(err_wu.mean())
    ci_mae = boot_ci_uniform(err_policy, err_wu, N_BOOT)
    d_mae = mae_p - mae_wu
    v_mae = "정책 우월" if ci_mae[1] < 0 else ("wu 우월" if ci_mae[0] > 0 else "유의차 없음")
    print(f"\n=== MAE ===")
    print(f"  policy = {mae_p:.4f}, wu = {mae_wu:.4f}, diff = {d_mae:+.4f}  CI95 [{ci_mae[0]:+.4f}, {ci_mae[1]:+.4f}]  → {v_mae}")

    print(f"\n=== gwmae ===")
    print(f"{'σ':>6}  {'policy':>9}  {'wu':>9}  {'diff':>10}  {'CI95':>22}  verdict")
    print("-" * 78)
    for sigma in SIGMAS:
        w = gaussian_weight_array(gt_vec, sigma)
        gw_p = float((err_policy * w).sum() / w.sum())
        gw_wu = float((err_wu * w).sum() / w.sum())
        ci = boot_ci_weighted(err_policy, err_wu, w, N_BOOT)
        d = gw_p - gw_wu
        v = "정책 우월" if ci[1] < 0 else ("wu 우월" if ci[0] > 0 else "유의차 없음")
        print(f"  {sigma:.2f}  {gw_p:9.4f}  {gw_wu:9.4f}  {d:+10.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]  {v}")


if __name__ == "__main__":
    sys.exit(main())
