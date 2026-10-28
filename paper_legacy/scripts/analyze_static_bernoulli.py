#!/usr/bin/env python3
"""정적 FEN 휴리스틱(S/M/L 버킷) 정책 vs wu — MAE / gwmae σ=0.10 / 베르누이 비교.

position_complexity.evaluate_position_complexity 로 각 FEN 을 S/M/L 로 분류하고,
EVAL_TIME_BY_BUCKET 매핑(S=100, M=250, L=400)을 sweep 시간 격자에 가장 가까운
값으로 근사한다. wu 는 정책의 평균 시간을 buckets 비중으로 가중평균해 같은
예산을 부여한다(weighted_uniform).

목적: cp50_abs 동적 정책의 σ=0.10 fragility 확인 이후, 정적 휴리스틱이
베르누이/MAE 에서도 wu 를 이기는지 검증.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.gaussian_weighted_mae import (  # noqa: E402
    bernoulli_variance_weight_array,
    gaussian_weight_array,
)
from app.utils.position_complexity import (  # noqa: E402
    EVAL_TIME_BY_BUCKET,
    evaluate_position_complexity,
)

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


def closest_idx(t, times):
    return int(np.argmin([abs(x - t) for x in times]))


def weighted_mean(values, weights):
    sw = float(weights.sum())
    if sw <= 0:
        return float("nan")
    return float((values * weights).sum() / sw)


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
    return assign(hi)


def verdict(ci_lo, ci_hi):
    if ci_hi < 0:
        return "정책 우월"
    if ci_lo > 0:
        return "wu 우월"
    return "유의차 없음"


def main():
    fens, pred_mat, gt_vec = load_raw_grid(RAW_DIR, SEARCHTIMES_MS)
    N = len(fens)
    err_mat = np.abs(pred_mat - gt_vec[:, None])
    times_arr = np.array(SEARCHTIMES_MS, dtype=np.float64)
    t2i = {st: i for i, st in enumerate(SEARCHTIMES_MS)}

    print(f"FEN {N}개")
    print(f"S/M/L 매핑: S={EVAL_TIME_BY_BUCKET['S']}ms, "
          f"M={EVAL_TIME_BY_BUCKET['M']}ms, L={EVAL_TIME_BY_BUCKET['L']}ms")

    # 각 FEN 의 버킷 → 시간 → sweep 격자에서 가장 가까운 인덱스
    buckets = []
    for f in fens:
        buckets.append(evaluate_position_complexity(f).bucket)
    bucket_arr = np.array(buckets)

    counts = {b: int((bucket_arr == b).sum()) for b in ["S", "M", "L"]}
    print(f"버킷 분포: S={counts['S']} ({100*counts['S']/N:.1f}%), "
          f"M={counts['M']} ({100*counts['M']/N:.1f}%), "
          f"L={counts['L']} ({100*counts['L']/N:.1f}%)")

    policy_ms = np.array([EVAL_TIME_BY_BUCKET[b] for b in buckets])
    policy_idx = np.array([closest_idx(t, SEARCHTIMES_MS) for t in policy_ms])
    policy_actual_t = times_arr[policy_idx]
    mean_t_policy = float(policy_actual_t.mean())
    print(f"정책 평균 시간 (sweep 격자 근사 후) = {mean_t_policy:.2f}ms")

    # wu: 같은 평균 예산을 단일 시간으로 (가장 가까운 격자값)
    wu_idx_scalar = closest_idx(mean_t_policy, SEARCHTIMES_MS)
    wu_t = SEARCHTIMES_MS[wu_idx_scalar]
    print(f"wu 단일 시간 = {wu_t}ms")

    err_policy = err_mat[np.arange(N), policy_idx]
    err_wu = err_mat[:, wu_idx_scalar]

    # gt 분포 by bucket — 정책이 어떤 gt 영역을 어떻게 다루나
    print("\n=== 버킷별 gt(win%) 분포 ===")
    for b in ["S", "M", "L"]:
        mask = bucket_arr == b
        if mask.sum() == 0:
            continue
        g = gt_vec[mask]
        print(f"  {b} (n={int(mask.sum())}, t={EVAL_TIME_BY_BUCKET[b]}ms): "
              f"gt mean={g.mean():.1f}, |gt-50| mean={np.abs(g-50).mean():.1f}, "
              f"|gt-50|<10 비중={100*(np.abs(g-50)<10).mean():.1f}%")

    # 세 가중치
    w_mae = np.ones(N, dtype=np.float64)
    w_gw = gaussian_weight_array(gt_vec, 0.10)
    w_bv = bernoulli_variance_weight_array(gt_vec)

    rows = []
    for name, w, label in [
        ("MAE", w_mae, ""),
        ("gwmae", w_gw, "σ=0.10"),
        ("Bernoulli", w_bv, ""),
    ]:
        p_m = weighted_mean(err_policy, w)
        wu_m = weighted_mean(err_wu, w)
        ci = boot_ci_weighted(err_policy, err_wu, w, N_BOOT)
        d = p_m - wu_m
        v = verdict(*ci)

        ceil_idx, ceil_mt = ceiling_lagrangian(err_mat, w, times_arr, mean_t_policy)
        err_ceil = err_mat[np.arange(N), ceil_idx]
        ceil_m = weighted_mean(err_ceil, w)

        rows.append({
            "name": name, "label": label,
            "policy": p_m, "wu": wu_m, "diff": d, "ci": ci, "verdict": v,
            "ceiling": ceil_m, "ceil_mt": ceil_mt,
        })

    print("\n" + "=" * 90)
    print(f"정적 FEN 휴리스틱(S/M/L) 정책 vs wu({wu_t}ms)")
    print("=" * 90)
    print(f"{'metric':>14}  {'policy':>9}  {'wu':>9}  {'diff':>10}  {'CI95':>22}  verdict")
    print("-" * 90)
    for r in rows:
        m = f"{r['name']} {r['label']}".strip()
        ci = r["ci"]
        print(f"  {m:>12}  {r['policy']:9.4f}  {r['wu']:9.4f}  "
              f"{r['diff']:+10.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]  {r['verdict']}")

    print("\n" + "=" * 90)
    print(f"천장(평균≤{mean_t_policy:.0f}ms)")
    print("=" * 90)
    print(f"{'metric':>14}  {'ceiling':>9}  {'mean_t':>8}  "
          f"{'policy-ceil':>11}  {'wu-ceil':>9}")
    print("-" * 90)
    for r in rows:
        m = f"{r['name']} {r['label']}".strip()
        print(f"  {m:>12}  {r['ceiling']:9.4f}  {r['ceil_mt']:8.1f}  "
              f"{r['policy']-r['ceiling']:+11.4f}  {r['wu']-r['ceiling']:+9.4f}")

    # 상대 개선
    print("\n" + "=" * 90)
    print("wu 대비 상대 개선 (% improvement, 양수=정책 우월)")
    print("=" * 90)
    for r in rows:
        m = f"{r['name']} {r['label']}".strip()
        if r['wu'] > 0:
            rel = -r['diff'] / r['wu'] * 100
            print(f"  {m:>14}: {rel:+.2f}%   ({r['verdict']})")


if __name__ == "__main__":
    sys.exit(main())
