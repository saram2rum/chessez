#!/usr/bin/env python3
"""cp50_abs quintile 정책 vs wu(250) — MAE / gwmae σ=0.10 / 베르누이 분산 비교.

기존 analyze_cp50_policy.py 와 동일한 정책·데이터를 쓰되, 세 가지 가중 메트릭으로
나란히 측정해 베르누이 분산 가중이 σ=0.10 결과를 재현하는지 검증한다.

세 가중치:
  - MAE        : w = 1 (균등)
  - gwmae 0.10 : w = exp(-((gt/100 - 0.5)^2) / (2 * 0.1^2))
  - Bernoulli  : w = (gt/100) * (1 - gt/100)   ← 하이퍼파라미터 없음

각 메트릭에서 (a) 정책 평균, (b) wu(250) 평균, (c) diff = policy - wu,
(d) paired bootstrap 95% CI 를 출력. 라그랑주 천장도 세 메트릭별로 계산.
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

RAW_DIR = PROJECT_ROOT / "searchtime_raw_local"
SEARCHTIMES_MS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
POLICY_TIMES = [50, 150, 250, 350, 450]  # quintile 평균 250ms
N_BOOT = 10_000
TARGET_MS = 250.0
MATE_CP_SENTINEL = 10_000.0


def load_raw_grid(raw_dir, search_times):
    pred_by_st, gt_map, cp_by_st = {}, {}, {}
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
    """values 큰 순서로 5분할 → times[0]~times[4]. 시드 고정."""
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


def weighted_mean(values, weights):
    sw = float(weights.sum())
    if sw <= 0:
        return float("nan")
    return float((values * weights).sum() / sw)


def boot_ci_weighted(a, b, w, n_boot, seed=42, chunk=1000):
    """mean_w(A) - mean_w(B) paired bootstrap CI95."""
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
    """평균 시간 제약 하 라그랑주 최적 분배. 반환: (time_idx[N], 평균 시간)."""
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


def analyze_metric(name, w, err_policy, err_wu, err_mat, times_arr, sigma_label=""):
    """단일 메트릭 분석: policy/wu/ceiling 평균, diff, CI, 천장 시간 분포."""
    p_mean = weighted_mean(err_policy, w)
    wu_mean = weighted_mean(err_wu, w)
    ci_pv = boot_ci_weighted(err_policy, err_wu, w, N_BOOT)
    d_pv = p_mean - wu_mean
    v_pv = verdict(*ci_pv)

    # ceiling
    ceil_idx, ceil_mt = ceiling_lagrangian(err_mat, w, times_arr, TARGET_MS)
    err_ceil = err_mat[np.arange(len(err_mat)), ceil_idx]
    ceil_mean = weighted_mean(err_ceil, w)
    ci_pc = boot_ci_weighted(err_policy, err_ceil, w, N_BOOT)
    ci_wc = boot_ci_weighted(err_wu, err_ceil, w, N_BOOT)
    hist = {int(t): int((times_arr[ceil_idx] == t).sum()) for t in SEARCHTIMES_MS}

    return {
        "name": name,
        "sigma_label": sigma_label,
        "policy": p_mean,
        "wu": wu_mean,
        "ceiling": ceil_mean,
        "ceil_mean_time_ms": ceil_mt,
        "diff_policy_minus_wu": d_pv,
        "ci_policy_minus_wu": ci_pv,
        "verdict_policy_vs_wu": v_pv,
        "diff_policy_minus_ceil": p_mean - ceil_mean,
        "ci_policy_minus_ceil": ci_pc,
        "diff_wu_minus_ceil": wu_mean - ceil_mean,
        "ci_wu_minus_ceil": ci_wc,
        "ceil_time_histogram": hist,
        "sum_weights": float(w.sum()),
    }


def main():
    fens, pred_mat, cp_mat, gt_vec = load_raw_grid(RAW_DIR, SEARCHTIMES_MS)
    N = len(fens)
    err_mat = np.abs(pred_mat - gt_vec[:, None])
    times_arr = np.array(SEARCHTIMES_MS, dtype=np.float64)
    t2i = {st: i for i, st in enumerate(SEARCHTIMES_MS)}

    cp50_abs = np.abs(cp_mat[:, t2i[50]])
    print(f"FEN {N}개")
    print(f"|cp50| min={cp50_abs.min():.0f}, max={cp50_abs.max():.0f}, mean={cp50_abs.mean():.1f}")

    policy_ms = assign_quintile_desc(cp50_abs, POLICY_TIMES)
    print(f"policy 평균 시간 = {policy_ms.mean():.2f}ms")

    policy_idx = np.array([t2i[int(t)] for t in policy_ms])
    err_policy = err_mat[np.arange(N), policy_idx]
    err_wu = err_mat[:, t2i[250]]

    # 세 가중치 정의
    w_mae = np.ones(N, dtype=np.float64)
    w_gw = gaussian_weight_array(gt_vec, 0.10)
    w_bv = bernoulli_variance_weight_array(gt_vec)

    # gt 분포 별 가중치 모양 확인
    print("\n=== 가중치 형태 (gt 별 단위 가중치, max=1로 정규화) ===")
    print(f"{'gt%':>4}  {'MAE':>8}  {'gwmae 0.10':>11}  {'Bernoulli':>10}")
    bv_max = 0.25  # max of p(1-p) at p=0.5
    for gt in [0, 10, 25, 40, 50, 60, 75, 90, 100]:
        g = float(np.exp(-((gt/100 - 0.5)**2) / (2 * 0.10**2)))
        b = (gt/100) * (1 - gt/100) / bv_max
        print(f"  {gt:>3}  {1.0:>8.4f}  {g:>11.4f}  {b:>10.4f}")

    print(f"\n총 가중치 합:")
    print(f"  MAE       = {w_mae.sum():.2f}  (= N)")
    print(f"  gwmae0.10 = {w_gw.sum():.2f}  (effective n)")
    print(f"  Bernoulli = {w_bv.sum():.2f}  (effective n)")

    # 세 메트릭 분석
    results = []
    results.append(analyze_metric("MAE", w_mae, err_policy, err_wu, err_mat, times_arr))
    results.append(analyze_metric("gwmae", w_gw, err_policy, err_wu, err_mat, times_arr, sigma_label="σ=0.10"))
    results.append(analyze_metric("Bernoulli", w_bv, err_policy, err_wu, err_mat, times_arr))

    # 출력
    print("\n" + "=" * 90)
    print("정책(cp50_abs quintile) vs wu(250)  —  세 가중 메트릭 비교")
    print("=" * 90)
    print(f"{'metric':>14}  {'policy':>9}  {'wu(250)':>9}  {'diff':>10}  {'CI95':>22}  verdict")
    print("-" * 90)
    for r in results:
        label = f"{r['name']} {r['sigma_label']}".strip()
        ci = r["ci_policy_minus_wu"]
        print(f"  {label:>12}  {r['policy']:9.4f}  {r['wu']:9.4f}  "
              f"{r['diff_policy_minus_wu']:+10.4f}  "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]  {r['verdict_policy_vs_wu']}")

    print("\n" + "=" * 90)
    print("천장(라그랑주, 평균≤250ms) 비교")
    print("=" * 90)
    print(f"{'metric':>14}  {'ceiling':>9}  {'mean_t':>8}  "
          f"{'policy-ceil':>11}  {'wu-ceil':>9}")
    print("-" * 90)
    for r in results:
        label = f"{r['name']} {r['sigma_label']}".strip()
        print(f"  {label:>12}  {r['ceiling']:9.4f}  {r['ceil_mean_time_ms']:8.1f}  "
              f"{r['diff_policy_minus_ceil']:+11.4f}  "
              f"{r['diff_wu_minus_ceil']:+9.4f}")

    print("\n" + "=" * 90)
    print("천장 시간 분포")
    print("=" * 90)
    header = "  " + " ".join(f"{t:>5}ms" for t in SEARCHTIMES_MS)
    print(f"  {'metric':>14} " + header)
    for r in results:
        label = f"{r['name']} {r['sigma_label']}".strip()
        cells = " ".join(f"{r['ceil_time_histogram'].get(t, 0):>7d}" for t in SEARCHTIMES_MS)
        print(f"  {label:>14}  {cells}")

    # 핵심: gwmae 와 Bernoulli 결과 비교 — 우월성 유지 여부
    gw = results[1]
    bv = results[2]
    print("\n" + "=" * 90)
    print("핵심 비교: gwmae σ=0.10 → Bernoulli 전환 영향")
    print("=" * 90)
    print(f"  gwmae σ=0.10  policy-wu = {gw['diff_policy_minus_wu']:+.4f}  "
          f"({gw['verdict_policy_vs_wu']})")
    print(f"  Bernoulli     policy-wu = {bv['diff_policy_minus_wu']:+.4f}  "
          f"({bv['verdict_policy_vs_wu']})")
    # 상대적 우월 비율 (% of wu)
    if gw['wu'] > 0 and bv['wu'] > 0:
        gw_rel = -gw['diff_policy_minus_wu'] / gw['wu'] * 100.0
        bv_rel = -bv['diff_policy_minus_wu'] / bv['wu'] * 100.0
        print(f"\n  wu 대비 상대 개선:")
        print(f"    gwmae σ=0.10 : {gw_rel:+.2f}%")
        print(f"    Bernoulli    : {bv_rel:+.2f}%")


if __name__ == "__main__":
    sys.exit(main())
