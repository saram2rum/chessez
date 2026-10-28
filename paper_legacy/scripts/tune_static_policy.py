#!/usr/bin/env python3
"""정적 휴리스틱 정책 튜닝 — 임계값/시간매핑 변형들의 wu 대비 성능.

feature 한 번만 추출, 5개 변형 구성으로 분류 다시. 각 변형을 MAE / gwmae σ=0.10 /
베르누이 세 메트릭으로 측정해 wu 를 이기는 조합을 탐색한다.

변형:
  V0 baseline       : S(imb≥9)=100, M=250, L(forcing≥5)=400
  V1 extreme times  : S=50,   L=500,  같은 임계값
  V2 relaxed thr    : S(imb≥6), L(forcing≥3), 같은 시간
  V3 both           : V1+V2 결합
  V4 aggressive     : S(imb≥4)=50, L(forcing≥2)=500
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
N_BOOT = 10_000

VARIANTS = [
    # name, S_imb_thr, L_forcing_thr, S_time, M_time, L_time
    ("V0 baseline",       9, 5, 100, 250, 400),
    ("V1 extreme times",  9, 5,  50, 250, 500),
    ("V2 relaxed thr",    6, 3, 100, 250, 400),
    ("V3 both",           6, 3,  50, 250, 500),
    ("V4 aggressive",     4, 2,  50, 250, 500),
]


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


def extract_all_features(fens):
    """각 FEN 에 대해 (imbalance, min_forcing) 만 뽑아 numpy 로."""
    imb = np.zeros(len(fens), dtype=np.float64)
    minfor = np.zeros(len(fens), dtype=np.float64)
    for i, f in enumerate(fens):
        board = chess.Board(f)
        feat = _extract_features(board)
        imb[i] = feat.material_imbalance
        minfor[i] = min(feat.white_forcing, feat.black_forcing)
    return imb, minfor


def classify(imb, minfor, imb_thr, forcing_thr):
    """S/M/L 라벨 배열 반환. S 우선."""
    bucket = np.full(len(imb), "M", dtype="<U1")
    bucket[imb >= imb_thr] = "S"
    is_L = (minfor >= forcing_thr) & (bucket != "S")
    bucket[is_L] = "L"
    return bucket


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

    print(f"FEN {N}개 — feature 추출 중...")
    imb, minfor = extract_all_features(fens)
    print(f"  imbalance range [{int(imb.min())}, {int(imb.max())}], mean {imb.mean():.1f}")
    print(f"  min_forcing range [{int(minfor.min())}, {int(minfor.max())}], mean {minfor.mean():.1f}")

    # 세 가중치 사전 계산
    w_mae = np.ones(N, dtype=np.float64)
    w_gw = gaussian_weight_array(gt_vec, 0.10)
    w_bv = bernoulli_variance_weight_array(gt_vec)
    wu_idx_250 = SEARCHTIMES_MS.index(250)
    err_wu = err_mat[:, wu_idx_250]

    print("\n" + "=" * 110)
    print("변형별 결과 (wu = 250ms 단일 시간 baseline, 모든 변형 평균 시간 ≈250ms 유지 확인)")
    print("=" * 110)
    print(f"{'variant':<20}  {'mean_t':>7}  {'S/M/L':>14}  "
          f"{'metric':>10}  {'policy':>8}  {'wu':>8}  {'diff':>8}  {'CI95':>20}  verdict")
    print("-" * 110)

    summary = []
    for name, imb_thr, force_thr, s_t, m_t, l_t in VARIANTS:
        bucket = classify(imb, minfor, imb_thr, force_thr)
        time_map = {"S": s_t, "M": m_t, "L": l_t}
        policy_ms = np.array([time_map[b] for b in bucket], dtype=np.float64)
        # 평균 시간 (sweep 격자 근사 전)
        mean_t_raw = float(policy_ms.mean())
        # sweep 격자 근사
        policy_idx = np.array([closest_idx(t, SEARCHTIMES_MS) for t in policy_ms])
        mean_t_used = float(times_arr[policy_idx].mean())
        err_policy = err_mat[np.arange(N), policy_idx]

        counts = {b: int((bucket == b).sum()) for b in ["S", "M", "L"]}
        smL = f"{counts['S']}/{counts['M']}/{counts['L']}"

        for metric_name, w in [
            ("MAE", w_mae), ("gwmae0.10", w_gw), ("Bernoulli", w_bv),
        ]:
            p_m = weighted_mean(err_policy, w)
            wu_m = weighted_mean(err_wu, w)
            ci = boot_ci_weighted(err_policy, err_wu, w, N_BOOT)
            d = p_m - wu_m
            v = verdict(*ci)
            rel = -d / wu_m * 100 if wu_m > 0 else 0.0
            summary.append({
                "variant": name, "metric": metric_name,
                "mean_t": mean_t_used, "smL": smL,
                "policy": p_m, "wu": wu_m, "diff": d, "ci": ci,
                "verdict": v, "rel_pct": rel,
            })
            print(f"  {name:<20}  {mean_t_used:7.1f}  {smL:>14}  "
                  f"{metric_name:>10}  {p_m:8.4f}  {wu_m:8.4f}  "
                  f"{d:+8.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]  {v}")
        print("-" * 110)

    # 정렬: Bernoulli 기준으로 best variant
    print("\n" + "=" * 90)
    print("Bernoulli 메트릭 기준 정렬 (정책 우월 → 큰 순)")
    print("=" * 90)
    bv_rows = [r for r in summary if r["metric"] == "Bernoulli"]
    bv_rows.sort(key=lambda r: r["rel_pct"], reverse=True)
    print(f"{'variant':<20}  {'mean_t':>7}  {'rel_pct':>8}  verdict")
    for r in bv_rows:
        print(f"  {r['variant']:<20}  {r['mean_t']:7.1f}  {r['rel_pct']:+7.2f}%  {r['verdict']}")

    print("\n" + "=" * 90)
    print("gwmae σ=0.10 기준 정렬")
    print("=" * 90)
    gw_rows = [r for r in summary if r["metric"] == "gwmae0.10"]
    gw_rows.sort(key=lambda r: r["rel_pct"], reverse=True)
    for r in gw_rows:
        print(f"  {r['variant']:<20}  {r['mean_t']:7.1f}  {r['rel_pct']:+7.2f}%  {r['verdict']}")

    print("\n" + "=" * 90)
    print("MAE 기준 정렬")
    print("=" * 90)
    mae_rows = [r for r in summary if r["metric"] == "MAE"]
    mae_rows.sort(key=lambda r: r["rel_pct"], reverse=True)
    for r in mae_rows:
        print(f"  {r['variant']:<20}  {r['mean_t']:7.1f}  {r['rel_pct']:+7.2f}%  {r['verdict']}")


if __name__ == "__main__":
    sys.exit(main())
