#!/usr/bin/env python3
"""정적 휴리스틱 — 평균 시간 250ms 로 고정한 공정 비교.

이전 tune_static_policy.py 는 변형들이 250ms 이상을 썼다(V4 = 322ms). 시간 예산
자체가 더 많아서 이긴 거라 wu(250) 와 공정 비교 X. 이 스크립트는 모든 변형의
평균 시간을 250ms 로 강제한다.

방법:
  - 각 (imb_thr, force_thr) 조합에 대해 S/M/L 개수 산출
  - S=50, L=500 으로 양 끝을 최대로 펼치고
  - M 을 풀어서 평균 시간 = 250ms 가 되도록 결정
  - M ∈ [50, 500] 범위 안이면 유효 정책. sweep 격자 근사 후 mean 확인
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
TARGET_MS = 250.0

# 임계값 조합 — (imb_thr, force_thr)
THRESHOLDS = [
    (9, 5),   # 원래 baseline
    (6, 5),   # S 만 완화
    (9, 3),   # L 만 완화
    (6, 3),   # 둘 다 약간 완화
    (4, 3),
    (6, 2),
    (4, 2),
    (3, 2),
    (4, 4),
]
# S/L 시간 후보
S_L_PAIRS = [
    (50, 500),   # 최대 확장
    (100, 400),  # 원래
    (50, 400),
    (100, 500),
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
    imb = np.zeros(len(fens), dtype=np.float64)
    minfor = np.zeros(len(fens), dtype=np.float64)
    for i, f in enumerate(fens):
        board = chess.Board(f)
        feat = _extract_features(board)
        imb[i] = feat.material_imbalance
        minfor[i] = min(feat.white_forcing, feat.black_forcing)
    return imb, minfor


def classify(imb, minfor, imb_thr, force_thr):
    bucket = np.full(len(imb), "M", dtype="<U1")
    bucket[imb >= imb_thr] = "S"
    is_L = (minfor >= force_thr) & (bucket != "S")
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

    # 사전 가중치
    w_mae = np.ones(N, dtype=np.float64)
    w_gw = gaussian_weight_array(gt_vec, 0.10)
    w_bv = bernoulli_variance_weight_array(gt_vec)
    wu_idx_250 = SEARCHTIMES_MS.index(250)
    err_wu = err_mat[:, wu_idx_250]

    # 모든 변형 시도
    candidates = []
    for imb_thr, force_thr in THRESHOLDS:
        bucket = classify(imb, minfor, imb_thr, force_thr)
        n_s = int((bucket == "S").sum())
        n_m = int((bucket == "M").sum())
        n_l = int((bucket == "L").sum())
        if n_m == 0:
            continue
        for s_t, l_t in S_L_PAIRS:
            # M 시간 = 평균 250 만족하는 값
            # n_s * s_t + n_m * m + n_l * l_t = 250 * N
            m_raw = (250.0 * N - n_s * s_t - n_l * l_t) / n_m
            if not (50.0 <= m_raw <= 500.0):
                continue
            # sweep 격자에 매핑 (가장 가까운 값)
            m_idx = closest_idx(m_raw, SEARCHTIMES_MS)
            m_t = SEARCHTIMES_MS[m_idx]
            time_map = {"S": s_t, "M": m_t, "L": l_t}
            policy_ms = np.array([time_map[b] for b in bucket], dtype=np.float64)
            policy_idx = np.array([closest_idx(t, SEARCHTIMES_MS) for t in policy_ms])
            mean_t_used = float(times_arr[policy_idx].mean())
            # 격자 근사 후 평균이 250 ±20 이내여야 공정
            if abs(mean_t_used - 250.0) > 20.0:
                continue
            err_policy = err_mat[np.arange(N), policy_idx]

            row = {
                "thr": f"imb≥{imb_thr},f≥{force_thr}",
                "smL": f"{n_s}/{n_m}/{n_l}",
                "times": f"{s_t}/{m_t}/{l_t}",
                "mean_t": mean_t_used,
            }
            for met_name, w in [("MAE", w_mae), ("gwmae", w_gw), ("Bern", w_bv)]:
                p_m = weighted_mean(err_policy, w)
                wu_m = weighted_mean(err_wu, w)
                ci = boot_ci_weighted(err_policy, err_wu, w, N_BOOT)
                d = p_m - wu_m
                v = verdict(*ci)
                rel = -d / wu_m * 100 if wu_m > 0 else 0.0
                row[f"{met_name}_pol"] = p_m
                row[f"{met_name}_diff"] = d
                row[f"{met_name}_ci"] = ci
                row[f"{met_name}_v"] = v
                row[f"{met_name}_rel"] = rel
            candidates.append(row)

    if not candidates:
        print("유효 변형 없음")
        return 1

    print(f"\n{len(candidates)}개 변형 평가됨 (평균 시간 250±20ms 범위 내)\n")

    # 각 메트릭별 best
    for met in ["Bern", "gwmae", "MAE"]:
        print("=" * 110)
        print(f"{met} 기준 정렬 (상위 8개)")
        print("=" * 110)
        sorted_c = sorted(candidates, key=lambda r: r[f"{met}_rel"], reverse=True)
        print(f"{'thr':<14}  {'S/M/L':>14}  {'times':>14}  {'mean_t':>7}  "
              f"{'rel':>7}  {'diff':>8}  {'CI95':>22}  verdict")
        for r in sorted_c[:8]:
            ci = r[f"{met}_ci"]
            print(f"  {r['thr']:<14}  {r['smL']:>14}  {r['times']:>14}  "
                  f"{r['mean_t']:7.1f}  {r[f'{met}_rel']:+7.2f}%  "
                  f"{r[f'{met}_diff']:+8.4f}  "
                  f"[{ci[0]:+.4f}, {ci[1]:+.4f}]  {r[f'{met}_v']}")
        print()

    # 통합 best: 세 메트릭 모두에서 0 이상 (혹은 유의차 없음 이상)
    print("=" * 110)
    print("종합: 세 메트릭 모두 정책 우월 또는 유의차 없음 (rel% 합 큰 순)")
    print("=" * 110)
    survive = [r for r in candidates
               if r["MAE_v"] != "wu 우월"
               and r["gwmae_v"] != "wu 우월"
               and r["Bern_v"] != "wu 우월"]
    if not survive:
        print("세 메트릭 모두에서 wu 를 안 진 변형 없음.")
        # 차선: 두 메트릭 이상에서 안 짐
        print("\n차선: gwmae + Bernoulli 둘 다에서 정책 우월 또는 유의차 없음")
        sub = [r for r in candidates
               if r["gwmae_v"] != "wu 우월" and r["Bern_v"] != "wu 우월"]
        sub.sort(key=lambda r: r["Bern_rel"] + r["gwmae_rel"], reverse=True)
        print(f"{'thr':<14}  {'S/M/L':>14}  {'times':>14}  "
              f"{'MAE rel':>8}  {'gw rel':>8}  {'Bern rel':>8}")
        for r in sub[:8]:
            print(f"  {r['thr']:<14}  {r['smL']:>14}  {r['times']:>14}  "
                  f"{r['MAE_rel']:+8.2f}%  {r['gwmae_rel']:+8.2f}%  "
                  f"{r['Bern_rel']:+8.2f}%")
    else:
        survive.sort(key=lambda r: r["MAE_rel"] + r["gwmae_rel"] + r["Bern_rel"], reverse=True)
        print(f"{'thr':<14}  {'S/M/L':>14}  {'times':>14}  "
              f"{'MAE rel':>8}  {'gw rel':>8}  {'Bern rel':>8}")
        for r in survive[:8]:
            print(f"  {r['thr']:<14}  {r['smL']:>14}  {r['times']:>14}  "
                  f"{r['MAE_rel']:+8.2f}%  {r['gwmae_rel']:+8.2f}%  "
                  f"{r['Bern_rel']:+8.2f}%")


if __name__ == "__main__":
    sys.exit(main())
