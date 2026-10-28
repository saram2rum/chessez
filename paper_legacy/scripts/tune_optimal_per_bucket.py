#!/usr/bin/env python3
"""(imb, forcing) 버킷별 최적 시간 brute force — 정적 정보의 이론적 상한.

각 (imb, forcing) 조합을 하나의 버킷으로 보고, 그 버킷의 mean error 가
최소인 시간을 best_t 로 부여. 이게 (imb, forcing) feature 만으로 도달
가능한 최선의 inductive 정책.

두 시나리오:
  ① 무제약 (각 버킷 best_t 그대로 — 평균 시간 자유)
  ② 250ms 평균 시간 제약 (Lagrangian 으로 시간 가격 α 조정)
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
TARGET_MS = 250.0

# 버킷 정의 — clamp 로 sparse 버킷 합치기
IMB_CLAMP = 9   # imb ≥ 9 는 한 그룹
FORCING_CLAMP = 5  # forcing ≥ 5 는 한 그룹


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
    times_arr = np.array(SEARCHTIMES_MS, dtype=np.float64)
    t2i = {st: i for i, st in enumerate(SEARCHTIMES_MS)}
    T = len(SEARCHTIMES_MS)

    print(f"FEN {N}개 — feature 추출 중...")
    imb_values = []
    forcing_values = []
    for fen in fens:
        b = chess.Board(fen)
        f = _extract_features(b)
        imb_values.append(f.material_imbalance)
        forcing_values.append(min(f.white_forcing, f.black_forcing))
    imb_arr = np.minimum(np.array(imb_values, dtype=np.int64), IMB_CLAMP)
    forcing_arr = np.minimum(np.array(forcing_values, dtype=np.int64), FORCING_CLAMP)

    # 각 (imb, forcing) 버킷에 속하는 FEN 인덱스 수집
    buckets = {}
    for i in range(N):
        key = (int(imb_arr[i]), int(forcing_arr[i]))
        buckets.setdefault(key, []).append(i)

    print(f"\n총 버킷 수: {len(buckets)}")
    print(f"버킷 분포 (상위 10):")
    sorted_buckets = sorted(buckets.items(), key=lambda x: -len(x[1]))
    print(f"{'(imb, force)':>14}  {'FEN 수':>8}  {'비중':>8}")
    for key, idx_list in sorted_buckets[:10]:
        print(f"  ({key[0]:>2}, {key[1]:>2})  {len(idx_list):>8}  {len(idx_list)/N*100:>6.1f}%")
    print(f"  ... 외 {len(buckets)-10} 버킷")

    # 각 버킷의 시간별 mean error 계산
    bucket_errors = {}  # key → array of 10 mean errors
    for key, idx_list in buckets.items():
        errors = err_mat[idx_list, :]  # (n_bucket, 10)
        bucket_errors[key] = errors.mean(axis=0)  # (10,)

    # 시나리오 ① 무제약 — 각 버킷 best_t
    print(f"\n{'='*70}")
    print(f"  시나리오 ① 무제약 (각 버킷 자기 best_t)")
    print(f"{'='*70}")
    bucket_best_t_idx = {key: int(np.argmin(bucket_errors[key])) for key in buckets}
    policy_idx = np.zeros(N, dtype=np.int64)
    for key, idx_list in buckets.items():
        for i in idx_list:
            policy_idx[i] = bucket_best_t_idx[key]
    err_policy_uncons = err_mat[np.arange(N), policy_idx]
    mean_t_uncons = float(times_arr[policy_idx].mean())
    mae_uncons = float(err_policy_uncons.mean())
    print(f"  평균 시간: {mean_t_uncons:.2f}ms")
    print(f"  MAE: {mae_uncons:.4f}")

    # 버킷별 best_t 출력 (큰 버킷 위주)
    print(f"\n  버킷별 best_t (상위 10):")
    print(f"{'(imb, force)':>14}  {'FEN 수':>8}  {'best_t':>8}  {'MAE':>8}")
    for key, idx_list in sorted_buckets[:10]:
        bt_idx = bucket_best_t_idx[key]
        bt_ms = SEARCHTIMES_MS[bt_idx]
        mae_bucket = bucket_errors[key][bt_idx]
        print(f"  ({key[0]:>2}, {key[1]:>2})  {len(idx_list):>8}  {bt_ms:>6}ms  {mae_bucket:>8.4f}")

    # 시나리오 ② 250ms 평균 시간 제약 — Lagrangian
    print(f"\n{'='*70}")
    print(f"  시나리오 ② 평균 시간 250ms 제약 (Lagrangian)")
    print(f"{'='*70}")

    def assign_with_alpha(alpha):
        bucket_choices = {}
        for key in buckets:
            costs = bucket_errors[key] + alpha * times_arr
            bucket_choices[key] = int(np.argmin(costs))
        # mean time
        total_time = 0.0
        for key, idx_list in buckets.items():
            total_time += len(idx_list) * times_arr[bucket_choices[key]]
        mean_t = total_time / N
        return bucket_choices, mean_t

    # 이분탐색
    lo, hi = 0.0, 1.0
    for _ in range(40):
        _, mt_hi = assign_with_alpha(hi)
        if mt_hi <= TARGET_MS:
            break
        hi *= 4.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        _, mt = assign_with_alpha(mid)
        if mt > TARGET_MS:
            lo = mid
        else:
            hi = mid
    bucket_choices_cons, mean_t_cons = assign_with_alpha(hi)

    policy_idx_cons = np.zeros(N, dtype=np.int64)
    for key, idx_list in buckets.items():
        for i in idx_list:
            policy_idx_cons[i] = bucket_choices_cons[key]
    err_policy_cons = err_mat[np.arange(N), policy_idx_cons]
    mae_cons = float(err_policy_cons.mean())
    print(f"  평균 시간: {mean_t_cons:.2f}ms (≤ {TARGET_MS}ms 제약)")
    print(f"  MAE: {mae_cons:.4f}")
    print(f"  α* (시간 가격): {hi:.6f}")

    # 비교
    wu250_mae = float(err_mat[:, t2i[250]].mean())
    print(f"\n{'='*70}")
    print(f"  결과 비교 (vs wu(250) MAE = {wu250_mae:.4f})")
    print(f"{'='*70}")
    print(f"\n[시나리오 ① 무제약]")
    diff1 = mae_uncons - wu250_mae
    ci1 = boot_ci(err_policy_uncons, err_mat[:, t2i[250]], N_BOOT)
    rel1 = -diff1 / wu250_mae * 100
    v1 = "정책 우월" if ci1[1] < 0 else ("wu 우월" if ci1[0] > 0 else "유의차 없음")
    print(f"  mean_t={mean_t_uncons:.1f}ms, MAE={mae_uncons:.4f}")
    print(f"  vs wu(250): diff={diff1:+.4f}, CI95=[{ci1[0]:+.4f}, {ci1[1]:+.4f}], "
          f"rel={rel1:+.2f}%, {v1}")

    print(f"\n[시나리오 ② 250ms 제약]")
    diff2 = mae_cons - wu250_mae
    ci2 = boot_ci(err_policy_cons, err_mat[:, t2i[250]], N_BOOT)
    rel2 = -diff2 / wu250_mae * 100
    v2 = "정책 우월" if ci2[1] < 0 else ("wu 우월" if ci2[0] > 0 else "유의차 없음")
    print(f"  mean_t={mean_t_cons:.1f}ms, MAE={mae_cons:.4f}")
    print(f"  vs wu(250): diff={diff2:+.4f}, CI95=[{ci2[0]:+.4f}, {ci2[1]:+.4f}], "
          f"rel={rel2:+.2f}%, {v2}")

    # 공정 비교 (무제약의 평균 시간 보간)
    print(f"\n[시나리오 ① 공정 비교 — wu({mean_t_uncons:.0f}ms 보간)]")
    j_lo = max((j for j, t in enumerate(SEARCHTIMES_MS) if t <= mean_t_uncons), default=0)
    j_hi = min(T - 1, j_lo + 1)
    t_lo, t_hi = SEARCHTIMES_MS[j_lo], SEARCHTIMES_MS[j_hi]
    if t_lo == t_hi:
        v_interp = float(err_mat[:, j_lo].mean())
    else:
        alpha = (mean_t_uncons - t_lo) / (t_hi - t_lo)
        v_interp = float(err_mat[:, j_lo].mean()) + alpha * float(
            err_mat[:, j_hi].mean() - err_mat[:, j_lo].mean())
    diff_fair = mae_uncons - v_interp
    rel_fair = -diff_fair / v_interp * 100
    print(f"  wu({mean_t_uncons:.0f}ms) = {v_interp:.4f}")
    print(f"  diff: {diff_fair:+.4f}, rel: {rel_fair:+.2f}%")

    # 천장 (FEN-level Oracle) 와 비교
    err_oracle = err_mat[np.arange(N), err_mat.argmin(axis=1)]
    oracle_mae = float(err_oracle.mean())
    oracle_mt = float(times_arr[err_mat.argmin(axis=1)].mean())
    print(f"\n[참고] FEN-level Oracle (이론적 절대 하한):")
    print(f"  mean_t={oracle_mt:.1f}ms, MAE={oracle_mae:.4f}")
    print(f"  → 무제약 bucket Oracle 의 MAE 격차: {mae_uncons - oracle_mae:+.4f}")
    print(f"  → 250 제약 bucket Oracle 의 MAE 격차: {mae_cons - oracle_mae:+.4f}")


if __name__ == "__main__":
    sys.exit(main())
