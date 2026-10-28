#!/usr/bin/env python3
"""Bucket Oracle 의 train/test split 검증 — over-fitting 확인.

방법:
  1. 500 FEN 랜덤 샘플링 = train set
  2. train 에서 (imb, forcing) 버킷별 best_t + Lagrangian 으로 250ms 제약 매핑
  3. 나머지 2427 FEN = test set 에 그 매핑 적용
  4. test MAE 측정 + wu(250) 와 비교
  5. 10 seed 반복으로 안정성 확인
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
TARGET_MS = 250.0
N_TRAIN = 500
IMB_CLAMP = 9
FORCING_CLAMP = 5
N_SEEDS = 10
N_BOOT = 5_000


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


def extract_keys(fens):
    keys = []
    for fen in fens:
        b = chess.Board(fen)
        f = _extract_features(b)
        imb = min(f.material_imbalance, IMB_CLAMP)
        force = min(min(f.white_forcing, f.black_forcing), FORCING_CLAMP)
        keys.append((imb, force))
    return keys


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
    keys = extract_keys(fens)

    # Whole-data global best_t (fallback)
    global_errors = err_mat.mean(axis=0)
    global_best_t_idx = int(np.argmin(global_errors))

    print(f"\n{'='*78}")
    print(f"Train/Test Split 검증 (train {N_TRAIN}, test {N-N_TRAIN}, {N_SEEDS} seeds)")
    print(f"{'='*78}")

    results = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N)
        train_idx = perm[:N_TRAIN]
        test_idx = perm[N_TRAIN:]

        # 1. Train 에서 버킷별 mean errors
        train_buckets = {}
        for i in train_idx:
            train_buckets.setdefault(keys[i], []).append(i)
        bucket_errors = {}
        for key, idx_list in train_buckets.items():
            bucket_errors[key] = err_mat[idx_list, :].mean(axis=0)
        train_sizes = {k: len(v) for k, v in train_buckets.items()}

        # 2. Lagrangian: train 에서 mean_t = TARGET_MS 인 α 찾기
        def assign_alpha(alpha):
            choices = {}
            for key, errs in bucket_errors.items():
                costs = errs + alpha * times_arr
                choices[key] = int(np.argmin(costs))
            # train mean time
            total = sum(train_sizes[k] * times_arr[choices[k]] for k in choices)
            return choices, total / N_TRAIN

        lo, hi = 0.0, 1.0
        for _ in range(40):
            _, mt_hi = assign_alpha(hi)
            if mt_hi <= TARGET_MS:
                break
            hi *= 4.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            _, mt = assign_alpha(mid)
            if mt > TARGET_MS:
                lo = mid
            else:
                hi = mid
        bucket_time_idx, train_mt = assign_alpha(hi)

        # 3. Test 에 적용
        test_policy_idx = np.zeros(len(test_idx), dtype=np.int64)
        fallback_count = 0
        for ti, fen_i in enumerate(test_idx):
            key = keys[fen_i]
            if key in bucket_time_idx:
                test_policy_idx[ti] = bucket_time_idx[key]
            else:
                test_policy_idx[ti] = global_best_t_idx
                fallback_count += 1

        test_err_policy = err_mat[test_idx, test_policy_idx]
        test_err_wu250 = err_mat[test_idx, t2i[250]]
        test_mt = float(times_arr[test_policy_idx].mean())
        test_mae_policy = float(test_err_policy.mean())
        test_mae_wu250 = float(test_err_wu250.mean())

        diff = test_mae_policy - test_mae_wu250
        rel = -diff / test_mae_wu250 * 100
        ci = boot_ci(test_err_policy, test_err_wu250, N_BOOT, seed=42+seed)
        v = "정책 우월" if ci[1] < 0 else ("wu 우월" if ci[0] > 0 else "유의차 없음")

        results.append({
            "seed": seed,
            "train_mt": train_mt,
            "test_mt": test_mt,
            "test_mae_policy": test_mae_policy,
            "test_mae_wu250": test_mae_wu250,
            "diff": diff,
            "rel": rel,
            "ci": ci,
            "verdict": v,
            "fallback_count": fallback_count,
            "train_buckets": len(train_buckets),
        })

    # 결과 출력
    print(f"\n{'seed':>4}  {'train_mt':>9}  {'test_mt':>8}  {'test_MAE':>9}  "
          f"{'wu(250)':>8}  {'diff':>8}  {'rel':>7}  {'fallback':>9}  verdict")
    print("-" * 95)
    for r in results:
        print(f"  {r['seed']:>2}  {r['train_mt']:9.2f}  {r['test_mt']:8.2f}  "
              f"{r['test_mae_policy']:9.4f}  {r['test_mae_wu250']:8.4f}  "
              f"{r['diff']:+8.4f}  {r['rel']:+6.2f}%  {r['fallback_count']:>6}  {r['verdict']}")

    # 요약 통계
    diffs = np.array([r["diff"] for r in results])
    rels = np.array([r["rel"] for r in results])
    print(f"\n{'='*78}")
    print(f"  요약 ({N_SEEDS} seeds)")
    print(f"{'='*78}")
    print(f"  test diff (정책 - wu): 평균 {diffs.mean():+.4f},  std {diffs.std():.4f}")
    print(f"  test rel improvement:  평균 {rels.mean():+.2f}%,  std {rels.std():.2f}%")
    print(f"  test mean_t 평균:      {np.mean([r['test_mt'] for r in results]):.2f}ms")
    print(f"  fallback 평균:          {np.mean([r['fallback_count'] for r in results]):.1f} FEN")
    n_win = sum(1 for r in results if r["verdict"] == "정책 우월")
    n_tie = sum(1 for r in results if r["verdict"] == "유의차 없음")
    n_lose = sum(1 for r in results if r["verdict"] == "wu 우월")
    print(f"\n  판정 분포: 정책 우월 {n_win}회, 유의차 없음 {n_tie}회, wu 우월 {n_lose}회")

    # 참고: 무제약 (전체 데이터) Bucket Oracle 결과
    print(f"\n{'='*78}")
    print(f"  참고: 전체 데이터 Bucket Oracle (over-fitting 가능)")
    print(f"{'='*78}")
    print(f"  Lagrangian 250ms 제약: MAE 1.7373 (vs wu(250) 1.8029, rel +3.64% 정책 우월)")


if __name__ == "__main__":
    sys.exit(main())
