#!/usr/bin/env python3
"""Bucket Oracle (제약 없음) + Train/Test 검증.

제약 없이 각 버킷의 best_t 부여 (Lagrangian 없음).
정책의 평균 시간 T 와 매칭하는 wu(T 보간) 와 공정 비교.
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


def ceiling_wu(err_mat, mean_t, searchtimes):
    """평균 시간 mean_t 이상에서 가장 가까운 격자 wu MAE (보수적 비교).

    예: mean_t=451 → wu(500) 사용 (wu 에 약간 유리한 시간 부여)
    """
    if mean_t >= searchtimes[-1]:
        j = len(searchtimes) - 1
    else:
        j = min(jj for jj, t in enumerate(searchtimes) if t >= mean_t)
    return float(err_mat[:, j].mean()), searchtimes[j]


def main():
    fens, pred_mat, gt_vec = load_raw_grid(RAW_DIR, SEARCHTIMES_MS)
    N = len(fens)
    err_mat = np.abs(pred_mat - gt_vec[:, None])
    times_arr = np.array(SEARCHTIMES_MS, dtype=np.float64)
    T = len(SEARCHTIMES_MS)

    print(f"FEN {N}개 — feature 추출 중...")
    keys = extract_keys(fens)
    n_buckets = len(set(keys))
    print(f"총 버킷 수: {n_buckets}\n")

    # Whole-data global best_t (fallback)
    global_errors = err_mat.mean(axis=0)
    global_best_t_idx = int(np.argmin(global_errors))

    # ─── 전체 데이터 무제약 Bucket Oracle ───
    print(f"{'='*78}")
    print("전체 데이터 Bucket Oracle (무제약 — 각 버킷 자기 best_t)")
    print(f"{'='*78}")

    buckets = {}
    for i in range(N):
        buckets.setdefault(keys[i], []).append(i)
    bucket_errors = {key: err_mat[idx_list, :].mean(axis=0) for key, idx_list in buckets.items()}
    bucket_best = {key: int(np.argmin(errs)) for key, errs in bucket_errors.items()}

    policy_idx = np.zeros(N, dtype=np.int64)
    for key, idx_list in buckets.items():
        for i in idx_list:
            policy_idx[i] = bucket_best[key]
    err_policy = err_mat[np.arange(N), policy_idx]
    mae_policy = float(err_policy.mean())
    mean_t = float(times_arr[policy_idx].mean())

    wu_match, wu_t = ceiling_wu(err_mat, mean_t, SEARCHTIMES_MS)
    diff = mae_policy - wu_match
    rel = -diff / wu_match * 100
    print(f"  정책 MAE: {mae_policy:.4f}")
    print(f"  평균 시간: {mean_t:.2f}ms")
    print(f"  매칭 wu({wu_t}ms — 한 단계 위 격자): {wu_match:.4f}")
    print(f"  diff: {diff:+.4f}, 개선율: {rel:+.2f}%")

    # ─── Train/Test 검증 ───
    print(f"\n{'='*78}")
    print(f"Train/Test 검증 (train {N_TRAIN}, test {N-N_TRAIN}, {N_SEEDS} seeds, 무제약)")
    print(f"{'='*78}")

    results = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N)
        train_idx = perm[:N_TRAIN]
        test_idx = perm[N_TRAIN:]

        train_buckets = {}
        for i in train_idx:
            train_buckets.setdefault(keys[i], []).append(i)
        # 각 train 버킷의 best_t (무제약)
        bucket_choices = {}
        for key, idx_list in train_buckets.items():
            errs = err_mat[idx_list, :].mean(axis=0)
            bucket_choices[key] = int(np.argmin(errs))

        # test 적용
        test_policy_idx = np.zeros(len(test_idx), dtype=np.int64)
        fallback_count = 0
        for ti, fen_i in enumerate(test_idx):
            key = keys[fen_i]
            if key in bucket_choices:
                test_policy_idx[ti] = bucket_choices[key]
            else:
                test_policy_idx[ti] = global_best_t_idx
                fallback_count += 1

        test_err_policy = err_mat[test_idx, test_policy_idx]
        test_err_mat_full = err_mat[test_idx, :]
        test_mt = float(times_arr[test_policy_idx].mean())

        # 한 단계 위 격자 wu 사용 (test set 의 wu 별 MAE)
        if test_mt >= SEARCHTIMES_MS[-1]:
            j_ceil = T - 1
        else:
            j_ceil = min(jj for jj, t in enumerate(SEARCHTIMES_MS) if t >= test_mt)
        wu_match = float(test_err_mat_full[:, j_ceil].mean())
        wu_ceil_t = SEARCHTIMES_MS[j_ceil]

        test_mae_policy = float(test_err_policy.mean())
        diff = test_mae_policy - wu_match
        rel_test = -diff / wu_match * 100

        # CI: 정책 vs 한 단계 위 격자 wu (격자 값이라 직접 비교 가능)
        ci = boot_ci(test_err_policy, test_err_mat_full[:, j_ceil], N_BOOT, seed=42+seed)
        wu_closest_t = wu_ceil_t
        wu_closest_mae = wu_match
        diff_closest = diff

        if ci[1] < 0:
            v = "정책 우월"
        elif ci[0] > 0:
            v = "wu 우월"
        else:
            v = "유의차 없음"

        results.append({
            "seed": seed,
            "test_mt": test_mt,
            "test_mae_policy": test_mae_policy,
            "wu_match": wu_match,
            "diff_match": diff,
            "rel_match": rel_test,
            "wu_closest_t": wu_closest_t,
            "wu_closest_mae": wu_closest_mae,
            "diff_closest": diff_closest,
            "ci": ci,
            "verdict": v,
            "fallback": fallback_count,
        })

    # 출력
    print(f"\n{'seed':>4}  {'test_mt':>8}  {'test_MAE':>9}  {'wu(매칭)':>10}  "
          f"{'diff':>8}  {'개선율':>8}  {'CI vs wu_grid':>22}  {'judge':>10}")
    print("-" * 100)
    for r in results:
        ci = r["ci"]
        print(f"  {r['seed']:>2}  {r['test_mt']:8.1f}  {r['test_mae_policy']:9.4f}  "
              f"{r['wu_match']:10.4f}  {r['diff_match']:+8.4f}  {r['rel_match']:+7.2f}%  "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]  {r['verdict']:>10}")

    # 요약
    rels = np.array([r["rel_match"] for r in results])
    mts = np.array([r["test_mt"] for r in results])
    print(f"\n{'='*78}")
    print(f"  요약 ({N_SEEDS} seeds)")
    print(f"{'='*78}")
    print(f"  test 평균 mean_t: {mts.mean():.2f}ms (std {mts.std():.2f})")
    print(f"  test rel (vs 매칭 wu 보간): 평균 {rels.mean():+.2f}%, std {rels.std():.2f}%")
    n_win = sum(1 for r in results if r["verdict"] == "정책 우월")
    n_tie = sum(1 for r in results if r["verdict"] == "유의차 없음")
    n_lose = sum(1 for r in results if r["verdict"] == "wu 우월")
    print(f"  판정 분포 (vs 가까운 grid wu): 정책 우월 {n_win}, 유의차 없음 {n_tie}, wu 우월 {n_lose}")


if __name__ == "__main__":
    sys.exit(main())
