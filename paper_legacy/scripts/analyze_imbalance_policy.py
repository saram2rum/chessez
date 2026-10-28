#!/usr/bin/env python3
"""material_imbalance 기준 quintile 정책의 weighted_uniform 대비 우열.

각 FEN의 material_imbalance 를 큰 순으로 정렬하여 정확히 20% 씩 5분할.
큰 imbalance → 50ms, 작은 imbalance → 450ms 매핑.
시간 집합 [50, 150, 250, 350, 450]ms 각 20% 씩 ⇒ 평균 시간 정확히 250ms.

비교 대상: weighted_uniform(250ms 균등).
메트릭: MAE(unweighted) + gwmae(σ ∈ {0.05~0.25}).
통계: paired bootstrap 95% CI (N=10000).

Usage:
    python3 scripts/analyze_imbalance_policy.py
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

from app.utils.gaussian_weighted_mae import gaussian_weight_array  # noqa: E402
from app.utils.position_complexity import _extract_features  # noqa: E402

RAW_DIR = PROJECT_ROOT / "searchtime_raw_local"
SEARCHTIMES_MS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.25]
POLICY_TIMES_QUINTILE = [50, 150, 250, 350, 450]  # 평균 250ms 정확


def load_raw_grid(raw_dir: Path, search_times: list[int]):
    pred_by_st: dict[int, dict[str, float]] = {}
    gt_map: dict[str, float] = {}
    for st in search_times:
        path = raw_dir / f"searchtime_{st}ms_raw.csv"
        pred_map: dict[str, float] = {}
        with path.open() as fp:
            for row in csv.DictReader(fp):
                fen = row["fen"]
                pred_map[fen] = float(row["pred_win_pct_0_100"])
                gt_map.setdefault(fen, float(row["ground_truth_win_pct_0_100"]))
        pred_by_st[st] = pred_map
    fens_common = set(gt_map)
    for st in search_times:
        fens_common &= set(pred_by_st[st])
    fens = sorted(fens_common)
    N, T = len(fens), len(search_times)
    pred_mat = np.zeros((N, T), dtype=np.float64)
    gt_vec = np.array([gt_map[f] for f in fens], dtype=np.float64)
    for j, st in enumerate(search_times):
        col = pred_by_st[st]
        pred_mat[:, j] = [col[f] for f in fens]
    return fens, pred_mat, gt_vec


def imbalance_of_fen(fen: str) -> int:
    board = chess.Board(fen)
    feat = _extract_features(board)
    return feat.material_imbalance


def assign_quintile(imbalances: np.ndarray, times: list[int]) -> np.ndarray:
    """imbalance 큰 순서로 5분할 (각 20%) → times[0]~times[4] 배정.
    동률은 stable sort 순서 따라 분할. 남는 % 는 앞 그룹에 분배.
    """
    N = len(imbalances)
    # 무작위 shuffle 후 stable sort 로 동률 그룹 안 무작위성 확보
    rng = np.random.default_rng(0)
    shuffle_idx = rng.permutation(N)
    order_in_shuffle = np.argsort(-imbalances[shuffle_idx], kind="stable")
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


def paired_bootstrap_ci_weighted(
    err_a: np.ndarray, err_b: np.ndarray, weights: np.ndarray,
    n_boot: int = 10_000, seed: int = 42, chunk: int = 1000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    N = len(err_a)
    diffs = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = rng.integers(0, N, size=(size, N))
        w_b = weights[idx]
        sw = w_b.sum(axis=1)
        a = (err_a[idx] * w_b).sum(axis=1) / sw
        b = (err_b[idx] * w_b).sum(axis=1) / sw
        diffs[done:done + size] = a - b
        done += size
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def paired_bootstrap_ci_uniform(
    err_a: np.ndarray, err_b: np.ndarray,
    n_boot: int = 10_000, seed: int = 42, chunk: int = 1000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    N = len(err_a)
    diffs = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = rng.integers(0, N, size=(size, N))
        a = err_a[idx].mean(axis=1)
        b = err_b[idx].mean(axis=1)
        diffs[done:done + size] = a - b
        done += size
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> int:
    fens, pred_mat, gt_vec = load_raw_grid(RAW_DIR, SEARCHTIMES_MS)
    N = len(fens)
    err_mat = np.abs(pred_mat - gt_vec[:, None])

    imbalances = np.array([imbalance_of_fen(f) for f in fens], dtype=np.int64)
    print(f"FEN {N}개 | imbalance: min={imbalances.min()}, max={imbalances.max()}, mean={imbalances.mean():.2f}")
    print("imbalance percentiles:")
    for q in (0, 25, 50, 75, 90, 95, 99, 100):
        print(f"  p{q:3d}: {int(np.percentile(imbalances, q))}")

    policy_ms = assign_quintile(imbalances, POLICY_TIMES_QUINTILE)
    print(f"\nquintile policy 평균 시간: {policy_ms.mean():.2f}ms")
    print("quintile policy 시간 분포:")
    for t in POLICY_TIMES_QUINTILE:
        n_t = int((policy_ms == t).sum())
        # 해당 그룹의 imbalance 범위
        ib_in_group = imbalances[policy_ms == t]
        print(f"  {t}ms: n={n_t}, imbalance range [{ib_in_group.min()}, {ib_in_group.max()}], "
              f"mean={ib_in_group.mean():.2f}, gt_win% mean={gt_vec[policy_ms==t].mean():.2f}")

    time_to_idx = {st: i for i, st in enumerate(SEARCHTIMES_MS)}
    policy_idx = np.array([time_to_idx[int(t)] for t in policy_ms])
    wu_idx = time_to_idx[250]
    err_policy = err_mat[np.arange(N), policy_idx]
    err_wu = err_mat[:, wu_idx]

    print("\n=== MAE (가중치 없음) ===")
    mae_policy = float(err_policy.mean())
    mae_wu = float(err_wu.mean())
    ci_mae = paired_bootstrap_ci_uniform(err_policy, err_wu)
    diff_mae = mae_policy - mae_wu
    verdict_mae = "정책 우월" if ci_mae[1] < 0 else ("wu 우월" if ci_mae[0] > 0 else "유의차 없음")
    print(f"  policy MAE = {mae_policy:.4f}")
    print(f"  wu(250)MAE = {mae_wu:.4f}")
    print(f"  diff(policy - wu) = {diff_mae:+.4f}  CI95 [{ci_mae[0]:+.4f}, {ci_mae[1]:+.4f}]  → {verdict_mae}")

    print("\n=== gwmae (σ별) ===")
    print(f"{'σ':>6}  {'policy':>9}  {'wu':>9}  {'diff':>10}  {'CI95':>22}  verdict")
    print("-" * 78)
    for sigma in SIGMAS:
        w = gaussian_weight_array(gt_vec, sigma)
        gw_policy = float((err_policy * w).sum() / w.sum())
        gw_wu = float((err_wu * w).sum() / w.sum())
        diff = gw_policy - gw_wu
        ci = paired_bootstrap_ci_weighted(err_policy, err_wu, w)
        verdict = "정책 우월" if ci[1] < 0 else ("wu 우월" if ci[0] > 0 else "유의차 없음")
        print(f"  {sigma:.2f}  {gw_policy:9.4f}  {gw_wu:9.4f}  {diff:+10.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]  {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
