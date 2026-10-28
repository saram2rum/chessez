#!/usr/bin/env python3
"""MAE 기준 천장 + 다양한 feature × unweighted best_t 상관 분석.

1. MAE 기준 라그랑주 천장 (가중치 1) — 메트릭 독립적 절대 천장
2. 각 FEN 의 unweighted best_t (argmin |pred-gt| over t) 추출
3. 후보 feature 와 best_t 상관 (Spearman ρ) + 분산 분석 (ANOVA-like F)
4. 천장 시간 분포의 feature 별 평균 — 어떤 feature 가 신호인지
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
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
N_BOOT = 10_000


def load_raw_grid(raw_dir, search_times):
    pred_by_st, gt_map = {}, {}
    cp_by_st: dict[int, dict[str, float]] = {}
    for st in search_times:
        path = raw_dir / f"searchtime_{st}ms_raw.csv"
        pmap, cmap = {}, {}
        with path.open() as fp:
            for row in csv.DictReader(fp):
                f = row["fen"]
                pmap[f] = float(row["pred_win_pct_0_100"])
                cp_str = row.get("cp", "")
                cmap[f] = float(cp_str) if cp_str else float("nan")
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
    j, mt = assign(hi)
    return j, mt


def weighted_mean(values, weights):
    return float((values * weights).sum() / weights.sum())


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


def spearman_rho(x, y):
    """Pure numpy Spearman ρ. 동률 average rank."""
    def rankdata(a):
        a = np.asarray(a, dtype=np.float64)
        order = np.argsort(a, kind="stable")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(a) + 1)
        # average ranks for ties
        sorted_a = a[order]
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
                j += 1
            if j > i:
                avg = (i + j + 2) / 2  # (i+1 + j+1) / 2
                ranks[order[i:j+1]] = avg
            i = j + 1
        return ranks
    rx = rankdata(x)
    ry = rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / (np.sqrt((rx * rx).sum()) * np.sqrt((ry * ry).sum())))


def anova_f(values, groups):
    """numpy F-statistic for one-way ANOVA. groups: int array of group labels."""
    vals = np.asarray(values, dtype=np.float64)
    grp = np.asarray(groups)
    unique = np.unique(grp)
    overall = vals.mean()
    ss_between = 0.0
    ss_within = 0.0
    for g in unique:
        m = grp == g
        n_g = m.sum()
        if n_g == 0:
            continue
        v_g = vals[m]
        ss_between += n_g * (v_g.mean() - overall) ** 2
        ss_within += ((v_g - v_g.mean()) ** 2).sum()
    k = len(unique)
    n = len(vals)
    if k <= 1 or n - k <= 0:
        return float("nan")
    ms_b = ss_between / (k - 1)
    ms_w = ss_within / (n - k)
    return float(ms_b / ms_w) if ms_w > 0 else float("inf")


def compute_features(fens, cp_50_arr):
    """모든 후보 feature 를 매트릭스로. cp_50_arr 은 50ms 평가 cp."""
    feats: dict[str, list] = {
        "imbalance": [],
        "capture_count": [],
        "legal_moves": [],
        "total_material": [],
        "forcing_total": [],
        "is_check": [],
        "is_endgame": [],
        "cp50_abs": [],
        "cp50_sign_neutral": [],  # |cp50| 절대값과 동일 (대체)
    }
    for fen, cp50 in zip(fens, cp_50_arr):
        board = chess.Board(fen)
        f = _extract_features(board)
        feats["imbalance"].append(f.material_imbalance)
        feats["capture_count"].append(f.capture_count)
        feats["legal_moves"].append(f.legal_moves)
        feats["total_material"].append(f.total_material)
        feats["forcing_total"].append(f.white_forcing + f.black_forcing)
        feats["is_check"].append(int(f.is_check))
        feats["is_endgame"].append(int(f.is_endgame))
        cp_val = 0.0 if (cp50 is None or np.isnan(cp50)) else abs(float(cp50))
        feats["cp50_abs"].append(cp_val)
        feats["cp50_sign_neutral"].append(cp_val)
    return {k: np.asarray(v) for k, v in feats.items()}


def main():
    fens, pred_mat, cp_mat, gt_vec = load_raw_grid(RAW_DIR, SEARCHTIMES_MS)
    N = len(fens)
    err_mat = np.abs(pred_mat - gt_vec[:, None])
    times_arr = np.array(SEARCHTIMES_MS, dtype=np.float64)
    t2i = {st: i for i, st in enumerate(SEARCHTIMES_MS)}
    cp_50 = cp_mat[:, t2i[50]]

    # baseline (사용자 정의 weighted target — wu = 250ms 균등)
    target_ms = 250.0

    # ----- 1. MAE 기준 천장 -----
    print(f"FEN {N}개 | target 평균시간 = {target_ms}ms\n")

    err_wu = err_mat[:, t2i[250]]
    mae_wu = float(err_wu.mean())

    w_uniform = np.ones(N)
    ceil_idx_mae, ceil_mt_mae = ceiling_lagrangian(err_mat, w_uniform, times_arr, target_ms)
    err_ceil_mae = err_mat[np.arange(N), ceil_idx_mae]
    mae_ceil = float(err_ceil_mae.mean())

    # bootstrap CI: wu - ceil
    rng = np.random.default_rng(42)
    diffs = np.empty(N_BOOT)
    for k in range(N_BOOT):
        idx = rng.integers(0, N, size=N)
        diffs[k] = err_wu[idx].mean() - err_ceil_mae[idx].mean()
    ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    print(f"=== 1) MAE 기준 천장 (가중치 1, 평균시간 ≤ {target_ms}ms) ===")
    print(f"  wu(250)  MAE = {mae_wu:.4f}")
    print(f"  ceiling  MAE = {mae_ceil:.4f}  (mean_t={ceil_mt_mae:.1f}ms)")
    print(f"  wu - ceil    = {mae_wu - mae_ceil:+.4f}  CI95 [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # gwmae 천장 σ=0.10 비교 (참고)
    w_gw = gaussian_weight_array(gt_vec, 0.10)
    ceil_idx_gw, ceil_mt_gw = ceiling_lagrangian(err_mat, w_gw, times_arr, target_ms)
    err_ceil_gw = err_mat[np.arange(N), ceil_idx_gw]
    gw_ceil = weighted_mean(err_ceil_gw, w_gw)
    gw_wu = weighted_mean(err_wu, w_gw)
    print(f"\n[참고] gwmae σ=0.10 천장 = {gw_ceil:.4f}, wu = {gw_wu:.4f}, diff = {gw_wu - gw_ceil:+.4f}")

    # 천장 시간 분포 (MAE 기준 vs gwmae 기준)
    print("\n천장 시간 분포 비교 (MAE 기준 vs gwmae 기준 σ=0.10):")
    print(f"{'time':>6}  {'MAE천장':>10}  {'gwmae천장':>10}")
    for t in SEARCHTIMES_MS:
        n_mae = int((times_arr[ceil_idx_mae] == t).sum())
        n_gw = int((times_arr[ceil_idx_gw] == t).sum())
        print(f"  {t:>4}  {n_mae:>10}  {n_gw:>10}")

    # ----- 2. unweighted best_t × feature -----
    print("\n=== 2) unweighted best_t × feature 상관 ===")
    best_t_idx = np.argmin(err_mat, axis=1)
    best_t_ms = times_arr[best_t_idx]

    print(f"best_t 분포 (unweighted argmin MAE per FEN):")
    bt_counter = Counter(int(t) for t in best_t_ms)
    for t in SEARCHTIMES_MS:
        c = bt_counter.get(t, 0)
        print(f"  {t}ms: {c} ({c/N*100:.1f}%)")

    feats = compute_features(fens, cp_50)
    print(f"\n{'feature':<22}{'Spearman ρ(feat, best_t)':>28}{'ANOVA F (best_t bucket)':>28}")
    print("-" * 78)
    results = []
    for name, vals in feats.items():
        rho = spearman_rho(vals, best_t_ms)
        f_stat = anova_f(vals, best_t_idx)
        results.append((name, rho, f_stat))
        print(f"{name:<22}{rho:>+28.4f}{f_stat:>28.2f}")

    # 정렬: |ρ| 큰 순
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\n|ρ| 큰 순:")
    for name, rho, f_stat in results[:5]:
        print(f"  {name:<22}  ρ={rho:+.4f},  F={f_stat:.2f}")

    # ----- 3. winrate 와의 상관도 같이 (가중치 트릭 의심 feature 표시) -----
    print("\n=== 3) feature × gt_winrate 상관 (가중치 트릭 진단) ===")
    print("높을수록 (정/부 무관) winrate 와 강하게 연동 → 가중치 트릭 가능성")
    for name, vals in feats.items():
        rho_wr = spearman_rho(vals, gt_vec)
        rho_wr_dist = spearman_rho(vals, np.abs(gt_vec - 50.0))
        print(f"  {name:<22}  ρ(feat, gt)={rho_wr:+.4f}  ρ(feat, |gt-50|)={rho_wr_dist:+.4f}")


if __name__ == "__main__":
    sys.exit(main())
