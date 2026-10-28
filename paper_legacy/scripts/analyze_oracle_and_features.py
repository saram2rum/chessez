#!/usr/bin/env python3
"""Oracle 정책 + 정적 FEN feature 한계 분석 — win% 와 cp 두 공간 모두.

Lagrangian 천장 대신 단순 oracle (각 FEN 의 best_t) 으로 분석.
이 데이터에서 250ms 예산이 non-binding 이라 oracle = 라그랑주 천장.

두 공간으로 같이 분석해 robustness 확인:
  - win% 공간: Lichess 변환식 적용, 단위 %p
  - cp 공간: 직접 cp 값, 단위 cp (mate 는 ±10000 sentinel)

출력:
  1. Oracle 결과 vs 모든 uniform 정책 (50~500ms)
  2. 정적 FEN feature 의 best_t 와의 Spearman ρ
  3. 두 공간 결과 비교 → 결론 일관성 검증
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

from app.utils.position_complexity import _extract_features  # noqa: E402

RAW_DIR = PROJECT_ROOT / "searchtime_raw_local"
SEARCHTIMES_MS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
MATE_CP_SENTINEL = 10_000.0
N_BOOT = 10_000


def load_raw_grid(raw_dir, search_times):
    """returns (fens, pred_win_mat[N,T], pred_cp_mat[N,T], gt_win[N], gt_cp[N])."""
    pred_win_by_st, gt_win_map = {}, {}
    pred_cp_by_st = {}
    for st in search_times:
        path = raw_dir / f"searchtime_{st}ms_raw.csv"
        pmap_w, pmap_cp = {}, {}
        with path.open() as fp:
            for row in csv.DictReader(fp):
                f = row["fen"]
                pred_w = float(row["pred_win_pct_0_100"])
                pmap_w[f] = pred_w
                cs = row.get("cp", "")
                if cs:
                    pmap_cp[f] = float(cs)
                else:
                    pmap_cp[f] = MATE_CP_SENTINEL if pred_w >= 50 else -MATE_CP_SENTINEL
                gt_win_map.setdefault(f, float(row["ground_truth_win_pct_0_100"]))
        pred_win_by_st[st] = pmap_w
        pred_cp_by_st[st] = pmap_cp
    fens_common = set(gt_win_map)
    for st in search_times:
        fens_common &= set(pred_win_by_st[st])
    fens = sorted(fens_common)
    N, T = len(fens), len(search_times)
    pred_win = np.zeros((N, T))
    pred_cp = np.zeros((N, T))
    gt_win = np.array([gt_win_map[f] for f in fens])
    # gt_cp 는 gt_win 의 Lichess 역변환 — gt_win 이 (0, 100) 일 때만 유효
    # 극단 (≈ 0 또는 ≈ 100) 은 mate 가능성 → sentinel 처리
    gt_cp = np.zeros(N)
    for i, w in enumerate(gt_win):
        if w >= 99.999:
            gt_cp[i] = MATE_CP_SENTINEL
        elif w <= 0.001:
            gt_cp[i] = -MATE_CP_SENTINEL
        else:
            # Win% = 100 / (1 + exp(-0.00368208 * cp))
            # => cp = -ln(100/Win% - 1) / 0.00368208
            ratio = 100.0 / w - 1.0
            gt_cp[i] = -np.log(ratio) / 0.00368208
    for j, st in enumerate(search_times):
        pw, pc = pred_win_by_st[st], pred_cp_by_st[st]
        pred_win[:, j] = [pw[f] for f in fens]
        pred_cp[:, j] = [pc[f] for f in fens]
    return fens, pred_win, pred_cp, gt_win, gt_cp


def rankdata(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    # tie 처리: 평균
    sorted_a = a[order]
    i = 0
    while i < len(sorted_a):
        j = i
        while j + 1 < len(sorted_a) and sorted_a[j+1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j+1]] = avg
        i = j + 1
    return ranks


def spearman_rho(x, y):
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def anova_f(values, groups):
    vals = np.asarray(values, dtype=np.float64)
    grp = np.asarray(groups)
    overall_mean = vals.mean()
    ss_between = 0.0
    ss_within = 0.0
    df_between = -1
    df_within = 0
    for g in np.unique(grp):
        mask = grp == g
        n_g = int(mask.sum())
        if n_g == 0:
            continue
        gv = vals[mask]
        ss_between += n_g * (gv.mean() - overall_mean) ** 2
        ss_within += ((gv - gv.mean()) ** 2).sum()
        df_between += 1
        df_within += n_g - 1
    if df_between <= 0 or df_within <= 0 or ss_within == 0:
        return float("nan")
    return float((ss_between / df_between) / (ss_within / df_within))


def compute_features(fens):
    feats = {
        "imbalance": [],
        "capture_count": [],
        "legal_moves": [],
        "total_material": [],
        "forcing_total": [],
        "is_check": [],
        "is_endgame": [],
    }
    for fen in fens:
        board = chess.Board(fen)
        f = _extract_features(board)
        feats["imbalance"].append(f.material_imbalance)
        feats["capture_count"].append(f.capture_count)
        feats["legal_moves"].append(f.legal_moves)
        feats["total_material"].append(f.total_material)
        feats["forcing_total"].append(f.white_forcing + f.black_forcing)
        feats["is_check"].append(int(f.is_check))
        feats["is_endgame"].append(int(f.is_endgame))
    return {k: np.asarray(v) for k, v in feats.items()}


def boot_ci_uniform(a, b, n_boot, seed=42, chunk=1000):
    """mean(A) - mean(B) paired bootstrap CI95."""
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


def analyze_space(name, err_mat, gt_vec, fens, feats, times_arr):
    """단일 공간 (win% 또는 cp) 의 분석을 출력."""
    N, T = err_mat.shape

    print(f"\n{'=' * 78}")
    print(f"  공간: {name}")
    print(f"{'=' * 78}")

    # 1. Oracle 정책 — 각 FEN 의 best_t
    best_t_idx = np.argmin(err_mat, axis=1)
    best_t_ms = times_arr[best_t_idx]
    err_oracle = err_mat[np.arange(N), best_t_idx]
    mae_oracle = float(err_oracle.mean())
    mean_t_oracle = float(best_t_ms.mean())

    # 2. Uniform 정책별 — 각 시간 j 에서의 평균 오차
    uniform_maes = err_mat.mean(axis=0)

    # 3. Oracle vs 각 uniform CI
    print(f"\n[ Oracle 정책 (각 FEN 의 best_t) ]")
    print(f"  MAE = {mae_oracle:.4f}  |  평균 시간 = {mean_t_oracle:.1f} ms")

    print(f"\n[ Uniform 정책 비교 ]")
    print(f"{'time(ms)':>10}  {'uniform MAE':>12}  {'diff vs oracle':>15}  "
          f"{'rel (%)':>10}  {'CI95':>22}")
    print("-" * 78)
    for j, t in enumerate(SEARCHTIMES_MS):
        u_mae = float(uniform_maes[j])
        diff = u_mae - mae_oracle
        rel = diff / u_mae * 100
        # CI on (uniform - oracle)
        ci_lo, ci_hi = boot_ci_uniform(err_mat[:, j], err_oracle, N_BOOT)
        print(f"  {t:>8}  {u_mae:12.4f}  {diff:+15.4f}  {rel:+10.2f}  "
              f"[{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # 4. best_t 분포
    print(f"\n[ best_t 분포 ]")
    bt_counter = Counter(int(t) for t in best_t_ms)
    print(f"{'time(ms)':>10}  {'FEN 수':>8}  {'비중 (%)':>10}")
    for t in SEARCHTIMES_MS:
        c = bt_counter.get(t, 0)
        print(f"  {t:>8}  {c:>8}  {c/N*100:>10.1f}")

    # 5. Spearman ρ(feature, best_t) + ANOVA F
    print(f"\n[ 정적 FEN feature × best_t 상관 ]")
    print(f"{'feature':<22}{'Spearman ρ':>14}{'|ρ|':>10}{'ANOVA F':>12}")
    print("-" * 78)
    results = []
    for fname, vals in feats.items():
        rho = spearman_rho(vals, best_t_ms)
        f_stat = anova_f(vals, best_t_idx)
        results.append((fname, rho, f_stat))
        print(f"  {fname:<22}{rho:>+14.4f}{abs(rho):>10.4f}{f_stat:>12.2f}")
    print("\n  |ρ| 큰 순 top 5:")
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    for fname, rho, f_stat in results[:5]:
        print(f"    {fname:<22} ρ={rho:+.4f}, F={f_stat:.2f}")

    # 6. feature × gt 거리 상관 (가중치 트릭 진단)
    print(f"\n[ 진단: feature × |gt 중앙거리| 상관 ]")
    print(f"  (feature 가 gt 와 강하게 상관되면 정책이 gt 정보를 우회 사용)")
    for fname, vals in feats.items():
        rho_gtd = spearman_rho(vals, np.abs(gt_vec - gt_vec.mean()))
        print(f"  {fname:<22} ρ(feat, |gt-mean|) = {rho_gtd:+.4f}")

    return {
        "name": name,
        "mae_oracle": mae_oracle,
        "mean_t_oracle": mean_t_oracle,
        "uniform_maes": dict(zip(SEARCHTIMES_MS, uniform_maes.tolist())),
        "best_t_distribution": dict(bt_counter),
        "feature_rho": [(n, r, f) for n, r, f in results],
    }


def main():
    fens, pred_win, pred_cp, gt_win, gt_cp = load_raw_grid(RAW_DIR, SEARCHTIMES_MS)
    N = len(fens)
    times_arr = np.array(SEARCHTIMES_MS, dtype=np.float64)

    print(f"FEN {N}개 로드 완료")
    print(f"시간 격자: {SEARCHTIMES_MS} ms")
    print(f"\nGround truth 분포:")
    print(f"  win% — min: {gt_win.min():.2f}, max: {gt_win.max():.2f}, "
          f"mean: {gt_win.mean():.2f}")
    print(f"  cp   — min: {gt_cp.min():.1f}, max: {gt_cp.max():.1f}, "
          f"mean: {gt_cp.mean():.1f}")
    sentinel_count = int(((gt_cp == MATE_CP_SENTINEL) | (gt_cp == -MATE_CP_SENTINEL)).sum())
    print(f"  cp sentinel (mate 추정) FEN 수: {sentinel_count} ({sentinel_count/N*100:.1f}%)")

    # 두 공간의 오차 행렬
    err_win = np.abs(pred_win - gt_win[:, None])
    err_cp = np.abs(pred_cp - gt_cp[:, None])

    # feature 추출 (한 번만)
    print(f"\nfeature 추출 중...")
    feats = compute_features(fens)

    # 각 공간 분석
    res_win = analyze_space("win% (Lichess 변환)", err_win, gt_win, fens, feats, times_arr)
    res_cp = analyze_space("cp (centipawn, mate=±10000)", err_cp, gt_cp, fens, feats, times_arr)

    # 두 공간 비교 — 핵심 결론 일관성
    print(f"\n{'=' * 78}")
    print(f"  두 공간 결론 일관성 검증")
    print(f"{'=' * 78}")

    print(f"\n[ Oracle vs wu(250) ]")
    wu250_win = res_win["uniform_maes"][250]
    wu250_cp = res_cp["uniform_maes"][250]
    print(f"  win%: oracle={res_win['mae_oracle']:.4f}, wu(250)={wu250_win:.4f}, "
          f"개선={100*(wu250_win - res_win['mae_oracle'])/wu250_win:.1f}%")
    print(f"  cp  : oracle={res_cp['mae_oracle']:.1f},  wu(250)={wu250_cp:.1f},  "
          f"개선={100*(wu250_cp - res_cp['mae_oracle'])/wu250_cp:.1f}%")

    print(f"\n[ Oracle vs wu(500) — '시간 2배' 비교 ]")
    wu500_win = res_win["uniform_maes"][500]
    wu500_cp = res_cp["uniform_maes"][500]
    print(f"  win%: oracle={res_win['mae_oracle']:.4f}, wu(500)={wu500_win:.4f}, "
          f"개선={100*(wu500_win - res_win['mae_oracle'])/wu500_win:.1f}%")
    print(f"  cp  : oracle={res_cp['mae_oracle']:.1f},  wu(500)={wu500_cp:.1f},  "
          f"개선={100*(wu500_cp - res_cp['mae_oracle'])/wu500_cp:.1f}%")

    print(f"\n[ 정적 feature 최강 신호 비교 ]")
    win_top = max(res_win["feature_rho"], key=lambda x: abs(x[1]))
    cp_top = max(res_cp["feature_rho"], key=lambda x: abs(x[1]))
    print(f"  win% : {win_top[0]:<20} ρ={win_top[1]:+.4f}, F={win_top[2]:.2f}")
    print(f"  cp   : {cp_top[0]:<20} ρ={cp_top[1]:+.4f}, F={cp_top[2]:.2f}")
    all_below_005_win = all(abs(r) < 0.05 for _, r, _ in res_win["feature_rho"])
    all_below_005_cp = all(abs(r) < 0.05 for _, r, _ in res_cp["feature_rho"])
    print(f"\n  win% 공간: 모든 정적 feature |ρ| < 0.05? {all_below_005_win}")
    print(f"  cp   공간: 모든 정적 feature |ρ| < 0.05? {all_below_005_cp}")

    print(f"\n[ best_t 분포 50ms 비중 비교 ]")
    bt_50_win = res_win["best_t_distribution"].get(50, 0)
    bt_50_cp = res_cp["best_t_distribution"].get(50, 0)
    print(f"  win% : 50ms 가 best 인 FEN {bt_50_win} ({bt_50_win/N*100:.1f}%)")
    print(f"  cp   : 50ms 가 best 인 FEN {bt_50_cp} ({bt_50_cp/N*100:.1f}%)")

    print(f"\n→ 두 공간에서 결론이 일치하면 robustness 확인")


if __name__ == "__main__":
    sys.exit(main())
