#!/usr/bin/env python3
"""정적 feature 선형 결합 + 임계값 기반 시간 배정 — wu(250) 와 공정 비교.

핵심:
  - 각 FEN 독립 처리 (transductive X, inductive O)
  - score = α₁·feat₁ + α₂·feat₂ + ... (선형 결합)
  - 훈련 데이터에서 percentile thresholds 추출 (20/40/60/80%)
  - 새 FEN 은 그 임계값과 자기 score 비교해 시간 결정
  - 분위수 임계값이라 평균 시간 = 250ms 자동 보장

목표:
  - wu(250) 대비 동등 또는 우월한 조합 찾기
  - 못 이기면 모든 시도 보고 → negative result 견고화
"""
from __future__ import annotations

import csv
import sys
from itertools import product
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
POLICY_TIMES = [50, 150, 250, 350, 450]  # quintile 평균 250ms


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


def extract_features(fens):
    feats = {
        "imbalance": [], "forcing_total": [], "is_endgame": [],
        "is_check": [], "total_material": [], "legal_moves": [],
        "capture_count": [],
    }
    for fen in fens:
        b = chess.Board(fen)
        f = _extract_features(b)
        feats["imbalance"].append(f.material_imbalance)
        feats["forcing_total"].append(f.white_forcing + f.black_forcing)
        feats["is_endgame"].append(int(f.is_endgame))
        feats["is_check"].append(int(f.is_check))
        feats["total_material"].append(f.total_material)
        feats["legal_moves"].append(f.legal_moves)
        feats["capture_count"].append(f.capture_count)
    return {k: np.asarray(v, dtype=np.float64) for k, v in feats.items()}


def assign_by_quintile_threshold(scores, times):
    """훈련 데이터의 quintile (20/40/60/80%) 로 임계값 추출 후 시간 배정.

    times: 5개 (낮은 score → times[0], 높은 score → times[-1])
    """
    thresholds = np.percentile(scores, [20, 40, 60, 80])
    bucket_idx = np.digitize(scores, thresholds)  # 0,1,2,3,4
    assignments = np.array([times[b] for b in bucket_idx], dtype=np.float64)
    return assignments, thresholds.tolist()


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


def verdict(ci):
    if ci[1] < 0:
        return "정책 우월"
    if ci[0] > 0:
        return "wu 우월"
    return "유의차 없음"


def main():
    fens, pred_mat, gt_vec = load_raw_grid(RAW_DIR, SEARCHTIMES_MS)
    N = len(fens)
    err_mat = np.abs(pred_mat - gt_vec[:, None])
    t2i = {st: i for i, st in enumerate(SEARCHTIMES_MS)}

    print(f"FEN {N}개 — feature 추출 중...")
    feats = extract_features(fens)
    err_wu250 = err_mat[:, t2i[250]]
    wu250_mae = float(err_wu250.mean())
    print(f"wu(250) MAE = {wu250_mae:.4f}\n")

    # ρ 부호 (음/양) — score 방향 정할 때 참고
    # ρ < 0 인 feature: high feature → small best_t → short time (정상 방향)
    # ρ > 0 인 feature: high feature → large best_t → long time (역방향)

    # 시도할 조합들 — (이름, score 계산 함수)
    # score 가 높을수록 best_t 가 작다 (short time 필요) 라는 가정
    # times = [450, 350, 250, 150, 50] (low score → long time)

    combinations = [
        # 단일 feature (sanity check)
        ("imbalance",                lambda f: f["imbalance"]),
        ("forcing_total",            lambda f: f["forcing_total"]),
        ("is_endgame_neg",           lambda f: -f["is_endgame"]),  # ρ > 0 이라 부호 반대
        # 2-feature 조합
        ("imb + forcing",            lambda f: f["imbalance"] + f["forcing_total"]),
        ("imb + 2·forcing",          lambda f: f["imbalance"] + 2*f["forcing_total"]),
        ("2·imb + forcing",          lambda f: 2*f["imbalance"] + f["forcing_total"]),
        ("imb − endgame",            lambda f: f["imbalance"] - f["is_endgame"]),
        ("forcing − endgame",        lambda f: f["forcing_total"] - f["is_endgame"]),
        ("imb + forcing − endgame",  lambda f: f["imbalance"] + f["forcing_total"] - f["is_endgame"]),
        # 3-feature 조합
        ("imb + forcing + check",    lambda f: f["imbalance"] + f["forcing_total"] + f["is_check"]),
        ("imb + forcing − total_mat 보정",
         lambda f: f["imbalance"] + f["forcing_total"] + 0.1*f["total_material"]),  # total_mat ρ < 0
        # 모든 feature ρ < 0 가중 합
        ("multi-neg-ρ combo",
         lambda f: (f["imbalance"] + f["forcing_total"] + f["is_check"]
                    + 0.1*f["total_material"] + 0.1*f["legal_moves"])),
    ]

    # times 방향 두 가지 시도 (high score → short / high score → long)
    times_directions = [
        ("dec (high→short)", [450, 350, 250, 150, 50]),  # low score 가 best_t 큼
        ("inc (high→long)",  [50, 150, 250, 350, 450]),
    ]

    rows = []
    for combo_name, score_fn in combinations:
        scores = score_fn(feats)
        for dir_name, times in times_directions:
            assignments, thresholds = assign_by_quintile_threshold(scores, times)
            # sweep 격자 매핑
            policy_idx = np.array([t2i[int(t)] for t in assignments])
            err_policy = err_mat[np.arange(N), policy_idx]
            mae_policy = float(err_policy.mean())
            mean_t = float(assignments.mean())

            ci = boot_ci(err_policy, err_wu250, N_BOOT)
            d = mae_policy - wu250_mae
            v = verdict(ci)
            rel = -d / wu250_mae * 100

            rows.append({
                "combo": combo_name,
                "direction": dir_name,
                "thresholds": thresholds,
                "mean_t": mean_t,
                "mae": mae_policy,
                "diff": d,
                "ci": ci,
                "verdict": v,
                "rel_pct": rel,
            })

    # 정렬: rel_pct 큰 순 (= 정책에 좋은 순)
    rows.sort(key=lambda r: r["rel_pct"], reverse=True)

    print("=" * 110)
    print(f"임계값 기반 선형 결합 정책 vs wu(250) — 평균 시간 = 250ms (quintile 임계값)")
    print("=" * 110)
    print(f"{'조합':<30}  {'방향':<18}  {'mean_t':>7}  {'MAE':>7}  {'diff':>7}  "
          f"{'CI95':>20}  verdict          rel%")
    print("-" * 130)
    for r in rows[:15]:
        ci = r["ci"]
        print(f"  {r['combo']:<28}  {r['direction']:<18}  {r['mean_t']:7.1f}  "
              f"{r['mae']:7.4f}  {r['diff']:+7.4f}  "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]  {r['verdict']:<15} {r['rel_pct']:+6.2f}%")

    # best 변형 상세
    best = rows[0]
    print(f"\n[BEST 조합]")
    print(f"  {best['combo']} ({best['direction']})")
    print(f"  임계값: {[f'{t:.2f}' for t in best['thresholds']]}")
    print(f"  평균 시간: {best['mean_t']:.2f}ms")
    print(f"  MAE: {best['mae']:.4f}")
    print(f"  vs wu(250)({wu250_mae:.4f}): diff = {best['diff']:+.4f}, "
          f"CI95 = [{best['ci'][0]:+.4f}, {best['ci'][1]:+.4f}]")
    print(f"  판정: {best['verdict']}")

    # 정책 우월 또는 유의차 없음 만 따로
    print(f"\n[wu 못 진 조합 (정책 우월 또는 유의차 없음)]")
    survive = [r for r in rows if r["verdict"] != "wu 우월"]
    if survive:
        for r in survive[:5]:
            print(f"  {r['combo']:<30} {r['direction']:<18} diff {r['diff']:+.4f} {r['verdict']}")
    else:
        print(f"  없음. 모든 조합이 wu(250) 에 패배 — 정적 정보 한계 실증 확인.")


if __name__ == "__main__":
    sys.exit(main())
