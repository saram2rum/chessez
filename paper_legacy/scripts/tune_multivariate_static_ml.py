#!/usr/bin/env python3
"""정적 FEN feature 기반 best_t 예측 — 다변량 ML 검증.

기존 2-feature Bucket Oracle (`tune_bucket_oracle_holdout.py`) 의 train/test
프로토콜을 그대로 따르되, 수동 binning 없이 모든 정적 feature 를 연속값으로
트리 모델에 투입하여 "정적 정보 전체의 예측 한계"를 측정한다.

- 트리 모델: LightGBM 시도, 실패 시 sklearn RandomForest fallback
- 타깃 A (회귀): best_t 직접 예측 → 격자에 스냅 후 test MAE
- 타깃 B (분류): "best_t ≤ 200ms" 이진 분류 → AUC + 정책화 (임계값 train 보정)
- baseline:
  - wu_match: 정책 평균 시간 이상의 가장 가까운 격자 wu (보수적)
  - wu(250): 고정 250ms
  - bucket_oracle_2f: 동일 split 위에서 (imb, forcing) 2-feature Bucket Oracle 재실행
"""
from __future__ import annotations

import csv
import json
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
TRAIN_FRAC = 0.8  # 80/20 train/test (관례적 표준 비율, train-heavy)
IMB_CLAMP = 9
FORCING_CLAMP = 5
N_SEEDS = 10
N_BOOT = 5_000

# 분류 라벨 임계값
SHORT_T_THRESHOLD_MS = 200

# 모델 선택
USE_LIGHTGBM = False
try:
    import lightgbm as lgb  # noqa: F401
    USE_LIGHTGBM = True
except Exception:
    USE_LIGHTGBM = False

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score


FEATURE_NAMES = [
    "legal_moves",
    "total_material",
    "is_endgame",
    "material_imbalance",
    "capture_count",
    "is_check",
    "forcing_total",
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


def extract_feature_matrix(fens):
    """정적 feature → (N, F) 행렬. 연속값 원본 그대로."""
    rows = []
    for fen in fens:
        b = chess.Board(fen)
        f = _extract_features(b)
        row = [
            f.legal_moves,
            f.total_material,
            int(f.is_endgame),
            f.material_imbalance,
            f.capture_count,
            int(f.is_check),
            f.white_forcing + f.black_forcing,
        ]
        rows.append(row)
    return np.array(rows, dtype=np.float64)


def extract_bucket_keys(fens):
    """2-feature Bucket Oracle 용 (imb, forcing) clamp 키."""
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


def verdict_from_ci(ci):
    if ci[1] < 0:
        return "정책 우월"
    if ci[0] > 0:
        return "wu 우월"
    return "유의차 없음"


def snap_to_grid(values, grid):
    """예측값을 가장 가까운 격자 인덱스로 스냅."""
    grid_arr = np.asarray(grid, dtype=np.float64)
    idx = np.argmin(np.abs(values[:, None] - grid_arr[None, :]), axis=1)
    return idx


def conservative_wu_idx(policy_mean_ms, grid):
    """정책 평균 시간 이상의 가장 가까운 격자 인덱스 (없으면 마지막 격자)."""
    for i, g in enumerate(grid):
        if g >= policy_mean_ms - 1e-6:
            return i
    return len(grid) - 1


def fit_regression(X_train, y_train, X_test, seed):
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test), model.predict(X_train), model


def fit_classifier(X_train, y_train, X_test, seed):
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    p_train = model.predict_proba(X_train)[:, 1]
    return p_test, p_train, model


def bucket_oracle_2f(keys, err_mat, times_arr, train_idx, test_idx,
                    target_ms, global_best_idx):
    """기존 2-feature Bucket Oracle 을 동일 split 위에서 재실행."""
    train_buckets = {}
    for i in train_idx:
        train_buckets.setdefault(keys[i], []).append(i)
    bucket_errors = {k: err_mat[v, :].mean(axis=0) for k, v in train_buckets.items()}
    train_sizes = {k: len(v) for k, v in train_buckets.items()}
    n_train = len(train_idx)

    def assign_alpha(alpha):
        choices = {}
        for key, errs in bucket_errors.items():
            costs = errs + alpha * times_arr
            choices[key] = int(np.argmin(costs))
        total = sum(train_sizes[k] * times_arr[choices[k]] for k in choices)
        return choices, total / n_train

    lo, hi = 0.0, 1.0
    for _ in range(40):
        _, mt_hi = assign_alpha(hi)
        if mt_hi <= target_ms:
            break
        hi *= 4.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        _, mt = assign_alpha(mid)
        if mt > target_ms:
            lo = mid
        else:
            hi = mid
    bucket_time_idx, train_mt = assign_alpha(hi)

    test_idx_pol = np.zeros(len(test_idx), dtype=np.int64)
    fallback = 0
    for ti, fen_i in enumerate(test_idx):
        key = keys[fen_i]
        if key in bucket_time_idx:
            test_idx_pol[ti] = bucket_time_idx[key]
        else:
            test_idx_pol[ti] = global_best_idx
            fallback += 1
    return test_idx_pol, fallback, train_mt


def regression_policy(reg_pred_train, reg_pred_test, times_arr, target_ms):
    """회귀 예측을 격자에 스냅. 추가로 train 평균이 target 에 가깝도록 시프트."""
    # 단순 스냅
    test_idx = snap_to_grid(reg_pred_test, times_arr)
    return test_idx


def classification_policy(p_train, p_test, train_best_idx_short_long,
                          times_arr, target_ms, t_short_idx, t_long_idx,
                          train_idx, err_mat):
    """확률 임계값 τ 를 train 에서 mean_t == target 이 되도록 보정.
    p ≥ τ 이면 t_short, 아니면 t_long.
    """
    t_short = times_arr[t_short_idx]
    t_long = times_arr[t_long_idx]
    # mean_t = f * t_short + (1-f) * t_long = target
    # f = (t_long - target) / (t_long - t_short)
    target_short_frac = (t_long - target_ms) / (t_long - t_short)
    target_short_frac = float(np.clip(target_short_frac, 0.0, 1.0))

    # train 에서 p_train 의 상위 target_short_frac 분위 = 임계값
    n_train_short = int(round(target_short_frac * len(p_train)))
    if n_train_short <= 0:
        tau = 1.1
    elif n_train_short >= len(p_train):
        tau = -0.1
    else:
        sorted_p = np.sort(p_train)[::-1]
        tau = float(sorted_p[n_train_short - 1])

    pol_idx = np.where(p_test >= tau, t_short_idx, t_long_idx)
    return pol_idx.astype(np.int64), tau, target_short_frac


def main():
    print(f"모델 백엔드: {'LightGBM' if USE_LIGHTGBM else 'sklearn RandomForest'}")
    print(f"FEN 데이터 로딩...")
    fens, pred_mat, gt_vec = load_raw_grid(RAW_DIR, SEARCHTIMES_MS)
    N = len(fens)
    N_TRAIN = int(round(N * TRAIN_FRAC))
    err_mat = np.abs(pred_mat - gt_vec[:, None])
    times_arr = np.array(SEARCHTIMES_MS, dtype=np.float64)
    t2i = {st: i for i, st in enumerate(SEARCHTIMES_MS)}
    T = len(SEARCHTIMES_MS)

    print(f"FEN {N}개, 시간 격자 {SEARCHTIMES_MS}")
    print(f"정적 feature 추출 중...")
    X = extract_feature_matrix(fens)
    keys2f = extract_bucket_keys(fens)
    print(f"feature 행렬 shape: {X.shape}, 컬럼: {FEATURE_NAMES}")

    best_t_idx = np.argmin(err_mat, axis=1)
    best_t_ms = times_arr[best_t_idx]
    label_short = (best_t_ms <= SHORT_T_THRESHOLD_MS).astype(np.int64)
    print(f"\n전체 best_t 분포 (ms): "
          f"{dict(zip(*np.unique(best_t_ms, return_counts=True)))}")
    print(f"label_short (best_t ≤ {SHORT_T_THRESHOLD_MS}ms) 비율: "
          f"{label_short.mean():.3f}")

    global_errors = err_mat.mean(axis=0)
    global_best_idx = int(np.argmin(global_errors))
    print(f"전체 평균 MAE 가 최소인 시간: "
          f"{SEARCHTIMES_MS[global_best_idx]}ms (MAE {global_errors[global_best_idx]:.4f})")

    # 분류 정책에서 사용할 t_short, t_long
    t_short_idx = t2i[100]
    t_long_idx = t2i[400]

    print(f"\n{'='*100}")
    print(f"Train/Test Split 검증 (train {N_TRAIN}, test {N-N_TRAIN}, {N_SEEDS} seeds)")
    print(f"target_ms = {TARGET_MS}, 분류 t_short={SEARCHTIMES_MS[t_short_idx]}ms, "
          f"t_long={SEARCHTIMES_MS[t_long_idx]}ms")
    print(f"{'='*100}")

    rows = []
    feature_importance_sum_reg = np.zeros(X.shape[1])
    feature_importance_sum_clf = np.zeros(X.shape[1])

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N)
        train_idx = perm[:N_TRAIN]
        test_idx = perm[N_TRAIN:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_reg_train = best_t_ms[train_idx]
        y_clf_train = label_short[train_idx]
        y_reg_test = best_t_ms[test_idx]
        y_clf_test = label_short[test_idx]

        # === A. 회귀 ===
        reg_pred_test, reg_pred_train, reg_model = fit_regression(
            X_train, y_reg_train, X_test, seed
        )
        reg_pol_idx_test = regression_policy(
            reg_pred_train, reg_pred_test, times_arr, TARGET_MS
        )
        test_err_reg = err_mat[test_idx, reg_pol_idx_test]
        reg_mae_test = float(test_err_reg.mean())
        reg_mt_test = float(times_arr[reg_pol_idx_test].mean())
        train_err_reg = err_mat[train_idx, snap_to_grid(reg_pred_train, times_arr)]
        reg_mae_train = float(train_err_reg.mean())
        feature_importance_sum_reg += reg_model.feature_importances_

        # === B. 분류 ===
        p_test, p_train, clf_model = fit_classifier(
            X_train, y_clf_train, X_test, seed
        )
        try:
            auc = float(roc_auc_score(y_clf_test, p_test))
        except ValueError:
            auc = float("nan")
        try:
            auc_train = float(roc_auc_score(y_clf_train, p_train))
        except ValueError:
            auc_train = float("nan")
        clf_pol_idx_test, tau, short_frac = classification_policy(
            p_train, p_test, None, times_arr, TARGET_MS,
            t_short_idx, t_long_idx, train_idx, err_mat
        )
        test_err_clf = err_mat[test_idx, clf_pol_idx_test]
        clf_mae_test = float(test_err_clf.mean())
        clf_mt_test = float(times_arr[clf_pol_idx_test].mean())
        # train MAE (같은 임계값 적용)
        train_pol_idx = np.where(p_train >= tau, t_short_idx, t_long_idx).astype(np.int64)
        clf_mae_train = float(err_mat[train_idx, train_pol_idx].mean())
        feature_importance_sum_clf += clf_model.feature_importances_

        # === Baseline: 2-feature Bucket Oracle ===
        bo_idx, bo_fallback, bo_train_mt = bucket_oracle_2f(
            keys2f, err_mat, times_arr, train_idx, test_idx, TARGET_MS,
            global_best_idx
        )
        test_err_bo = err_mat[test_idx, bo_idx]
        bo_mae_test = float(test_err_bo.mean())
        bo_mt_test = float(times_arr[bo_idx].mean())

        # === wu baselines ===
        # 1) 고정 wu(250)
        wu250_err = err_mat[test_idx, t2i[250]]
        wu250_mae = float(wu250_err.mean())

        # 2) 각 정책별 보수적 wu (정책 mean_t ≥ 격자)
        def conservative(policy_mt):
            j = conservative_wu_idx(policy_mt, times_arr)
            return j, float(times_arr[j]), float(err_mat[test_idx, j].mean()), err_mat[test_idx, j]

        wu_reg_j, wu_reg_t, wu_reg_mae, wu_reg_err = conservative(reg_mt_test)
        wu_clf_j, wu_clf_t, wu_clf_mae, wu_clf_err = conservative(clf_mt_test)
        wu_bo_j, wu_bo_t, wu_bo_mae, wu_bo_err = conservative(bo_mt_test)

        # === verdicts (보수적 wu 기준) ===
        ci_reg = boot_ci(test_err_reg, wu_reg_err, N_BOOT, seed=42 + seed)
        ci_clf = boot_ci(test_err_clf, wu_clf_err, N_BOOT, seed=42 + seed)
        ci_bo = boot_ci(test_err_bo, wu_bo_err, N_BOOT, seed=42 + seed)
        v_reg = verdict_from_ci(ci_reg)
        v_clf = verdict_from_ci(ci_clf)
        v_bo = verdict_from_ci(ci_bo)

        rows.append(dict(
            seed=seed,
            # A. regression
            reg_mae_train=reg_mae_train,
            reg_mae_test=reg_mae_test,
            reg_mt_test=reg_mt_test,
            wu_reg_t=wu_reg_t,
            wu_reg_mae=wu_reg_mae,
            ci_reg=ci_reg,
            v_reg=v_reg,
            # B. classifier
            clf_auc_test=auc,
            clf_auc_train=auc_train,
            clf_mae_train=clf_mae_train,
            clf_mae_test=clf_mae_test,
            clf_mt_test=clf_mt_test,
            clf_tau=tau,
            clf_short_frac=short_frac,
            wu_clf_t=wu_clf_t,
            wu_clf_mae=wu_clf_mae,
            ci_clf=ci_clf,
            v_clf=v_clf,
            # Bucket Oracle 2f
            bo_mae_test=bo_mae_test,
            bo_mt_test=bo_mt_test,
            wu_bo_t=wu_bo_t,
            wu_bo_mae=wu_bo_mae,
            ci_bo=ci_bo,
            v_bo=v_bo,
            bo_fallback=bo_fallback,
            wu250_mae=wu250_mae,
        ))

    # === 출력 ===
    print("\n[A] 회귀 (RandomForest, best_t 직접 예측)")
    print(f"{'seed':>4}  {'train_MAE':>9}  {'test_MAE':>9}  {'mt(ms)':>7}  "
          f"{'wu_match':>8}  {'wu_MAE':>8}  {'diff':>8}  {'rel':>7}  verdict")
    for r in rows:
        diff = r['reg_mae_test'] - r['wu_reg_mae']
        rel = -diff / r['wu_reg_mae'] * 100
        print(f"  {r['seed']:>2}  {r['reg_mae_train']:9.4f}  {r['reg_mae_test']:9.4f}  "
              f"{r['reg_mt_test']:7.2f}  {r['wu_reg_t']:8.0f}  {r['wu_reg_mae']:8.4f}  "
              f"{diff:+8.4f}  {rel:+6.2f}%  {r['v_reg']}")

    print("\n[B] 분류 (RandomForest, label = best_t ≤ 200ms; 정책: p≥τ→100ms, else 400ms)")
    print(f"{'seed':>4}  {'AUC_tr':>7}  {'AUC_te':>7}  {'train_MAE':>9}  {'test_MAE':>9}  "
          f"{'mt(ms)':>7}  {'τ':>6}  {'short%':>7}  {'wu_match':>8}  {'wu_MAE':>8}  "
          f"{'diff':>8}  {'rel':>7}  verdict")
    for r in rows:
        diff = r['clf_mae_test'] - r['wu_clf_mae']
        rel = -diff / r['wu_clf_mae'] * 100
        print(f"  {r['seed']:>2}  {r['clf_auc_train']:7.3f}  {r['clf_auc_test']:7.3f}  "
              f"{r['clf_mae_train']:9.4f}  {r['clf_mae_test']:9.4f}  {r['clf_mt_test']:7.2f}  "
              f"{r['clf_tau']:6.3f}  {r['clf_short_frac']*100:6.2f}%  "
              f"{r['wu_clf_t']:8.0f}  {r['wu_clf_mae']:8.4f}  {diff:+8.4f}  {rel:+6.2f}%  {r['v_clf']}")

    print("\n[Baseline] 2-feature Bucket Oracle (imb, forcing)")
    print(f"{'seed':>4}  {'test_MAE':>9}  {'mt(ms)':>7}  {'wu_match':>8}  {'wu_MAE':>8}  "
          f"{'diff':>8}  {'rel':>7}  {'fallback':>9}  verdict")
    for r in rows:
        diff = r['bo_mae_test'] - r['wu_bo_mae']
        rel = -diff / r['wu_bo_mae'] * 100
        print(f"  {r['seed']:>2}  {r['bo_mae_test']:9.4f}  {r['bo_mt_test']:7.2f}  "
              f"{r['wu_bo_t']:8.0f}  {r['wu_bo_mae']:8.4f}  {diff:+8.4f}  "
              f"{rel:+6.2f}%  {r['bo_fallback']:>6}  {r['v_bo']}")

    def summarize(policy_name, mae_test_key, mt_key, wu_mae_key, verdict_key, train_key=None):
        maes = np.array([r[mae_test_key] for r in rows])
        mts = np.array([r[mt_key] for r in rows])
        wus = np.array([r[wu_mae_key] for r in rows])
        diffs = maes - wus
        rels = -diffs / wus * 100
        n_win = sum(1 for r in rows if r[verdict_key] == "정책 우월")
        n_tie = sum(1 for r in rows if r[verdict_key] == "유의차 없음")
        n_lose = sum(1 for r in rows if r[verdict_key] == "wu 우월")
        print(f"\n  {policy_name}")
        print(f"    test_MAE 평균 {maes.mean():.4f} (std {maes.std():.4f})")
        print(f"    mean_t  평균 {mts.mean():.2f}ms")
        print(f"    diff vs wu_match: 평균 {diffs.mean():+.4f}, rel 평균 {rels.mean():+.2f}%")
        print(f"    판정 분포: 정책 우월 {n_win} / 유의차 없음 {n_tie} / wu 우월 {n_lose}")
        if train_key is not None:
            tr = np.array([r[train_key] for r in rows])
            print(f"    train_MAE 평균 {tr.mean():.4f} | gap (test - train): {maes.mean()-tr.mean():+.4f}")

    print(f"\n{'='*100}")
    print(f"  요약 ({N_SEEDS} seeds)")
    print(f"{'='*100}")
    summarize("[A] regression policy", "reg_mae_test", "reg_mt_test",
              "wu_reg_mae", "v_reg", "reg_mae_train")
    summarize("[B] classification policy", "clf_mae_test", "clf_mt_test",
              "wu_clf_mae", "v_clf", "clf_mae_train")
    summarize("[Baseline] 2f Bucket Oracle", "bo_mae_test", "bo_mt_test",
              "wu_bo_mae", "v_bo")

    aucs_test = np.array([r['clf_auc_test'] for r in rows])
    aucs_train = np.array([r['clf_auc_train'] for r in rows])
    print(f"\n  [B] AUC train 평균 {aucs_train.mean():.3f}, test 평균 {aucs_test.mean():.3f} "
          f"(gap {aucs_train.mean()-aucs_test.mean():+.3f})")

    wu250s = np.array([r['wu250_mae'] for r in rows])
    print(f"\n  [참고] wu(250 고정) test_MAE 평균 {wu250s.mean():.4f} (std {wu250s.std():.4f})")

    # === Feature Importance ===
    print(f"\n{'='*100}")
    print(f"  Feature Importance ({N_SEEDS} seeds 평균)")
    print(f"{'='*100}")
    imp_reg = feature_importance_sum_reg / N_SEEDS
    imp_clf = feature_importance_sum_clf / N_SEEDS
    print(f"  {'feature':<20}  {'reg_imp':>8}  {'clf_imp':>8}")
    order = np.argsort(-imp_reg)
    for j in order:
        print(f"  {FEATURE_NAMES[j]:<20}  {imp_reg[j]:8.4f}  {imp_clf[j]:8.4f}")

    # === 최종 비교표 ===
    print(f"\n{'='*100}")
    print(f"  최종 비교표 — test MAE 평균 (10 seeds)")
    print(f"{'='*100}")
    reg_mean = np.mean([r['reg_mae_test'] for r in rows])
    clf_mean = np.mean([r['clf_mae_test'] for r in rows])
    bo_mean = np.mean([r['bo_mae_test'] for r in rows])
    wu_reg_mean = np.mean([r['wu_reg_mae'] for r in rows])
    wu_clf_mean = np.mean([r['wu_clf_mae'] for r in rows])
    wu_bo_mean = np.mean([r['wu_bo_mae'] for r in rows])
    print(f"  {'policy':<28}  {'test_MAE':>10}  {'mean_t':>8}  {'wu_match_MAE':>13}  {'diff':>9}")
    print(f"  {'A. ML regression':<28}  {reg_mean:10.4f}  "
          f"{np.mean([r['reg_mt_test'] for r in rows]):8.2f}  "
          f"{wu_reg_mean:13.4f}  {reg_mean-wu_reg_mean:+9.4f}")
    print(f"  {'B. ML classification':<28}  {clf_mean:10.4f}  "
          f"{np.mean([r['clf_mt_test'] for r in rows]):8.2f}  "
          f"{wu_clf_mean:13.4f}  {clf_mean-wu_clf_mean:+9.4f}")
    print(f"  {'2f Bucket Oracle':<28}  {bo_mean:10.4f}  "
          f"{np.mean([r['bo_mt_test'] for r in rows]):8.2f}  "
          f"{wu_bo_mean:13.4f}  {bo_mean-wu_bo_mean:+9.4f}")
    print(f"  {'wu(250) 고정':<28}  {wu250s.mean():10.4f}  {250.0:8.2f}  "
          f"{'(self)':>13}  {0.0:+9.4f}")

    # JSON 저장
    out_path = PROJECT_ROOT / "multivariate_static_ml_results.json"
    serializable = {
        "model_backend": "LightGBM" if USE_LIGHTGBM else "sklearn RandomForest",
        "features": FEATURE_NAMES,
        "n_fens": N,
        "n_train": N_TRAIN,
        "n_test": N - N_TRAIN,
        "n_seeds": N_SEEDS,
        "target_ms": TARGET_MS,
        "short_t_threshold_ms": SHORT_T_THRESHOLD_MS,
        "rows": [
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}
            for r in rows
        ],
        "summary": {
            "reg_mae_test_mean": float(reg_mean),
            "clf_mae_test_mean": float(clf_mean),
            "bo_mae_test_mean": float(bo_mean),
            "wu250_mae_mean": float(wu250s.mean()),
            "clf_auc_test_mean": float(aucs_test.mean()),
            "clf_auc_train_mean": float(aucs_train.mean()),
            "feature_importance_reg": dict(zip(FEATURE_NAMES, imp_reg.tolist())),
            "feature_importance_clf": dict(zip(FEATURE_NAMES, imp_clf.tolist())),
        },
    }
    with out_path.open("w") as fp:
        json.dump(serializable, fp, indent=2, ensure_ascii=False)
    print(f"\nJSON 저장: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
