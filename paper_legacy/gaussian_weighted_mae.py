"""Gaussian-weighted MAE on win percent (0~100).

가중치 w(p_gt) = exp(-((p_gt_norm - 0.5)^2) / (2 * sigma^2))
  - p_gt_norm = p_gt / 100  (0~1 스케일로 정규화)
  - sigma 도 0~1 스케일. 예: sigma=0.1 ⇒ 정규화 winrate 10%p 폭.

가중 평균 MAE:
    Σ w_i * |p_pred_i - p_gt_i|  /  Σ w_i
입력은 0~100 스케일을 그대로 받고 내부에서 정규화한다.

sigma 가 작을수록 0.5 근방에 더 집중, sigma 가 클수록 평평한 가중치.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


def gaussian_weight(p_gt_pct: float, sigma: float) -> float:
    """0~100 스케일 winrate에 대한 가우시안 가중치. sigma 는 0~1 스케일."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    diff = float(p_gt_pct) / 100.0 - 0.5
    return math.exp(-(diff * diff) / (2.0 * sigma * sigma))


def gaussian_weighted_abs_error(
    p_pred_pct: float, p_gt_pct: float, sigma: float
) -> tuple[float, float]:
    """(weight, weight * |err|) 튜플. err 는 0~100 스케일."""
    w = gaussian_weight(p_gt_pct, sigma)
    err = abs(float(p_pred_pct) - float(p_gt_pct))
    return w, w * err


def mean_gwmae(
    pred_pcts: Sequence[float], gt_pcts: Sequence[float], sigma: float
) -> float:
    """가중 평균 MAE. 빈 입력은 NaN."""
    if not pred_pcts or not gt_pcts:
        return float("nan")
    if len(pred_pcts) != len(gt_pcts):
        raise ValueError("pred_pcts and gt_pcts must have the same length")
    num = 0.0
    den = 0.0
    for pred, gt in zip(pred_pcts, gt_pcts):
        w, we = gaussian_weighted_abs_error(pred, gt, sigma)
        num += we
        den += w
    return num / den if den > 0 else float("nan")


def gaussian_weight_array(p_gt_pcts, sigma: float):
    """numpy 벡터화: 0~100 스케일 ndarray → 가중치 ndarray."""
    if np is None:
        raise ImportError("numpy is required for gaussian_weight_array")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    diff = np.asarray(p_gt_pcts, dtype=np.float64) / 100.0 - 0.5
    return np.exp(-(diff * diff) / (2.0 * sigma * sigma))


def bernoulli_variance_weight_array(p_gt_pcts):
    """베르누이 시행의 분산 기반 가중치: w(gt) = p(1-p), p = gt/100.

    하이퍼파라미터가 없는 가중치. gt=50%에서 최대(0.25), gt=0%/100%에서 0.
    리체스 win% 변환 시그모이드의 도함수와 비례. 결과 불확실성이 큰
    위치일수록 가중치가 크다.
    """
    if np is None:
        raise ImportError("numpy is required for bernoulli_variance_weight_array")
    p = np.asarray(p_gt_pcts, dtype=np.float64) / 100.0
    return p * (1.0 - p)


def gwmae_per_pair_array(pred_pcts, gt_pcts, sigma: float):
    """numpy 벡터화: (weight, weighted_abs_err) 두 ndarray 반환."""
    if np is None:
        raise ImportError("numpy is required for gwmae_per_pair_array")
    pred = np.asarray(pred_pcts, dtype=np.float64)
    gt = np.asarray(gt_pcts, dtype=np.float64)
    w = gaussian_weight_array(gt, sigma)
    err = np.abs(pred - gt)
    return w, w * err


def mean_gwmae_array(pred_pcts, gt_pcts, sigma: float) -> float:
    """numpy 벡터화 평균. 빈/총가중치 0이면 NaN."""
    w, we = gwmae_per_pair_array(pred_pcts, gt_pcts, sigma)
    total_w = float(w.sum())
    if total_w <= 0:
        return float("nan")
    return float(we.sum()) / total_w


def mean_of_weighted_pairs(pairs: Iterable[tuple[float, float]]) -> float:
    """(weight, weighted_abs_err) 누적 평균 (체크포인트용)."""
    num = 0.0
    den = 0.0
    for w, we in pairs:
        num += float(we)
        den += float(w)
    return num / den if den > 0 else float("nan")
