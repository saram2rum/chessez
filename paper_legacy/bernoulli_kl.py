"""베르누이 KL divergence 유틸.

승률(0~100)을 베르누이 확률로 보고 한 FEN의 KL을 계산하거나,
N개 FEN에 대한 평균 KL을 구하는 데 사용한다.

평균 KL (자연로그 기준, 단위: nats):
    D̄_KL = (1/N) Σ_i [ p_i log(p_i/q_i) + (1 - p_i) log((1 - p_i)/(1 - q_i)) ]
where p_i = gt_i / 100, q_i = pred_i / 100.

p, q ∈ {0, 1} (예: mate 평가) 인 경계 케이스를 위해 양쪽 모두 [EPS, 1-EPS]로 clip 한다.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

try:
    import numpy as np
except ImportError:  # numpy 미설치 환경(scripts 일부)에서도 동작하도록
    np = None  # type: ignore[assignment]


EPS = 1e-6


def _clip_prob(p: float) -> float:
    if p < EPS:
        return EPS
    if p > 1.0 - EPS:
        return 1.0 - EPS
    return p


def bernoulli_kl_nats(pred_pct: float, gt_pct: float) -> float:
    """단일 FEN의 베르누이 KL D(p || q). pred_pct, gt_pct ∈ [0, 100]."""
    p = _clip_prob(float(gt_pct) / 100.0)
    q = _clip_prob(float(pred_pct) / 100.0)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def mean_bernoulli_kl_nats(pred_pcts: Sequence[float], gt_pcts: Sequence[float]) -> float:
    """평균 베르누이 KL. 빈 입력은 NaN."""
    if not pred_pcts or not gt_pcts:
        return float("nan")
    if len(pred_pcts) != len(gt_pcts):
        raise ValueError("pred_pcts and gt_pcts must have the same length")
    total = 0.0
    for pred, gt in zip(pred_pcts, gt_pcts):
        total += bernoulli_kl_nats(pred, gt)
    return total / len(pred_pcts)


def mean_of_kl_values(kl_values: Iterable[float]) -> float:
    """이미 per-FEN KL 리스트가 있을 때의 평균 (체크포인트 누적용)."""
    vals = list(kl_values)
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def bernoulli_kl_nats_array(pred_pcts, gt_pcts):
    """numpy 벡터화 버전. pred_pcts, gt_pcts 는 0~100 스케일의 ndarray."""
    if np is None:
        raise ImportError("numpy is required for bernoulli_kl_nats_array")
    p = np.clip(np.asarray(gt_pcts, dtype=np.float64) / 100.0, EPS, 1.0 - EPS)
    q = np.clip(np.asarray(pred_pcts, dtype=np.float64) / 100.0, EPS, 1.0 - EPS)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))
