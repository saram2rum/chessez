#!/usr/bin/env python3
"""
오라클 분석: sweep 데이터를 활용한 complexity 시스템 천장(ceiling) 분석.

각 FEN에 대해 100ms, 250ms, 400ms 에서의 예측 베르누이 KL(nats)을 비교하여:
1. 이론적 최대 edge (완벽한 분류 시 평균 KL)
2. 현재 분류의 오라클 일치율
3. "시간에 민감한" 포지션의 특성 분석

API 호출 없이 기존 sweep_3times_raw.csv + feature 추출만으로 동작한다.

Usage:
    python3 scripts/oracle_analysis.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.bernoulli_kl import bernoulli_kl_nats
from app.utils.position_complexity import (
    EVAL_TIME_BY_BUCKET,
    evaluate_position_complexity,
)

SWEEP_CSV = PROJECT_ROOT / "sweep_3times_raw.csv"
POOL_CSV = PROJECT_ROOT / "mixed_workload_pool.csv"

BUCKET_TIME = {"S": 100, "M": 250, "L": 400}
TIME_TO_BUCKET = {100: "S", 250: "M", 400: "L"}
TIMES = [100, 250, 400]


def load_sweep() -> list[dict]:
    rows = []
    with open(SWEEP_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fen = row["fen"].strip()
            gt = float(row["ground_truth"])
            p100 = float(row["pred_100ms"])
            p250 = float(row["pred_250ms"])
            p400 = float(row["pred_400ms"])
            rows.append({
                "fen": fen,
                "gt": gt,
                "kl": {
                    100: bernoulli_kl_nats(p100, gt),
                    250: bernoulli_kl_nats(p250, gt),
                    400: bernoulli_kl_nats(p400, gt),
                },
            })
    return rows


def oracle_bucket(kl: dict[int, float]) -> int:
    """시간 예산 대비 최적 할당: 같은 KL이면 더 짧은 시간을 선호."""
    return min(TIMES, key=lambda t: (kl[t], t))


def oracle_budget_constrained(data: list[dict], target_avg_ms: float) -> dict:
    """
    시간 예산 제약 하에 최적 할당 (탐욕적).
    각 FEN을 250ms(M)에서 시작하고, 시간 예산 여유분으로
    가장 이득이 큰 FEN을 L로 승격하거나 가장 손해가 적은 FEN을 S로 강등.
    """
    n = len(data)
    assignments = [250] * n

    gain_to_l = []
    cost_to_s = []
    for i, d in enumerate(data):
        gain_to_l.append((d["kl"][250] - d["kl"][400], i))
        cost_to_s.append((d["kl"][100] - d["kl"][250], i))

    gain_to_l.sort(reverse=True)
    cost_to_s.sort()

    current_total = 250 * n
    target_total = target_avg_ms * n

    li = 0
    si = 0
    used_l = set()
    used_s = set()

    while current_total > target_total and si < n:
        _, idx = cost_to_s[si]
        if idx not in used_l:
            assignments[idx] = 100
            current_total -= 150
            used_s.add(idx)
        si += 1

    while current_total < target_total and li < n:
        gain, idx = gain_to_l[li]
        if idx not in used_s and gain > 0:
            assignments[idx] = 400
            current_total += 150
            used_l.add(idx)
        li += 1

    return {
        "assignments": assignments,
        "avg_ms": sum(assignments) / n,
    }


def main() -> int:
    data = load_sweep()
    n = len(data)
    print(f"FEN {n}개 로드\n")

    # ─── 1. Uniform baselines ───
    print("=" * 65)
    print("  1. Uniform baselines (모든 FEN에 동일 시간)")
    print("=" * 65)
    for t in TIMES:
        mean_kl = sum(d["kl"][t] for d in data) / n
        print(f"  {t:4d}ms uniform  mean KL = {mean_kl:.6f} nats")

    # ─── 2. 현재 분류 성능 ───
    print("\n" + "=" * 65)
    print("  2. 현재 complexity 분류 성능")
    print("=" * 65)
    current_assignments = []
    for d in data:
        ev = evaluate_position_complexity(d["fen"])
        current_assignments.append(BUCKET_TIME[ev.bucket])

    current_kl = sum(
        d["kl"][t] for d, t in zip(data, current_assignments)
    ) / n
    current_avg = sum(current_assignments) / n
    uniform_at_current = sum(d["kl"][250] for d in data) / n

    dist = Counter(current_assignments)
    print(f"  mean KL = {current_kl:.6f} nats")
    print(f"  avg time= {current_avg:.1f} ms")
    print(f"  uniform@{current_avg:.0f}ms ≈ {uniform_at_current:.6f} nats (250ms 기준)")
    print(f"  edge    = {uniform_at_current - current_kl:+.6f} nats")
    print(f"  분포     S={dist[100]} M={dist[250]} L={dist[400]}")

    # ─── 3. 무제약 오라클 (각 FEN에 가장 좋은 시간) ───
    print("\n" + "=" * 65)
    print("  3. 무제약 오라클 (각 FEN에 최적 시간 — 천장)")
    print("=" * 65)
    oracle_times = [oracle_bucket(d["kl"]) for d in data]
    oracle_kl = sum(d["kl"][t] for d, t in zip(data, oracle_times)) / n
    oracle_avg = sum(oracle_times) / n
    oracle_dist = Counter(oracle_times)

    print(f"  mean KL = {oracle_kl:.6f} nats  ← 이론적 천장")
    print(f"  avg time= {oracle_avg:.1f} ms")
    print(f"  분포     S={oracle_dist[100]} M={oracle_dist[250]} L={oracle_dist[400]}")
    print(f"  현재 대비 개선 가능량 = {current_kl - oracle_kl:.6f} nats")

    # ─── 4. 예산 제약 오라클 (현재와 동일한 avg_ms) ───
    print("\n" + "=" * 65)
    print(f"  4. 예산 제약 오라클 (avg ≤ {current_avg:.0f}ms)")
    print("=" * 65)
    bc = oracle_budget_constrained(data, current_avg)
    bc_kl = sum(
        d["kl"][t] for d, t in zip(data, bc["assignments"])
    ) / n
    bc_dist = Counter(bc["assignments"])
    bc_avg = bc["avg_ms"]

    print(f"  mean KL = {bc_kl:.6f} nats  ← 같은 시간 예산에서의 천장")
    print(f"  avg time= {bc_avg:.1f} ms")
    print(f"  분포     S={bc_dist[100]} M={bc_dist[250]} L={bc_dist[400]}")
    print(f"  현재 대비 개선 가능량 = {current_kl - bc_kl:.6f} nats")
    print(f"  uniform 대비 edge  = {uniform_at_current - bc_kl:+.6f} nats")

    # ─── 5. 현재 분류 vs 오라클 일치율 ───
    print("\n" + "=" * 65)
    print("  5. 현재 분류 vs 오라클 일치율")
    print("=" * 65)
    match = sum(1 for c, o in zip(current_assignments, oracle_times) if c == o)
    print(f"  일치    = {match}/{n} ({100*match/n:.1f}%)")

    confusion: dict[str, Counter] = {b: Counter() for b in "SML"}
    for c, o in zip(current_assignments, oracle_times):
        cb = TIME_TO_BUCKET[c]
        ob = TIME_TO_BUCKET[o]
        confusion[cb][ob] += 1

    print(f"\n  현재\\오라클   S       M       L")
    for cb in "SML":
        row = confusion[cb]
        total = sum(row.values())
        if total == 0:
            continue
        parts = [f"{row[ob]:>5}" for ob in "SML"]
        print(f"     {cb}      {'  '.join(parts)}  (n={total})")

    # ─── 6. 시간 민감도 분석 ───
    print("\n" + "=" * 65)
    print("  6. 시간 민감도 (100ms→400ms KL 변화)")
    print("=" * 65)

    sensitivities = []
    for d in data:
        sens = d["kl"][100] - d["kl"][400]
        ev = evaluate_position_complexity(d["fen"])
        sensitivities.append({
            "fen": d["fen"],
            "sensitivity": sens,
            "kl_100": d["kl"][100],
            "kl_400": d["kl"][400],
            "bucket": ev.bucket,
            "reason": ev.features.reason,
            "legal_moves": ev.features.legal_moves,
        })

    sensitivities.sort(key=lambda x: x["sensitivity"], reverse=True)

    pos_sens = [s for s in sensitivities if s["sensitivity"] > 0]
    neg_sens = [s for s in sensitivities if s["sensitivity"] < 0]
    zero_sens = [s for s in sensitivities if s["sensitivity"] == 0]

    print(f"  시간이 도움되는 포지션 (KL↓): {len(pos_sens)}개 ({100*len(pos_sens)/n:.1f}%)")
    print(f"  시간이 해치는 포지션 (KL↑):  {len(neg_sens)}개 ({100*len(neg_sens)/n:.1f}%)")
    print(f"  변화 없음:                    {len(zero_sens)}개")

    avg_pos = sum(s["sensitivity"] for s in pos_sens) / len(pos_sens) if pos_sens else 0
    avg_neg = sum(s["sensitivity"] for s in neg_sens) / len(neg_sens) if neg_sens else 0
    print(f"  도움 평균: {avg_pos:.6f} nats | 해침 평균: {avg_neg:.6f} nats")

    # Top 20 가장 시간에 민감한 포지션
    print(f"\n  ── 시간에 가장 민감한 상위 20개 (100ms→400ms KL 감소 큰 순) ──")
    print(f"  {'kl@100':>10}  {'kl@400':>10}  {'Δ':>10}  {'bkt':>3}  {'reason':<20}  {'lgl':>4}")
    for s in sensitivities[:20]:
        print(
            f"  {s['kl_100']:10.6f}  {s['kl_400']:10.6f}  "
            f"{s['sensitivity']:+10.6f}  {s['bucket']:>3}  {s['reason']:<20}  "
            f"{s['legal_moves']:4d}"
        )

    # 시간 민감도 vs feature 상관관계
    print(f"\n  ── 시간 민감도와 feature 상관관계 (높을수록 시간이 중요) ──")
    sens_vals = [s["sensitivity"] for s in sensitivities]
    for feat_name in ["legal_moves"]:
        feat_vals = [s[feat_name] for s in sensitivities]
        corr = _pearson(sens_vals, feat_vals)
        bar = "█" * int(abs(corr) * 30)
        sign = "+" if corr >= 0 else "-"
        print(f"  {feat_name:>20}  r={corr:+.3f}  {sign}{bar}")

    # ─── 7. 요약 ───
    print("\n" + "=" * 65)
    print("  7. 요약")
    print("=" * 65)
    theoretical_max_edge = uniform_at_current - oracle_kl
    budget_max_edge = uniform_at_current - bc_kl
    current_edge = uniform_at_current - current_kl
    pct_of_ceiling = (current_edge / budget_max_edge * 100) if budget_max_edge > 0 else 0

    print(f"  uniform@250ms mean KL = {uniform_at_current:.6f} nats")
    print(f"  현재 complexity KL    = {current_kl:.6f} nats  (edge {current_edge:+.6f})")
    print(f"  예산 제약 오라클 KL   = {bc_kl:.6f} nats  (edge {budget_max_edge:+.6f})")
    print(f"  무제약 오라클 KL      = {oracle_kl:.6f} nats  (edge {theoretical_max_edge:+.6f})")
    print(f"\n  현재 분류는 예산 제약 천장의 {pct_of_ceiling:.0f}% 를 달성 중")

    return 0


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    sy = sum((yi - my) ** 2 for yi in y) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


if __name__ == "__main__":
    raise SystemExit(main())
