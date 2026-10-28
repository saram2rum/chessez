#!/usr/bin/env python3
"""
확장 오라클 분석: S=50 / M=250 / L=750 시간 배분의 천장 분석.

sweep_5times_raw.csv (50/100/250/400/750ms) 를 활용하여
다양한 S/M/L 시간 조합의 천장과 현재 분류 대비 개선 여지를 베르누이 KL(nats) 기준으로 비교한다.

Usage:
    python3 scripts/oracle_analysis_wide.py
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
from app.utils.position_complexity import evaluate_position_complexity

SWEEP_CSV = PROJECT_ROOT / "sweep_5times_raw.csv"

ALL_TIMES = [50, 100, 250, 400, 750]

CONFIGS = [
    {"label": "현재 (S=100/M=250/L=400)", "S": 100, "M": 250, "L": 400},
    {"label": "제안 (S=50/M=250/L=750)",   "S": 50,  "M": 250, "L": 750},
    {"label": "넓음 (S=50/M=100/L=750)",   "S": 50,  "M": 100, "L": 750},
    {"label": "극단 (S=50/M=250/L=400)",   "S": 50,  "M": 250, "L": 400},
]


def load_sweep() -> list[dict]:
    rows = []
    with open(SWEEP_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fen = row["fen"].strip()
            gt = float(row["ground_truth"])
            preds = {
                50: float(row["pred_50ms"]),
                100: float(row["pred_100ms"]),
                250: float(row["pred_250ms"]),
                400: float(row["pred_400ms"]),
                750: float(row["pred_750ms"]),
            }
            kl = {t: bernoulli_kl_nats(preds[t], gt) for t in ALL_TIMES}
            rows.append({"fen": fen, "gt": gt, "kl": kl})
    return rows


def oracle_unconstrained(data: list[dict], times: list[int]) -> dict:
    """무제약 오라클: 각 FEN에 가장 KL 낮은 시간 배정."""
    n = len(data)
    assignments = [min(times, key=lambda t: (d["kl"][t], t)) for d in data]
    mean_kl = sum(d["kl"][t] for d, t in zip(data, assignments)) / n
    avg_ms = sum(assignments) / n
    dist = Counter(assignments)
    return {"mean_kl": mean_kl, "avg_ms": avg_ms, "dist": dist, "assignments": assignments}


def oracle_budget_constrained(
    data: list[dict], s_time: int, m_time: int, l_time: int, target_avg: float,
) -> dict:
    """
    예산 제약 오라클 (탐욕적).
    모든 FEN을 M에서 시작 → 이득이 큰 순으로 L로 승격, 비용을 S로 충당.
    """
    n = len(data)
    assignments = [m_time] * n

    gain_to_l = [(d["kl"][m_time] - d["kl"][l_time], i) for i, d in enumerate(data)]
    cost_to_s = [(d["kl"][s_time] - d["kl"][m_time], i) for i, d in enumerate(data)]

    gain_to_l.sort(reverse=True)
    cost_to_s.sort()

    current_total = m_time * n
    target_total = target_avg * n

    used = set()

    si = 0
    while current_total > target_total and si < n:
        _, idx = cost_to_s[si]
        assignments[idx] = s_time
        current_total -= (m_time - s_time)
        used.add(idx)
        si += 1

    li = 0
    while li < n:
        gain, idx = gain_to_l[li]
        if idx in used or gain <= 0:
            li += 1
            continue
        if current_total + (l_time - m_time) > target_total:
            break
        assignments[idx] = l_time
        current_total += (l_time - m_time)
        li += 1

    mean_kl = sum(d["kl"][t] for d, t in zip(data, assignments)) / n
    avg_ms = sum(assignments) / n
    dist = Counter(assignments)
    return {"mean_kl": mean_kl, "avg_ms": avg_ms, "dist": dist, "assignments": assignments}


def current_policy_kl(data: list[dict], times: dict[str, int]) -> dict:
    """현재 complexity 분류를 사용하되, S/M/L 시간은 인자로 받음."""
    n = len(data)
    assignments = []
    for d in data:
        ev = evaluate_position_complexity(d["fen"])
        t = times[ev.bucket]
        assignments.append(t)

    mean_kl = sum(d["kl"][t] for d, t in zip(data, assignments)) / n
    avg_ms = sum(assignments) / n
    dist = Counter(assignments)
    return {"mean_kl": mean_kl, "avg_ms": avg_ms, "dist": dist}


def main() -> int:
    data = load_sweep()
    n = len(data)
    print(f"FEN {n}개 로드\n")

    # ─── Uniform baselines ───
    print("=" * 70)
    print("  Uniform baselines (mean KL, nats)")
    print("=" * 70)
    uniform_kl = {}
    for t in ALL_TIMES:
        mean_kl = sum(d["kl"][t] for d in data) / n
        uniform_kl[t] = mean_kl
        print(f"  {t:4d}ms  mean KL = {mean_kl:.6f} nats")

    for cfg in CONFIGS:
        label = cfg["label"]
        s, m, l = cfg["S"], cfg["M"], cfg["L"]

        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}")

        # 현재 분류 적용
        cur = current_policy_kl(data, {"S": s, "M": m, "L": l})
        uniform_at_avg = _interp_uniform(cur["avg_ms"], uniform_kl)

        print(f"\n  ── 현재 분류 적용 (S={s}/M={m}/L={l}) ──")
        print(f"  mean KL   = {cur['mean_kl']:.6f} nats")
        print(f"  avg time  = {cur['avg_ms']:.1f} ms")
        print(f"  uniform@{cur['avg_ms']:.0f}ms ≈ {uniform_at_avg:.6f} nats")
        print(f"  edge      = {uniform_at_avg - cur['mean_kl']:+.6f} nats")
        _print_dist(cur["dist"], n)

        # 무제약 오라클
        orc = oracle_unconstrained(data, [s, m, l])
        print(f"\n  ── 무제약 오라클 ──")
        print(f"  mean KL   = {orc['mean_kl']:.6f} nats  ← 천장")
        print(f"  avg time  = {orc['avg_ms']:.1f} ms")
        _print_dist(orc["dist"], n)

        # 예산 제약 오라클 (현재 avg와 동일)
        bc = oracle_budget_constrained(data, s, m, l, cur["avg_ms"])
        uniform_at_bc = _interp_uniform(bc["avg_ms"], uniform_kl)
        print(f"\n  ── 예산 제약 오라클 (avg ≤ {cur['avg_ms']:.0f}ms) ──")
        print(f"  mean KL   = {bc['mean_kl']:.6f} nats")
        print(f"  avg time  = {bc['avg_ms']:.1f} ms")
        print(f"  edge      = {uniform_at_bc - bc['mean_kl']:+.6f} nats")
        _print_dist(bc["dist"], n)

        room = cur["mean_kl"] - bc["mean_kl"]
        print(f"\n  ▶ 개선 여지 = {room:.6f} nats (현재→예산 제약 천장)")

    # ─── 최종 비교표 ───
    print(f"\n{'=' * 70}")
    print("  최종 비교 요약 (mean KL, nats)")
    print(f"{'=' * 70}")
    print(f"  {'설정':<28} {'현재 KL':>11} {'천장 KL':>11} {'개선 여지':>11} {'천장 edge':>12}")
    for cfg in CONFIGS:
        s, m, l = cfg["S"], cfg["M"], cfg["L"]
        cur = current_policy_kl(data, {"S": s, "M": m, "L": l})
        bc = oracle_budget_constrained(data, s, m, l, cur["avg_ms"])
        uniform_at_bc = _interp_uniform(bc["avg_ms"], uniform_kl)
        room = cur["mean_kl"] - bc["mean_kl"]
        edge = uniform_at_bc - bc["mean_kl"]
        print(f"  {cfg['label']:<28} {cur['mean_kl']:>11.6f} {bc['mean_kl']:>11.6f} {room:>11.6f} {edge:>+12.6f}")

    return 0


def _interp_uniform(ms: float, uniform_kl: dict[int, float]) -> float:
    times = sorted(uniform_kl.keys())
    if ms <= times[0]:
        return uniform_kl[times[0]]
    if ms >= times[-1]:
        return uniform_kl[times[-1]]
    for i in range(len(times) - 1):
        if times[i] <= ms <= times[i + 1]:
            t = (ms - times[i]) / (times[i + 1] - times[i])
            return uniform_kl[times[i]] + t * (uniform_kl[times[i + 1]] - uniform_kl[times[i]])
    return uniform_kl[times[-1]]


def _print_dist(dist: Counter, n: int) -> None:
    parts = []
    for t in sorted(dist.keys()):
        pct = 100 * dist[t] / n
        parts.append(f"{t}ms={dist[t]}({pct:.1f}%)")
    print(f"  분포     {' | '.join(parts)}")


if __name__ == "__main__":
    raise SystemExit(main())
