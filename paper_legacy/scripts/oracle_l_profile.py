#!/usr/bin/env python3
"""
오라클이 L=750ms로 분류한 69개 FEN의 프로필 분석.

질문: 이 69개가 나머지 1265개와 feature 수준에서 구별 가능한가?
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import chess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.bernoulli_kl import bernoulli_kl_nats
from app.utils.position_complexity import evaluate_position_complexity

SWEEP_CSV = PROJECT_ROOT / "sweep_5times_raw.csv"


def load_data() -> list[dict]:
    rows = []
    with open(SWEEP_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fen = row["fen"].strip()
            gt = float(row["ground_truth"])
            p250 = float(row["pred_250ms"])
            p750 = float(row["pred_750ms"])
            kl_250 = bernoulli_kl_nats(p250, gt)
            kl_750 = bernoulli_kl_nats(p750, gt)
            gain = kl_250 - kl_750

            ev = evaluate_position_complexity(fen)
            board = chess.Board(fen)

            rows.append({
                "fen": fen,
                "kl_250": kl_250,
                "kl_750": kl_750,
                "gain": gain,
                "bucket": ev.bucket,
                "reason": ev.features.reason,
                "legal_moves": ev.features.legal_moves,
                "piece_count": chess.popcount(board.occupied),
                "has_queens": bool(board.pieces(chess.QUEEN, chess.WHITE))
                    and bool(board.pieces(chess.QUEEN, chess.BLACK)),
            })
    return rows


def oracle_budget_l_set(data: list[dict], target_avg: float = 276.0) -> set[int]:
    """예산 제약 오라클에서 L=750으로 배정된 FEN 인덱스."""
    n = len(data)
    gains = sorted(enumerate(data), key=lambda x: x[1]["gain"], reverse=True)
    budget = target_avg * n
    current = 250 * n
    l_set = set()
    for idx, d in gains:
        if d["gain"] <= 0:
            break
        if current + 500 > budget:
            break
        l_set.add(idx)
        current += 500
    return l_set


def main() -> int:
    data = load_data()
    n = len(data)

    l_indices = oracle_budget_l_set(data)
    m_indices = set(range(n)) - l_indices

    l_data = [data[i] for i in sorted(l_indices)]
    m_data = [data[i] for i in sorted(m_indices)]

    print(f"전체 {n}개 | 오라클 L={len(l_data)} | 오라클 M={len(m_data)}\n")

    # ─── 1. Feature 분포 비교 ───
    print("=" * 70)
    print("  1. Feature 평균 비교 (오라클 L vs M)")
    print("=" * 70)

    features = [
        "legal_moves", "piece_count", "gain",
    ]
    bool_features = ["has_queens"]

    print(f"  {'feature':<16} {'오라클 L':>10} {'오라클 M':>10} {'차이':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}")
    for feat in features:
        avg_l = sum(d[feat] for d in l_data) / len(l_data)
        avg_m = sum(d[feat] for d in m_data) / len(m_data)
        diff = avg_l - avg_m
        marker = " ◀" if abs(diff) > 0.5 * avg_m and avg_m != 0 else ""
        print(f"  {feat:<16} {avg_l:>10.2f} {avg_m:>10.2f} {diff:>+10.2f}{marker}")

    for feat in bool_features:
        pct_l = 100 * sum(1 for d in l_data if d[feat]) / len(l_data)
        pct_m = 100 * sum(1 for d in m_data if d[feat]) / len(m_data)
        print(f"  {feat:<16} {pct_l:>9.1f}% {pct_m:>9.1f}% {pct_l - pct_m:>+9.1f}%p")

    # ─── 2. 현재 분류 vs 오라클 ───
    print(f"\n{'=' * 70}")
    print("  2. 현재 분류가 오라클 L을 얼마나 잡는가")
    print("=" * 70)

    current_l_in_oracle_l = sum(1 for d in l_data if d["bucket"] == "L")
    current_m_in_oracle_l = sum(1 for d in l_data if d["bucket"] == "M")
    current_s_in_oracle_l = sum(1 for d in l_data if d["bucket"] == "S")

    print(f"  오라클 L 중 현재 L로 분류: {current_l_in_oracle_l}/{len(l_data)} ({100*current_l_in_oracle_l/len(l_data):.1f}%)")
    print(f"  오라클 L 중 현재 M로 분류: {current_m_in_oracle_l}/{len(l_data)} ({100*current_m_in_oracle_l/len(l_data):.1f}%)")
    print(f"  오라클 L 중 현재 S로 분류: {current_s_in_oracle_l}/{len(l_data)} ({100*current_s_in_oracle_l/len(l_data):.1f}%)")

    # ─── 3. Gain 분포 ───
    print(f"\n{'=' * 70}")
    print("  3. 250ms→750ms KL gain 분포 (전체, nats)")
    print("=" * 70)

    gains = sorted([d["gain"] for d in data], reverse=True)
    brackets = [
        ("> 0.10",        lambda g: g > 0.10),
        ("0.05~0.10",     lambda g: 0.05 < g <= 0.10),
        ("0.02~0.05",     lambda g: 0.02 < g <= 0.05),
        ("0.005~0.02",    lambda g: 0.005 < g <= 0.02),
        ("0~0.005",       lambda g: 0 < g <= 0.005),
        ("< 0 (해로움)",   lambda g: g <= 0),
    ]
    for label, pred in brackets:
        cnt = sum(1 for g in gains if pred(g))
        print(f"  {label:<16} {cnt:>5}개 ({100*cnt/n:5.1f}%)")

    # ─── 4. 오라클 L의 gain 통계 ───
    print(f"\n{'=' * 70}")
    print("  4. 오라클 L의 KL gain 통계 (nats)")
    print("=" * 70)
    l_gains = sorted([d["gain"] for d in l_data], reverse=True)
    print(f"  최소 gain: {l_gains[-1]:.6f}")
    print(f"  최대 gain: {l_gains[0]:.6f}")
    print(f"  평균 gain: {sum(l_gains)/len(l_gains):.6f}")
    print(f"  → 이 {len(l_data)}개 FEN에 750ms를 주면 FEN당 평균 {sum(l_gains)/len(l_gains):.6f} nats KL 감소")

    # ─── 5. 간단한 rule 기반 분류 시뮬레이션 ───
    print(f"\n{'=' * 70}")
    print("  5. 간단한 rule 기반 분류 시뮬레이션")
    print("=" * 70)

    rules = [
        ("현재 분류 (legal≥40 → L)", lambda d: d["legal_moves"] >= 40),
        ("legal_moves ≥ 35", lambda d: d["legal_moves"] >= 35),
        ("legal_moves ≥ 45", lambda d: d["legal_moves"] >= 45),
        ("piece_count ≥ 25", lambda d: d["piece_count"] >= 25),
        ("piece_count ≥ 28", lambda d: d["piece_count"] >= 28),
        ("has_queens & legal ≥ 35", lambda d: d["has_queens"] and d["legal_moves"] >= 35),
    ]

    target_avg = 276.0
    uniform_kl_250 = sum(d["kl_250"] for d in data) / n
    uniform_kl_750 = sum(d["kl_750"] for d in data) / n
    uniform_kl_target = (
        uniform_kl_250
        + (target_avg - 250) * (uniform_kl_750 - uniform_kl_250) / (750 - 250)
    )
    print(f"  시간 예산: avg ≤ {target_avg:.0f}ms (S=50/M=250/L=750)")
    print(f"  uniform@250ms = {uniform_kl_250:.6f} nats | uniform@750ms = {uniform_kl_750:.6f} nats")
    print(f"  uniform@{target_avg:.0f}ms ≈ {uniform_kl_target:.6f} nats (선형 보간)\n")
    print(f"  {'rule':<36} {'n_L':>5} {'KL':>10} {'edge':>10} {'정밀도':>7} {'재현율':>7}")
    print(f"  {'-'*36} {'-'*5} {'-'*10} {'-'*10} {'-'*7} {'-'*7}")

    for label, rule in rules:
        l_candidates = [i for i, d in enumerate(data) if rule(d)]
        n_l = len(l_candidates)
        if n_l == 0:
            continue

        avg_ms = (250 * (n - n_l) + 750 * n_l) / n
        if avg_ms > target_avg + 5:
            status = "(예산초과)"
        else:
            status = ""

        kls = []
        l_set = set(l_candidates)
        for i, d in enumerate(data):
            if i in l_set:
                kls.append(d["kl_750"])
            else:
                kls.append(d["kl_250"])
        mean_kl = sum(kls) / n

        uniform_kl_approx = uniform_kl_250 + (avg_ms - 250) * (uniform_kl_750 - uniform_kl_250) / (750 - 250)
        edge = uniform_kl_approx - mean_kl

        tp = len(l_set & l_indices)
        precision = tp / n_l if n_l > 0 else 0
        recall = tp / len(l_indices) if l_indices else 0

        print(
            f"  {label:<36} {n_l:>5} {mean_kl:>10.6f} {edge:>+10.6f} "
            f"{100*precision:>6.1f}% {100*recall:>6.1f}% {status}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
