#!/usr/bin/env python3
"""
EC2 서버 Stockfish 1-thread NPS 벤치마크.

다양한 searchTimeMs로 isTestMode 요청을 보내 nodes, nps, depth를 수집한 뒤
시간→노드 매핑 테이블을 산출한다.

Usage:
    python3 scripts/benchmark_nps.py
    python3 scripts/benchmark_nps.py --url http://localhost:8000
    python3 scripts/benchmark_nps.py --rounds 5
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import requests

API_BASE_URL = "https://chessez.com"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppppppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "r2qk2r/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 b kq - 5 7",
    "2r2rk1/pp2bppp/2n1bn2/2qpp3/8/1BNP1N2/PPP2PPP/R1BQR1K1 w - - 0 11",
    "r1b2rk1/2q1bppp/p2p1n2/np2p3/3PP3/2N1BN1P/PPB2PP1/R2QR1K1 w - - 0 13",
    "8/5pk1/4p1p1/3pP1Pp/2pP3P/2P2K2/8/8 w - - 0 40",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 35",
]

SEARCH_TIMES_MS = [50, 100, 250, 400, 750, 1000]


def run_benchmark(base_url: str, rounds: int) -> list[dict]:
    endpoint = f"{base_url.rstrip('/')}/api/ai/evaluate"
    rows: list[dict] = []

    total = len(BENCHMARK_FENS) * len(SEARCH_TIMES_MS) * rounds
    done = 0

    for search_time_ms in SEARCH_TIMES_MS:
        for fen in BENCHMARK_FENS:
            for r in range(rounds):
                done += 1
                try:
                    resp = requests.post(
                        endpoint,
                        json={
                            "fen": fen,
                            "searchTimeMs": search_time_ms,
                            "isTestMode": True,
                        },
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    rows.append({
                        "searchTimeMs": search_time_ms,
                        "fen_short": fen[:30],
                        "round": r + 1,
                        "nodes": data.get("nodes"),
                        "nps": data.get("nps"),
                        "depth": data.get("depth"),
                        "type": data.get("type"),
                        "value": data.get("value"),
                    })
                    nodes_str = f"nodes={data.get('nodes', '?'):>10}"
                    nps_str = f"nps={data.get('nps', '?'):>10}"
                    depth_str = f"d={data.get('depth', '?')}"
                    print(
                        f"  [{done:3d}/{total}] {search_time_ms:4d}ms  "
                        f"{nodes_str}  {nps_str}  {depth_str}"
                    )
                except Exception as exc:
                    print(f"  [{done:3d}/{total}] {search_time_ms:4d}ms  FAILED: {exc}")

    return rows


def summarize(rows: list[dict]) -> None:
    by_time: dict[int, list[dict]] = {}
    for row in rows:
        by_time.setdefault(row["searchTimeMs"], []).append(row)

    print("\n" + "=" * 70)
    print("  EC2 Stockfish NPS 벤치마크 결과 (Threads=1, Hash=16)")
    print("=" * 70)
    print(f"  {'ms':>6}  {'avg_nodes':>12}  {'avg_nps':>10}  {'avg_depth':>9}  {'samples':>7}")
    print("-" * 70)

    mapping: dict[int, int] = {}

    for search_time_ms in sorted(by_time.keys()):
        group = by_time[search_time_ms]
        nodes_list = [r["nodes"] for r in group if r["nodes"] is not None]
        nps_list = [r["nps"] for r in group if r["nps"] is not None]
        depth_list = [r["depth"] for r in group if r["depth"] is not None]

        avg_nodes = int(statistics.mean(nodes_list)) if nodes_list else 0
        avg_nps = int(statistics.mean(nps_list)) if nps_list else 0
        avg_depth = round(statistics.mean(depth_list), 1) if depth_list else 0
        n = len(nodes_list)

        mapping[search_time_ms] = avg_nodes
        print(f"  {search_time_ms:>6}  {avg_nodes:>12,}  {avg_nps:>10,}  {avg_depth:>9}  {n:>7}")

    all_nps = [r["nps"] for r in rows if r["nps"] is not None]
    if all_nps:
        global_avg_nps = int(statistics.mean(all_nps))
        global_median_nps = int(statistics.median(all_nps))
        global_stdev_nps = int(statistics.stdev(all_nps)) if len(all_nps) > 1 else 0
        print("-" * 70)
        print(f"  전체 평균 NPS: {global_avg_nps:,}")
        print(f"  전체 중앙값 NPS: {global_median_nps:,}")
        print(f"  전체 표준편차 NPS: {global_stdev_nps:,}")

    print("\n" + "=" * 70)
    print("  시간 → 노드 매핑 (실험 스크립트용)")
    print("=" * 70)
    print("  TIME_TO_NODES = {")
    for ms in sorted(mapping.keys()):
        print(f"      {ms}: {mapping[ms]:,},")
    print("  }")

    out_path = PROJECT_ROOT / "nps_benchmark_result.json"
    result = {
        "global_avg_nps": int(statistics.mean(all_nps)) if all_nps else 0,
        "global_median_nps": int(statistics.median(all_nps)) if all_nps else 0,
        "time_to_avg_nodes": {str(k): v for k, v in mapping.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n  결과 저장: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="EC2 Stockfish 1-thread NPS 벤치마크")
    parser.add_argument("--url", default=API_BASE_URL, help="API base URL")
    parser.add_argument("--rounds", type=int, default=3, help="FEN당 반복 횟수")
    args = parser.parse_args()

    print(f"대상: {args.url}/api/ai/evaluate")
    print(f"FEN {len(BENCHMARK_FENS)}개 × 시간 {len(SEARCH_TIMES_MS)}종 × {args.rounds}라운드")
    print(f"= 총 {len(BENCHMARK_FENS) * len(SEARCH_TIMES_MS) * args.rounds}회 호출\n")

    rows = run_benchmark(args.url, args.rounds)
    if not rows:
        print("결과 없음")
        return 1

    summarize(rows)

    raw_path = PROJECT_ROOT / "nps_benchmark_raw.csv"
    with open(raw_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  raw 데이터: {raw_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
