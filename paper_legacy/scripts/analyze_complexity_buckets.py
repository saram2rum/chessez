#!/usr/bin/env python3
"""
Classify all FEN rows in mixed_workload_pool.csv into S/M/L buckets and report:

- bucket counts
- weighted average baseline move time
- a few sample positions per bucket

This is meant to be the quickest way to iterate on `app/utils/position_complexity.py`.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.position_complexity import (
    EVAL_TIME_BY_BUCKET,
    evaluate_position_complexity,
    get_complexity_config_snapshot,
)


POOL_CSV = PROJECT_ROOT / "mixed_workload_pool.csv"
OUTPUT_DIR = PROJECT_ROOT / "complexity_bucket_reports"
LATEST_OUTPUT_JSON = OUTPUT_DIR / "complexity_bucket_report_latest.json"


def resolve_fen_column(header: list[str]) -> int:
    header = [column.strip() for column in header]
    if "fen" in header:
        return header.index("fen")
    return 0


def load_fens(sample: int | None) -> list[str]:
    fens: list[str] = []
    with open(POOL_CSV, encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        fen_index = resolve_fen_column(header)
        for row in reader:
            if not row or len(row) <= fen_index:
                continue
            fen = row[fen_index].strip()
            if fen:
                fens.append(fen)

    if sample:
        random.seed(42)
        fens = random.sample(fens, min(sample, len(fens)))
    return fens


def build_report(fens: list[str], example_limit: int, sample: int | None) -> dict:
    counts = {"S": 0, "M": 0, "L": 0}
    examples: dict[str, list[dict]] = defaultdict(list)
    reason_counts: dict[str, int] = defaultdict(int)
    reason_counts_by_bucket: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for fen in fens:
        evaluation = evaluate_position_complexity(fen)
        bucket = evaluation.bucket
        reason = evaluation.features.reason or "unknown"
        counts[bucket] += 1
        reason_counts[reason] += 1
        reason_counts_by_bucket[bucket][reason] += 1

        if example_limit > 0 and len(examples[bucket]) < example_limit:
            examples[bucket].append(
                {
                    "fen": fen,
                    "reason": reason,
                    "moveTimeMs": evaluation.move_time_ms,
                    "features": evaluation.features.__dict__,
                }
            )

    total_fens = len(fens)
    weighted_sum = sum(counts[bucket] * EVAL_TIME_BY_BUCKET[bucket] for bucket in counts)
    baseline_move_time = weighted_sum / total_fens if total_fens else 0.0

    report = {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "sourceCsv": str(POOL_CSV),
        "analysisConfig": {
            "sample": sample,
            "exampleLimit": example_limit,
        },
        "heuristicConfig": get_complexity_config_snapshot(),
        "totalFens": total_fens,
        "bucketCounts": counts,
        "bucketRatios": {
            bucket: (counts[bucket] / total_fens if total_fens else 0.0)
            for bucket in counts
        },
        "weightedBaselineMoveTimeMs": baseline_move_time,
        "reasonCounts": dict(reason_counts),
        "reasonCountsByBucket": {
            bucket: dict(bucket_reasons)
            for bucket, bucket_reasons in reason_counts_by_bucket.items()
        },
        "moveTimeByBucket": EVAL_TIME_BY_BUCKET,
    }
    if example_limit > 0:
        report["examples"] = examples
    return report


def resolve_output_path(output: str | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"complexity_bucket_report_{timestamp}.json"


def write_report(report: dict, output_path: Path) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LATEST_OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return output_path, LATEST_OUTPUT_JSON


def print_summary(report: dict) -> None:
    print("=== Complexity Bucket Summary ===")
    print(f"Total FENs: {report['totalFens']}")
    print(
        "Weighted baseline move time (ms): "
        f"{report['weightedBaselineMoveTimeMs']:.2f}"
    )
    print("")

    for bucket in ("S", "M", "L"):
        count = report["bucketCounts"].get(bucket, 0)
        ratio = report["bucketRatios"].get(bucket, 0.0) * 100
        move_time = report["moveTimeByBucket"][bucket]
        print(f"{bucket}: count={count}, ratio={ratio:.2f}%, moveTimeMs={move_time}")

    print("")
    print("Reason distribution:")
    for reason, count in sorted(report.get("reasonCounts", {}).items()):
        print(f"  {reason}: {count}")

    examples = report.get("examples")
    if examples:
        print("")
        print("Example positions per bucket:")
        for bucket in ("S", "M", "L"):
            print(f"[{bucket}]")
            for example in examples.get(bucket, []):
                print(
                    f"  reason={example['reason']} moveTimeMs={example['moveTimeMs']} "
                    f"fen={example['fen']}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze S/M/L bucket distribution and weighted baseline move time",
    )
    parser.add_argument("--sample", type=int, default=None, help="Optional sample size")
    parser.add_argument("--examples", type=int, default=0, help="Examples shown per bucket")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path. If omitted, saves a timestamped report.",
    )
    args = parser.parse_args()

    if not POOL_CSV.exists():
        print(f"mixed_workload_pool.csv not found: {POOL_CSV}", file=sys.stderr)
        return 1

    fens = load_fens(args.sample)
    report = build_report(
        fens,
        example_limit=max(0, args.examples),
        sample=args.sample,
    )
    print_summary(report)

    output_path = resolve_output_path(args.output)
    saved_path, latest_path = write_report(report, output_path)
    print("")
    print(f"Saved versioned JSON report: {saved_path}")
    print(f"Updated latest JSON report: {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
