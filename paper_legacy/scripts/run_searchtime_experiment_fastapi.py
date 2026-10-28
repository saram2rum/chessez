#!/usr/bin/env python3
"""
searchTimeMs 별 평균 베르누이 KL 실험 - 현재 FastAPI 평가 API용.

mixed_workload_pool.csv 의 FEN을 대상으로 `/api/ai/evaluate`를 호출한 뒤,
CSV의 `new_win_rate`(정답, 0~100)와 현재 FastAPI 응답의 예측 승률을 비교해
FEN별 베르누이 KL(nats)을 누적하고, 평균 KL을 결과로 저장한다.

모드 (--mode):
  sweep             기존: SEARCHTIMES_MS 스윕 (체크포인트·출력 파일 기존과 동일)
  complexity        FEN마다 `app.utils.position_complexity` 버킷 기준 ms (100/250/400)
  weighted_uniform  풀 전체 가중 평균 ms 한 번만 사용 (버킷 비율과 동일한 공식)

현재 FastAPI 평가 API는 백 기준 점수를 반환하므로 턴에 따른 부호 보정 없이
Lichess Win% 공식에 바로 넣는다.

API 요청은 항상 프로덕션 베이스 URL(고정)로 보낸다.

Usage:
    python3 scripts/run_searchtime_experiment_fastapi.py --sample 500
    python3 scripts/run_searchtime_experiment_fastapi.py --mode complexity --sample 500
    python3 scripts/run_searchtime_experiment_fastapi.py --mode weighted_uniform --sample 500
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import requests
from chess import Board
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.bernoulli_kl import bernoulli_kl_nats  # noqa: E402
from app.utils.position_complexity import (  # noqa: E402
    EVAL_TIME_BY_BUCKET,
    evaluate_position_complexity,
)

API_BASE_URL = "https://chessez.com"

POOL_CSV = PROJECT_ROOT / "mixed_workload_pool.csv"
OUTPUT_JSON = PROJECT_ROOT / "searchtime_kl_results_fastapi.json"
OUTPUT_JSON_COMPLEXITY = PROJECT_ROOT / "searchtime_kl_results_fastapi_complexity.json"
OUTPUT_JSON_WEIGHTED_UNIFORM = PROJECT_ROOT / "searchtime_kl_results_fastapi_weighted_uniform.json"
OUTPUT_RAW_DIR = PROJECT_ROOT / "searchtime_raw_fastapi"
CHECKPOINT_PATH = PROJECT_ROOT / "searchtime_experiment_checkpoint_fastapi_kl.json"
CHECKPOINT_COMPLEXITY = PROJECT_ROOT / "searchtime_experiment_checkpoint_fastapi_complexity_kl.json"
CHECKPOINT_WEIGHTED_UNIFORM = PROJECT_ROOT / "searchtime_experiment_checkpoint_fastapi_weighted_uniform_kl.json"
RAW_CSV_COMPLEXITY = OUTPUT_RAW_DIR / "searchtime_complexity_policy_raw_kl.csv"
RAW_CSV_WEIGHTED_UNIFORM = OUTPUT_RAW_DIR / "searchtime_weighted_uniform_raw_kl.csv"

SEARCHTIMES_MS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
REQUEST_DELAY_SEC = 0.1
LICHESS_CP_SCALE = 0.00368208

EC2_NPS_1THREAD = 360_000


def ms_to_nodes(ms: int) -> int:
    """EC2 1-thread NPS 기준 ms → nodes 변환. benchmark_nps.py 측정값 기반."""
    return max(1, int(round(ms * EC2_NPS_1THREAD / 1000)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def lichess_cp_to_win_percent(cp: float) -> float:
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-LICHESS_CP_SCALE * float(cp))) - 1.0)


def eval_to_win_percent_lichess(data: dict) -> float | None:
    if not data or "type" not in data or "value" not in data:
        return None

    eval_type = data["type"]
    raw_value = data["value"]
    value = int(raw_value) if isinstance(raw_value, (int, float)) else raw_value

    if eval_type == "cp":
        return lichess_cp_to_win_percent(float(value))
    if eval_type == "mate":
        return 100.0 if int(value) > 0 else 0.0
    return None


def _resolve_gt_column(header: list[str]) -> int:
    header = [h.strip() for h in header]
    if "new_win_rate" in header:
        return header.index("new_win_rate")
    if len(header) >= 4:
        log.warning(
            "헤더에 new_win_rate 없음 → 4번째 열을 정답으로 사용합니다. "
            "권장 헤더: fen,ground_truth,eval_value,new_win_rate"
        )
        return 3
    raise ValueError(
        "mixed_workload_pool.csv에 new_win_rate 열이 없습니다.\n"
        "헤더 예: fen,ground_truth,eval_value,new_win_rate"
    )


def _resolve_fen_column(header: list[str]) -> int:
    if "fen" in header:
        return header.index("fen")
    return 0


def weighted_baseline_move_time_ms_for_fens(fens: list[str]) -> float:
    """analyze_complexity_buckets.build_report 과 동일: 버킷별 건수 가중 평균 ms."""
    counts = {"S": 0, "M": 0, "L": 0}
    for fen in fens:
        bucket = evaluate_position_complexity(fen).bucket
        counts[bucket] += 1
    total = len(fens)
    if not total:
        return 0.0
    weighted = sum(counts[b] * EVAL_TIME_BY_BUCKET[b] for b in counts)
    return weighted / total


def load_pool(sample: int | None, exclude_game_over: bool = True) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    excluded = 0

    with open(POOL_CSV, encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        header = [h.strip() for h in header]
        fen_index = _resolve_fen_column(header)
        gt_index = _resolve_gt_column(header)

        for row in reader:
            if not row or len(row) <= max(fen_index, gt_index):
                continue
            fen = row[fen_index].strip()
            if not fen:
                continue
            try:
                gt = float(row[gt_index].strip())
            except (ValueError, IndexError):
                continue

            if exclude_game_over:
                try:
                    if Board(fen).is_game_over():
                        excluded += 1
                        continue
                except Exception:
                    excluded += 1
                    continue

            rows.append((fen, gt))

    if excluded:
        log.info("게임 종료 국면 %d개 제외", excluded)
    if sample:
        random.seed(42)
        rows = random.sample(rows, min(sample, len(rows)))
    return rows


def load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        return {"kls": {}, "completed_st": []}
    try:
        with open(CHECKPOINT_PATH, encoding="utf-8") as file:
            raw = json.load(file)
    except Exception:
        return {"kls": {}, "completed_st": []}
    normalized_kls = {
        int(search_time_ms): [float(kl) for kl in kls]
        for search_time_ms, kls in raw.get("kls", {}).items()
    }
    normalized_completed = [int(search_time_ms) for search_time_ms in raw.get("completed_st", [])]
    return {"kls": normalized_kls, "completed_st": normalized_completed}


def save_checkpoint(data: dict) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_single_pass_checkpoint(path: Path) -> list[float]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as file:
            raw = json.load(file)
    except Exception:
        return []
    kls = raw.get("kls", [])
    return [float(x) for x in kls]


def save_single_pass_checkpoint(path: Path, kls: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"kls": kls}, file, indent=2)


def _api_search_time_ms(value: float) -> int:
    """요청 바디용: FastAPI AiEvaluateRequest ge=50., 정수로 반올림."""
    return max(50, min(30_000, int(round(float(value)))))


def _weighted_uniform_ms_from_baseline(baseline: float) -> int:
    """가중 평균 baseline만 올림(동일 ms 대비 조금 더 까다로운 Stockfish). complexity 버킷은 정수라 별도 처리 없음."""
    return max(50, min(30_000, int(math.ceil(float(baseline)))))


def run_experiment(base_url: str, pool: list[tuple[str, float]], resume: bool = True) -> dict:
    endpoint = f"{base_url.rstrip('/')}/api/ai/evaluate"
    OUTPUT_RAW_DIR.mkdir(exist_ok=True)

    checkpoint = load_checkpoint() if resume else {"kls": {}, "completed_st": []}
    kls_by_st: dict[int, list[float]] = checkpoint.get("kls", {})
    completed_st: list[int] = list(checkpoint.get("completed_st", []))
    for search_time_ms in SEARCHTIMES_MS:
        if search_time_ms not in kls_by_st:
            kls_by_st[search_time_ms] = []

    total_calls = len(pool) * len(SEARCHTIMES_MS)
    done_count = sum(len(kls_by_st.get(search_time_ms, [])) for search_time_ms in SEARCHTIMES_MS)

    raw_fieldnames = [
        "fen",
        "pred_win_pct_0_100",
        "ground_truth_win_pct_0_100",
        "kl_nats",
    ]

    try:
        with logging_redirect_tqdm(), tqdm(total=total_calls, desc="searchtime 실험", unit="call") as progress:
            progress.update(done_count)
            for search_time_ms in SEARCHTIMES_MS:
                if search_time_ms in completed_st:
                    continue

                output_path = OUTPUT_RAW_DIR / f"searchtime_{search_time_ms}ms_raw_kl.csv"
                exists = output_path.exists()
                file = open(output_path, "a", encoding="utf-8", newline="")
                writer = csv.DictWriter(file, fieldnames=raw_fieldnames)
                if not exists:
                    writer.writeheader()

                try:
                    nodes = ms_to_nodes(search_time_ms)
                    start_index = len(kls_by_st.get(search_time_ms, []))
                    for index, (fen, gt) in enumerate(pool):
                        if index < start_index:
                            continue

                        try:
                            response = requests.post(
                                endpoint,
                                json={"fen": fen, "searchNodes": nodes, "isTestMode": True},
                                timeout=30,
                            )
                            response.raise_for_status()
                            data = response.json()
                            pred = eval_to_win_percent_lichess(data)
                            if pred is not None:
                                kl = bernoulli_kl_nats(pred, gt)
                                kls_by_st.setdefault(search_time_ms, []).append(kl)
                                writer.writerow(
                                    {
                                        "fen": fen,
                                        "pred_win_pct_0_100": f"{pred:.6f}",
                                        "ground_truth_win_pct_0_100": f"{gt:.6f}",
                                        "kl_nats": f"{kl:.6f}",
                                    }
                                )
                        except Exception as exc:
                            log.warning("FEN %s st=%d 실패: %s", fen[:40], search_time_ms, exc)

                        progress.update(1)
                        if REQUEST_DELAY_SEC:
                            time.sleep(REQUEST_DELAY_SEC)
                        if (index + 1) % 100 == 0:
                            file.flush()
                            save_checkpoint({"kls": kls_by_st, "completed_st": completed_st})
                finally:
                    file.close()

                completed_st.append(search_time_ms)
                save_checkpoint({"kls": kls_by_st, "completed_st": completed_st})
    except Exception:
        save_checkpoint({"kls": kls_by_st, "completed_st": completed_st})
        raise

    results = {}
    for search_time_ms in SEARCHTIMES_MS:
        kls = kls_by_st.get(search_time_ms, [])
        mean_kl = sum(kls) / len(kls) if kls else float("nan")
        results[search_time_ms] = {"mean_kl_nats": mean_kl, "n": len(kls)}
    return results


def run_per_fen_search_time_experiment(
    base_url: str,
    pool: list[tuple[str, float]],
    *,
    search_time_ms_for_fen: list[int],
    checkpoint_path: Path,
    output_raw_path: Path,
    resume: bool,
    progress_desc: str,
) -> dict[str, float | int]:
    if len(search_time_ms_for_fen) != len(pool):
        raise ValueError("search_time_ms_for_fen 길이가 pool과 같아야 합니다.")
    endpoint = f"{base_url.rstrip('/')}/api/ai/evaluate"
    OUTPUT_RAW_DIR.mkdir(exist_ok=True)

    kls: list[float] = [] if not resume else load_single_pass_checkpoint(checkpoint_path)
    start_index = len(kls)

    raw_fieldnames = [
        "fen",
        "searchTimeMs",
        "searchNodes",
        "pred_win_pct_0_100",
        "ground_truth_win_pct_0_100",
        "kl_nats",
    ]
    total = len(pool)
    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if resume and start_index > 0 else "w"
    file_exists_before = output_raw_path.exists() and file_mode == "a"

    with open(output_raw_path, file_mode, encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=raw_fieldnames)
        if not file_exists_before or file_mode == "w":
            writer.writeheader()

        with logging_redirect_tqdm():
            with tqdm(total=total, desc=progress_desc, unit="call", initial=0) as progress:
                progress.update(start_index)
                for index in range(start_index, total):
                    fen, gt = pool[index]
                    st = search_time_ms_for_fen[index]
                    nodes = ms_to_nodes(st)
                    try:
                        response = requests.post(
                            endpoint,
                            json={"fen": fen, "searchNodes": nodes, "isTestMode": True},
                            timeout=30,
                        )
                        response.raise_for_status()
                        data = response.json()
                        pred = eval_to_win_percent_lichess(data)
                        if pred is not None:
                            kl = bernoulli_kl_nats(pred, gt)
                            kls.append(kl)
                            writer.writerow(
                                {
                                    "fen": fen,
                                    "searchTimeMs": st,
                                    "searchNodes": nodes,
                                    "pred_win_pct_0_100": f"{pred:.6f}",
                                    "ground_truth_win_pct_0_100": f"{gt:.6f}",
                                    "kl_nats": f"{kl:.6f}",
                                }
                            )
                    except Exception as exc:
                        log.warning("FEN %s nodes=%d 실패: %s", fen[:40], nodes, exc)

                    progress.update(1)
                    if REQUEST_DELAY_SEC:
                        time.sleep(REQUEST_DELAY_SEC)
                    if (index + 1) % 100 == 0:
                        file.flush()
                        save_single_pass_checkpoint(checkpoint_path, kls)

    save_single_pass_checkpoint(checkpoint_path, kls)
    mean_kl = sum(kls) / len(kls) if kls else float("nan")
    return {"mean_kl_nats": mean_kl, "n": len(kls)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="현재 FastAPI `/api/ai/evaluate` 기준 searchtime 별 평균 베르누이 KL 측정",
    )
    parser.add_argument(
        "--mode",
        choices=("sweep", "complexity", "weighted_uniform"),
        default="sweep",
        help="sweep: 고정 ms 스윕 | complexity: FEN별 버킷 ms | weighted_uniform: 풀 가중 평균 ms",
    )
    parser.add_argument("--sample", type=int, default=None, help="FEN 샘플 수")
    parser.add_argument("--no-resume", action="store_true", help="체크포인트 무시")
    parser.add_argument(
        "--uniform-ms",
        type=int,
        default=None,
        help="weighted_uniform 모드에서만 사용: 가중 평균 대신 이 ms 고정 (API 최소 50)",
    )
    args = parser.parse_args()

    if not POOL_CSV.exists():
        log.error("mixed_workload_pool.csv 없음: %s", POOL_CSV)
        return 1

    try:
        pool = load_pool(args.sample)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    log.info("evaluate API: %s/api/ai/evaluate (testMode, nodes 기반)", API_BASE_URL.rstrip("/"))
    log.info("NPS=%d → ms→nodes 변환 적용", EC2_NPS_1THREAD)

    if args.mode == "sweep":
        log.info(
            "FEN %d개 로드 | 정답: new_win_rate (0~100) | 예측: 현재 FastAPI Win%% | searchtime %d종",
            len(pool),
            len(SEARCHTIMES_MS),
        )

        results = run_experiment(API_BASE_URL, pool, resume=not args.no_resume)

        with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)

        log.info("KL 결과(JSON): %s", OUTPUT_JSON)
        log.info("raw CSV: %s", OUTPUT_RAW_DIR)

        print("\n=== searchtime별 평균 베르누이 KL (현재 FastAPI API 기준, 단위: nats) ===")
        print("    예측: 현재 API의 백 기준 eval -> Lichess Win%")
        print("    정답: CSV new_win_rate\n")
        for search_time_ms in sorted(results.keys(), reverse=True):
            result = results[search_time_ms]
            mean_kl = result["mean_kl_nats"]
            kl_text = f"{mean_kl:.6f}" if math.isfinite(mean_kl) else "nan"
            print(f"  {search_time_ms:4d} ms: mean KL = {kl_text} nats  (n={result['n']})")
        return 0

    resume = not args.no_resume
    if args.mode == "complexity":
        per_fen_ms = [evaluate_position_complexity(fen).move_time_ms for fen, _gt in pool]
        if not resume:
            if RAW_CSV_COMPLEXITY.exists():
                RAW_CSV_COMPLEXITY.unlink()
            if CHECKPOINT_COMPLEXITY.exists():
                CHECKPOINT_COMPLEXITY.unlink()
        nodes_by_bucket = {b: ms_to_nodes(ms) for b, ms in EVAL_TIME_BY_BUCKET.items()}
        log.info(
            "mode=complexity | FEN %d개 | 버킷 ms %s → nodes %s",
            len(pool),
            dict(EVAL_TIME_BY_BUCKET),
            nodes_by_bucket,
        )
        out = run_per_fen_search_time_experiment(
            API_BASE_URL,
            pool,
            search_time_ms_for_fen=per_fen_ms,
            checkpoint_path=CHECKPOINT_COMPLEXITY,
            output_raw_path=RAW_CSV_COMPLEXITY,
            resume=resume,
            progress_desc="complexity policy",
        )
        payload = {
            "mode": "complexity",
            "mean_kl_nats": out["mean_kl_nats"],
            "n": out["n"],
            "moveTimeByBucket": dict(EVAL_TIME_BY_BUCKET),
        }
        with open(OUTPUT_JSON_COMPLEXITY, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        log.info("KL 결과(JSON): %s", OUTPUT_JSON_COMPLEXITY)
        log.info("raw CSV: %s", RAW_CSV_COMPLEXITY)
        mean_kl = out["mean_kl_nats"]
        kl_text = f"{mean_kl:.6f}" if math.isfinite(mean_kl) else "nan"
        print(f"\n=== complexity policy mean KL = {kl_text} nats (n={out['n']}) ===")
        return 0

    # weighted_uniform
    fens_only = [fen for fen, _gt in pool]
    baseline = weighted_baseline_move_time_ms_for_fens(fens_only)
    if args.uniform_ms is not None:
        uniform_ms = _api_search_time_ms(args.uniform_ms)
    else:
        uniform_ms = _weighted_uniform_ms_from_baseline(baseline)
    if not resume:
        if RAW_CSV_WEIGHTED_UNIFORM.exists():
            RAW_CSV_WEIGHTED_UNIFORM.unlink()
        if CHECKPOINT_WEIGHTED_UNIFORM.exists():
            CHECKPOINT_WEIGHTED_UNIFORM.unlink()
    uniform_nodes = ms_to_nodes(uniform_ms)
    log.info(
        "mode=weighted_uniform | FEN %d개 | baseline=%.2f ms → %d ms → %d nodes",
        len(pool),
        baseline,
        uniform_ms,
        uniform_nodes,
    )
    per_fen_uniform = [uniform_ms for _ in pool]
    out = run_per_fen_search_time_experiment(
        API_BASE_URL,
        pool,
        search_time_ms_for_fen=per_fen_uniform,
        checkpoint_path=CHECKPOINT_WEIGHTED_UNIFORM,
        output_raw_path=RAW_CSV_WEIGHTED_UNIFORM,
        resume=resume,
        progress_desc="weighted uniform ms",
    )
    payload = {
        "mode": "weighted_uniform",
        "weightedBaselineMoveTimeMs": baseline,
        "searchTimeMsUsed": uniform_ms,
        "uniformMsOverride": args.uniform_ms,
        "mean_kl_nats": out["mean_kl_nats"],
        "n": out["n"],
        "moveTimeByBucket": dict(EVAL_TIME_BY_BUCKET),
    }
    with open(OUTPUT_JSON_WEIGHTED_UNIFORM, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    log.info("KL 결과(JSON): %s", OUTPUT_JSON_WEIGHTED_UNIFORM)
    log.info("raw CSV: %s", RAW_CSV_WEIGHTED_UNIFORM)
    mean_kl = out["mean_kl_nats"]
    kl_text = f"{mean_kl:.6f}" if math.isfinite(mean_kl) else "nan"
    print(
        f"\n=== weighted uniform (baseline {baseline:.2f} ms → {uniform_ms} ms) "
        f"mean KL = {kl_text} nats (n={out['n']}) ==="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
