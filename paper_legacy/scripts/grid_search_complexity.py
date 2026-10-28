#!/usr/bin/env python3
"""
오프라인 그리드 서치: complexity scoring 파라미터 최적화.

collect_3time_sweep.py 가 생성한 sweep_3times_raw.csv 와
mixed_workload_pool.csv 의 FEN에서 체스 특징을 추출하여,
API 호출 없이 최적 파라미터 조합을 탐색한다.

2-Phase 탐색:
  Phase 1 — 기존 6개 파라미터 (weights + thresholds)
  Phase 2 — 새 feature (legal_move_count, queen_bonus) + pp_bonus 포함 전체 탐색

Usage:
    python3 scripts/grid_search_complexity.py
    python3 scripts/grid_search_complexity.py --top 30
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import chess
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.bernoulli_kl import bernoulli_kl_nats_array  # noqa: E402

POOL_CSV = PROJECT_ROOT / "mixed_workload_pool.csv"
SWEEP_CSV = PROJECT_ROOT / "sweep_3times_raw.csv"
OUTPUT_JSON = PROJECT_ROOT / "grid_search_results.json"

BUCKET_TIMES = {0: 100, 1: 250, 2: 400}  # S=0, M=1, L=2

MOBILITY_PIECE_TYPES = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
CURRENT_PARAMS = {
    "mob_div": 7,
    "dar_div": 5,
    "int_div": 72,
    "check_bonus": 2,
    "med_th": 6,
    "lar_th": 12,
    "legal_div": 0,
    "queen_bonus": 0,
    "pp_bonus": 15,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction (mirrors position_complexity.py + new features)
# ---------------------------------------------------------------------------

@dataclass
class FenFeatures:
    safe_mobility: int
    pawn_danger_area: int
    interaction_density: int
    is_check: int          # 0 or 1
    is_endgame: int        # 0 or 1
    advanced_passed_pawns: int
    legal_move_count: int
    has_both_queens: int   # 0 or 1


def _pawn_attack_map(board: chess.Board, color: chess.Color) -> chess.Bitboard:
    m = chess.BB_EMPTY
    for sq in chess.scan_forward(board.pieces_mask(chess.PAWN, color)):
        m |= chess.BB_PAWN_ATTACKS[color][sq]
    return m


def _safe_mobility(board: chess.Board, color: chess.Color, opp_pawn_att: chess.Bitboard) -> int:
    own = board.occupied_co[color]
    total = 0
    for pt in MOBILITY_PIECE_TYPES:
        bb = board.pieces_mask(pt, color)
        while bb:
            sq = chess.lsb(bb)
            targets = board.attacks_mask(sq) & ~own
            total += chess.popcount(targets & ~opp_pawn_att)
            bb &= bb - 1
    return total


def _interaction_density(board: chess.Board) -> int:
    acc = 0
    occ = board.occupied
    while occ:
        sq = chess.lsb(occ)
        w = chess.popcount(board.attackers_mask(chess.WHITE, sq))
        b = chess.popcount(board.attackers_mask(chess.BLACK, sq))
        acc += w * b
        occ &= occ - 1
    return acc


def _adjacent_files_bb(file_idx: int) -> chess.Bitboard:
    m = chess.BB_FILES[file_idx]
    if file_idx > 0:
        m |= chess.BB_FILES[file_idx - 1]
    if file_idx < 7:
        m |= chess.BB_FILES[file_idx + 1]
    return m


def _ranks_above(rank: int) -> chess.Bitboard:
    m = chess.BB_EMPTY
    for r in range(rank + 1, 8):
        m |= chess.BB_RANKS[r]
    return m


def _ranks_below(rank: int) -> chess.Bitboard:
    m = chess.BB_EMPTY
    for r in range(0, rank):
        m |= chess.BB_RANKS[r]
    return m


ENDGAME_NNK_THRESHOLD = 6
ADV_PASS_WHITE_RANK_MIN = 5
ADV_PASS_BLACK_RANK_MAX = 2


def _count_advanced_passed_pawns(board: chess.Board) -> int:
    wp = board.pieces_mask(chess.PAWN, chess.WHITE)
    bp = board.pieces_mask(chess.PAWN, chess.BLACK)
    count = 0
    bb = wp
    while bb:
        sq = chess.lsb(bb)
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if (_adjacent_files_bb(f) & _ranks_above(r) & bp) == 0:
            if r >= ADV_PASS_WHITE_RANK_MIN:
                count += 1
        bb &= bb - 1
    bb = bp
    while bb:
        sq = chess.lsb(bb)
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if (_adjacent_files_bb(f) & _ranks_below(r) & wp) == 0:
            if r <= ADV_PASS_BLACK_RANK_MAX:
                count += 1
        bb &= bb - 1
    return count


def extract_features(fen: str) -> FenFeatures:
    board = chess.Board(fen)
    w_pa = _pawn_attack_map(board, chess.WHITE)
    b_pa = _pawn_attack_map(board, chess.BLACK)

    mob_w = _safe_mobility(board, chess.WHITE, b_pa)
    mob_b = _safe_mobility(board, chess.BLACK, w_pa)

    occ = board.occupied
    pawns = board.pieces_mask(chess.PAWN, chess.WHITE) | board.pieces_mask(chess.PAWN, chess.BLACK)
    kings = board.pieces_mask(chess.KING, chess.WHITE) | board.pieces_mask(chess.KING, chess.BLACK)
    nnk = chess.popcount(occ & ~pawns & ~kings)
    is_endgame = int(nnk <= ENDGAME_NNK_THRESHOLD)

    app = _count_advanced_passed_pawns(board) if is_endgame else 0

    wq = bool(board.pieces(chess.QUEEN, chess.WHITE))
    bq = bool(board.pieces(chess.QUEEN, chess.BLACK))

    return FenFeatures(
        safe_mobility=mob_w + mob_b,
        pawn_danger_area=chess.popcount(w_pa) + chess.popcount(b_pa),
        interaction_density=_interaction_density(board),
        is_check=int(board.is_check()),
        is_endgame=is_endgame,
        advanced_passed_pawns=app,
        legal_move_count=sum(1 for _ in board.legal_moves),
        has_both_queens=int(wq and bq),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sweep_data() -> dict[str, dict]:
    """sweep_3times_raw.csv → {fen: {gt, p100, p250, p400}}"""
    data: dict[str, dict] = {}
    with open(SWEEP_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fen = row["fen"].strip()
            data[fen] = {
                "gt": float(row["ground_truth"]),
                "p100": float(row["pred_100ms"]),
                "p250": float(row["pred_250ms"]),
                "p400": float(row["pred_400ms"]),
            }
    return data


def load_pool_fens() -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with open(POOL_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        fi = header.index("fen")
        gi = header.index("new_win_rate")
        for row in reader:
            fen = row[fi].strip()
            try:
                gt = float(row[gi].strip())
            except (ValueError, IndexError):
                continue
            from chess import Board
            try:
                if Board(fen).is_game_over():
                    continue
            except Exception:
                continue
            rows.append((fen, gt))
    return rows


# ---------------------------------------------------------------------------
# Grid search engine
# ---------------------------------------------------------------------------

def build_arrays(
    pool: list[tuple[str, float]],
    sweep: dict[str, dict],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, int]:
    """FEN별 특징 + 예측 오차 배열을 numpy로 구성한다."""

    fens_in_both = [fen for fen, _ in pool if fen in sweep]
    n = len(fens_in_both)
    log.info("sweep & pool 교집합: %d FENs", n)

    feat_keys = [
        "safe_mobility", "pawn_danger_area", "interaction_density",
        "is_check", "is_endgame", "advanced_passed_pawns",
        "legal_move_count", "has_both_queens",
    ]
    arrays: dict[str, list[int]] = {k: [] for k in feat_keys}
    gt_list: list[float] = []
    p100_list: list[float] = []
    p250_list: list[float] = []
    p400_list: list[float] = []

    log.info("특징 추출 중 (%d FENs) ...", n)
    from tqdm import tqdm
    for fen in tqdm(fens_in_both, desc="feature extraction"):
        feat = extract_features(fen)
        for k in feat_keys:
            arrays[k].append(getattr(feat, k))

        s = sweep[fen]
        gt_list.append(s["gt"])
        p100_list.append(s["p100"])
        p250_list.append(s["p250"])
        p400_list.append(s["p400"])

    feat_np = {k: np.array(v, dtype=np.int32) for k, v in arrays.items()}
    gt = np.array(gt_list, dtype=np.float64)
    kl_100 = bernoulli_kl_nats_array(np.array(p100_list, dtype=np.float64), gt)
    kl_250 = bernoulli_kl_nats_array(np.array(p250_list, dtype=np.float64), gt)
    kl_400 = bernoulli_kl_nats_array(np.array(p400_list, dtype=np.float64), gt)

    return feat_np, kl_100, kl_250, kl_400, n


def _interpolate_uniform_kl(
    avg_ms: float,
    u100: float,
    u250: float,
    u400: float,
) -> float:
    """sweep 3-point 기준 uniform 평균 KL 선형 보간."""
    if avg_ms <= 100:
        return u100
    if avg_ms <= 250:
        t = (avg_ms - 100) / 150
        return u100 + t * (u250 - u100)
    if avg_ms <= 400:
        t = (avg_ms - 250) / 150
        return u250 + t * (u400 - u250)
    return u400


def grid_search(
    feat: dict[str, np.ndarray],
    kl_100: np.ndarray,
    kl_250: np.ndarray,
    kl_400: np.ndarray,
    n: int,
    search_space: dict[str, list[int]],
    top_k: int = 20,
    max_avg_ms: float = 280.0,
) -> list[dict]:
    """
    전 파라미터 조합에 대해 평균 베르누이 KL(nats)을 계산한다.
    avg_ms <= max_avg_ms 제약 하에 uniform 대비 edge 기준 상위 결과를 반환.
    """

    mob = feat["safe_mobility"]
    dar = feat["pawn_danger_area"]
    inter = feat["interaction_density"]
    chk = feat["is_check"]
    eg = feat["is_endgame"]
    app = feat["advanced_passed_pawns"]
    lmc = feat["legal_move_count"]
    bq = feat["has_both_queens"]

    endgame_pp = eg * app

    u100 = float(kl_100.mean())
    u250 = float(kl_250.mean())
    u400 = float(kl_400.mean())
    log.info("uniform 평균 KL(nats) — 100ms: %.6f  250ms: %.6f  400ms: %.6f", u100, u250, u400)

    param_names = list(search_space.keys())
    param_values = [search_space[k] for k in param_names]
    total_combos = 1
    for v in param_values:
        total_combos *= len(v)

    log.info("그리드 서치: %d 조합 × %d FENs (max_avg_ms=%.0f)", total_combos, n, max_avg_ms)

    best: list[tuple[float, float, dict]] = []  # (neg_edge, mean_kl, params)
    best_edge_cutoff = -float("inf")

    t0 = time.time()
    evaluated = 0
    skipped_budget = 0

    for combo in product(*param_values):
        params = dict(zip(param_names, combo))

        if params["lar_th"] <= params["med_th"]:
            continue

        md = params["mob_div"]
        dd = params["dar_div"]
        nd = params["int_div"]
        cb = params["check_bonus"]
        mt = params["med_th"]
        lt = params["lar_th"]
        ld = params["legal_div"]
        qb = params["queen_bonus"]
        pb = params["pp_bonus"]

        score = mob // md + dar // dd + inter // nd + cb * chk + pb * endgame_pp
        if ld > 0:
            score = score + lmc // ld
        if qb > 0:
            score = score + qb * bq

        is_s = score < mt
        is_l = score >= lt

        n_s = int(is_s.sum())
        n_l = int(is_l.sum())
        n_m = n - n_s - n_l
        avg_ms = (n_s * 100 + n_m * 250 + n_l * 400) / n

        evaluated += 1

        if avg_ms > max_avg_ms:
            skipped_budget += 1
            continue

        kls = np.where(is_s, kl_100, np.where(is_l, kl_400, kl_250))
        mean_kl = float(kls.mean())

        uniform_kl = _interpolate_uniform_kl(avg_ms, u100, u250, u400)
        edge = uniform_kl - mean_kl

        if len(best) < top_k:
            best.append((-edge, mean_kl, params))
            if len(best) == top_k:
                best.sort(key=lambda x: x[0])
                best_edge_cutoff = -best[-1][0]
        elif edge > best_edge_cutoff:
            best.append((-edge, mean_kl, params))
            best.sort(key=lambda x: x[0])
            best = best[:top_k]
            best_edge_cutoff = -best[-1][0]

    elapsed = time.time() - t0
    log.info(
        "평가 완료: %d 조합 (budget 초과 %d 제외), %.1f초 (%.0f combo/s)",
        evaluated, skipped_budget, elapsed, evaluated / elapsed,
    )

    results = []
    for neg_edge, mean_kl, params in best:
        md = params["mob_div"]
        dd = params["dar_div"]
        nd = params["int_div"]
        cb = params["check_bonus"]
        mt = params["med_th"]
        lt = params["lar_th"]
        ld = params["legal_div"]
        qb = params["queen_bonus"]
        pb = params["pp_bonus"]

        score = mob // md + dar // dd + inter // nd + cb * chk + pb * (eg * app)
        if ld > 0:
            score = score + lmc // ld
        if qb > 0:
            score = score + qb * bq

        n_s = int((score < mt).sum())
        n_m = int(((score >= mt) & (score < lt)).sum())
        n_l = int((score >= lt).sum())
        avg_ms = (n_s * 100 + n_m * 250 + n_l * 400) / n
        uniform_kl = _interpolate_uniform_kl(avg_ms, u100, u250, u400)

        results.append({
            "mean_kl_nats": round(mean_kl, 8),
            "edge": round(-neg_edge, 8),
            "uniform_kl_nats": round(uniform_kl, 8),
            "params": params,
            "buckets": {"S": n_s, "M": n_m, "L": n_l},
            "bucket_pct": {
                "S": round(100 * n_s / n, 1),
                "M": round(100 * n_m / n, 1),
                "L": round(100 * n_l / n, 1),
            },
            "avg_ms": round(avg_ms, 1),
        })

    return results


def compute_current_kl(
    feat: dict[str, np.ndarray],
    kl_100: np.ndarray,
    kl_250: np.ndarray,
    kl_400: np.ndarray,
    n: int,
) -> dict:
    """현재 파라미터로 평균 베르누이 KL 계산 (비교 기준선)."""
    p = CURRENT_PARAMS
    mob = feat["safe_mobility"]
    dar = feat["pawn_danger_area"]
    inter = feat["interaction_density"]
    chk = feat["is_check"]
    eg = feat["is_endgame"]
    app = feat["advanced_passed_pawns"]

    score = (mob // p["mob_div"] + dar // p["dar_div"] + inter // p["int_div"]
             + p["check_bonus"] * chk + p["pp_bonus"] * eg * app)

    is_s = score < p["med_th"]
    is_l = score >= p["lar_th"]
    kls = np.where(is_s, kl_100, np.where(is_l, kl_400, kl_250))
    mean_kl = float(kls.mean())

    n_s = int(is_s.sum())
    n_l = int(is_l.sum())
    n_m = n - n_s - n_l
    avg_ms = (n_s * 100 + n_m * 250 + n_l * 400) / n

    u100 = float(kl_100.mean())
    u250 = float(kl_250.mean())
    u400 = float(kl_400.mean())
    uniform_kl = _interpolate_uniform_kl(avg_ms, u100, u250, u400)

    return {
        "mean_kl_nats": round(mean_kl, 8),
        "edge": round(uniform_kl - mean_kl, 8),
        "uniform_kl_nats": round(uniform_kl, 8),
        "params": dict(CURRENT_PARAMS),
        "buckets": {"S": n_s, "M": n_m, "L": n_l},
        "bucket_pct": {
            "S": round(100 * n_s / n, 1),
            "M": round(100 * n_m / n, 1),
            "L": round(100 * n_l / n, 1),
        },
        "avg_ms": round(avg_ms, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SEARCH_SPACE_FULL = {
    "mob_div":      [5, 6, 7, 8, 9, 10],
    "dar_div":      [3, 4, 5, 6, 7, 8],
    "int_div":      [36, 48, 60, 72, 84, 96],
    "check_bonus":  [0, 1, 2, 3],
    "med_th":       [4, 5, 6, 7, 8, 9],
    "lar_th":       [9, 10, 11, 12, 13, 14, 15, 16],
    "legal_div":    [0, 6, 8, 10, 12, 15, 20],
    "queen_bonus":  [0, 1, 2, 3],
    "pp_bonus":     [0, 5, 10, 15, 20],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Complexity 파라미터 그리드 서치")
    parser.add_argument("--top", type=int, default=20, help="상위 N개 결과 출력")
    parser.add_argument(
        "--max-avg-ms", type=float, default=280.0,
        help="평균 시간 예산 상한 (기본 280ms)",
    )
    parser.add_argument(
        "--min-avg-ms", type=float, default=0.0,
        help="평균 시간 예산 하한",
    )
    parser.add_argument(
        "--sort-by", choices=("edge", "kl"), default="edge",
        help="정렬 기준: edge=uniform 대비 이점 | kl=절대 평균 KL",
    )
    args = parser.parse_args()

    if not SWEEP_CSV.exists():
        log.error("sweep 데이터 없음: %s", SWEEP_CSV)
        log.error("먼저 collect_3time_sweep.py 를 실행하세요.")
        return 1

    sweep = load_sweep_data()
    pool = load_pool_fens()
    feat_np, kl_100, kl_250, kl_400, n = build_arrays(pool, sweep)

    # ── 현재 파라미터 기준선 ──
    current = compute_current_kl(feat_np, kl_100, kl_250, kl_400, n)
    print("\n" + "=" * 70)
    print("  현재 파라미터 기준선")
    print("=" * 70)
    _print_result(current, n)

    # ── 전체 그리드 서치 ──
    print("\n" + "=" * 70)
    print(f"  전체 그리드 서치 (max_avg_ms={args.max_avg_ms}ms)")
    print(f"  정렬 기준: uniform 대비 edge (큰 edge = 더 스마트한 분류)")
    print("=" * 70)

    results = grid_search(
        feat_np, kl_100, kl_250, kl_400, n,
        SEARCH_SPACE_FULL,
        top_k=max(args.top, 200),
        max_avg_ms=args.max_avg_ms,
    )

    if args.min_avg_ms > 0:
        results = [r for r in results if r["avg_ms"] >= args.min_avg_ms]

    if args.sort_by == "kl":
        results.sort(key=lambda r: r["mean_kl_nats"])
    results = results[:args.top]

    sort_label = "절대 평균 KL 기준" if args.sort_by == "kl" else "edge 기준"
    for rank, r in enumerate(results, 1):
        kl_delta = r["mean_kl_nats"] - current["mean_kl_nats"]
        edge_delta = r["edge"] - current["edge"]
        ks = "+" if kl_delta >= 0 else ""
        es = "+" if edge_delta >= 0 else ""
        print(f"\n── #{rank} (KL {ks}{kl_delta:.6f} | edge {es}{edge_delta:.6f}) [{sort_label}] ──")
        _print_result(r, n)

    # ── 결과 저장 ──
    output = {
        "max_avg_ms": args.max_avg_ms,
        "current": current,
        "top_results": results,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info("결과 저장: %s", OUTPUT_JSON)

    return 0


def _print_result(r: dict, n: int) -> None:
    p = r["params"]
    bp = r["bucket_pct"]
    print(f"  mean KL   = {r['mean_kl_nats']:.6f} nats")
    print(f"  uniform   = {r['uniform_kl_nats']:.6f} nats (at {r['avg_ms']:.0f}ms)")
    print(f"  edge      = {r['edge']:.6f} nats  ← uniform보다 이만큼 좋음")
    print(f"  avg time  = {r['avg_ms']:.1f} ms")
    print(f"  buckets   = S {bp['S']}% | M {bp['M']}% | L {bp['L']}%  "
          f"({r['buckets']['S']}/{r['buckets']['M']}/{r['buckets']['L']})")
    parts = []
    parts.append(f"mob÷{p['mob_div']} dar÷{p['dar_div']} int÷{p['int_div']}")
    parts.append(f"chk+{p['check_bonus']}")
    parts.append(f"th={p['med_th']}/{p['lar_th']}")
    if p.get("legal_div", 0) > 0:
        parts.append(f"legal÷{p['legal_div']}")
    else:
        parts.append("legal=off")
    parts.append(f"queen+{p.get('queen_bonus', 0)}")
    parts.append(f"pp×{p.get('pp_bonus', 15)}")
    print(f"  params    = {' | '.join(parts)}")


if __name__ == "__main__":
    raise SystemExit(main())
