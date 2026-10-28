from dataclasses import dataclass

import chess


# ---------------------------------------------------------------------------
# Bucket → search time / nodes
# ---------------------------------------------------------------------------

EVAL_TIME_BY_BUCKET = {
    "S": 100,
    "M": 250,
    "L": 400,
}

_NODES_PER_MS = 360

EVAL_NODES_BY_BUCKET = {
    bucket: ms * _NODES_PER_MS
    for bucket, ms in EVAL_TIME_BY_BUCKET.items()
}

# ---------------------------------------------------------------------------
# Endgame detection — 기물 가치 합산 (폰·킹 제외)
# ---------------------------------------------------------------------------

_PIECE_VALUES = {
    chess.QUEEN: 9,
    chess.ROOK: 5,
    chess.BISHOP: 3,
    chess.KNIGHT: 3,
}

ENDGAME_MATERIAL_THRESHOLD = 24

# ---------------------------------------------------------------------------
# Material imbalance — 6/7랭크 폰 보정 포함
# ---------------------------------------------------------------------------

RANK6_PAWN_VALUE = 3
RANK7_PAWN_VALUE = 5
IMBALANCE_THRESHOLD = 9

# ---------------------------------------------------------------------------
# L-bucket thresholds
# ---------------------------------------------------------------------------

MUTUAL_FORCING_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComplexityFeatures:
    legal_moves: int
    total_material: int
    is_endgame: bool
    material_imbalance: int
    capture_count: int
    is_check: bool
    white_forcing: int
    black_forcing: int
    reason: str | None


@dataclass(frozen=True)
class ComplexityEvaluation:
    bucket: str
    move_time_ms: int
    features: ComplexityFeatures
    skip_eval: bool = False
    skip_value: int | None = None


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _material_value(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for piece_type, value in _PIECE_VALUES.items():
        total += chess.popcount(board.pieces_mask(piece_type, color)) * value
    return total


def _material_with_advanced_pawns(board: chess.Board, color: chess.Color) -> int:
    base = _material_value(board, color)

    pawn_bb = board.pieces_mask(chess.PAWN, color)
    while pawn_bb:
        sq = chess.lsb(pawn_bb)
        rank = chess.square_rank(sq)

        if color == chess.WHITE:
            if rank == 5:
                base += RANK6_PAWN_VALUE
            elif rank == 6:
                base += RANK7_PAWN_VALUE
        else:
            if rank == 2:
                base += RANK6_PAWN_VALUE
            elif rank == 1:
                base += RANK7_PAWN_VALUE

        pawn_bb &= pawn_bb - 1

    return base


def _count_captures(board: chess.Board) -> int:
    return sum(1 for move in board.legal_moves if board.is_capture(move))


def _count_forcing_moves(board: chess.Board, color: chess.Color) -> int:
    count = 0
    original_turn = board.turn
    board.turn = color
    for move in board.legal_moves:
        if board.is_capture(move) or board.gives_check(move):
            count += 1
    board.turn = original_turn
    return count


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_features(board: chess.Board) -> ComplexityFeatures:
    legal_moves = len(list(board.legal_moves))

    w_mat = _material_value(board, chess.WHITE)
    b_mat = _material_value(board, chess.BLACK)
    total_mat = w_mat + b_mat
    endgame = total_mat <= ENDGAME_MATERIAL_THRESHOLD

    if endgame:
        w_adjusted = _material_with_advanced_pawns(board, chess.WHITE)
        b_adjusted = _material_with_advanced_pawns(board, chess.BLACK)
        imbalance = abs(w_adjusted - b_adjusted)
    else:
        imbalance = abs(w_mat - b_mat)

    capture_count = _count_captures(board)

    w_forcing = _count_forcing_moves(board, chess.WHITE)
    b_forcing = _count_forcing_moves(board, chess.BLACK)

    return ComplexityFeatures(
        legal_moves=legal_moves,
        total_material=total_mat,
        is_endgame=endgame,
        material_imbalance=imbalance,
        capture_count=capture_count,
        is_check=board.is_check(),
        white_forcing=w_forcing,
        black_forcing=b_forcing,
        reason=None,
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify(
    board: chess.Board,
    features: ComplexityFeatures,
) -> tuple[str, str]:
    """Returns (bucket, reason)."""

    # S — 확정 승리
    if features.material_imbalance >= IMBALANCE_THRESHOLD:
        return "S", "quiet_winning_endgame"

    min_forcing = min(features.white_forcing, features.black_forcing)

    # L — 양쪽 다 forcing moves 높음
    if min_forcing >= MUTUAL_FORCING_THRESHOLD:
        return "L", "mutual_forcing_with_queens"

    # M — 나머지
    return "M", "default"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_position_complexity(fen: str) -> ComplexityEvaluation:
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError("Invalid FEN")

    features = _extract_features(board)
    bucket, reason = _classify(board, features)

    features = ComplexityFeatures(
        legal_moves=features.legal_moves,
        total_material=features.total_material,
        is_endgame=features.is_endgame,
        material_imbalance=features.material_imbalance,
        capture_count=features.capture_count,
        is_check=features.is_check,
        white_forcing=features.white_forcing,
        black_forcing=features.black_forcing,
        reason=reason,
    )

    return ComplexityEvaluation(
        bucket=bucket,
        move_time_ms=EVAL_TIME_BY_BUCKET[bucket],
        features=features,
    )


def get_complexity_config_snapshot() -> dict:
    return {
        "evalTimeByBucket": dict(EVAL_TIME_BY_BUCKET),
        "evalNodesByBucket": dict(EVAL_NODES_BY_BUCKET),
        "nodesPerMs": _NODES_PER_MS,
        "algorithm": "cascade_s_ml_v2",
        "endgame": {
            "materialThreshold": ENDGAME_MATERIAL_THRESHOLD,
            "pieceValues": {
                "queen": 9,
                "rook": 5,
                "bishop": 3,
                "knight": 3,
            },
        },
        "imbalance": {
            "threshold": IMBALANCE_THRESHOLD,
            "rank6PawnValue": RANK6_PAWN_VALUE,
            "rank7PawnValue": RANK7_PAWN_VALUE,
        },
        "lBucket": {
            "mutualForcingThreshold": MUTUAL_FORCING_THRESHOLD,
            "requireBothQueens": True,
        },
    }
