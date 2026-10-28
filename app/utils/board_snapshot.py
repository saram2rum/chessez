import chess


PIECE_NAME_MAP = {
    chess.PAWN: "PAWN",
    chess.KNIGHT: "KNIGHT",
    chess.BISHOP: "BISHOP",
    chess.ROOK: "ROOK",
    chess.QUEEN: "QUEEN",
    chess.KING: "KING",
}


def board_to_snapshot(board: chess.Board) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for square, piece in board.piece_map().items():
        color = "WHITE" if piece.color == chess.WHITE else "BLACK"
        snapshot[chess.square_name(square)] = f"{color}_{PIECE_NAME_MAP[piece.piece_type]}"
    return snapshot
