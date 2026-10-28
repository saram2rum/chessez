"""
Lichess 래피드 게임 수집
- 양쪽 플레이어 레이팅 1200~1600
- 서로 다른 50명의 유저, 1게임씩
- 한 유저가 2개 이상 게임에 포함되면 안 됨
"""

import sys, time, random, requests, chess, chess.pgn, io, csv
from collections import deque
from pathlib import Path

ROOT = Path(__file__).parent.parent
RATING_MIN, RATING_MAX = 1200, 1600
TARGET = 50
PGN_OUT  = ROOT / "lichess_rapid_1200_1600.pgn"
CSV_OUT  = ROOT / "mixed_workload_pool.csv"


# ── PGN 유틸 ──────────────────────────────────────────────────────────────────

def split_pgn(raw: str) -> list[str]:
    games, cur = [], []
    for line in raw.splitlines():
        if line.startswith("[Event ") and cur:
            block = "\n".join(cur).strip()
            if block:
                games.append(block)
            cur = []
        cur.append(line)
    if cur:
        block = "\n".join(cur).strip()
        if block:
            games.append(block)
    return games


def get_games_pgn(username: str, max_n: int = 100) -> list[str]:
    try:
        r = requests.get(
            f"https://lichess.org/api/games/user/{username}",
            params={"perfType": "rapid", "rated": "true", "max": max_n,
                    "clocks": "false", "evals": "false", "opening": "false"},
            headers={"Accept": "application/x-chess-pgn"},
            timeout=30,
        )
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return split_pgn(r.text)
    except Exception as e:
        print(f"  [{username}] 오류: {e}", file=sys.stderr)
        return []


def parse_game(pgn_text: str):
    """(game_obj, white_lc, black_lc, w_elo, b_elo) 또는 None"""
    g = chess.pgn.read_game(io.StringIO(pgn_text))
    if not g:
        return None
    h = g.headers
    try:
        ws, bs = h.get("WhiteElo", "?"), h.get("BlackElo", "?")
        if "?" in ws or "?" in bs:
            return None
        we, be = int(ws), int(bs)
    except Exception:
        return None
    w = h.get("White", "").lower().strip()
    b = h.get("Black", "").lower().strip()
    if not w or not b:
        return None
    return g, w, b, we, be


def in_range(elo: int) -> bool:
    return RATING_MIN <= elo <= RATING_MAX


def fens_from_game(game) -> list[str]:
    board = game.board()
    fens = [board.fen()]
    for m in game.mainline_moves():
        board.push(m)
        fens.append(board.fen())
    return fens


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1단계: 앵커 유저 게임에서 1200~1600 유저명을 시드로 추출 ──────────────
    # german11, alireza2003 등 이전 실행에서 1200~1600 상대가 확인된 계정 활용.
    # 이 게임들 자체는 수집 대상이 아님(앵커 본인이 범위 밖).
    ANCHORS = ["german11", "alireza2003"]
    print("[1단계] 앵커 게임에서 1200~1600 범위 유저 시드 추출...")

    seeds: set[str] = set()
    for anchor in ANCHORS:
        pgns = get_games_pgn(anchor, max_n=200)
        time.sleep(1.2)
        for pgn in pgns:
            info = parse_game(pgn)
            if not info:
                continue
            _, w, b, we, be = info
            if in_range(we):
                seeds.add(w)
            if in_range(be):
                seeds.add(b)
        print(f"  {anchor} → 누적 시드 {len(seeds)}명")

    print(f"  시드 총 {len(seeds)}명\n")

    # ── 2단계: BFS로 게임 선택 ────────────────────────────────────────────────
    seed_list = list(seeds)
    random.shuffle(seed_list)
    queue: deque[str] = deque(seed_list)

    visited: set[str] = set(seeds)   # 큐에 한 번이라도 들어간 유저
    used: set[str]    = set()        # 선택된 게임에 등장한 유저 (white + black)

    selected: list[tuple] = []       # (pgn_text, w, b, we, be, game_obj)
    pgn_blocks: list[str] = []

    print(f"[2단계] 게임 선택 시작 (큐 {len(queue)}명)...")

    while queue and len(selected) < TARGET:
        username = queue.popleft()

        # 이미 다른 게임에 등장한 유저라면 스킵
        if username in used:
            continue

        pgns = get_games_pgn(username, max_n=50)
        time.sleep(1.1)

        found = None
        new_seeds: list[str] = []

        for pgn in pgns:
            info = parse_game(pgn)
            if not info:
                continue
            g, w, b, we, be = info

            # 스노우볼: 상대방을 후보 큐에 추가
            opp = b if w == username else w
            if opp not in visited:
                new_seeds.append(opp)
                visited.add(opp)

            # 이미 이 유저의 게임을 찾았으면 상대방 수집만 계속
            if found is not None:
                continue

            # 레이팅 조건 & 유저 중복 검사
            if not (in_range(we) and in_range(be)):
                continue
            if w in used or b in used:
                continue

            found = (pgn, w, b, we, be, g)

        if found is not None:
            pgn_text, w, b, we, be, g = found
            selected.append(found)
            pgn_blocks.append(pgn_text)
            used.add(w)
            used.add(b)
            print(f"  [{len(selected):2d}] {w}({we}) vs {b}({be})")

        # 새 후보를 큐 앞쪽에 추가해 빠르게 탐색
        random.shuffle(new_seeds)
        for s in new_seeds[:8]:
            queue.appendleft(s)

    print()
    n = len(selected)
    if n == 0:
        print("게임 수집 실패", file=sys.stderr)
        sys.exit(1)
    if n < TARGET:
        print(f"경고: 목표 {TARGET}개 중 {n}개만 수집됨", file=sys.stderr)

    print(f"=== 수집 완료: {n}게임 ===\n")

    # ── PGN 저장 ──────────────────────────────────────────────────────────────
    with open(PGN_OUT, "w", encoding="utf-8") as f:
        f.write("\n\n".join(pgn_blocks) + "\n")
    print(f"PGN 저장: {PGN_OUT}")

    # ── FEN 추출 (중복 제거) ──────────────────────────────────────────────────
    seen_fens: set[str] = set()
    all_fens:  list[str] = []
    for _, _, _, _, _, game_obj in selected:
        for fen in fens_from_game(game_obj):
            if fen not in seen_fens:
                seen_fens.add(fen)
                all_fens.append(fen)

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "ground_truth", "eval_value", "new_win_rate"])
        for fen in all_fens:
            writer.writerow([fen, "", "", ""])

    print(f"mixed_workload_pool.csv 덮어쓰기 완료")
    print(f"\n최종 FEN 개수: {len(all_fens)}")


if __name__ == "__main__":
    main()
