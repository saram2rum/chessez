"""
현재 FastAPI `/api/ai/evaluate` 부하 테스트 - Locust.

캐시 왜곡을 줄이기 위해 mixed_workload_pool.csv 의 FEN을 순차 순환하며 POST 요청을 보낸다.

실행 예시:
    python3 -m locust -f scripts/locust_eval_load_fastapi.py --host=http://localhost:8000
    python3 -m locust -f scripts/locust_eval_load_fastapi.py --headless -u 50 -r 10 -t 60s --host=http://localhost:8000
"""
from pathlib import Path
import threading

from locust import HttpUser, between, task


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FEN_POOL: list[str] = []
FEN_LOCK = threading.Lock()
FEN_INDEX = 0
PRINTED_ONCE = False


def load_fen_pool() -> list[str]:
    csv_path = PROJECT_ROOT / "mixed_workload_pool.csv"
    if not csv_path.exists():
        return [DEFAULT_FEN]

    fens: list[str] = []
    with open(csv_path, encoding="utf-8") as file:
        next(file, None)
        for line in file:
            parts = line.strip().split(",", 1)
            if parts and parts[0]:
                fens.append(parts[0])
    return fens if fens else [DEFAULT_FEN]


def next_fen() -> str:
    global FEN_INDEX
    with FEN_LOCK:
        fen = FEN_POOL[FEN_INDEX % len(FEN_POOL)]
        FEN_INDEX += 1
        return fen


FEN_POOL = load_fen_pool()


class ChessEvalLoadTest(HttpUser):
    host = "http://localhost:8000"
    wait_time = between(0, 0)

    @task
    def evaluate_endpoint(self) -> None:
        global PRINTED_ONCE
        fen = next_fen()
        search_time_ms = 1700

        with self.client.post(
            "/api/ai/evaluate",
            json={"fen": fen, "searchTimeMs": search_time_ms},
            name=f"POST /api/ai/evaluate ({search_time_ms}ms)",
            catch_response=True,
        ) as response:
            if response.status_code >= 400:
                response.failure(f"status={response.status_code} body={response.text}")
                return

            if not PRINTED_ONCE:
                print(f"서버 응답: {response.text}")
                PRINTED_ONCE = True
