from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass(frozen=True)
class Settings:
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_decode_responses: bool = True
    stockfish_path: str = os.getenv("STOCKFISH_PATH", "/opt/homebrew/bin/stockfish")
    frontend_dir: str = os.getenv("FRONTEND_DIR", "frontend")
    ai_max_concurrent: int = int(os.getenv("AI_MAX_CONCURRENT", "3"))
    ai_engine_threads: int = int(os.getenv("AI_ENGINE_THREADS", "2"))
    ai_engine_hash_mb: int = int(os.getenv("AI_ENGINE_HASH_MB", "16"))
    ai_move_max_time_ms: int = int(os.getenv("AI_MOVE_MAX_TIME_MS", "800"))
    ai_eval_min_search_ms: int = int(os.getenv("AI_EVAL_MIN_SEARCH_MS", "50"))
    ai_eval_max_search_ms: int = int(os.getenv("AI_EVAL_MAX_SEARCH_MS", "2000"))
    # 라이브 1대1 WS 브로드캐스트용 Stockfish 시간(ms); 양쪽이 동일 값 수신
    game_online_eval_search_ms: int = int(os.getenv("GAME_ONLINE_EVAL_SEARCH_MS", "500"))
    # 평가 큐 예산(ms). 대기 총합이 이 값을 초과하면 신규 요청 드랍.
    eval_queue_budget_ms: int = int(os.getenv("EVAL_QUEUE_BUDGET_MS", "2000"))
    # Ablation study toggles
    enable_complexity_bucket: bool = os.getenv("ENABLE_COMPLEXITY_BUCKET", "true").lower() in ("true", "1", "yes")
    enable_queue_supersede: bool = os.getenv("ENABLE_QUEUE_SUPERSEDE", "true").lower() in ("true", "1", "yes")
    baseline_fixed_ms: int = int(os.getenv("BASELINE_FIXED_MS", "288"))
    default_time_limit_sec: int = int(os.getenv("DEFAULT_TIME_LIMIT_SEC", "600"))
    default_increment_sec: int = int(os.getenv("DEFAULT_INCREMENT_SEC", "0"))
    max_active_rooms: int = int(os.getenv("MAX_ACTIVE_ROOMS", "50"))
    max_active_connections: int = int(os.getenv("MAX_ACTIVE_CONNECTIONS", "50"))
    waiting_room_ttl_sec: int = int(os.getenv("WAITING_ROOM_TTL_SEC", "1800"))
    finished_room_ttl_sec: int = int(os.getenv("FINISHED_ROOM_TTL_SEC", "300"))
    playing_room_disconnect_ttl_sec: int = int(os.getenv("PLAYING_ROOM_DISCONNECT_TTL_SEC", "60"))
    # playing 중 한쪽만 WS 끊김일 때 유기 패까지 허용 시간(ms). 복귀 시 마감 시각은 다시 연장됨.
    abandonment_grace_ms: int = int(os.getenv("ABANDONMENT_GRACE_MS", "60000"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
