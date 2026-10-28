from fastapi import APIRouter

from app.core.redis import redis_client


router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str | bool]:
    redis_ok = False
    try:
        redis_ok = bool(redis_client.ping())
    except Exception:
        redis_ok = False
    return {"status": "ok", "redis": redis_ok}
