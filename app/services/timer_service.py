import time


class TimerService:
    def __init__(self) -> None:
        self.millis_per_second = 1000

    def now_ms(self) -> int:
        return int(time.time() * self.millis_per_second)

    def apply_elapsed(self, room: dict, now_ms: int | None = None) -> tuple[int, int, str | None]:
        now_ms = now_ms if now_ms is not None else self.now_ms()
        turn = room["turn"]
        last_move_at = int(room.get("lastMoveAt") or now_ms)
        elapsed = max(0, now_ms - last_move_at)
        white_time = int(room.get("whiteTimeMs") or 0)
        black_time = int(room.get("blackTimeMs") or 0)

        if turn == "WHITE":
            white_time = max(0, white_time - elapsed)
            timeout_color = "WHITE" if white_time == 0 else None
        else:
            black_time = max(0, black_time - elapsed)
            timeout_color = "BLACK" if black_time == 0 else None

        return white_time, black_time, timeout_color

    def apply_increment(
        self,
        white_time: int,
        black_time: int,
        mover_color: str,
        increment_sec: int,
    ) -> tuple[int, int]:
        increment_ms = max(0, increment_sec) * self.millis_per_second
        if mover_color == "WHITE":
            white_time += increment_ms
        else:
            black_time += increment_ms
        return white_time, black_time
