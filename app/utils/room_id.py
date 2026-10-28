import random
import string


ROOM_ID_CHARS = string.ascii_uppercase + string.digits


def generate_room_id(length: int = 5) -> str:
    return "".join(random.choice(ROOM_ID_CHARS) for _ in range(length))
