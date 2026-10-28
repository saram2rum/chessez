# Chessez

FastAPI 기반 실시간 체스 멀티플레이 + AI 평가 서버.
프론트엔드는 정적 자산으로 같은 서버에서 서빙한다.

논문 시기에 쓰던 실험 코드/데이터/문서는 모두 `paper_legacy/` 로 분리해두었다 — 운영에 필요한 것만 루트에 남는다.

---

## 1. 전체 흐름 한눈에

```
브라우저(index.html, JS)
   │
   │  HTTP  /api/health, /api/ai/status, /api/ai/move, /api/ai/evaluate
   │  WS    /ws  (room.* , game.*  이벤트)
   ▼
Nginx (80/443)  ──►  systemd: chessez-fastapi  ──►  uvicorn  main:app
                                                       │
                                                       ├── app/routers   (HTTP/WS 진입점)
                                                       ├── app/services  (도메인 로직)
                                                       ├── app/utils     (보드/포지션 헬퍼)
                                                       ├── Redis         (방 상태 단일 진실)
                                                       └── Stockfish     (AI 수/평가)
```

핵심 원칙
- **실시간 멀티플레이**는 WebSocket, **AI 계산/평가**는 HTTP API. 두 채널을 섞지 않는다.
- **방 상태**는 Redis 가 단일 진실 원천. 서버 메모리는 WebSocket 연결 목록만 가진다.
- **시간 계산**의 기준은 서버. 클라이언트는 표시용 카운트다운만 한다.

---

## 2. 디렉토리 구조

```
chessez/
├── main.py                  # uvicorn 엔트리: `from app.main import app`
├── requirements.txt
├── deploy.sh                # EC2 한방 배포 스크립트
├── (SSH 키는 저장소에 포함하지 않음 — KEY_PATH 환경변수로 지정)
│
├── app/
│   ├── main.py              # FastAPI 앱 조립, 정적 파일 mount, 라우터 등록
│   ├── core/
│   │   ├── config.py        # 환경변수 → Settings
│   │   └── redis.py         # Redis 클라이언트 싱글톤
│   ├── routers/
│   │   ├── pages.py         # index.html 서빙
│   │   ├── health.py        # GET /api/health
│   │   ├── session.py       # 세션 관련 HTTP
│   │   ├── ai.py            # POST /api/ai/move, /api/ai/evaluate, GET /api/ai/status
│   │   └── ws.py            # WebSocket /ws — room.*/game.* 이벤트 라우팅
│   ├── services/
│   │   ├── connection_manager.py   # 방별 WS 연결 보관
│   │   ├── room_service.py         # 방 생성/입장/퇴장/설정/재동기화
│   │   ├── game_service.py         # 수 검증, 턴 처리, 종료 판정
│   │   ├── timer_service.py        # 서버 기준 시간 차감
│   │   ├── engine_service.py       # Stockfish 호출
│   │   ├── eval_queue_service.py   # 평가 요청 큐 + 복잡도 버킷 입장 제어
│   │   └── live_position_eval.py   # 실시간 포지션 평가
│   ├── repositories/
│   │   └── room_repository.py      # Redis hash 입출력 캡슐화
│   ├── schemas/
│   │   ├── ai.py, game.py, room.py # 요청/응답 Pydantic 모델
│   └── utils/
│       ├── position_complexity.py  # 포지션 복잡도 점수 → eval_queue 가 사용
│       ├── board_snapshot.py       # board → 클라이언트 전송용 스냅샷
│       └── room_id.py              # 5자리 방 코드 생성
│
├── config/
│   ├── systemd/
│   │   ├── chessez-fastapi.service
│   │   └── chessez-fastapi.env.example
│   └── nginx/
│       └── chessez-fastapi.conf
│
├── frontend/
│   ├── index.html
│   ├── css/, js/, images/, sounds/
│
└── paper_legacy/           # 논문 작업 전체. 운영과 무관, 보관용.
    ├── scripts/            # 실험/측정/슬라이드 빌드 스크립트
    ├── data/               # 실험 결과 json/csv
    ├── raw/                # 원본 측정 raw 폴더들
    ├── docs/               # pptx/docx + 통합 전 원본 md (ARCHITECTURE_SPEC, DEPLOY_GUIDE)
    ├── bernoulli_kl.py             # 논문 전용 유틸 (운영 import 없음)
    └── gaussian_weighted_mae.py    # 논문 전용 유틸 (운영 import 없음)
```

---

## 3. WebSocket 계약 (요약)

모든 메시지 봉투:

```json
{ "event": "game.move", "roomId": "AB123", "requestId": "optional", "payload": {} }
```

| 방향 | event | 의미 |
|---|---|---|
| C→S | `room.create` / `room.join` / `room.leave` / `room.sync` / `room.settings.update` | 방 라이프사이클 |
| C→S | `game.start` / `game.move` / `game.timeout` | 게임 진행 |
| C→S | `ping` | 연결 확인 |
| S→C | `room.created` / `room.state` / `player.left` | 방 상태 알림 |
| S→C | `game.started` / `game.move_applied` / `game.over` | 게임 진행 알림 |
| S→C | `error` / `pong` | 메타 |

자세한 payload 형식은 `app/schemas/{room,game,ai}.py` 가 단일 진실 원천.

---

## 4. HTTP API

| Method | Path | 목적 |
|---|---|---|
| GET  | `/api/health` | 서버 살아있는지 |
| GET  | `/api/ai/status` | AI 처리 가능 여부 (`available`, `activeRequests`, `maxConcurrent`) |
| POST | `/api/ai/move` | `{fen, skillLevel, moveTimeMs}` → `{bestMove, fen}` |
| POST | `/api/ai/evaluate` | `{fen, searchTimeMs}` → `{type, value, fen}` |

---

## 5. 로컬 실행

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Redis 가 로컬에 떠 있어야 함 (brew services start redis 등)
# Stockfish 도 PATH 에 있어야 함

uvicorn main:app --reload --port 8010
```

브라우저에서 `http://localhost:8010` 으로 접속.

---

## 6. 배포 (EC2)

### 6.1 구조

```
/opt/chessez/
├── releases/<timestamp>/   # 매 배포마다 새 폴더
├── current -> 최신 release
└── shared/
    ├── venv/               # 재사용
    └── logs/
```

- `nginx` : 공인 80/443 종단
- `systemd` 유닛 `chessez-fastapi` : uvicorn 실행
- `redis-server`, `stockfish` : 같은 인스턴스에 로컬 설치

### 6.2 최초 부트스트랩

```bash
# 1) EC2 에 런타임 패키지 설치
sudo apt-get update -y
sudo apt-get install -y nginx redis-server stockfish python3 python3-venv python3-pip rsync

# 2) 첫 배포 (systemd 유닛 + nginx 사이트 자동 설치)
BOOTSTRAP=1 RELOAD_NGINX=1 ./deploy.sh

# 3) /etc/chessez/chessez-fastapi.env 가 example 에서 복사돼 있음 → 편집 후 재배포
sudo nano /etc/chessez/chessez-fastapi.env
BOOTSTRAP=1 RELOAD_NGINX=1 ./deploy.sh
```

### 6.3 일반 재배포

```bash
./deploy.sh
```

스크립트가 자동으로:
1. 새 release 디렉토리 업로드 (rsync)
2. venv 갱신 (pip install -r requirements.txt)
3. `current` 심볼릭 링크 교체
4. `chessez-fastapi` 재시작
5. health check 3개 (`/api/health`, `/api/ai/status`, `/api/ai/evaluate` POST)

### 6.4 환경변수 오버라이드

```bash
KEY_PATH=<PATH_TO_SSH_KEY> \
SERVER_IP=<SERVER_IP> \
SERVER_USER=ubuntu \
APP_PORT=8010 \
./deploy.sh
```

### 6.5 운영 명령어

```bash
sudo systemctl restart chessez-fastapi
sudo systemctl status  chessez-fastapi --no-pager
journalctl -u chessez-fastapi -n 100 --no-pager
```

### 6.6 롤백

```bash
ls -1dt /opt/chessez/releases/*/                # 이전 release 확인
ln -sfn /opt/chessez/releases/<prev> /opt/chessez/current
sudo systemctl restart chessez-fastapi
```

---

## 7. Redis 레이아웃

| Key | 타입 | 내용 |
|---|---|---|
| `room:{roomId}` | hash | 방 상태 전체 (status, host/guest, turn, fen, 시간 등) |
| `user_room:{userId}` | string | 유저가 속한 방 ID |

---

## 8. 운영 시 자주 보는 것

- 로그 : `journalctl -u chessez-fastapi -f`
- 헬스 : `curl http://localhost:8010/api/health`
- AI 동작 : `curl -X POST http://localhost:8010/api/ai/evaluate -H 'Content-Type: application/json' -d '{"fen":"<fen>","searchTimeMs":50}'`
- Stockfish 존재 : `which stockfish`
- Redis 존재 : `redis-cli ping`

---

## 9. paper_legacy/ 메모

논문 발표 후 보관용. 운영 코드 어디서도 import 하지 않는다.

- 다시 실험을 돌릴 일이 생기면 `paper_legacy/scripts/` 하위에 모든 스크립트가 그대로 있다.
- `bernoulli_kl.py`, `gaussian_weighted_mae.py` 는 원래 `app/utils/` 에 있었으나 운영 의존이 없어 옮겼다. 실험 재현 시 import 경로를 `paper_legacy.bernoulli_kl` 같은 식으로 고쳐야 한다.
- 한 달쯤 더 안 건드리면 `tar czf paper_legacy.tar.gz paper_legacy/` 로 압축해 외부 보관 권장.
