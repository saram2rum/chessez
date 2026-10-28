# Chessez Architecture Spec

## Goals

- `FastAPI` 기준으로 서버 구조를 재정의한다.
- 기존 `js`를 절대 기준으로 삼지 않고, 필요한 UI/기능만 보존한다.
- 멀티플레이는 `WebSocket`, AI 계산/평가는 `HTTP API`로 분리한다.
- `Redis`는 실시간 방 상태의 단일 진실 원천으로 사용한다.
- 게임 기록 아카이브는 나중에 붙일 수 있게 경계만 미리 만든다.

## Non-Goals For Now

- 지금 단계에서 `MongoDB`를 도입하지 않는다.
- 로그인/회원 시스템을 바로 만들지 않는다.
- 과거 스프링부트 메시지 형식과의 완전 호환을 목표로 하지 않는다.
- 기존 프론트 구조를 최대한 보존하는 것을 목표로 하지 않는다.

## Core Decisions

| 항목 | 결정 |
| --- | --- |
| 실시간 멀티플레이 | `WebSocket` |
| AI 수 계산 | `HTTP POST /api/ai/move` |
| 평가 바 | `HTTP POST /api/ai/evaluate` |
| 방 상태 저장 | `Redis` |
| WebSocket 연결 목록 | 서버 메모리 |
| 게임 규칙 검증 | `python-chess` |
| 엔진 호출 | `Stockfish` + 서비스 레이어 |
| 게임 기록 저장 | 나중에 추가, 지금은 인터페이스만 준비 |

## Backend Target Structure

```text
app/
  main.py
  core/
    config.py
    redis.py
  routers/
    pages.py
    ws.py
    ai.py
    health.py
  services/
    connection_manager.py
    room_service.py
    game_service.py
    engine_service.py
    timer_service.py
    archive_service.py
  repositories/
    room_repository.py
  schemas/
    common.py
    ws.py
    http.py
    room.py
    game.py
    ai.py
  utils/
    room_id.py
    board_snapshot.py
```

## Backend Responsibility Table

| 경로 | 책임 |
| --- | --- |
| `app/main.py` | 앱 생성, 라우터 등록, 정적 파일 mount |
| `app/core/config.py` | 환경변수/설정 로딩 |
| `app/core/redis.py` | Redis 클라이언트 생성 |
| `app/routers/pages.py` | `index.html` 및 정적 페이지 응답 |
| `app/routers/ws.py` | WebSocket 수신, 이벤트 라우팅 |
| `app/routers/ai.py` | AI 계산/평가 HTTP API |
| `app/routers/health.py` | 헬스체크 |
| `app/services/connection_manager.py` | 방별 WebSocket 연결 관리 |
| `app/services/room_service.py` | 방 생성/입장/퇴장/설정/재동기화 |
| `app/services/game_service.py` | 수 검증, 턴 처리, 게임 종료 판정 |
| `app/services/engine_service.py` | Stockfish 호출 |
| `app/services/timer_service.py` | 서버 기준 시간 차감 계산 |
| `app/services/archive_service.py` | 향후 아카이브 확장 포인트 |
| `app/repositories/room_repository.py` | Redis 입출력 캡슐화 |
| `app/schemas/*` | 요청/응답 계약 정의 |

## Frontend Target Structure

```text
frontend/js/
  app.js
  state/
    session-store.js
    lobby-store.js
    game-store.js
  network/
    ws-client.js
    ai-api.js
  controllers/
    lobby-controller.js
    game-controller.js
    ai-controller.js
    timer-controller.js
    sound-controller.js
  views/
    lobby-view.js
    waiting-room-view.js
    board-view.js
    game-view.js
    modal-view.js
  utils/
    user-id.js
    nickname.js
    fen.js
```

## Frontend Responsibility Table

| 경로 | 책임 |
| --- | --- |
| `frontend/js/app.js` | 앱 초기화, 의존성 조립 |
| `state/*` | 전역 상태 저장과 갱신 |
| `network/ws-client.js` | WebSocket 연결, 이벤트 송수신 |
| `network/ai-api.js` | AI/평가 HTTP 요청 |
| `controllers/lobby-controller.js` | 방 생성/입장/대기실 흐름 |
| `controllers/game-controller.js` | 게임 시작/수 입력/동기화 |
| `controllers/ai-controller.js` | AI 모드 로직 |
| `controllers/timer-controller.js` | 클라이언트 표시용 타이머 |
| `controllers/sound-controller.js` | 이벤트 기반 사운드 처리 |
| `views/*` | DOM 렌더링 |

## Domain Model

### Room State

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `roomId` | string | 5자리 방 코드 |
| `status` | string | `waiting`, `playing`, `finished` |
| `hostUserId` | string | 방장 ID |
| `guestUserId` | string or null | 참가자 ID |
| `hostNickname` | string | 방장 닉네임 |
| `guestNickname` | string or null | 참가자 닉네임 |
| `hostColor` | string or null | `WHITE` 또는 `BLACK` |
| `turn` | string | 현재 턴 |
| `fen` | string | 현재 보드 상태 |
| `moveSequence` | int | 수 진행 순서 |
| `whiteTimeMs` | int | 백 남은 시간 |
| `blackTimeMs` | int | 흑 남은 시간 |
| `lastMoveAt` | int | 직전 수 확정 시각 |
| `winner` | string or null | 게임 종료 시 승자 |
| `endReason` | string or null | 종료 사유 |
| `settings` | object | 시간/진영 설정 |

### Lobby Settings

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `startingSide` | string | `WHITE`, `BLACK`, `RANDOM` |
| `timeLimitSec` | int | 기본 시간 |
| `incrementSec` | int | 증분 |

## Redis Layout

| Key | 타입 | 내용 |
| --- | --- | --- |
| `room:{roomId}` | hash | 방 상태 전체 |
| `user_room:{userId}` | string | 유저가 속한 방 ID |

## WebSocket Envelope

모든 WebSocket 메시지는 아래 형식을 따른다.

```json
{
  "event": "game.move",
  "roomId": "AB123",
  "requestId": "optional-client-id",
  "payload": {}
}
```

## WebSocket Event Contract

### Client -> Server

| event | payload | 설명 |
| --- | --- | --- |
| `room.create` | `userId`, `nickname` | 방 생성 |
| `room.join` | `userId`, `nickname` | 방 입장 |
| `room.leave` | `userId` | 방 퇴장 |
| `room.sync` | `userId` | 재연결/상태 재동기화 |
| `room.settings.update` | `userId`, `settings` | 방 설정 변경 |
| `game.start` | `userId` | 게임 시작 |
| `game.move` | `userId`, `move` | 수 두기 |
| `game.timeout` | `userId` | 시간 초과 신고 |
| `ping` | 없음 | 연결 확인 |

### Server -> Client

| event | payload | 설명 |
| --- | --- | --- |
| `room.created` | `roomId` | 방 생성 성공 |
| `room.state` | room snapshot | 대기실/재동기화 상태 |
| `game.started` | game snapshot | 게임 시작 알림 |
| `game.move_applied` | move result | 수 적용 결과 |
| `game.over` | result snapshot | 게임 종료 |
| `player.left` | user info | 상대 퇴장 |
| `error` | `code`, `message` | 에러 |
| `pong` | 없음 | 연결 확인 응답 |

## WebSocket Payload Reference

### `room.state`

```json
{
  "event": "room.state",
  "roomId": "AB123",
  "payload": {
    "roomId": "AB123",
    "status": "waiting",
    "host": {
      "userId": "u-host",
      "nickname": "host"
    },
    "guest": {
      "userId": null,
      "nickname": null
    },
    "settings": {
      "startingSide": "RANDOM",
      "timeLimitSec": 600,
      "incrementSec": 0
    }
  }
}
```

### `game.started`

```json
{
  "event": "game.started",
  "roomId": "AB123",
  "payload": {
    "roomId": "AB123",
    "fen": "startpos fen here",
    "hostColor": "WHITE",
    "guestColor": "BLACK",
    "turn": "WHITE",
    "moveSequence": 0,
    "whiteTimeMs": 600000,
    "blackTimeMs": 600000
  }
}
```

### `game.move_applied`

```json
{
  "event": "game.move_applied",
  "roomId": "AB123",
  "payload": {
    "fen": "updated fen here",
    "move": {
      "source": "e2",
      "target": "e4",
      "promotion": null
    },
    "nextTurn": "BLACK",
    "moveSequence": 1,
    "whiteTimeMs": 598000,
    "blackTimeMs": 600000,
    "isCheck": false,
    "isGameOver": false,
    "winner": null,
    "endReason": null
  }
}
```

## HTTP API Contract

| Method | Path | 목적 |
| --- | --- | --- |
| `GET` | `/api/health` | 서버 상태 확인 |
| `GET` | `/api/ai/status` | AI 처리 가능 여부 확인 |
| `POST` | `/api/ai/move` | 현재 FEN에서 AI 최선 수 계산 |
| `POST` | `/api/ai/evaluate` | 현재 FEN 평가치 계산 |

## HTTP Request/Response Reference

### `GET /api/ai/status`

```json
{
  "available": true,
  "activeRequests": 0,
  "maxConcurrent": 3
}
```

### `POST /api/ai/move`

Request:

```json
{
  "fen": "fen string",
  "skillLevel": 10,
  "moveTimeMs": 500
}
```

Response:

```json
{
  "bestMove": "e2e4",
  "fen": "fen string"
}
```

### `POST /api/ai/evaluate`

Request:

```json
{
  "fen": "fen string",
  "searchTimeMs": 500
}
```

Response:

```json
{
  "type": "cp",
  "value": 34,
  "fen": "fen string"
}
```

## Timer Rules

- 시간 계산의 기준은 서버다.
- 클라이언트는 표시용 카운트다운만 수행한다.
- 서버는 수가 들어올 때마다 `lastMoveAt` 기준으로 시간을 차감한다.
- 서버는 수 적용 응답마다 `whiteTimeMs`, `blackTimeMs`를 같이 보낸다.
- 최종 시간 초과 판정은 서버가 수행한다.

## Sound Rules

- 사운드는 함수 내부 임의 호출이 아니라 이벤트 기반으로 처리한다.
- `my confirmed move`와 `opponent confirmed move`를 구분한다.
- 낙관적 UI와 서버 확정 이벤트를 분리하지 못하면 사운드 중복 버그가 생기므로, 최종 기준은 서버 확정 이벤트다.

## Archive Extension Point

지금은 DB 저장을 구현하지 않지만, 아래 데이터 모델을 기준으로 나중에 `MongoDB`를 붙일 수 있게 한다.

| 필드 | 설명 |
| --- | --- |
| `gameId` | 고유 게임 ID |
| `roomId` | 원본 방 ID |
| `startedAt` | 시작 시각 |
| `endedAt` | 종료 시각 |
| `players` | 플레이어 정보 |
| `timeControl` | 시간 설정 |
| `initialFen` | 시작 FEN |
| `movesUci` | 전체 UCI 기보 |
| `finalFen` | 종료 FEN |
| `winner` | 승자 |
| `termination` | 종료 사유 |
| `pgn` | 필요 시 PGN |

## Migration Phases

| 단계 | 목표 | 완료 기준 |
| --- | --- | --- |
| 1 | 백엔드 기본 구조 분리 | `main.py`가 앱 조립만 담당 |
| 2 | 새 WebSocket 계약 적용 | 방 생성/입장/시작/수 두기가 새 이벤트명으로 동작 |
| 3 | 프론트 로비 흐름 재작성 | 방 생성/입장/대기실 UI가 새 클라이언트로 동작 |
| 4 | 프론트 게임 흐름 재작성 | 보드 동기화, 턴 전환, 종료 처리 동작 |
| 5 | 사운드/타이머 정리 | 중복 사운드 제거, 서버 기준 시간 반영 |
| 6 | AI HTTP API 연결 | AI 수 계산/평가 바 동작 |
| 7 | 레거시 코드 제거 | 기존 불필요 JS 및 임시 호환 코드 정리 |

## Immediate Next Step

다음 구현 단계에서는 아래 범위를 먼저 처리한다.

1. `app/` 폴더 구조 생성
2. `main.py`를 `app/main.py` 진입점 구조로 정리
3. `connection_manager`, `room_repository`, `room_service`, `game_service` 뼈대 작성
4. WebSocket에서 `room.create`, `room.join`, `room.sync`, `game.start`, `game.move`까지만 먼저 옮긴다
