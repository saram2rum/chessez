# 졸업논문 수정본 재현 자료

「체스 형세 평가치 품질 향상을 위한 FEN 기반 엔진 탐색 시간 차등 배분의 검증과 한계」(한국외국어대학교 정보통신공학과, 윤지호) 수정본의 실험과 분석을 재현하기 위한 자료입니다.

## 폴더 구성

| 위치 | 내용 |
|---|---|
| `../app/` | Chessez 서비스 코드 (FastAPI). S/M/L 휴리스틱: `app/utils/position_complexity.py`, 입장 제어·대기열: `app/services/eval_queue_service.py` |
| `../paper_legacy/scripts/` | 원래 실험 스크립트. 데이터 수집 `fetch_lichess_rapid.py`, 기준값 `stockfish_eval_pool.py`, 본 측정 `measure_searchtime_local.py`, NPS 측정 `benchmark_nps.py` |
| `../paper_legacy/data/` | 데이터셋 `mixed_workload_pool.csv`(2,927 FEN, 기준값), 원본 PGN, NPS 측정 원자료 |
| `../paper_legacy/raw/searchtime_raw_local/` | 본 측정 원자료 (10개 예산 × 2,927 FEN) |
| `scripts/` | 수정본에서 추가한 측정·분석 스크립트 |
| `data/grid_p25.csv` | 두 번째 격자 {75, …, 525} ms 상당 노드 측정값 (논문 4.6절) |
| `data/grid_105.csv` | 노드 수 ×1.05 격자 측정값 (참고용, 원래 격자와 81%가 같은 값) |
| `data/grid_orig_sub.csv` | 원래 격자 재측정 600 FEN × 10 (깊이 기록, 재현성 확인) |
| `data/trace_500.jsonl.gz` | 180,000 nodes 탐색의 반복 심화 깊이별 평가치 (논문 5.7절) |
| `data/results.json`, `data/tables.json` | 논문에 들어간 수치와 표 |

## 재현 방법

1. Stockfish 18과 Python 패키지를 설치합니다: `pip install chess==1.11.2 numpy scipy scikit-learn matplotlib`
2. 스크립트 상단의 경로(`/home/claude/...`)를 이 저장소 위치로 바꿉니다.
3. 측정: `scripts/run_grid.py`(격자), `scripts/run_grid_off.py`(두 번째 격자), `scripts/run_trace.py`(깊이별 평가치)
4. 분석: `scripts/compute_all.py`가 `results.json`과 `tables.json`을 만들고, `scripts/fig_wu.py`가 그림 9를 그립니다.

엔진 설정은 Threads=1, Hash=16MB, 요청마다 해시 초기화(`ucinewgame`), 탐색량은 `nodes = 360 × t`입니다. 이 설정에서는 실행하는 컴퓨터와 관계없이 같은 평가치가 나옵니다(600 FEN × 10 예산, 6,000회 일치 확인).

## 주의

서버 접속용 SSH 키와 서버 주소는 포함하지 않았습니다.
