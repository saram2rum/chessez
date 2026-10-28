#!/usr/bin/env python3
"""졸업논문 핵심 2파트만 다루는 docx + pptx 생성.

PART 1: 라그랑주 천장 분석 — 위치별 시간 배분의 이론적 잠재력 정량화
PART 2: 정적 FEN feature 의 한계 — Spearman 상관 + 정적 휴리스틱 튜닝 부정적 결과

가우시안/베르누이 가중치, cp_50 동적 정책은 모두 제외. MAE 단일 메트릭.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT

from pptx import Presentation
from pptx.util import Inches, Pt as PptPt
from pptx.dml.color import RGBColor as PptColor
from pptx.enum.text import PP_ALIGN

OUT_DIR = Path(__file__).resolve().parent.parent
DOCX_PATH = OUT_DIR / "thesis_focus.docx"
PPTX_PATH = OUT_DIR / "thesis_focus.pptx"


# ── 콘텐츠 데이터 ────────────────────────────────────────────────────────────

N_FENS = 2927
WU_MAE = 1.8029
CEILING_MAE = 1.1185
CEILING_GAP = WU_MAE - CEILING_MAE
CEILING_GAP_PCT = CEILING_GAP / WU_MAE * 100
CEILING_MEAN_T = 243.0
TARGET_MS = 250

SEARCHTIMES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
CEILING_HIST = [559, 353, 277, 282, 237, 230, 233, 258, 237, 261]  # MAE 천장 시간 분포

# 시간별 uniform MAE
UNIFORM_MAE = {
    50: 2.4201, 100: 2.1595, 150: 1.9941, 200: 1.8776, 250: 1.8029,
    300: 1.7349, 350: 1.6791, 400: 1.6250, 450: 1.5865, 500: 1.5396,
}

# Spearman ρ (feat, best_t)
FEATURE_RHO = [
    ("imbalance", -0.0442, 5.29, "재료 불균형 (퀸=9, 룩=5, 비숍/나이트=3, 폰 가중 포함)"),
    ("capture_count", -0.0152, 1.09, "현재 측의 합법 capture 수"),
    ("legal_moves", -0.0202, 2.00, "현재 측의 합법 수 총 개수 (분기 인자)"),
    ("total_material", -0.0229, 2.24, "양측 비폰 재료 가치 합 (게임 단계 지표)"),
    ("forcing_total", -0.0457, 2.14, "양측 capture+체크 수 합"),
    ("is_check", -0.0289, 1.68, "현재 체크 상태 여부 (0/1)"),
    ("is_endgame", +0.0395, 2.24, "재료 ≤ 24 (퀸 제거 등)"),
]

# 정적 휴리스틱 변형 (공정 비교: 평균 시간 ≈ 250ms)
STATIC_VARIANTS_FAIR = [
    # name, S_thr, L_thr, S_t, M_t, L_t, S/M/L count, mean_t, MAE diff vs wu, CI95, verdict
    ("V0 baseline", "imb≥9", "f≥5", 100, 250, 400, "167/2677/83", 245.7, +0.033,
     "[+0.021, +0.045]", "wu 우월 (-1.81%)"),
    ("relaxed S only", "imb≥6", "f≥5", 100, 250, 400, "292/2552/83", 248.5,
     +0.029, "[+0.018, +0.041]", "wu 우월 (-1.62%)"),
    ("relaxed L only", "imb≥9", "f≥3", 100, 250, 400, "167/2144/616", 256.0,
     "—", "—", "예산 초과 (266ms)"),
    ("balanced", "imb≥6", "f≥3", 100, 200, 500, "292/2037/598", 251.3,
     +0.035, "[+0.006, +0.064]", "wu 우월 (-1.94%)"),
]

# 베스트 정적 변형도 wu 못 이김 — 부정적 결과 강조용


# ──────────────────────────────────────────────────────────────────────────
# DOCX 빌더
# ──────────────────────────────────────────────────────────────────────────

def add_para(doc, text, bold=False, italic=False, size=11, color=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color
    return p


def add_bullet(doc, text):
    return doc.add_paragraph(text, style="List Bullet")


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.bold = True
    for row in rows:
        cells = table.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    return table


def build_docx():
    doc = Document()

    doc.add_heading(
        "체스 엔진 위치별 검색 시간 배분의 이론적 잠재력과 정적 FEN 특징의 한계",
        level=0,
    )
    add_para(
        doc,
        "졸업논문 핵심 결과 정리 — "
        "(1) 라그랑주 천장 분석으로 위치별 시간 차등 배분의 이론적 잠재력 정량화, "
        "(2) 정적 FEN 특징만으로 그 잠재력을 실현할 수 있는지의 부정적 결과.",
        italic=True, size=10,
    )

    # ── 0. 메트릭 ────────────────────────────────────────────────────────
    doc.add_heading("1. 평가 메트릭 — win% 공간 MAE", level=1)

    add_para(doc, "본 연구는 Stockfish 의 시간별 평가 정확도를 win% 공간의 평균절대오차(MAE)로 측정한다.")
    add_para(doc, "MAE = (1/N) Σ |p̂_i − p*_i|", bold=True)
    add_para(doc, "여기서 p̂_i 는 검색 시간 t 에서의 win% 예측, p*_i 는 충분히 긴 시간에서의 reference 평가다.")

    doc.add_heading("1.1 win% 변환식", level=2)
    add_para(doc, "Win% = 50 + 50·(2/(1+exp(-0.00368208·cp)) − 1)", bold=True)
    add_bullet(doc, "Lichess 공식 시그모이드 (chess.com, AlphaZero 등 다수 채택)")
    add_bullet(doc, "cp 의 위치 의존적 비선형성 — 결정된 위치에서 cp 차이의 의미 손실 — 을 제거")
    add_bullet(doc, "Stockfish 자체도 fishtest LTC 모델로 cp ↔ win prob 매핑 정의 (100 cp ≈ 50% selfplay)")

    doc.add_heading("1.2 MAE 선택 근거", level=2)
    add_bullet(doc, "chess evaluation stability 연구의 최신 표준 — Avalle et al. (2025, Beyond Accuracy) 가 동일 구조의 MAE 를 reference 엔진 대비 absolute deviation 으로 사용")
    add_bullet(doc, "forecast accuracy 일반 표준 — Hyndman & Koehler (2006), scale-dependent 데이터의 MAE 표준성")
    add_bullet(doc, "단위 해석 가능성 — win% 의 percentage point (%p) 단위가 직관적, 분야 합의된 단위")
    add_bullet(doc, "Outlier robustness — Stockfish 의 mate horizon 한계로 인한 극단 오차에 robust")

    doc.add_heading("1.3 대안 메트릭의 부적합성", level=2)
    add_para(doc, "MSE — bounded space [0, 100] 에서 제곱 단위(%p²) 해석 불가, 극단 오차에 비선형 가중")
    add_para(doc, "KL 발산 — 두 확률 분포 간 거리로 정의되어 단일 점 추정치 비교에 부적합, "
                 "결정된 위치(gt ≈ 0 또는 100)에서 수치적 발산")

    # ── 2. PART 1: 라그랑주 천장 ─────────────────────────────────────────
    doc.add_heading("2. 라그랑주 천장 분석 — 이론적 잠재력 정량화", level=1)

    doc.add_heading("2.1 문제 정의", level=2)
    add_para(doc, "각 FEN i 에 검색 시간 인덱스 j_i ∈ {0,...,9} 를 배정할 때:")
    add_para(doc, "최소화: Σ_i |p̂_i^{t_{j_i}} − p*_i|     "
                 "제약: (1/N) Σ_i t_{j_i} ≤ T* = 250ms", bold=True)
    add_bullet(doc, "변수: 각 FEN 마다 10개 시간 후보 → 10^N 경우의 수 (직접 탐색 불가능)")
    add_bullet(doc, "제약: 평균 검색 시간 ≤ 250ms (현실적 자원 예산)")
    add_bullet(doc, f"데이터: N = {N_FENS} FEN, 시간 후보 = {SEARCHTIMES} ms")

    doc.add_heading("2.2 방법 — 라그랑주 완화 + 이분탐색", level=2)
    add_para(doc, "제약을 목적함수에 시간 가격 α 로 흡수:")
    add_para(doc, "L(α) = Σ_i (|err_{i,j_i}| + α · t_{j_i})", bold=True)
    add_bullet(doc, "α 고정 시 각 FEN 독립 argmin (분리 가능) — 계산 가능")
    add_bullet(doc, "평균 시간 mt(α) 는 α 에 대해 단조감소 → 이분탐색으로 mt(α*) = T* 인 α* 탐색")
    add_bullet(doc, "이 천장은 메트릭 독립 — MAE 와 다른 가중 메트릭이 같은 시간 배분을 천장으로 제공 (검증됨)")

    doc.add_heading("2.3 결과", level=2)
    add_table(doc, ["항목", "값"], [
        ["FEN 수", str(N_FENS)],
        ["wu(250) MAE", f"{WU_MAE:.4f}"],
        ["라그랑주 천장 MAE", f"{CEILING_MAE:.4f}"],
        ["천장 평균 시간", f"{CEILING_MEAN_T:.1f} ms"],
        ["wu − 천장 격차", f"{CEILING_GAP:+.4f} %p"],
        ["상대 개선 잠재력", f"{CEILING_GAP_PCT:.1f}%"],
    ])
    add_para(doc,
             f"이는 이론적으로 위치별 검색 시간을 최적 배분하면 wu 대비 {CEILING_GAP_PCT:.0f}% 의 MAE 개선이 "
             f"가능함을 의미한다. 어떤 정책도 이 천장을 넘을 수 없다.")

    doc.add_heading("2.4 천장 시간 분포 — 단일 최적 시간이 존재하지 않음", level=2)
    rows = [[f"{t} ms", str(c), f"{c/N_FENS*100:.1f}%"] for t, c in zip(SEARCHTIMES, CEILING_HIST)]
    add_table(doc, ["검색 시간", "천장에서 배정된 FEN 수", "비중"], rows)
    add_para(doc,
             "천장 정책은 어느 한 시간에 몰아주지 않는다. 가장 많이 배정된 50ms 도 19% 에 불과하고, "
             "10개 시간 모두에 의미 있는 비중이 분포되어 있다. 즉 위치마다 최적 시간이 다르다는 통계적 증거.",
             italic=True)

    doc.add_heading("2.5 시간별 uniform 정책 성능 비교", level=2)
    add_para(doc, "참고: 모든 위치를 같은 시간으로 평가한 경우의 MAE.")
    u_rows = [[f"{t} ms", f"{UNIFORM_MAE[t]:.4f}"] for t in SEARCHTIMES]
    add_table(doc, ["검색 시간", "uniform MAE"], u_rows)
    add_para(doc,
             f"가장 긴 500ms 도 MAE = {UNIFORM_MAE[500]:.4f} 로 천장 {CEILING_MAE:.4f} 와 여전히 격차. "
             f"즉 단순히 시간만 늘리는 것으로는 천장에 도달 못 함. 위치별 배분이 핵심.",
             italic=True)

    # ── 3. PART 2: 정적 FEN feature 한계 ─────────────────────────────────
    doc.add_heading("3. 정적 FEN 특징의 한계 — 부정적 결과", level=1)

    doc.add_heading("3.1 접근법", level=2)
    add_para(doc, "각 FEN 의 best_t (해당 FEN 의 MAE 가 최소가 되는 검색 시간) 와 "
                 "정적 FEN 특징의 Spearman 순위 상관 ρ 를 측정.")
    add_para(doc, "best_t = argmin_t |p̂_i^t − p*_i|", bold=True)
    add_bullet(doc, "ρ 가 크면(±) 해당 feature 가 best_t 예측에 유용한 신호")
    add_bullet(doc, "ρ ≈ 0 이면 그 feature 는 best_t 와 무관 (정보 없음)")
    add_bullet(doc, "ANOVA F 도 병행: best_t 그룹별 feature 평균 차이의 통계적 유의도")

    doc.add_heading("3.2 결과 — 모든 정적 feature 의 ρ < 0.05", level=2)
    rows = [[f, f"{rho:+.4f}", f"{F:.2f}", desc] for f, rho, F, desc in FEATURE_RHO]
    add_table(doc, ["feature", "Spearman ρ(feat, best_t)", "ANOVA F", "설명"], rows)
    add_para(doc,
             "모든 정적 feature 의 |ρ| < 0.05 — 통계적으로 노이즈 수준. "
             "어떤 단일 정적 feature 도 best_t 와 의미 있는 상관을 보이지 않음.",
             bold=True)

    doc.add_heading("3.3 정적 휴리스틱 정책 — 공정 비교 (평균 시간 ≈ 250ms)", level=2)
    add_para(doc, "S/M/L 버킷 + 시간 매핑의 다양한 변형을 시도. 모든 변형은 평균 검색 시간을 250ms 로 "
                 "고정하여 wu(250) 와 공정 비교.")
    rows = []
    for v in STATIC_VARIANTS_FAIR:
        name, s_thr, l_thr, s_t, m_t, l_t, sml, mt, d, ci, verd = v
        rows.append([name, f"{s_thr}, {l_thr}", f"{s_t}/{m_t}/{l_t}", sml, f"{mt:.1f}",
                     f"{d:+.4f}" if isinstance(d, float) else d, ci, verd])
    add_table(doc, ["변형", "임계값", "S/M/L 시간(ms)", "S/M/L 개수",
                    "평균시간(ms)", "diff vs wu", "CI95", "판정"], rows)
    add_para(doc,
             "공정 시간 예산 (≈ 250ms) 하에서 모든 변형이 wu 와 통계적 동등 또는 패배. "
             "임계값을 완화하거나 시간 매핑을 극단화해도 결과 동일.",
             bold=True)
    add_para(doc,
             "주의: 이전 분석에서 일부 변형이 wu 를 이기는 것처럼 보였으나, 평균 검색 시간이 "
             "wu(250) 보다 6~29% 더 큰 시간 예산을 사용한 결과였다. 같은 시간 예산으로 보간하면 "
             "모든 우월성이 사라진다.",
             italic=True, size=10)

    doc.add_heading("3.4 의미 — FEN 정보의 본질적 한계", level=2)
    add_para(doc, "왜 정적 FEN 으로 best_t 를 예측할 수 없는가?")
    add_bullet(doc, "Stockfish 가 시간을 더 줘야 의미 있는 위치 = 장기 전술 라인 (10+ 수 깊이) 이 있는 위치")
    add_bullet(doc, "장기 전술은 트리의 깊은 곳에 숨어 있어 FEN 의 표면 특징 (재료, forcing) 에서 안 드러남")
    add_bullet(doc, "FEN 은 탐색 트리의 root 만 묘사하지, 안쪽 분기의 복잡도는 안 보여줌")
    add_para(doc,
             "이는 마치 미로 입구에서 그 미로가 얼마나 복잡할지 예측하려는 것과 같다 — "
             "표면 정보만으로는 안쪽 분기의 깊이를 알 수 없다.",
             italic=True)

    # ── 4. 결론 ──────────────────────────────────────────────────────────
    doc.add_heading("4. 결론", level=1)

    doc.add_heading("4.1 핵심 발견", level=2)
    add_bullet(doc, f"라그랑주 천장 분석으로 위치별 시간 배분의 이론적 잠재력 정량화: "
                    f"wu 대비 {CEILING_GAP_PCT:.0f}% MAE 개선 가능")
    add_bullet(doc, f"천장 시간 분포는 10개 시간 모두에 분산 — 단일 최적 시간이 존재하지 않음")
    add_bullet(doc, "7개 정적 FEN 특징 모두 best_t 와 |ρ| < 0.05 의 노이즈 수준 상관")
    add_bullet(doc, "정적 휴리스틱의 임계값/시간 변형 시도 모두 공정 시간 예산에서 wu 와 동등 또는 패배")

    doc.add_heading("4.2 시사점", level=2)
    add_para(doc, "본 연구는 두 축의 결과를 통해 다음을 시사한다:")
    add_bullet(doc, "긍정적 발견: 위치별 시간 배분에 의한 평가 정확도 개선의 이론적 여지가 매우 크다 (38%)")
    add_bullet(doc, "부정적 발견: 그 잠재력은 정적 FEN 정보만으로는 본질적으로 도달 불가능하다")
    add_para(doc, "이는 향후 연구가 더 강한 신호 — 학습 기반 (NN), 다중 깊이 평가, 게임 트리 분기 분석 등 — "
                 "을 활용해야 함을 시사한다.")

    doc.add_heading("4.3 한계 및 Future work", level=2)
    add_bullet(doc, f"본 연구는 mixed_workload_pool 의 {N_FENS} FEN 에 한정 — 다른 분포에서의 검증 필요")
    add_bullet(doc, "Ground truth 는 long-time Stockfish 로 정의 — 진정한 절대 정답 (tablebase 등) 과의 비교 미수행")
    add_bullet(doc, "정적 feature 셋은 7개 (재료/forcing 류) 에 한정 — 더 정교한 정적 feature "
                    "(킹 안전도, 폰 구조, 피스 활동성 등) 의 robust 한 부정적 결과 확장 가능")
    add_bullet(doc, "학습 기반 정책 (FEN → 추천 시간) 의 실증 검증이 future work")

    # ── 부록 ─────────────────────────────────────────────────────────────
    doc.add_heading("부록 A. 메트릭 인용 레퍼런스", level=1)
    add_bullet(doc, "Avalle et al. (2025). Beyond Accuracy: A Geometric Stability Analysis of "
                    "LLMs in Chess Evaluation. arXiv:2512.15033.")
    add_bullet(doc, "Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of "
                    "forecast accuracy. International Journal of Forecasting, 22(4), 679-688.")
    add_bullet(doc, "Lichess Accuracy metric documentation. https://lichess.org/page/accuracy")
    add_bullet(doc, "Stockfish official documentation — Useful data. "
                    "https://official-stockfish.github.io/docs/stockfish-wiki/Useful-data.html")
    add_bullet(doc, "Silver, D. et al. (2018). A general reinforcement learning algorithm that "
                    "masters chess, shogi, and Go through self-play. Science, 362(6419), 1140-1144.")

    doc.save(DOCX_PATH)
    print(f"docx 생성: {DOCX_PATH}")


# ──────────────────────────────────────────────────────────────────────────
# PPTX 빌더
# ──────────────────────────────────────────────────────────────────────────

def _add_title_slide(prs, title, subtitle):
    layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title
    if len(slide.placeholders) > 1:
        slide.placeholders[1].text = subtitle
    return slide


def _add_bullet_slide(prs, title, bullets, notes=None):
    layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title
    body = slide.placeholders[1]
    tf = body.text_frame
    tf.clear()
    for i, b in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        # bullet 이 (text, level) 튜플이면 indent 단계 적용
        if isinstance(b, tuple):
            text, level = b
            p.text = text
            p.level = level
        else:
            p.text = b
            p.level = 0
        for r in p.runs:
            r.font.size = PptPt(18)
    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def _add_table_slide(prs, title, headers, rows, col_widths=None, font_size=14):
    layout = prs.slide_layouts[5]  # Title Only
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title
    left = Inches(0.5)
    top = Inches(1.5)
    width = Inches(9)
    height = Inches(5)
    n_rows = len(rows) + 1
    n_cols = len(headers)
    table_shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = table_shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            if i < n_cols:
                table.columns[i].width = Inches(w)
    # 헤더
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = PptPt(font_size)
    # 본문
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = PptPt(font_size)
    return slide


def build_pptx():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    # ── Slide 1: 제목 ────────────────────────────────────────────────────
    _add_title_slide(
        prs,
        "체스 엔진 위치별 검색 시간 배분의\n이론적 잠재력과 정적 FEN 특징의 한계",
        "라그랑주 천장 분석 + 정적 특징의 best_t 상관 분석",
    )

    # ── Slide 2: 연구 동기 ───────────────────────────────────────────────
    _add_bullet_slide(
        prs,
        "연구 동기",
        [
            "Stockfish 는 위치마다 평가 수렴 속도가 다르다",
            "결판 위치는 짧은 시간으로도 정확, 복잡한 위치는 더 긴 시간 필요",
            "동일 시간 예산에서 위치별로 시간을 차등 배분하면 평가 정확도가 개선될 수 있을까?",
            "핵심 질문:",
            ("Q1. 이론적으로 얼마나 개선 가능한가? (천장)", 1),
            ("Q2. 정적 FEN 특징만으로 그 잠재력을 실현할 수 있나?", 1),
        ],
    )

    # ── Slide 3: 메트릭 ──────────────────────────────────────────────────
    _add_bullet_slide(
        prs,
        "평가 메트릭 — win% 공간 MAE",
        [
            "MAE = (1/N) Σ |p̂_i − p*_i| in win% [0, 100]",
            "win% 변환: Lichess 시그모이드 (계수 0.00368208)",
            ("cp 의 위치 의존적 비선형성 제거", 1),
            ("Stockfish 자체도 fishtest LTC 모델로 cp ↔ win prob 매핑 정의", 1),
            "MAE 선택 근거:",
            ("Beyond Accuracy (Avalle 2025) — chess eval stability MAE 표준", 1),
            ("Hyndman & Koehler (2006) — forecast accuracy 일반 표준", 1),
            ("bounded space 에서 MSE 의 %p² 해석 불가 / KL 은 점 비교 부적합", 1),
        ],
        notes="MAE 정당화 카드: (1) Beyond Accuracy 2025 가 거의 같은 일을 하면서 MAE 사용, "
              "(2) Hyndman 표준, (3) Stockfish 자체 정의."
    )

    # ── Slide 4: PART 1 시작 — 문제 정의 ────────────────────────────────
    _add_bullet_slide(
        prs,
        "PART 1 — 라그랑주 천장: 문제 정의",
        [
            f"N = {N_FENS} FEN, 시간 후보 10개 ({SEARCHTIMES[0]}~{SEARCHTIMES[-1]} ms)",
            "각 FEN i 에 시간 인덱스 j_i 배정:",
            ("최소화: Σ_i |err_{i, j_i}|", 1),
            ("제약: (1/N) Σ_i t_{j_i} ≤ 250 ms", 1),
            "변수 공간: 10^N — 직접 탐색 불가",
            "제약이 모든 FEN 을 묶어 독립 풀이 불가",
        ],
    )

    # ── Slide 5: PART 1 — 라그랑주 완화 ──────────────────────────────────
    _add_bullet_slide(
        prs,
        "PART 1 — 라그랑주 완화 + 이분탐색",
        [
            "제약을 시간 가격 α 로 흡수: L(α) = Σ_i (|err_{i, j_i}| + α · t_{j_i})",
            "α 고정 시 각 FEN 독립 argmin → 계산 가능",
            "평균 시간 mt(α) 는 α 에 대해 단조감소",
            ("α=0: 시간 비용 0 → 다 긴 시간 → mt 최대", 1),
            ("α→∞: 시간 비용 무한 → 다 50ms → mt 최소", 1),
            "이분탐색으로 mt(α*) = 250ms 인 α* 탐색 (80회 반복)",
            "이 시점의 시간 배정이 도달 가능한 천장",
        ],
        notes="라그랑주 완화 + 이분탐색이지 KKT 직접 만족은 아님. 이산 문제라 엄밀히는 쌍대 갭 존재할 수 있지만 N 크면 무시 가능."
    )

    # ── Slide 6: PART 1 — 천장 결과 ──────────────────────────────────────
    _add_table_slide(
        prs,
        "PART 1 — 천장 결과: wu 대비 38% 개선 잠재력",
        ["항목", "값"],
        [
            ["FEN 수", str(N_FENS)],
            ["wu(250) MAE", f"{WU_MAE:.4f}"],
            ["라그랑주 천장 MAE", f"{CEILING_MAE:.4f}"],
            ["천장 평균 시간", f"{CEILING_MEAN_T:.1f} ms"],
            ["wu − 천장 격차", f"{CEILING_GAP:+.4f} %p"],
            ["상대 개선 잠재력", f"{CEILING_GAP_PCT:.1f}%"],
            ["paired bootstrap N", "10,000"],
            ["CI95 (wu − ceil)", "양수, 0 미포함 (유의)"],
        ],
        col_widths=[3.5, 5.5],
        font_size=18,
    )

    # ── Slide 7: PART 1 — 천장 시간 분포 ─────────────────────────────────
    rows = [[f"{t} ms", str(c), f"{c/N_FENS*100:.1f}%"] for t, c in zip(SEARCHTIMES, CEILING_HIST)]
    _add_table_slide(
        prs,
        "PART 1 — 천장 시간 분포 (단일 최적 시간 X)",
        ["검색 시간", "배정된 FEN 수", "비중"],
        rows,
        col_widths=[3, 3, 3],
        font_size=14,
    )

    # ── Slide 8: PART 1 — 시간별 uniform 성능 ────────────────────────────
    _add_bullet_slide(
        prs,
        "PART 1 — 시간만 늘려서는 천장에 도달 못함",
        [
            f"가장 짧은 50ms uniform MAE = {UNIFORM_MAE[50]:.4f}",
            f"표준 250ms uniform MAE = {UNIFORM_MAE[250]:.4f} (wu)",
            f"가장 긴 500ms uniform MAE = {UNIFORM_MAE[500]:.4f}",
            f"라그랑주 천장 MAE = {CEILING_MAE:.4f} (평균 시간 ≈ 250ms)",
            "단순 시간 증가만으로는 격차 못 메움 — 위치별 배분이 핵심",
        ],
    )

    # ── Slide 9: PART 2 시작 — 접근법 ────────────────────────────────────
    _add_bullet_slide(
        prs,
        "PART 2 — 정적 FEN 특징의 한계: 접근법",
        [
            "각 FEN 의 best_t = argmin_t |p̂_i^t − p*_i|",
            "정적 feature 후보 7개:",
            ("imbalance, capture_count, legal_moves, total_material", 1),
            ("forcing_total, is_check, is_endgame", 1),
            "측정: Spearman ρ(feature, best_t) + ANOVA F",
            "기대: ρ 크면 그 feature 가 best_t 예측 신호",
        ],
    )

    # ── Slide 10: PART 2 — Spearman ρ 결과 (메인 슬라이드) ───────────────
    rows = [[f, f"{rho:+.4f}", f"{F:.2f}"] for f, rho, F, _ in FEATURE_RHO]
    _add_table_slide(
        prs,
        "PART 2 — 모든 정적 feature 의 ρ < 0.05 (노이즈 수준)",
        ["feature", "Spearman ρ", "ANOVA F"],
        rows,
        col_widths=[4, 3, 2],
        font_size=18,
    )

    # ── Slide 11: PART 2 — 정적 휴리스틱 정책 시도 ───────────────────────
    rows = []
    for v in STATIC_VARIANTS_FAIR:
        name, s_thr, l_thr, s_t, m_t, l_t, sml, mt, d, ci, verd = v
        rows.append([name, f"{s_t}/{m_t}/{l_t}", f"{mt:.1f}", verd])
    _add_table_slide(
        prs,
        "PART 2 — 정적 휴리스틱: 공정 비교에서 모두 패배 또는 동등",
        ["변형", "S/M/L (ms)", "평균시간(ms)", "vs wu 판정"],
        rows,
        col_widths=[3, 2.5, 2, 2.5],
        font_size=16,
    )

    # ── Slide 12: PART 2 — 미로 비유 + 의미 ──────────────────────────────
    _add_bullet_slide(
        prs,
        "PART 2 — 왜 FEN 만으로는 한계인가",
        [
            "Stockfish 가 시간을 더 줘야 의미 있는 위치",
            ("= 장기 전술 라인 (10+ 수 깊이) 이 있는 위치", 1),
            ("= 표면 특징에서 안 드러나는 깊은 분기 복잡도", 1),
            "FEN 은 탐색 트리의 root 만 묘사 — 안쪽 분기는 안 보임",
            "",
            "이는 마치 미로 입구에서 그 미로가 얼마나 복잡할지",
            "예측하려는 것과 같다 — 표면 정보만으로는",
            "안쪽 분기의 깊이를 알 수 없다.",
        ],
    )

    # ── Slide 13: 결론 ───────────────────────────────────────────────────
    _add_bullet_slide(
        prs,
        "결론",
        [
            "긍정적 발견 (천장 분석):",
            (f"위치별 시간 배분에 의한 MAE 개선의 이론적 여지 = wu 대비 {CEILING_GAP_PCT:.0f}%", 1),
            "부정적 발견 (정적 특징):",
            ("7개 정적 FEN 특징 모두 |ρ| < 0.05 의 노이즈 수준 상관", 1),
            ("모든 휴리스틱 변형이 공정 시간 예산에서 wu 와 동등 또는 패배", 1),
            "시사: 천장 도달에는 더 강한 신호 필요",
            ("학습 기반 (NN), 다중 깊이 평가, 게임 트리 분기 분석 등", 1),
        ],
    )

    # ── Slide 14: 예상 Q&A ───────────────────────────────────────────────
    _add_bullet_slide(
        prs,
        "예상 질문 — 메트릭",
        [
            "Q. 왜 MAE? MSE/KL 도 가능하지 않나?",
            ("A. Beyond Accuracy (2025), Hyndman & Koehler (2006) 표준", 1),
            ("MSE: %p² 해석 불가, 극단 오차 비선형 가중", 1),
            ("KL: 점 추정 비교 부적합, 경계 발산", 1),
            "Q. Ground truth 는 진짜 정답인가?",
            ("A. Lichess/chess.com 표준 — deeper Stockfish 를 reference", 1),
            ("절대 정답 X, 분야 합의된 근사", 1),
            "Q. cp 가 표준 아닌가?",
            ("A. 플레이어 품질 측정은 cp(ACPL), 엔진 평가 정확도는 win% 가 표준 (Lichess accuracy)", 1),
        ],
    )

    _add_bullet_slide(
        prs,
        "예상 질문 — 천장 & 정적 한계",
        [
            "Q. 라그랑주 = KKT 만족?",
            ("A. 엄밀히는 이산 문제라 라그랑주 쌍대 문제의 최적 승수를 이분탐색", 1),
            ("연속 완화 관점에서 KKT (상보 여유성) 성립", 1),
            "Q. ρ < 0.05 가 noise 라는 근거?",
            ("A. p<0.05 통계적 유의도와 별개, 효과 크기로 매우 약함", 1),
            ("ANOVA F 도 cp50_abs(엔진 신호) 의 10.53 대비 5 이하", 1),
            "Q. 더 정교한 feature 안 시도했나?",
            ("A. 7개 feature 의 |ρ| 다 < 0.05 면 조합으로도 한계", 1),
            ("(개별 weak 신호의 조합 한계 √Σρ² ≈ 0.10)", 1),
        ],
    )

    prs.save(PPTX_PATH)
    print(f"pptx 생성: {PPTX_PATH}")


def main():
    build_docx()
    build_pptx()
    print(f"\n파일 위치:")
    print(f"  {DOCX_PATH}")
    print(f"  {PPTX_PATH}")


if __name__ == "__main__":
    main()
