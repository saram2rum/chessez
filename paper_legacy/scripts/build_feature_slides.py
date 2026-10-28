#!/usr/bin/env python3
"""PART 2 — 정적 FEN feature × best_t 상관 분석 슬라이드 (최소 2장).

수식은 PPT 수식 편집기에서 따로 입력 (LaTeX 별도 제공). 슬라이드는 텍스트만.

내용:
  1. 분석 방법 + 결과 (feature 설명 + ρ)
  2. 결론 + 조합 한계 + 미로 비유
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

OUT = Path(__file__).resolve().parent.parent / "feature_slides.pptx"


def add_title(slide, text, size=30):
    title = slide.shapes.title
    title.text = text
    for p in title.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = True


def add_table(slide, left_in, top_in, width_in, height_in, headers, rows,
              header_font_size=14, body_font_size=14,
              highlight_cells=None, col_widths_in=None):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    table_shape = slide.shapes.add_table(
        n_rows, n_cols,
        Inches(left_in), Inches(top_in),
        Inches(width_in), Inches(height_in),
    )
    table = table_shape.table
    if col_widths_in:
        for i, w in enumerate(col_widths_in):
            if i < n_cols:
                table.columns[i].width = Inches(w)
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(header_font_size)
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                for r in p.runs:
                    r.font.size = Pt(body_font_size)
    if highlight_cells:
        for r_idx, c_idx, color_hex in highlight_cells:
            cell = table.cell(r_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(color_hex)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.bold = True
    return table


def add_textbox(slide, left_in, top_in, width_in, height_in, text,
                font_size=16, bold=False, italic=False, color_hex=None):
    tb = slide.shapes.add_textbox(
        Inches(left_in), Inches(top_in),
        Inches(width_in), Inches(height_in),
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.text = text
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.size = Pt(font_size)
            r.font.bold = bold
            r.font.italic = italic
            if color_hex:
                r.font.color.rgb = RGBColor.from_string(color_hex)
    return tb


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    # ─────────────────────────────────────────────────────────────
    # Slide A: 분석 방법 + 결과
    # ─────────────────────────────────────────────────────────────
    slide1 = prs.slides.add_slide(prs.slide_layouts[5])
    add_title(slide1, "정적 FEN feature × best_t 상관 분석")

    add_textbox(
        slide1, 0.5, 1.05, 12.3, 0.5,
        "Oracle 정책의 38% 잠재력 — 정적 FEN 정보만으로 도달 가능한가?",
        font_size=14, italic=True,
    )

    # 측정 방법 (한 줄)
    add_textbox(
        slide1, 0.5, 1.65, 12.3, 0.5,
        "측정: Spearman 순위 상관 ρ (수식은 별도 슬라이드)",
        font_size=15, bold=True,
    )

    # 결과 표 — feature 설명 포함
    feature_results = [
        ("imbalance",      "재료 불균형 (퀸=9, 룩=5, 비숍/나이트=3)", -0.0442),
        ("capture_count",  "현재 측의 합법 capture 수",                -0.0152),
        ("legal_moves",    "합법 수 총 개수 (분기 인자)",              -0.0202),
        ("total_material", "양측 비폰 재료 가치 합 (게임 단계)",       -0.0229),
        ("forcing_total",  "양측 capture + 체크 수 합",                -0.0457),
        ("is_check",       "현재 체크 상태 (0/1)",                     -0.0289),
        ("is_endgame",     "재료 ≤ 24 (퀸 제거 등)",                   +0.0395),
    ]
    rows_result = []
    for fname, desc, rho in feature_results:
        rows_result.append([fname, desc, f"{rho:+.4f}"])

    add_table(
        slide1,
        left_in=0.5, top_in=2.3, width_in=12.3, height_in=3.5,
        headers=["feature", "설명", "Spearman ρ"],
        rows=rows_result,
        header_font_size=13, body_font_size=12,
        col_widths_in=[2.3, 7.5, 2.5],
    )

    # 결론 + Cohen 인용
    add_textbox(
        slide1, 0.5, 6.0, 12.3, 0.5,
        "모든 정적 feature 의 |ρ| < 0.05 — Cohen 의 'small effect' 임계값 0.10 의 절반 미만",
        font_size=14, bold=True,
    )
    add_textbox(
        slide1, 0.5, 6.55, 12.3, 0.6,
        "Cohen (1988): r ≈ 0.10 은 small, r ≈ 0.30 은 medium, r ≈ 0.50 은 large effect.\n"
        "|r| < 0.10 의 상관은 예측 신호로 사실상 무효과(negligible).",
        font_size=11, italic=True,
    )

    # ─────────────────────────────────────────────────────────────
    # Slide B: 결론 + 조합 한계 + 미로 비유
    # ─────────────────────────────────────────────────────────────
    slide2 = prs.slides.add_slide(prs.slide_layouts[5])
    add_title(slide2, "결론 — 정적 FEN 정보의 본질적 한계")

    # 좌측: 한계 두 축
    add_textbox(
        slide2, 0.6, 1.3, 6.3, 0.5,
        "단일 feature 한계",
        font_size=18, bold=True,
    )
    add_textbox(
        slide2, 0.6, 1.8, 6.3, 1.4,
        "▸ 7개 feature 모두 |ρ| < 0.05\n"
        "▸ Cohen (1988) 의 small effect\n"
        "   임계값 0.10 미만 → 무효과",
        font_size=13,
    )

    add_textbox(
        slide2, 0.6, 3.4, 6.3, 0.5,
        "Feature 조합으로도 한계",
        font_size=18, bold=True,
    )
    add_textbox(
        slide2, 0.6, 3.9, 6.3, 2.6,
        "독립 가정 하 조합 상관 상한:\n"
        "  R ≤ √( Σ ρᵢ² )  ≈ 0.13\n\n"
        "Cohen small effect (0.10) 약간 넘는 수준\n"
        "→ 의미 있는 예측 신호로 부족\n\n"
        "실제 feature 들은 결정 여부 (|gt−50|)\n"
        "와 강한 상관 — 중복 정보\n"
        "→ 실제 조합 R 은 더 낮음",
        font_size=12,
    )

    # 우측: 미로 비유
    add_textbox(
        slide2, 7.2, 1.3, 5.8, 0.5,
        "직관적 비유",
        font_size=18, bold=True,
    )
    add_textbox(
        slide2, 7.2, 1.8, 5.8, 4.5,
        "FEN 만 보고 best_t 예측은\n\n"
        "마치 미로 입구에서\n"
        "그 미로가 얼마나 복잡할지\n"
        "예측하려는 것과 같다.\n\n"
        "입구 모양 (= FEN feature) 만으로는\n"
        "안쪽 분기의 깊이를 알 수 없다.",
        font_size=14, italic=True,
    )

    # 하단 시사점
    add_textbox(
        slide2, 0.5, 6.5, 12.3, 0.8,
        "→ Oracle (위치별 best_t) 의 38% 잠재력은 정적 FEN 정보로 도달 불가능. "
        "더 강한 신호 (학습 기반, 다중 깊이 평가 등) 필요.",
        font_size=12, bold=True,
    )

    prs.save(OUT)
    print(f"슬라이드 2장 생성: {OUT}")


if __name__ == "__main__":
    main()
