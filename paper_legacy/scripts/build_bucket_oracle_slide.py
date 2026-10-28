#!/usr/bin/env python3
"""Bucket Oracle 검증 슬라이드 — 무제약 + 한 단계 위 격자 wu (보수적)."""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

OUT = Path(__file__).resolve().parent.parent / "bucket_oracle_slide.pptx"


def add_title(slide, text, size=26):
    title = slide.shapes.title
    title.text = text
    for p in title.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = True


def add_table(slide, left_in, top_in, width_in, height_in, headers, rows,
              header_font_size=12, body_font_size=12,
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
                font_size=14, bold=False, italic=False, color_hex=None):
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

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    add_title(slide, "Bucket Oracle 검증 — 정적 feature 의 이론적 상한 일반화")

    # 부제
    add_textbox(
        slide, 0.5, 1.0, 12.3, 0.45,
        "그나마 Spearman 상관계수가 컸던 imbalance, forcing 으로 이론적 최선 정책 구성 — 일반화 가능?",
        font_size=13, italic=True,
    )

    # ── 1단: 방법 ──
    add_textbox(
        slide, 0.5, 1.55, 12.3, 0.4,
        "① 방법",
        font_size=15, bold=True,
    )
    add_textbox(
        slide, 0.7, 1.95, 12.0, 1.1,
        "• imbalance (0~9+) × forcing (0~5+) 조합 중 데이터에 등장한 54 그룹으로 FEN 분류\n"
        "• 각 그룹의 mean error 가 최소인 시간을 그 그룹에 부여 (제약 없음, 자유)\n"
        "  → 정적 (imb, forcing) feature 만으로 가능한 이론적 최선 정책\n"
        "• 비교: 정책 평균 시간 ≤ 격자 값 중 가장 가까운 wu (보수적 비교)",
        font_size=11,
    )

    # ── 2단: 전체 데이터 결과 ──
    add_textbox(
        slide, 0.5, 3.2, 12.3, 0.4,
        "② 전체 데이터 결과 — 외관상 우월",
        font_size=15, bold=True,
    )
    add_textbox(
        slide, 0.7, 3.6, 12.0, 0.9,
        "Bucket Oracle MAE 1.5307 (mean 492ms)   vs   매칭 wu(500ms) 1.5396\n"
        "→ 개선율 +0.58% (정책 우월?)\n"
        "⚠️ 그러나 학습 = 평가 같은 데이터 — over-fitting 가능성",
        font_size=12,
    )

    # ── 3단: Train/Test ──
    add_textbox(
        slide, 0.5, 4.6, 12.3, 0.4,
        "③ Train/Test 검증 — 500 train × 10 seeds",
        font_size=15, bold=True,
    )
    add_textbox(
        slide, 0.7, 5.0, 5.5, 1.4,
        "Train: 500 FEN 으로 그룹별\n"
        "         best_t 학습\n"
        "Test: 나머지 2427 FEN 에 적용\n"
        "10 seeds 반복\n"
        "각 정책 평균 시간 이상의\n"
        "가까운 격자 wu 와 공정 비교",
        font_size=11,
    )
    # 결과 표
    add_table(
        slide,
        left_in=6.3, top_in=5.0, width_in=6.7, height_in=1.4,
        headers=["지표", "Test 평균"],
        rows=[
            ["Test 정책 MAE",  "1.602"],
            ["Test mean_t",    "444ms"],
            ["매칭 wu",        "1.568"],
            ["개선율",         "−3.44% (wu 우월)"],
            ["판정 분포",      "정책 우월 0 / 유의차 없음 1 / wu 우월 9"],
        ],
        header_font_size=11, body_font_size=11,
        highlight_cells=[
            (5, 1, "FFB3B3"),  # 판정 분포 강조
        ],
        col_widths_in=[2.2, 4.5],
    )

    # ── 4단: 해석 ──
    add_textbox(
        slide, 0.5, 6.55, 12.3, 0.4,
        "④ 해석",
        font_size=15, bold=True,
    )
    add_textbox(
        slide, 0.7, 6.95, 12.3, 0.5,
        "전체 데이터의 +0.58% 는 over-fitting — 새 데이터에서 평균 −3.44%, wu 우월 9/10.\n"
        "→ 정적 (imb, forcing) feature 의 이론적 최선도 실용적 한계. Cohen ρ<0.1 분석과 일관.",
        font_size=12,
    )

    prs.save(OUT)
    print(f"슬라이드 1장 생성: {OUT}")


if __name__ == "__main__":
    main()
