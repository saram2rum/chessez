#!/usr/bin/env python3
"""다변량 정적 ML 검증 — 단일 슬라이드 (7개 정적 feature, 10 seeds).

단순·핵심 메시지: 회귀(RF) 한 정책이 wu(균일)를 못 이긴다(0승).
분류·AUC·2f Bucket Oracle 은 시선 분산/개념부담 방지로 제외, 분류는 각주 한 줄로만 언급.
기존 build_bucket_oracle_slide.py 의 헬퍼/레이아웃(16:9) 스타일을 따른다.
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

OUT = Path(__file__).resolve().parent.parent / "multivariate_ml_slide.pptx"

ACCENT = "1F4E79"    # 진한 파랑
CALLOUT = "EAF1FA"   # 연한 파랑 (콜아웃)
WUROW = "D9E1F2"     # wu 기준 행


def add_title(slide, text, size=24):
    title = slide.shapes.title
    title.text = text
    for p in title.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = True
            r.font.color.rgb = RGBColor.from_string(ACCENT)


def add_table(slide, left_in, top_in, width_in, height_in, headers, rows,
              header_font_size=13, body_font_size=13,
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
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor.from_string(ACCENT)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(header_font_size)
                r.font.color.rgb = RGBColor.from_string("FFFFFF")
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
    return table


def add_textbox(slide, left_in, top_in, width_in, height_in, lines,
                font_size=10, italic=True, color_hex="666666"):
    """lines = 문자열 리스트 (각 줄). 각주용."""
    tb = slide.shapes.add_textbox(
        Inches(left_in), Inches(top_in), Inches(width_in), Inches(height_in)
    )
    tf = tb.text_frame
    tf.word_wrap = True
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln
        p.space_after = Pt(2)
        for r in p.runs:
            r.font.size = Pt(font_size)
            r.font.italic = italic
            r.font.color.rgb = RGBColor.from_string(color_hex)
    return tb


def add_bullets(slide, left_in, top_in, width_in, height_in, items,
                font_size=13, head=None, head_size=15, space_after=10):
    tb = slide.shapes.add_textbox(
        Inches(left_in), Inches(top_in), Inches(width_in), Inches(height_in)
    )
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    if head is not None:
        p = tf.paragraphs[0]
        p.text = head
        for r in p.runs:
            r.font.size = Pt(head_size)
            r.font.bold = True
            r.font.color.rgb = RGBColor.from_string(ACCENT)
        first = False
    for it in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(space_after)
        p.text = "•  " + it
        for r in p.runs:
            r.font.size = Pt(font_size)
    return tb


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    add_title(slide, "다변량 정적 ML 검증 — 7개 정적 feature로도 균일 정책(wu)을 이기지 못함")

    # 부제 (설계)
    sub = slide.shapes.add_textbox(Inches(0.5), Inches(0.95), Inches(12.4), Inches(0.55))
    sub.text_frame.word_wrap = True
    sp = sub.text_frame.paragraphs[0]
    sp.text = ("정적 feature 7종(legal_moves·total_material·is_endgame·material_imbalance·"
               "capture_count·is_check·forcing_total)을 RandomForest 회귀에 전량 투입해 "
               "국면별 적정 탐색시간(best_t)을 직접 예측. train/test = 2,342 / 585 (80/20) · 10 seeds.")
    for r in sp.runs:
        r.font.size = Pt(11.5)
        r.font.italic = True

    # ── 좌측: 핵심 비교 표 ──
    headers = ["정책", "test MAE (10회 평균)"]
    rows = [
        ["RF 회귀 (7 feature)", "1.811"],
        ["wu (균일 정책)", "1.810"],
    ]
    highlight = [(2, 0, WUROW), (2, 1, WUROW)]
    add_table(
        slide, 0.5, 1.75, 5.7, 1.15, headers, rows,
        header_font_size=13, body_font_size=13,
        highlight_cells=highlight,
        col_widths_in=[3.0, 2.7],
    )

    # 각주 (3줄)
    add_textbox(
        slide, 0.5, 3.0, 5.7, 0.6,
        [
            "· 비교: 회귀 평균 243ms vs 균일 정책 wu(250ms) — MAE는 낮을수록 좋음",
        ],
        font_size=10, italic=True, color_hex="666666",
    )

    # ── 좌측 하단: 결론 박스 ("왜 못 이기나" + 결론 통합) ──
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(3.85), Inches(6.4), Inches(2.45)
    )
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor.from_string(ACCENT)
    box.line.fill.background()
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.2)
    tf.margin_right = Inches(0.2)
    tf.margin_top = Inches(0.14)

    p0 = tf.paragraphs[0]
    p0.text = "결론"
    for r in p0.runs:
        r.font.size = Pt(15)
        r.font.bold = True
        r.font.color.rgb = RGBColor.from_string("FFFFFF")
    p1 = tf.add_paragraph()
    p1.text = "7개 정적 feature를 RF에 전량 투입해도 wu(250)를 이기지 못함."
    p1.space_before = Pt(6)
    for r in p1.runs:
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = RGBColor.from_string("FFFFFF")
    p2 = tf.add_paragraph()
    p2.text = ("회귀가 best_t를 못 맞혀 예측이 평균(~243ms)으로 수렴 → "
               "사실상 wu와 동일(1.811≈1.810) = 학습된 신호 없음.")
    p2.space_before = Pt(6)
    for r in p2.runs:
        r.font.size = Pt(12)
        r.font.color.rgb = RGBColor.from_string("FFFFFF")
    p3 = tf.add_paragraph()
    p3.text = "→ 정적 정보만으로는 국면별 적정 탐색시간을 예측할 수 없음 (단변량 |ρ|<0.15와 일관)."
    p3.space_before = Pt(6)
    for r in p3.runs:
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = RGBColor.from_string("FFFFFF")

    # ── 우측: 핵심 관찰 ──
    add_bullets(
        slide, 7.2, 1.75, 5.7, 4.4,
        [
            "train을 80%(2,342개)로 늘려도 wu를 못 이김 → \"데이터 부족\"이 아니라 "
            "신호 자체가 없음",
            "확인된 건 과적합뿐 — train 1.75 / test 1.81 "
            "(모델이 train을 외울 뿐 일반화 안 됨)",
        ],
        font_size=13.5, head="핵심 관찰", space_after=16,
    )

    prs.save(OUT)
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
