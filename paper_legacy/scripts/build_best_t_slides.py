#!/usr/bin/env python3
"""best_t 분포 분석 슬라이드 — 사용자 PPT 에 복붙용.

표 중심으로 4장. 결정적 카운트 (랜덤 X) 만 사용.

내용:
  1. 두 가지 측정 방법 소개
  2. "In best set" 카운트 — 단조 증가 + 50ms 특이
  3. "단독 best" 카운트 — U자 분포
  4. 함의 — 단일 시간 정책의 한계
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

OUT = Path(__file__).resolve().parent.parent / "best_t_slides.pptx"


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

    SEARCHTIMES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    N = 2927

    # ─────────────────────────────────────────────────────────────
    # Slide 1: best_t 정의 + 두 가지 카운트
    # ─────────────────────────────────────────────────────────────
    slide1 = prs.slides.add_slide(prs.slide_layouts[5])
    add_title(slide1, "best_t 정의 및 카운트 방법")

    add_textbox(
        slide1, 0.5, 1.1, 12.3, 0.6,
        "각 FEN 의 예측 오차를 최소화하는 시간",
        font_size=15, italic=True,
    )

    # best_t 정의
    add_textbox(
        slide1, 0.7, 1.8, 12.0, 0.7,
        "best_t  =  argmin  | 예측(t) − GT |",
        font_size=22, bold=True,
    )

    # 두 가지 카운트 안내
    add_textbox(
        slide1, 0.5, 2.9, 12.3, 0.5,
        "한 FEN 이 여러 시간에서 동률이면 어떻게 처리할까 — 두 가지 카운트",
        font_size=14, italic=True,
    )

    # 카운트 ① best_t 비율 (raw argmin)
    add_textbox(
        slide1, 0.7, 3.5, 12.0, 0.5,
        "①  best_t 비율 (단일 우선권)",
        font_size=18, bold=True,
    )
    add_textbox(
        slide1, 0.7, 4.0, 12.0, 1.2,
        "• 한 FEN 당 best_t 하나 — 동률 시 가장 짧은 시간 선택\n"
        "• 합계 = 100%\n"
        "• 의미: '이 시간이 단독 또는 우선권으로 선택된 비율'",
        font_size=13,
    )

    # 카운트 ② 비중 (best set 포함)
    add_textbox(
        slide1, 0.7, 5.3, 12.0, 0.5,
        "②  비중 (best set 포함)",
        font_size=18, bold=True,
    )
    add_textbox(
        slide1, 0.7, 5.8, 12.0, 1.2,
        "• 한 FEN 의 최저 오차와 같은 모든 시간 카운트 (동률이면 모두 +1)\n"
        "• 합계 > 100% 가능\n"
        "• 의미: '이 시간이 최저 후보에 들어가는 비율'",
        font_size=13,
    )

    add_textbox(
        slide1, 0.5, 7.0, 12.3, 0.4,
        f"본 데이터 N = {N} FEN | 시간 격자 50~500ms (10개)",
        font_size=12, italic=True,
    )

    # ─────────────────────────────────────────────────────────────
    # Slide 2: 두 표 — best_t 비율 vs 비중
    # ─────────────────────────────────────────────────────────────
    slide2 = prs.slides.add_slide(prs.slide_layouts[5])
    add_title(slide2, "best_t 분포 — 두 카운트 비교")

    # 표 데이터
    raw_argmin_counts = [559, 353, 277, 282, 237, 230, 233, 258, 237, 261]
    in_best_counts = [559, 537, 568, 643, 696, 758, 834, 979, 1064, 1226]

    # 좌측 표: best_t 비율 (raw argmin)
    add_textbox(
        slide2, 0.4, 1.2, 6.0, 0.5,
        "①  best_t 비율 (단일 우선권, 합 = 100%)",
        font_size=15, bold=True,
    )
    rows_left = []
    for t, c in zip(SEARCHTIMES, raw_argmin_counts):
        rows_left.append([f"{t} ms", str(c), f"{c/N*100:.1f}%"])
    add_table(
        slide2,
        left_in=0.4, top_in=1.8, width_in=6.0, height_in=5.0,
        headers=["검색 시간", "FEN 수", "best_t 비율"],
        rows=rows_left,
        header_font_size=13, body_font_size=12,
        highlight_cells=[
            (1, 0, "B3D9FF"), (1, 1, "B3D9FF"), (1, 2, "B3D9FF"),     # 50ms 강조
        ],
        col_widths_in=[1.8, 1.8, 2.4],
    )

    # 우측 표: 비중 (best set 포함)
    add_textbox(
        slide2, 6.9, 1.2, 6.0, 0.5,
        "②  비중 (best set 포함, 합 > 100%)",
        font_size=15, bold=True,
    )
    rows_right = []
    for t, c in zip(SEARCHTIMES, in_best_counts):
        rows_right.append([f"{t} ms", str(c), f"{c/N*100:.2f}%"])
    add_table(
        slide2,
        left_in=6.9, top_in=1.8, width_in=6.0, height_in=5.0,
        headers=["검색 시간", "FEN 수", "비중"],
        rows=rows_right,
        header_font_size=13, body_font_size=12,
        highlight_cells=[
            (10, 0, "FFD9B3"), (10, 1, "FFD9B3"), (10, 2, "FFD9B3"),  # 500ms 강조
        ],
        col_widths_in=[1.8, 1.8, 2.4],
    )

    # 하단 관찰 박스
    add_textbox(
        slide2, 0.4, 6.85, 12.5, 0.5,
        "관찰",
        font_size=15, bold=True,
    )
    add_textbox(
        slide2, 0.4, 7.2, 12.5, 0.3,
        "▸ ① best_t 비율: 시간 짧을수록 많음 (50ms = 19.1%, 1위)        "
        "▸ ② 비중: 시간 길수록 많음 (500ms = 41.89%, 1위) — 단, 50ms 도 19.1% 로 특이",
        font_size=12,
    )

    # ─────────────────────────────────────────────────────────────
    # Slide 3: 함의 — 단일 시간 정책의 한계 + Oracle
    # ─────────────────────────────────────────────────────────────
    slide4 = prs.slides.add_slide(prs.slide_layouts[5])
    add_title(slide4, "함의 — 단일 시간 정책의 한계")

    add_textbox(
        slide4, 0.5, 1.1, 12.3, 0.6,
        "위치는 두 그룹으로 나뉜다: 단순 vs 복잡. 단일 시간 정책으로는 둘 다 만족 X",
        font_size=14, italic=True,
    )

    # 정책별 비교 표
    rows_s4 = [
        ["wu(50)",  "모든 FEN 50ms",     "50 ms",   "2.4201", "−34.2%", "단순 위치만 만족, 나머지 큰 손해"],
        ["wu(250)", "모든 FEN 250ms",    "250 ms",  "1.8029", "기준",   "baseline"],
        ["wu(500)", "모든 FEN 500ms",    "500 ms",  "1.5396", "+14.6%", "시간 2배, 단순 위치 시간 낭비"],
        ["Oracle",  "각 FEN best_t",     "243 ms",  "1.1185", "+38.0%", "이론적 하한 — 어떤 정책도 못 넘음"],
    ]
    add_table(
        slide4,
        left_in=0.5, top_in=2.0, width_in=12.3, height_in=2.8,
        headers=["정책", "정의", "평균 시간", "MAE", "vs wu(250)", "비고"],
        rows=rows_s4,
        header_font_size=14, body_font_size=13,
        highlight_cells=[
            (1, 4, "FFB3B3"),
            (4, 0, "B3FFB3"), (4, 3, "B3FFB3"), (4, 4, "B3FFB3"),
        ],
        col_widths_in=[1.5, 2.2, 1.5, 1.5, 1.6, 4.0],
    )

    # 딜레마 표
    dilemma_rows = [
        ["모두 50ms",       "단순 그룹만 만족",    "복잡 그룹 큰 손해 → MAE 2.42"],
        ["모두 500ms",      "복잡 그룹만 만족",    "단순 그룹 시간 낭비 → MAE 1.54"],
        ["위치별 best_t",   "두 그룹 모두 만족",   "Oracle MAE 1.12 (38% 개선)"],
    ]
    add_table(
        slide4,
        left_in=0.5, top_in=5.1, width_in=12.3, height_in=1.9,
        headers=["전략", "해당 그룹", "결과"],
        rows=dilemma_rows,
        header_font_size=14, body_font_size=13,
        highlight_cells=[
            (3, 0, "C8FFC8"), (3, 1, "C8FFC8"), (3, 2, "C8FFC8"),
        ],
        col_widths_in=[2.5, 3.5, 6.3],
    )

    prs.save(OUT)
    print(f"슬라이드 3장 생성: {OUT}")


if __name__ == "__main__":
    main()
