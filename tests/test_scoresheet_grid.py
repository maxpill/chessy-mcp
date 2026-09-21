"""Tests for scoresheet grid layout and cell extraction."""

from __future__ import annotations

import base64
from PIL import Image

from core.chess_ocr.scoresheet import (
    RECTIFIED_HEIGHT,
    RECTIFIED_WIDTH,
    crop_cell,
    crop_header,
    detect_paper_quad,
    detect_scoresheet_layout,
    encode_crop_base64,
    extract_all_cell_crops,
    warp_paper_quad,
)


def test_scoresheet_layout_detection() -> None:
    # Create synthetic sheet
    img = Image.new("RGB", (RECTIFIED_WIDTH, RECTIFIED_HEIGHT), color="white")
    layout = detect_scoresheet_layout(img)

    assert layout.document_type == "polish_scoresheet"
    assert len(layout.move_cells) == 120  # 60 moves * 2 sides

    # Check ply 1 (White move 1) and ply 2 (Black move 1)
    cell1 = layout.move_cells[1]
    cell2 = layout.move_cells[2]
    assert cell1.side == "white"
    assert cell1.move_number == 1
    assert cell2.side == "black"
    assert cell2.move_number == 1

    # Check ply 61 (White move 31, Col 2)
    cell61 = layout.move_cells[61]
    assert cell61.side == "white"
    assert cell61.move_number == 31
    assert cell61.bbox[0] > cell1.bbox[0]  # In right column


def test_crop_cell_and_base64_encode() -> None:
    img = Image.new("RGB", (RECTIFIED_WIDTH, RECTIFIED_HEIGHT), color="gray")
    layout = detect_scoresheet_layout(img)

    cell1 = layout.move_cells[1]
    crop = crop_cell(img, cell1, upscale=2)
    assert crop.size[0] > 0 and crop.size[1] > 0

    b64 = encode_crop_base64(crop)
    raw = base64.b64decode(b64)
    assert raw.startswith(b"\x89PNG")


def test_crop_header() -> None:
    img = Image.new("RGB", (RECTIFIED_WIDTH, RECTIFIED_HEIGHT), color="white")
    layout = detect_scoresheet_layout(img)
    header = crop_header(img, layout)
    assert header.size[0] > 0 and header.size[1] > 0


def test_extract_all_cell_crops() -> None:
    img = Image.new("RGB", (RECTIFIED_WIDTH, RECTIFIED_HEIGHT), color="white")
    layout = detect_scoresheet_layout(img)
    crops = extract_all_cell_crops(img, layout, max_ply=10)
    assert len(crops) == 10
    assert 1 in crops
    assert 10 in crops


def test_detect_paper_quad() -> None:
    img = Image.new("RGB", (1000, 1000), color="black")
    sheet = Image.new("RGB", (500, 700), color="white")
    img.paste(sheet, (250, 150))
    quad = detect_paper_quad(img)
    assert quad is not None
    assert len(quad) == 8


def test_warp_paper_quad_fallback() -> None:
    img = Image.new("RGB", (500, 700), color="white")
    warped = warp_paper_quad(img, quad=None, target_size=(300, 420))
    assert warped.size == (300, 420)
