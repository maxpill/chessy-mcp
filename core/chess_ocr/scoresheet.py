"""Scoresheet layout detection, perspective warping, and cell extraction.

Supports structured chess scoresheet templates (PZSzach, KPZSzach, FIDE)
and generic two-column scoresheets. Performs:
    1. Paper quadrilateral detection & perspective rectification
    2. Header block and move-table grid segmentation
    3. Individual half-move (ply) cell extraction and enhancement
    4. Base64 encoding for ambiguous-move forensics
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# Standard A4 aspect ratio (height / width).
A4_ASPECT: Final[float] = 1.4142

# Default target rectified dimensions for normalized scoresheets.
RECTIFIED_WIDTH: Final[int] = 1440
RECTIFIED_HEIGHT: Final[int] = int(RECTIFIED_WIDTH * A4_ASPECT)  # ~2036


@dataclass(frozen=True)
class CellRegion:
    """Bounding coordinates and metadata for one half-move cell."""

    ply: int
    side: Literal["white", "black"]
    move_number: int
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1)


@dataclass(frozen=True)
class ScoresheetLayout:
    """Document structural layout with header and move cell coordinates."""

    document_type: str
    paper_quad: tuple[int, int, int, int, int, int, int, int] | None
    header_bbox: tuple[int, int, int, int]
    move_cells: dict[int, CellRegion]
    rectified_size: tuple[int, int]


def detect_paper_quad(img: Image.Image) -> tuple[int, int, int, int, int, int, int, int] | None:
    """Detect the quadrilateral bounding the paper sheet within a photo.

    Returns (tl_x, tl_y, bl_x, bl_y, br_x, br_y, tr_x, tr_y) in PIL QUAD
    transform order: top-left, bottom-left, bottom-right, top-right.
    Returns None if the paper occupies the entire frame or cannot be isolated.
    """
    from scipy.ndimage import label as cc_label

    w, h = img.size
    total_area = float(w * h)

    # Downsample for fast blob analysis
    scale = 800.0 / max(w, h) if max(w, h) > 800 else 1.0
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = img.resize((sw, sh), Image.Resampling.BILINEAR)

    gray = np.asarray(small.convert("L"), dtype=np.uint8)
    blurred = np.asarray(
        Image.fromarray(gray).filter(ImageFilter.GaussianBlur(3)),
        dtype=np.uint8,
    )

    # Otsu threshold to separate light paper from darker background
    hist = np.bincount(blurred.ravel(), minlength=256).astype(np.float64)
    tot = float(hist.sum())
    if tot <= 0:
        return None

    sum_t = float(np.dot(np.arange(256, dtype=np.float64), hist))
    wb, sb, best_var, thresh = 0.0, 0.0, -1.0, 127
    for t in range(256):
        wb += hist[t]
        if wb == 0:
            continue
        wf = tot - wb
        if wf == 0:
            break
        sb += t * hist[t]
        between = wb * wf * ((sb / wb) - ((sum_t - sb) / wf)) ** 2
        if between > best_var:
            best_var = between
            thresh = t

    bright_mask = blurred >= max(100, int(thresh))
    struct = np.ones((3, 3), dtype=np.uint8)
    lab, n_labels = cc_label(bright_mask, structure=struct)  # pyright: ignore[reportGeneralTypeIssues]
    if n_labels == 0:
        return None

    counts = np.bincount(lab.ravel().astype(np.int64))
    counts[0] = 0
    if counts.size == 0 or counts.max() == 0:
        return None

    best_label = int(counts.argmax())
    ys, xs = np.where(lab == best_label)
    if ys.size < (sw * sh * 0.20):  # At least 20% of frame
        return None

    # Determine corners via extremal sums:
    # top-left: min(x + y), bottom-right: max(x + y)
    # top-right: max(x - y), bottom-left: min(x - y)
    sum_coords = xs + ys
    diff_coords = xs - ys

    tl_idx = int(np.argmin(sum_coords))
    br_idx = int(np.argmax(sum_coords))
    tr_idx = int(np.argmax(diff_coords))
    bl_idx = int(np.argmin(diff_coords))

    inv = 1.0 / scale
    tl = (int(xs[tl_idx] * inv), int(ys[tl_idx] * inv))
    bl = (int(xs[bl_idx] * inv), int(ys[bl_idx] * inv))
    br = (int(xs[br_idx] * inv), int(ys[br_idx] * inv))
    tr = (int(xs[tr_idx] * inv), int(ys[tr_idx] * inv))

    # Check if quad occupies almost the full image (already cropped)
    min_x, max_x = min(tl[0], bl[0]), max(tr[0], br[0])
    min_y, max_y = min(tl[1], tr[1]), max(bl[1], br[1])
    box_area = float((max_x - min_x) * (max_y - min_y))
    if box_area > total_area * 0.90 and min_x < w * 0.05 and min_y < h * 0.05:
        return None

    return (tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1])


def warp_paper_quad(
    img: Image.Image,
    quad: tuple[int, int, int, int, int, int, int, int] | None = None,
    target_size: tuple[int, int] = (RECTIFIED_WIDTH, RECTIFIED_HEIGHT),
) -> Image.Image:
    """Rectify perspective of the scoresheet using PIL QUAD transform.

    QUAD transform maps (top-left, bottom-left, bottom-right, top-right)
    in the source image to the rectangular target_size.
    """
    if quad is None:
        quad = detect_paper_quad(img)

    if quad is None:
        # Resize to standard rectified size if needed
        return img.resize(target_size, Image.Resampling.LANCZOS)

    return img.transform(target_size, Image.Transform.QUAD, quad, resample=Image.Resampling.BILINEAR)


def detect_scoresheet_layout(
    img: Image.Image,
    document_type: str = "auto",
) -> ScoresheetLayout:
    """Segment rectified scoresheet into header and per-ply cell boxes.

    Standard Polish scoresheet format (PZSzach / KPZSzach / FIDE):
      - Header: y in [0.03, 0.20]
      - Move table: y in [0.20, 0.96]
      - 2 Main Columns:
          Col 1 (moves 1..30):  x in [0.04, 0.49]
          Col 2 (moves 31..60): x in [0.51, 0.96]
      - Sub-columns per group:
          Move number: 12% width
          White move:  44% width
          Black move:  44% width
      - 30 rows vertically for moves 1..30 and 31..60.
    """
    w, h = img.size

    # Base normalized proportions
    header_top = int(h * 0.02)
    header_bottom = int(h * 0.19)
    table_top = int(h * 0.20)
    table_bottom = int(h * 0.97)

    header_bbox = (int(w * 0.04), header_top, int(w * 0.96), header_bottom)

    # Left and right column group x-boundaries
    col1_left = int(w * 0.04)
    col1_right = int(w * 0.485)

    col2_left = int(w * 0.515)
    col2_right = int(w * 0.96)

    # Sub-column splits
    def subcol_boxes(x_left: int, x_right: int) -> tuple[tuple[int, int], tuple[int, int]]:
        width = x_right - x_left
        white_x0 = x_left + int(width * 0.14)
        white_x1 = x_left + int(width * 0.57)
        black_x0 = white_x1 + 1
        black_x1 = x_right
        return (white_x0, white_x1), (black_x0, black_x1)

    c1_white_x, c1_black_x = subcol_boxes(col1_left, col1_right)
    c2_white_x, c2_black_x = subcol_boxes(col2_left, col2_right)

    num_rows = 30
    row_height = (table_bottom - table_top) / num_rows

    move_cells: dict[int, CellRegion] = {}

    for row_idx in range(num_rows):
        move_num_col1 = row_idx + 1
        move_num_col2 = row_idx + 31

        y0 = int(table_top + row_idx * row_height)
        y1 = int(table_top + (row_idx + 1) * row_height)

        # Col 1: White
        ply_c1_w = (move_num_col1 - 1) * 2 + 1
        move_cells[ply_c1_w] = CellRegion(
            ply=ply_c1_w,
            side="white",
            move_number=move_num_col1,
            bbox=(c1_white_x[0], y0, c1_white_x[1], y1),
        )

        # Col 1: Black
        ply_c1_b = (move_num_col1 - 1) * 2 + 2
        move_cells[ply_c1_b] = CellRegion(
            ply=ply_c1_b,
            side="black",
            move_number=move_num_col1,
            bbox=(c1_black_x[0], y0, c1_black_x[1], y1),
        )

        # Col 2: White
        ply_c2_w = (move_num_col2 - 1) * 2 + 1
        move_cells[ply_c2_w] = CellRegion(
            ply=ply_c2_w,
            side="white",
            move_number=move_num_col2,
            bbox=(c2_white_x[0], y0, c2_white_x[1], y1),
        )

        # Col 2: Black
        ply_c2_b = (move_num_col2 - 1) * 2 + 2
        move_cells[ply_c2_b] = CellRegion(
            ply=ply_c2_b,
            side="black",
            move_number=move_num_col2,
            bbox=(c2_black_x[0], y0, c2_black_x[1], y1),
        )

    return ScoresheetLayout(
        document_type=document_type if document_type != "auto" else "polish_scoresheet",
        paper_quad=None,
        header_bbox=header_bbox,
        move_cells=move_cells,
        rectified_size=(w, h),
    )


def crop_cell(
    img: Image.Image,
    cell: CellRegion,
    *,
    margin: int = 4,
    upscale: int = 2,
    enhance: bool = True,
) -> Image.Image:
    """Crop and enhance an individual move cell."""
    w, h = img.size
    x0, y0, x1, y1 = cell.bbox

    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w, x1 + margin)
    y1 = min(h, y1 + margin)

    crop = img.crop((x0, y0, x1, y1))

    if upscale > 1:
        new_size = (crop.size[0] * upscale, crop.size[1] * upscale)
        crop = crop.resize(new_size, Image.Resampling.LANCZOS)

    if enhance:
        contrast = ImageEnhance.Contrast(crop)
        crop = contrast.enhance(1.25)
        sharpness = ImageEnhance.Sharpness(crop)
        crop = sharpness.enhance(1.20)

    return crop


def crop_header(img: Image.Image, layout: ScoresheetLayout) -> Image.Image:
    """Crop the metadata header region of the scoresheet."""
    return img.crop(layout.header_bbox)


def encode_crop_base64(crop: Image.Image) -> str:
    """Encode a PIL crop to PNG base64 string."""
    buf = io.BytesIO()
    crop.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_all_cell_crops(
    img: Image.Image,
    layout: ScoresheetLayout,
    max_ply: int = 60,
) -> dict[int, str]:
    """Extract base64-encoded crops for all half-moves up to max_ply."""
    crops: dict[int, str] = {}
    for ply in range(1, max_ply + 1):
        if ply in layout.move_cells:
            cell = layout.move_cells[ply]
            cropped = crop_cell(img, cell)
            crops[ply] = encode_crop_base64(cropped)
    return crops


__all__ = [
    "A4_ASPECT",
    "RECTIFIED_HEIGHT",
    "RECTIFIED_WIDTH",
    "CellRegion",
    "ScoresheetLayout",
    "crop_cell",
    "crop_header",
    "detect_paper_quad",
    "detect_scoresheet_layout",
    "encode_crop_base64",
    "extract_all_cell_crops",
    "warp_paper_quad",
]
