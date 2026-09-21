"""Image preprocessing for OCR.

The M3 multimodal model handles raw images well, but a few preprocessing
steps materially improve accuracy on handwritten Polish score sheets:
    - HEIC → RGB decode (pillow-heif)
    - Resize so the longest side is ≤ MAX_DIMENSION_px (default 2048) —
      M3 enforces a 10 MB upload cap and iPhone HEIC scans are routinely
      2-5 MB; without resize, the encoded PNG/quality JPEG blows past.
    - Light JPEG re-encoding (quality 90) for any input > 5 MB so we stay
      under M3's limit with comfortable headroom.

Heavy preprocessing (Otsu, morphological ops) was tested and DECREASED
M3 accuracy on handwritten sheets because it removed pencil-stroke
gradients that M3 uses for character disambiguation.

Opt-in enhancements (disabled by default, behind `rotation_strategy=`,
`contrast_strategy=`, `denoise_strategy=` parameters):
    - Multi-orientation: when a photo was taken with the camera held
      landscape + EXIF stripped, M3 reads sideways text and returns
      garbage. Detect the sheet rotation and (optionally) send M3 up to
      three orientation variants so its verifier picks the upright one.
    - CLAHE contrast: tile-based adaptive histogram equalisation.
      Preserves pen-stroke gradients (just stretches contrast locally)
      — should be safe for M3; on by opt-in only because the only test
      is "what does M3 actually do", and that varies.
    - Bilateral denoise: edge-preserving smoothing for noisy phone
      photos. Operates after CLAHE in the pipeline.
"""

from __future__ import annotations

import io
import math
from typing import Final, Literal

import numpy as np
from PIL import Image, ImageFilter, ImageOps


# Target longest-side pixel dimension after resize. 2048 preserves fine
# handwriting detail while keeping the encoded payload under M3's
# 10 MB upload cap (2048² RGBA ≈ 16 MB raw, JPEG q=90 ≈ 1-2 MB).
MAX_DIMENSION_PX: Final[int] = 2048

# JPEG quality for re-encoding when the image exceeds this size.
JPEG_QUALITY_THRESHOLD_BYTES: Final[int] = 5 * 1024 * 1024
JPEG_QUALITY: Final[int] = 90


# Public strategy type aliases.
RotationStrategy = Literal["off", "auto", "three"]
ContrastStrategy = Literal["off", "clahe"]
DenoiseStrategy = Literal["off", "bilateral"]


# Magic bytes for image format detection (kept here so callers don't need
# a separate import).
_MAGIC_BYTES_HEIC: tuple[bytes, ...] = (b"heic", b"heix", b"heim", b"heis", b"mif1")
_MAGIC_BYTES_JPEG: tuple[bytes, ...] = (b"\xff\xd8\xff",)
_MAGIC_BYTES_PNG: tuple[bytes, ...] = (b"\x89PNG\r\n\x1a\n",)
_MAGIC_BYTES_WEBP: tuple[bytes, ...] = (b"RIFF",)
_MAGIC_BYTES_GIF: tuple[bytes, ...] = (b"GIF87a", b"GIF89a")


# Detection tuning constants (used by _detect_best_orientation).
_TARGET_ASPECT: Final[float] = 1.4142  # A4 portrait
_ASPECT_TOL: Final[float] = 0.30
_BLUR_RADIUS: Final[int] = 3
_MIN_QUAD_AREA_FRAC: Final[float] = 0.06


# CLAHE defaults.
_CLAHE_TILE: Final[int] = 64
_CLAHE_CLIP_LIMIT: Final[float] = 2.0

# Bilateral defaults (applied at half-resolution for speed).
_BILATERAL_SIGMA_SPATIAL: Final[float] = 1.4
_BILATERAL_SIGMA_RANGE: Final[float] = 22.0


# -----------------------------------------------------------------------------
# Format detection (public)
# -----------------------------------------------------------------------------


def detect_image_format(image_bytes: bytes) -> str:
    """Return a canonical format string for the given image bytes.

    Recognized: ``"heic"``, ``"jpeg"``, ``"png"``, ``"webp"``, ``"gif"``,
    ``"unknown"``. Used by the OCR engine to pick the right base64
    payload encoding for M3.
    """
    if image_bytes.startswith(_MAGIC_BYTES_JPEG):
        return "jpeg"
    if image_bytes.startswith(_MAGIC_BYTES_PNG):
        return "png"
    if image_bytes.startswith(_MAGIC_BYTES_GIF):
        return "gif"
    if image_bytes.startswith(_MAGIC_BYTES_WEBP):
        return "webp"
    # HEIC: magic is at offset 4 (the "ftyp" box follows a 4-byte size).
    if len(image_bytes) >= 12 and image_bytes[4:8] == b"ftyp":
        brand = image_bytes[8:12]
        for candidate in _MAGIC_BYTES_HEIC:
            if brand.startswith(candidate):
                return "heic"
    return "unknown"


# -----------------------------------------------------------------------------
# Decode pipeline (public)
# -----------------------------------------------------------------------------


def decode_image_to_rgb(
    image_bytes: bytes,
    *,
    rotation_strategy: RotationStrategy = "off",
    contrast_strategy: ContrastStrategy = "off",
    denoise_strategy: DenoiseStrategy = "off",
) -> tuple[list[int], list[bytes], int]:
    """Decode, optionally rotate, optionally enhance.

    Args:
        image_bytes: Encoded image (HEIC/JPEG/PNG/WebP/GIF).
        rotation_strategy:
            ``"off"``   — return the image as-is, in 1 variant.
            ``"auto"``  — detect sheet rotation; return the corrected
                          image as 1 variant; ``detected_rotation`` is
                          the auto-correct angle in degrees (-90/0/90).
            ``"three"`` — return all 3 orientation variants (0°/90°/-90°)
                          for the multi-pass verifier; detected_rotation
                          will be 0 (let the verifier pick the upright).
        contrast_strategy: ``"off"`` or ``"clahe"``.
        denoise_strategy:   ``"off"`` or ``"bilateral"``.

    Returns:
        ``(rotations, png_or_jpeg_bytes_per_variant, detected_rotation)``.
        ``rotations[i]`` is the rotation in degrees CCW that was applied
        to the i-th variant (always 0 when ``rotation_strategy != "three"``).
        ``detected_rotation`` is the angle the auto-detector chose
        (only meaningful when ``rotation_strategy == "auto"``).

    Raises:
        ValueError: on unsupported format.
    """
    fmt = detect_image_format(image_bytes)
    if fmt == "unknown":
        raise ValueError(f"Unsupported image format: {fmt!r}")

    img = _decode_to_rgb_pil(image_bytes, fmt=fmt)
    img = _downscale_to_cap(img)

    if rotation_strategy == "off":
        rotations: list[int] = [0]
        variants = [_apply_optional_enhancements(img, contrast_strategy, denoise_strategy)]
    elif rotation_strategy == "auto":
        rot = _detect_best_rotation(img)
        rotated = _rotate_image(img, rot)
        rotations = [0]
        variants = [_apply_optional_enhancements(rotated, contrast_strategy, denoise_strategy)]
    elif rotation_strategy == "three":
        rotations = [0, 90, -90]
        variants = []
        for rot in rotations:
            rotated = _rotate_image(img, rot) if rot != 0 else img
            variants.append(
                _apply_optional_enhancements(rotated, contrast_strategy, denoise_strategy)
            )
    else:
        raise ValueError(f"Unknown rotation_strategy: {rotation_strategy!r}")

    encoded = [_pil_to_png_bytes(v) for v in variants]
    detected_rotation = rotations[0] if rotation_strategy == "auto" else 0
    return rotations, encoded, detected_rotation


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    """Encode a PIL image to lossless PNG bytes for the M3 data URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _decode_to_rgb_pil(image_bytes: bytes, *, fmt: str) -> Image.Image:
    """Decode encoded bytes to a single-mode-RGB PIL Image (handles HEIC)."""
    if fmt == "heic":
        try:
            from pillow_heif.as_plugin import register_heif_opener
        except ImportError as exc:
            raise ValueError(
                "HEIC images require pillow-heif; install with `pip install pillow-heif`."
            ) from exc
        register_heif_opener()
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _downscale_to_cap(img: Image.Image) -> Image.Image:
    longest = max(img.size)
    if longest > MAX_DIMENSION_PX:
        scale = MAX_DIMENSION_PX / longest
        new_size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img


def _apply_optional_enhancements(
    img: Image.Image,
    contrast_strategy: ContrastStrategy,
    denoise_strategy: DenoiseStrategy,
) -> Image.Image:
    out = img
    if contrast_strategy == "clahe":
        out = _apply_clahe(out)
    if denoise_strategy == "bilateral":
        out = _apply_bilateral(out)
    return out


# -----------------------------------------------------------------------------
# Rotation detection (private — port from enhanced-png)
# -----------------------------------------------------------------------------


def _otsu_threshold(arr_u8: np.ndarray) -> int:
    """Otsu threshold (0..255) splitting darker from lighter pixels."""
    hist = np.bincount(arr_u8.ravel(), minlength=256).astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return 127
    sum_total = float(np.dot(np.arange(256, dtype=np.float64), hist))
    w_b, sum_b, best_var, threshold = 0.0, 0.0, -1.0, 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / w_b
        mean_f = (sum_total - sum_b) / w_f
        between = w_b * w_f * (mean_b - mean_f) ** 2
        if between > best_var:
            best_var = between
            threshold = t
    return int(threshold)


def _score_blob_aspect_fill(
    arr_u8: np.ndarray,
    bbox: tuple[int, int, int, int],
    image_area: float,
) -> float:
    """Combined aspect + fill + ink-density score for a candidate blob.

    The chess scoresheet is the load-bearing signal: A4 aspect AND dense
    ink (handwriting + grid lines, ~10 % of pixel area). Empty paper,
    wood tables and carpet get low ink scores and lose to a real sheet.
    """
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return -1.0
    aspect = max(bw, bh) / min(bw, bh)
    aspect_err = abs(aspect - _TARGET_ASPECT) / _TARGET_ASPECT
    if aspect_err > _ASPECT_TOL:
        return -1.0
    area = bw * bh
    if area < image_area * _MIN_QUAD_AREA_FRAC:
        return -1.0
    h_img, w_img = arr_u8.shape
    crop = arr_u8[max(0, y0) : min(h_img, y1), max(0, x0) : min(w_img, x1)]
    if crop.size == 0:
        return -1.0
    ink = float((crop < 110).mean())
    # Sweet spot: 0.12 is typical for a scoresheet; fall off linearly to 0
    # within 0.25 either way. Off-target ink is the most useful single
    # signal that this blob ISN'T a scoresheet.
    ink_score = max(0.0, 1.0 - min(1.0, abs(ink - 0.12) / 0.25))
    aspect_score = 1.0 - aspect_err
    return aspect_score * (0.4 + 0.6 * ink_score)


def _detect_best_rotation(img: Image.Image) -> int:
    """Pick 0 / 90 / -90 degrees so the scoresheet appears upright.

    Brute-force small: downscale the image, threshold via Otsu, label
    the white components, score each by A4-aspect + ink-density, return
    the rotation whose best blob has the highest score. Falls back to 0
    when no orientation yields a candidate (low-contrast photo).
    """
    from scipy.ndimage import label as cc_label  # local import: heavy dep

    img_small = img
    if max(img.size) > 1200:
        scale = 1200 / max(img.size)
        img_small = img.resize(
            (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale))),
            Image.Resampling.LANCZOS,
        )

    rotations = (0, 90, -90)
    best_score = -1.0
    best_rot = 0
    image_area = float(img_small.size[1] * img_small.size[0])
    for rot in rotations:
        rotated = _rotate_image(img_small, rot)
        gray_u8 = np.asarray(rotated.convert("L"), dtype=np.uint8)
        blurred = np.asarray(
            Image.fromarray(gray_u8).filter(ImageFilter.GaussianBlur(_BLUR_RADIUS)),
            dtype=np.uint8,
        )
        thr = _otsu_threshold(blurred)
        white = blurred >= thr
        if int(white.sum()) < image_area * _MIN_QUAD_AREA_FRAC:
            continue
        structure = np.ones((3, 3), dtype=np.uint8)
        # scipy.ndimage.label's pyright stubs disagree with runtime on the
        # structure type; cast to Any to keep the static checker quiet.
        lab, n_labels = cc_label(white, structure=structure)  # pyright: ignore[reportGeneralTypeIssues]
        if n_labels == 0:
            continue
        counts = np.bincount(lab.ravel().astype(np.int64))
        counts[0] = 0
        if counts.size == 0 or counts.max() == 0:
            continue
        for label in np.argsort(-counts)[:5]:
            label = int(label)
            if label == 0:
                continue
            ys, xs = np.where(lab == label)
            if ys.size < 50:
                continue
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            s = _score_blob_aspect_fill(blurred, (x0, y0, x1, y1), image_area)  # pyright: ignore[reportArgumentType]
            if s > best_score:
                best_score = s
                best_rot = rot
    return best_rot


def _rotate_image(img: Image.Image, deg: int) -> Image.Image:
    """Rotate by ``deg`` (CCW) using PIL transpose when 90° multiple (faster), else .rotate()."""
    if deg % 360 == 0:
        return img
    # PIL < 10 used Image.ROTATE_90/ROTATE_270; PIL >= 10 uses Image.Transpose.ROTATE_90.
    transp = getattr(Image, "Transpose", None)
    if transp is not None:
        rotate_90 = getattr(transp, "ROTATE_90", None)
        rotate_270 = getattr(transp, "ROTATE_270", None)
        if deg == 90 and rotate_90 is not None:
            return img.transpose(rotate_90)  # pyright: ignore[reportArgumentType]
        if deg in (-90, 270) and rotate_270 is not None:
            return img.transpose(rotate_270)  # pyright: ignore[reportArgumentType]
    # PIL < 10 fallback (rare; the chess-ocr image uses 12+).
    legacy_90 = getattr(Image, "ROTATE_90", None)
    legacy_270 = getattr(Image, "ROTATE_270", None)
    if deg == 90 and legacy_90 is not None:
        return img.transpose(legacy_90)  # pyright: ignore[reportAttributeAccessIssue]
    if deg in (-90, 270) and legacy_270 is not None:
        return img.transpose(legacy_270)  # pyright: ignore[reportAttributeAccessIssue]
    return img.rotate(-deg, resample=Image.Resampling.BILINEAR, expand=True)


# -----------------------------------------------------------------------------
# CLAHE (private — port from enhanced-png)
# -----------------------------------------------------------------------------


def _apply_clahe(img: Image.Image) -> Image.Image:
    """Tile-based CLAHE on L channel, returns RGB.

    Preserves gradients (doesn't threshold) but lifts shadows. Pure
    numpy — no cv2 / opencv dependency.
    """
    h, w = img.size[1], img.size[0]
    tile = _CLAHE_TILE
    if tile < 8:
        tile = 8
    tiles_y = max(1, (h + tile - 1) // tile)
    tiles_x = max(1, (w + tile - 1) // tile)
    n_bins = 256
    clip = max(1, int(_CLAHE_CLIP_LIMIT * (tile * tile) / n_bins))

    gray_u8 = np.asarray(img.convert("L"), dtype=np.uint8)

    def tile_eq(sub: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(sub, bins=n_bins, range=(0.0, 256.0))
        excess = np.maximum(hist - clip, 0).sum()
        if excess > 0:
            redistribution = excess // n_bins + 1
            hist = np.minimum(hist, clip) + redistribution
        cdf = hist.cumsum()
        if cdf[-1] == 0:
            return sub.copy()
        cdf = (cdf * (n_bins - 1) / cdf[-1]).astype(np.uint8)
        return cdf[sub]

    h_pad = (tile - (h % tile)) % tile
    w_pad = (tile - (w % tile)) % tile
    padded = np.pad(gray_u8, ((0, h_pad), (0, w_pad)), mode="reflect")
    out_full = np.empty_like(padded)
    for ty in range(tiles_y):
        y0 = ty * tile
        for tx in range(tiles_x):
            x0 = tx * tile
            out_full[y0 : y0 + tile, x0 : x0 + tile] = tile_eq(
                padded[y0 : y0 + tile, x0 : x0 + tile]
            )
    eq = out_full[:h, :w]

    # Replace L channel; keep original RGB chroma so pen-ink colour
    # distinctions (black vs blue) still help the verifier.
    gray_orig = np.asarray(img.convert("L"), dtype=np.uint8)
    rgb = np.asarray(img.convert("RGB"), dtype=np.int16)
    delta = eq.astype(np.int16) - gray_orig.astype(np.int16)
    rgb = np.clip(rgb + delta[:, :, None], 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


# -----------------------------------------------------------------------------
# Bilateral denoise (private — port from enhanced-png)
# -----------------------------------------------------------------------------


def _apply_bilateral(img: Image.Image) -> Image.Image:
    """Edge-preserving bilateral filter on L channel; applied at 0.5x
    resolution for speed, then upsampled. Pure numpy; safe for M3
    because it smooths noise without destroying edges or gradients.
    """
    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    h, w = arr.shape

    # Work at half resolution to keep the O(N * radius²) loop bounded.
    h_half = max(8, h // 2)
    w_half = max(8, w // 2)
    small = Image.fromarray(arr, mode="L").resize((w_half, h_half), Image.Resampling.LANCZOS)
    small_arr = np.asarray(small, dtype=np.float32)

    radius = max(1, round(2.5 * _BILATERAL_SIGMA_SPATIAL))
    spatial = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial[dy + radius, dx + radius] = math.exp(
                -(dx * dx + dy * dy) / (2.0 * _BILATERAL_SIGMA_SPATIAL * _BILATERAL_SIGMA_SPATIAL)
            )

    out = np.zeros_like(small_arr)
    weights_sum = np.zeros_like(small_arr)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(small_arr, dy, axis=0), dx, axis=1)
            intensity_diff = small_arr - shifted
            range_w = np.exp(
                -(intensity_diff * intensity_diff)
                / (2.0 * _BILATERAL_SIGMA_RANGE * _BILATERAL_SIGMA_RANGE)
            )
            sp = spatial[dy + radius, dx + radius]
            out += sp * range_w * shifted
            weights_sum += sp * range_w
    out /= np.maximum(weights_sum, 1e-6)
    denoised_small = np.clip(out, 0, 255).astype(np.uint8)

    # Upsample to original size.
    denoised_pil = Image.fromarray(denoised_small, mode="L").resize(
        (w, h), Image.Resampling.BILINEAR
    )
    # Apply the per-pixel delta to original RGB so chroma is preserved.
    delta_l = np.asarray(denoised_pil, dtype=np.int16) - arr.astype(np.int16)
    rgb = np.asarray(img.convert("RGB"), dtype=np.int16)
    rgb = np.clip(rgb + delta_l[:, :, None], 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")
