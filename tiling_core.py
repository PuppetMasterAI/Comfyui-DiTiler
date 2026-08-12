"""
Shared tiling utilities for the Unified DiTiler.

Contains the core geometry, blending, and debug infrastructure used by both
tiling engines (ids_rope and matrix_rope).
"""

import torch
from torch import Tensor
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ceildiv(a: int, b: int) -> int:
    """Ceiling integer division."""
    return -(a // -b)


def repeat_to_batch_size(tensor: Tensor, batch_size: int) -> Tensor:
    """Repeat or truncate a tensor along dim 0 to match batch_size."""
    b = tensor.shape[0]
    if b == batch_size:
        return tensor
    if b == 1:
        return tensor.expand(batch_size, *tensor.shape[1:])
    if b > batch_size:
        return tensor.narrow(0, 0, batch_size)
    reps = ceildiv(batch_size, b)
    return tensor.repeat(reps, *([1] * (tensor.dim() - 1))).narrow(0, 0, batch_size)


# ---------------------------------------------------------------------------
# Bounding boxes and tile splitting
# ---------------------------------------------------------------------------

class BBox:
    """A tile bounding box with a precomputed slicer for N-dimensional tensors."""
    __slots__ = ("x", "y", "w", "h", "slicer")

    def __init__(self, x: int, y: int, w: int, h: int, ndim: int = 4):
        self.x, self.y, self.w, self.h = x, y, w, h
        n_lead = ndim - 2
        self.slicer = tuple([slice(None)] * n_lead) + (slice(y, y + h), slice(x, x + w))


_gauss_cache = {}


def gaussian_weight(w: int, h: int, device, ndim: int = 4) -> Tensor:
    """Cached 2D Gaussian feather weight for seamless tile blending."""
    key = (w, h, ndim, str(device))
    if key in _gauss_cache:
        return _gauss_cache[key]
    y = torch.linspace(-1, 1, h, device=device)
    x = torch.linspace(-1, 1, w, device=device)
    gy = torch.exp(-(y ** 2) / 0.6)
    gx = torch.exp(-(x ** 2) / 0.6)
    g = (gy[:, None] * gx[None, :]).clamp(min=1e-3)
    g = g.view(*([1] * (ndim - 2)), h, w)
    _gauss_cache[key] = g
    return g


def split_bboxes(w: int, h: int, tile_w: int, tile_h: int,
                 overlap_x: int, overlap_y: int,
                 device, ndim: int = 4) -> Tuple[List[BBox], Tensor]:
    """Split a canvas into overlapping tile bounding boxes with a Gaussian weight map.

    Args:
        w, h: Canvas dimensions (latent space).
        tile_w, tile_h: Tile dimensions (latent space).
        overlap_x, overlap_y: Overlap between tiles in each axis (latent space).
        device: Torch device for the weight map.
        ndim: Number of dimensions of the tensor being tiled (4 or 5).

    Returns:
        (bboxes, weight_map): List of BBox and the accumulated Gaussian weight map.
    """
    tile_w, tile_h = min(tile_w, w), min(tile_h, h)
    overlap_x = max(0, min(overlap_x, tile_w - 4))
    overlap_y = max(0, min(overlap_y, tile_h - 4))

    cols = ceildiv(max(w - overlap_x, 1), max(tile_w - overlap_x, 1))
    rows = ceildiv(max(h - overlap_y, 1), max(tile_h - overlap_y, 1))

    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    weight_shape = tuple([1] * (ndim - 2)) + (h, w)
    weight = torch.zeros(weight_shape, device=device, dtype=torch.float32)

    bboxes: List[BBox] = []
    for row in range(rows):
        y = min(int(round(row * dy)), h - tile_h)
        for col in range(cols):
            x = min(int(round(col * dx)), w - tile_w)
            bbox = BBox(x, y, tile_w, tile_h, ndim=ndim)
            bboxes.append(bbox)
            weight[bbox.slicer] += gaussian_weight(tile_w, tile_h, device, ndim=ndim)

    return bboxes, weight


# ---------------------------------------------------------------------------
# Tile grid computation
# ---------------------------------------------------------------------------

def compute_tile_grid(canvas_w_latent, canvas_h_latent, tile_size_px, overlap_fraction,
                      align_px, min_tile_px, compression):
    """Compute tile dimensions, overlap, and grid count for a canvas.

    Canvas dimensions are in LATENT space. Tile size, alignment, and minimum are in
    PIXEL space and converted to latent internally using the compression factor.
    The longer canvas side gets tile_size; the shorter side adapts proportionally
    to preserve the canvas aspect ratio.

    Args:
        canvas_w_latent: Canvas width in latent pixels.
        canvas_h_latent: Canvas height in latent pixels.
        tile_size_px: Max tile side in IMAGE pixels (applied to the longer canvas side).
        overlap_fraction: Overlap fraction (0.0-1.0) of the tile side.
        align_px: Pixel alignment for clean token boundaries (vae_compression * packing_size).
        min_tile_px: Minimum tile side in IMAGE pixels.
        compression: VAE compression factor (pixel to latent).

    Returns:
        (tile_w, tile_h, overlap_x, overlap_y, cols, rows) - all in latent units.
    """
    # Convert pixel-space parameters to latent space
    tile_size_latent = tile_size_px // compression
    align_latent = max(1, align_px // compression)
    min_tile_latent = max(1, min_tile_px // compression)

    # Determine which side is longer and compute proportional tile dimensions
    if canvas_w_latent >= canvas_h_latent:
        # Landscape or square: width is the longer side
        tile_w = min(tile_size_latent, canvas_w_latent)
        tile_h = max(1, int(round(tile_size_latent * canvas_h_latent / canvas_w_latent)))
    else:
        # Portrait: height is the longer side
        tile_h = min(tile_size_latent, canvas_h_latent)
        tile_w = max(1, int(round(tile_size_latent * canvas_w_latent / canvas_h_latent)))

    # Align to token boundaries (round down to nearest multiple of align_latent)
    tile_w = max(align_latent, (tile_w // align_latent) * align_latent)
    tile_h = max(align_latent, (tile_h // align_latent) * align_latent)

    # Enforce minimum tile size (but never exceed the canvas)
    tile_w = max(tile_w, min(min_tile_latent, canvas_w_latent))
    tile_h = max(tile_h, min(min_tile_latent, canvas_h_latent))

    # Final clamp to canvas
    tile_w = min(tile_w, canvas_w_latent)
    tile_h = min(tile_h, canvas_h_latent)

    # Compute overlap from the fraction
    overlap_x = max(0, int(tile_w * overlap_fraction))
    overlap_y = max(0, int(tile_h * overlap_fraction))

    # Ensure overlap doesn't consume the entire tile
    overlap_x = min(overlap_x, max(0, tile_w - align_latent))
    overlap_y = min(overlap_y, max(0, tile_h - align_latent))

    # Coverage-based tile count
    if tile_w >= canvas_w_latent:
        cols = 1
        overlap_x = 0
    else:
        cols = max(1, ceildiv(canvas_w_latent - overlap_x, tile_w - overlap_x))

    if tile_h >= canvas_h_latent:
        rows = 1
        overlap_y = 0
    else:
        rows = max(1, ceildiv(canvas_h_latent - overlap_y, tile_h - overlap_y))

    return tile_w, tile_h, overlap_x, overlap_y, cols, rows


# ---------------------------------------------------------------------------
# Debug / info printing
# ---------------------------------------------------------------------------

def print_run_info(H, W, vae_scale, tile_w, tile_h, n_tiles, patch_size,
                   overlap_x=0, overlap_y=0,
                   ctx_tokens=0, cond_text_tokens=None, cond_img_tokens=None):
    """Print the tiling target info and token counts.

    Args:
        H, W: Canvas dimensions in latent space.
        vae_scale: VAE compression factor (latent → pixel).
        tile_w, tile_h: Tile dimensions in latent space.
        n_tiles: Number of tiles.
        patch_size: Model patch size (latent → token).
        overlap_x, overlap_y: Overlap in latent space.
        ctx_tokens: Number of cross-tile context tokens.
        cond_text_tokens: Number of text conditioning tokens (None = unknown).
        cond_img_tokens: Number of image conditioning tokens (None or 0 = text-only).
    """
    px_h, px_w = H * vae_scale, W * vae_scale
    tile_tokens = (tile_h // patch_size) * (tile_w // patch_size)

    # Overlap info
    if overlap_x > 0 or overlap_y > 0:
        overlap_str = f", overlap {overlap_y * vae_scale}x{overlap_x * vae_scale}px"
    else:
        overlap_str = ", overlap 0px (edge-to-edge)"

    msg = (
        f"[DiTiler] target {px_h}x{px_w}px ({H}x{W} latent, "
        f"{n_tiles} tile(s), tile {tile_h * vae_scale}x{tile_w * vae_scale}px"
        f"{overlap_str})"
    )
    if n_tiles == 1:
        msg += "  [!] Only 1 tile -- tiling NOT happening."
    print(msg)

    # Token count line
    cond_str = ""
    if cond_text_tokens is not None:
        if cond_img_tokens is not None and cond_img_tokens > 0:
            cond_str = f", conditioning {cond_text_tokens} text + {cond_img_tokens} img"
        else:
            cond_str = f", conditioning {cond_text_tokens} text"

    print(f"[DiTiler] tokens: {tile_tokens} img, {ctx_tokens} context{cond_str}")