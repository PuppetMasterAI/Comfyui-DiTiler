"""
Matrix-RoPE Tiling Engine for Unified DiTiler

Handles DiT models where position information is baked into precomputed rotation
matrices (e.g. Cosmos Predict2's VideoRopePosition3DEmb) rather than stored as
explicit per-token position IDs.

Tiling mechanism:
  - RoPE offset: reconstruct rotation matrices via attn1_patch hook
  - Cross-tile context: extend spatial tensor via post_input hook (L-shape)
  - Model patches: set_model_unet_function_wrapper + set_model_attn1_patch + set_model_post_input_patch

Model-agnostic: works with any model using precomputed-matrix RoPE
(Anima/Cosmos Predict2 today, future Cosmos derivatives tomorrow).
"""

import math
import torch
from torch import Tensor
from typing import List, Tuple
from einops import rearrange, repeat

from .tiling_core import BBox, split_bboxes, gaussian_weight, repeat_to_batch_size, ceildiv, print_run_info


# ---------------------------------------------------------------------------
# RoPE reconstruction
# ---------------------------------------------------------------------------

def reconstruct_rope(pos_embedder, t_dim, h_dim, w_dim,
                     t_off, h_off, w_off, device, dtype=None):
    """Faithful reimplementation of VideoRopePosition3DEmb.generate_embeddings
    with a position offset baked in. Reads frequency buffers directly from the
    real pos_embedder instance."""
    seq_h = torch.arange(h_off, h_off + h_dim, dtype=torch.float, device=device)
    seq_w = torch.arange(w_off, w_off + w_dim, dtype=torch.float, device=device)
    seq_t = torch.arange(t_off, t_off + t_dim, dtype=torch.float, device=device)
    return reconstruct_rope_from_seqs(pos_embedder, seq_t, seq_h, seq_w, device, dtype=dtype)


def reconstruct_rope_from_seqs(pos_embedder, seq_t, seq_h, seq_w, device, dtype=None):
    """Core RoPE reconstruction from explicit position sequences (arbitrary float
    positions, not necessarily contiguous integers)."""
    h_theta = 10000.0 * pos_embedder.h_ntk_factor
    w_theta = 10000.0 * pos_embedder.w_ntk_factor
    t_theta = 10000.0 * pos_embedder.t_ntk_factor
    dim_spatial_range = pos_embedder.dim_spatial_range.to(device=device)
    dim_temporal_range = pos_embedder.dim_temporal_range.to(device=device)
    h_spatial_freqs = 1.0 / (h_theta ** dim_spatial_range)
    w_spatial_freqs = 1.0 / (w_theta ** dim_spatial_range)
    temporal_freqs = 1.0 / (t_theta ** dim_temporal_range)

    T, H, W = seq_t.shape[0], seq_h.shape[0], seq_w.shape[0]

    half_emb_h = torch.outer(seq_h, h_spatial_freqs)
    half_emb_w = torch.outer(seq_w, w_spatial_freqs)
    half_emb_t = torch.outer(seq_t, temporal_freqs)

    half_emb_h = torch.stack([torch.cos(half_emb_h), -torch.sin(half_emb_h),
                              torch.sin(half_emb_h), torch.cos(half_emb_h)], dim=-1)
    half_emb_w = torch.stack([torch.cos(half_emb_w), -torch.sin(half_emb_w),
                              torch.sin(half_emb_w), torch.cos(half_emb_w)], dim=-1)
    half_emb_t = torch.stack([torch.cos(half_emb_t), -torch.sin(half_emb_t),
                              torch.sin(half_emb_t), torch.cos(half_emb_t)], dim=-1)

    em_T_H_W_D = torch.cat([
        repeat(half_emb_t, "t d x -> t h w d x", h=H, w=W),
        repeat(half_emb_h, "h d x -> t h w d x", t=T, w=W),
        repeat(half_emb_w, "w d x -> t h w d x", t=T, h=H),
    ], dim=-2)

    out = rearrange(em_T_H_W_D, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()
    out = out.unsqueeze(0).unsqueeze(2)  # -> (1, L, 1, D_head, 2, 2)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MatrixRopeTilingImpl:
    """Tiling engine for models with precomputed rotation-matrix RoPE.

    Serves as model_function_wrapper (tiling loop + blending), attn1_patch
    (RoPE reconstruction with offset), and post_input (L-shape context extension).
    """

    def __init__(self, tile_width, tile_height, tile_overlap, tile_batch_size,
                 patch_spatial, patch_temporal, pos_embedder, x_embedder=None,
                 use_cross_tile_context=False, cross_tile_context=0.0,
                 cross_tile_context_index=0, debug=False,
                 concat_padding_mask=True, vae_scale=8,
                 thumb_weight=1.0, strip_weight=1.0,
                 source_latent=None, context_jitter=0.0):
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.tile_overlap = tile_overlap
        self.tile_batch_size = tile_batch_size
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.pos_embedder = pos_embedder
        self.x_embedder = x_embedder
        self.use_cross_tile_context = use_cross_tile_context
        self.cross_tile_context = cross_tile_context
        self.cross_tile_context_index = cross_tile_context_index
        self.debug = debug
        self.concat_padding_mask = concat_padding_mask
        self.vae_scale = vae_scale
        self.thumb_weight = thumb_weight 
        self.strip_weight = strip_weight 
        self.source_latent = source_latent 
        self.context_jitter = context_jitter 
        self._ctx_jitter_row = None 
        self._ctx_jitter_col = None  
        self._jitter_debug_printed_steps = set()

        # Token budget: context tokens capped at 1.0x tile tokens.
        # This prevents the quadratic thumb term from overwhelming the tile's
        # self-attention at high cross_tile_context values.
        self.max_context_ratio = 1.0

        self.h = self.w = self.ndim = None
        self.bboxes: List[BBox] = []
        self.weight_map: Tensor = None

        self._current_offset = (0, 0, 0)
        self._current_tile_shape = (1, 1, 1)
        self._current_context_col_centers = None
        self._current_context_row_centers = None
        self._current_context_shape = None
        self._current_bbox = None
        self._current_x_in = None

        self._debug_state = {"verified": False, "printed_apply": False,
                             "printed_fire": False, "printed_context": False,
                             "warned_batch": False}

    # ---------------------------------------------------------------- tiling

    def _maybe_init(self, x_in: Tensor):
        H, W = x_in.shape[-2], x_in.shape[-1]
        ndim = x_in.ndim
        if self.h == H and self.w == W and self.ndim == ndim and self.bboxes:
            return
        self.h, self.w, self.ndim = H, W, ndim
        device = x_in.device
        self.bboxes, self.weight_map = split_bboxes(
            W, H, self.tile_width, self.tile_height,
            self.tile_overlap, self.tile_overlap,
            device, ndim=ndim,
        )

        # Init context geometry BEFORE printing so token counts are available
        if self.use_cross_tile_context:
            self._init_cross_tile_context_geometry(H, W)

        # Pre-compute expected context tokens for the summary print
        ctx_tokens = 0
        if (self.use_cross_tile_context and self.cross_tile_context > 0
                and self.x_embedder is not None):
            ps = self.patch_spatial
            # Geometry is computed in working latent space
            ctx_h_latent, ctx_w_latent = self._compute_context_geometry(H, W)
            ctx_h_tok = ctx_h_latent // ps
            ctx_w_tok = ctx_w_latent // ps
            tile_h_tok = self.bboxes[0].h // ps
            tile_w_tok = self.bboxes[0].w // ps
            ctx_tokens = (tile_h_tok * ctx_w_tok) + (ctx_h_tok * tile_w_tok) + (ctx_h_tok * ctx_w_tok)

        print_run_info(
            H, W, self.vae_scale,
            self.bboxes[0].w, self.bboxes[0].h,
            len(self.bboxes), self.patch_spatial,
            overlap_x=self.tile_overlap,
            overlap_y=self.tile_overlap,
            ctx_tokens=ctx_tokens,
            cond_text_tokens=getattr(self, '_cond_text_tokens', None),
            cond_img_tokens=getattr(self, '_cond_img_tokens', None),
        )

        if len(self.bboxes) == 1:
            print("[DiTiler] [!] Only 1 tile -- tiling NOT happening.")

    # ------------------------------------------------ context geometry

    def _compute_context_geometry(self, H, W):
        """Compute the context grid so total context tokens ≈ cross_tile_context × tile tokens.

        cross_tile_context now directly controls the ratio of context tokens to the tile's
        own token count (0.1 = context is ~10% of the tile's tokens). This replaces the old
        'fraction of the native grid' interpretation, which produced a much higher effective
        ratio because h_strip/v_strip scale with the tile dimensions.
        """
        ps = self.patch_spatial
        tile_h_tok = self.bboxes[0].h // ps
        tile_w_tok = self.bboxes[0].w // ps
        tile_tokens = tile_h_tok * tile_w_tok

        full_h_tok = self.h // ps
        full_w_tok = self.w // ps

        # Target total context tokens
        target = self.cross_tile_context * tile_tokens

        # Context tokens for a ctx_h × ctx_w grid:
        #   h_strip = tile_h × ctx_w,  v_strip = ctx_h × tile_w,  thumb = ctx_h × ctx_w
        # Preserve the working latent's aspect ratio: ctx_h = ctx_w × aspect.
        aspect = full_h_tok / max(full_w_tok, 1)
        a = aspect
        b = aspect * tile_w_tok + tile_h_tok
        # Solve: a·ctx_w² + b·ctx_w − target = 0
        discriminant = b * b + 4 * a * target
        ctx_w_tok = (-b + math.sqrt(discriminant)) / (2 * a)
        ctx_h_tok = ctx_w_tok * aspect

        ctx_w_tok = max(1, int(round(ctx_w_tok)))
        ctx_h_tok = max(1, int(round(ctx_h_tok)))

        # Never exceed the native grid
        ctx_w_tok = min(ctx_w_tok, full_w_tok)
        ctx_h_tok = min(ctx_h_tok, full_h_tok)

        return ctx_h_tok * ps, ctx_w_tok * ps

    def _init_cross_tile_context_geometry(self, H, W):
        """Store the context geometry for later use. Actual computation is
        deferred to _compute_context_geometry which needs bboxes initialized."""
        pass  # Geometry is computed on-demand in _compute_context_geometry

    # ------------------------------------------------ cross-tile context

    def _build_context_embeddings(self, bbox):
        """Builds L-shape context: horizontal strip + vertical strip + 2D thumbnail.
        The cross_tile_context fraction is applied per-axis to the native resolution,
        then capped by the token budget to prevent attention domination."""

        x_in = self._current_x_in
        context_source = self.source_latent if self.source_latent is not None else x_in
        ps = self.patch_spatial

        # Context geometry is computed in the WORKING latent's coordinate space
        # (that's where RoPE positions and tile dimensions live).
        ctx_h_latent, ctx_w_latent = self._compute_context_geometry(self.h, self.w)

        # Source-latent region corresponding to the tile (scaled coordinates).
        native_H, native_W = context_source.shape[-2], context_source.shape[-1]
        if self.source_latent is not None:
            scale_h = native_H / self.h
            scale_w = native_W / self.w
            src_h_off = max(0, min(int(round(bbox.y * scale_h)), native_H - 1))
            src_w_off = max(0, min(int(round(bbox.x * scale_w)), native_W - 1))
            src_h_span = max(ps, min(int(round(bbox.h * scale_h)), native_H - src_h_off))
            src_w_span = max(ps, min(int(round(bbox.w * scale_w)), native_W - src_w_off))
        else:
            src_h_off, src_h_span = bbox.y, bbox.h
            src_w_off, src_w_span = bbox.x, bbox.w

        B, C, T = context_source.shape[0], context_source.shape[1], context_source.shape[2]

        weight_dtype = next(self.x_embedder.parameters()).dtype
        weight_device = next(self.x_embedder.parameters()).device

        def _embed(pooled):
            if self.concat_padding_mask:
                mask_ch = torch.zeros(pooled.shape[0], 1, pooled.shape[2],
                                    pooled.shape[3], pooled.shape[4],
                                    dtype=pooled.dtype, device=pooled.device)
                pooled = torch.cat([pooled, mask_ch], dim=1)
            return self.x_embedder(pooled.to(device=weight_device, dtype=weight_dtype))

        # 1. Horizontal strip: tile's rows from source, pooled in W.
        #    Target H = bbox.h (working latent) so it matches the tile's H tokens.
        row_band = context_source[:, :, :, src_h_off:src_h_off + src_h_span, :]
        band_bt = row_band.permute(0, 2, 1, 3, 4).reshape(B * T, C, src_h_span, native_W)
        h_strip_bt = torch.nn.functional.interpolate(
            band_bt, size=(bbox.h, ctx_w_latent), mode='bilinear', align_corners=False)
        h_strip = h_strip_bt.reshape(B, T, C, bbox.h, ctx_w_latent).permute(0, 2, 1, 3, 4)
        h_strip_emb = _embed(h_strip)

        # 2. Vertical strip: tile's columns from source, pooled in H.
        #    Target W = bbox.w (working latent) so it matches the tile's W tokens.
        col_band = context_source[:, :, :, :, src_w_off:src_w_off + src_w_span]
        band_bt = col_band.permute(0, 2, 1, 3, 4).reshape(B * T, C, native_H, src_w_span)
        v_strip_bt = torch.nn.functional.interpolate(
            band_bt, size=(ctx_h_latent, bbox.w), mode='bilinear', align_corners=False)
        v_strip = v_strip_bt.reshape(B, T, C, ctx_h_latent, bbox.w).permute(0, 2, 1, 3, 4)
        v_strip_emb = _embed(v_strip)

        # 3. Thumbnail: full source image pooled to context grid.
        full_bt = context_source.permute(0, 2, 1, 3, 4).reshape(B * T, C, native_H, native_W)
        thumb_bt = torch.nn.functional.interpolate(
            full_bt, size=(ctx_h_latent, ctx_w_latent), mode='bilinear', align_corners=False)
        thumb = thumb_bt.reshape(B, T, C, ctx_h_latent, ctx_w_latent).permute(0, 2, 1, 3, 4)
        thumb_emb = _embed(thumb)

        # RoPE center positions in WORKING latent token space
        ctx_w_tok = h_strip_emb.shape[3]
        ctx_h_tok = v_strip_emb.shape[2]
        full_w_tok = self.w // ps
        full_h_tok = self.h // ps

        col_centers = (torch.arange(ctx_w_tok, dtype=torch.float, device=weight_device) + 0.5) * (full_w_tok / ctx_w_tok)
        row_centers = (torch.arange(ctx_h_tok, dtype=torch.float, device=weight_device) + 0.5) * (full_h_tok / ctx_h_tok)

        # Mask overlapping regions (in working latent token space)
        w_off_tok = bbox.x // ps
        w_end_tok = w_off_tok + bbox.w // ps
        h_off_tok = bbox.y // ps
        h_end_tok = h_off_tok + bbox.h // ps

        h_overlap = (col_centers >= w_off_tok) & (col_centers < w_end_tok)
        if h_overlap.any():
            h_strip_emb = h_strip_emb.clone()
            h_strip_emb[:, :, :, h_overlap, :] = 0.0

        v_overlap = (row_centers >= h_off_tok) & (row_centers < h_end_tok)
        if v_overlap.any():
            v_strip_emb = v_strip_emb.clone()
            v_strip_emb[:, :, v_overlap, :, :] = 0.0

        thumb_ov_h = (row_centers >= h_off_tok) & (row_centers < h_end_tok)
        thumb_ov_w = (col_centers >= w_off_tok) & (col_centers < w_end_tok)
        thumb_overlap = thumb_ov_h[:, None] & thumb_ov_w[None, :]
        if thumb_overlap.any():
            thumb_emb = thumb_emb.clone()
            thumb_emb[:, :, thumb_overlap, :] = 0.0

        return {
            "h_strip": h_strip_emb, "v_strip": v_strip_emb, "thumb": thumb_emb,
            "col_centers": col_centers, "row_centers": row_centers,
            "ctx_h_tok": ctx_h_tok, "ctx_w_tok": ctx_w_tok,
        }

    def post_input(self, io_dict):
        if not (self.use_cross_tile_context and self.cross_tile_context > 0) or self.x_embedder is None:
            return io_dict
        # Skip entirely when both weights are zero — prevents attention dilution
        if self.thumb_weight <= 0.0 and self.strip_weight <= 0.0:
            return io_dict
        x = io_dict.get("img")
        if x is None or self._current_bbox is None:
            return io_dict

        ctx = self._build_context_embeddings(self._current_bbox)
        h_strip = ctx["h_strip"].to(device=x.device, dtype=x.dtype)
        v_strip = ctx["v_strip"].to(device=x.device, dtype=x.dtype)
        thumb = ctx["thumb"].to(device=x.device, dtype=x.dtype)

        if h_strip.shape[0] != x.shape[0]:
            h_strip = h_strip.expand(x.shape[0], *h_strip.shape[1:])
        if v_strip.shape[0] != x.shape[0]:
            v_strip = v_strip.expand(x.shape[0], *v_strip.shape[1:])
        if thumb.shape[0] != x.shape[0]:
            thumb = thumb.expand(x.shape[0], *thumb.shape[1:])

        # --- Apply component weights ---
        if self.strip_weight != 1.0:
            h_strip = h_strip * self.strip_weight
            v_strip = v_strip * self.strip_weight
        if self.thumb_weight != 1.0:
            thumb = thumb * self.thumb_weight
        # --- End weight application ---

        self._current_context_col_centers = ctx["col_centers"]
        self._current_context_row_centers = ctx["row_centers"]
        self._current_context_shape = (ctx["ctx_h_tok"], ctx["ctx_w_tok"])

        # Per-forward jitter: randomize context RoPE positions so the grid
        # pattern can't build up coherently across denoising steps.
        # Generated here (once per forward pass) so every attn1_patch layer
        # in this step sees the same jittered positions.
        if self.context_jitter > 0:
            ps = self.patch_spatial
            full_h_tok = self.h // ps
            full_w_tok = self.w // ps
            ctx_h_tok = max(ctx["ctx_h_tok"], 1)
            ctx_w_tok = max(ctx["ctx_w_tok"], 1)
            # context_jitter is a fraction of the context cell spacing (distance between
            # adjacent context tokens). 1.0 = jitter spans one full cell (±half cell),
            # 0.2 = ±20% of that. Scales automatically with the context grid size.
            row_spacing = full_h_tok / ctx_h_tok
            col_spacing = full_w_tok / ctx_w_tok
            self._ctx_jitter_row = (torch.rand_like(ctx["row_centers"]) - 0.5) * self.context_jitter * row_spacing
            self._ctx_jitter_col = (torch.rand_like(ctx["col_centers"]) - 0.5) * self.context_jitter * col_spacing
        else:
            self._ctx_jitter_row = None
            self._ctx_jitter_col = None

        top = torch.cat([x, h_strip], dim=3)
        bottom = torch.cat([v_strip, thumb], dim=3)
        combined = torch.cat([top, bottom], dim=2)

        self._current_context_col_centers = ctx["col_centers"]
        self._current_context_row_centers = ctx["row_centers"]
        self._current_context_shape = (ctx["ctx_h_tok"], ctx["ctx_w_tok"])

        io_dict["img"] = combined
        return io_dict

    # ---------------------------------------------------------- attn1_patch

    def attn1_patch(self, q, k, v, pe=None, attn_mask=None, extra_options=None):
        if pe is None or self.pos_embedder is None:
            return {}
        T, H, W = self._current_tile_shape

        t_off, h_off, w_off = self._current_offset
        context_active = (self._current_context_col_centers is not None
                          and self._current_context_row_centers is not None)

        if (t_off, h_off, w_off) == (0, 0, 0) and not context_active:
            return {}

        seq_t = torch.arange(t_off, t_off + T, dtype=torch.float, device=q.device)
        seq_h = torch.arange(h_off, h_off + H, dtype=torch.float, device=q.device)
        seq_w = torch.arange(w_off, w_off + W, dtype=torch.float, device=q.device)

        if context_active:
            ctx_row = self._current_context_row_centers.to(q.device)
            ctx_col = self._current_context_col_centers.to(q.device)
            if self._ctx_jitter_row is not None:
                ctx_row = ctx_row + self._ctx_jitter_row.to(q.device)
                ctx_col = ctx_col + self._ctx_jitter_col.to(q.device)

            # --- DEBUG: confirm jittered positions reach RoPE ---
            if self.debug and not self._debug_state.get("printed_jitter_apply"):
                self._debug_state["printed_jitter_apply"] = True
                print(f"[DiTiler][JITTER-APPLY] seq_h last3 (context)={ctx_row[-3:].tolist()} "
                      f"seq_w last3 (context)={ctx_col[-3:].tolist()} "
                      f"jitter_active={self._ctx_jitter_row is not None}")
            # --- END DEBUG ---

            seq_h = torch.cat([seq_h, ctx_row])
            seq_w = torch.cat([seq_w, ctx_col])

        new_pe = reconstruct_rope_from_seqs(self.pos_embedder, seq_t, seq_h, seq_w,
                                            q.device, dtype=pe.dtype)

        return {"pe": new_pe}

    # ---------------------------------------------------------------- call

    def __call__(self, model_function, args):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]

        self._maybe_init(x_in)
        N = x_in.shape[0]
        x_buffer = torch.zeros_like(x_in)

        self._current_timestep = float(t_in.flatten()[0].item())

        num_batches = ceildiv(len(self.bboxes), self.tile_batch_size)
        batches = [
            self.bboxes[i * self.tile_batch_size:(i + 1) * self.tile_batch_size]
            for i in range(num_batches)
        ]

        if self.tile_batch_size > 1 and self.debug and not self._debug_state["warned_batch"]:
            print("[DiTiler] [!] tile_batch_size > 1: RoPE offset is shared state. "
                  "Use tile_batch_size=1.")
            self._debug_state["warned_batch"] = True

        ps, pt = self.patch_spatial, self.patch_temporal

        for batch_idx, bboxes in enumerate(batches):
            x_tile = torch.cat([x_in[bbox.slicer] for bbox in bboxes], dim=0)
            t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])

            c_tile = {}
            for k, v in c_in.items():
                if k == "transformer_options":
                    continue
                if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] != x_tile.shape[0]:
                    v = repeat_to_batch_size(v, x_tile.shape[0])
                c_tile[k] = v

            # Forward transformer_options so model-level patches
            # (post_input for L-shape context, attn1_patch for RoPE offset) fire.
            c_tile["transformer_options"] = dict(c_in.get("transformer_options", {}))

            bbox = bboxes[0]
            self._current_offset = (0, bbox.y // ps, bbox.x // ps)
            t_tokens = x_in.shape[-3] // pt
            self._current_tile_shape = (t_tokens, bbox.h // ps, bbox.w // ps)
            self._current_bbox = bbox
            self._current_x_in = x_in
            self._current_context_col_centers = None
            self._current_context_row_centers = None
            self._current_context_shape = None

            x_tile_out = model_function(x_tile, t_tile, **c_tile)

            for i, bbox in enumerate(bboxes):
                w = gaussian_weight(bbox.w, bbox.h, x_in.device, ndim=x_in.ndim)
                x_buffer[bbox.slicer] += x_tile_out[i * N:(i + 1) * N] * w

            del x_tile, x_tile_out, c_tile

        x_buffer = x_buffer / self.weight_map.clamp(min=1e-6)
        return x_buffer