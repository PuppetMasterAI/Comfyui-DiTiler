"""
IDs-RoPE Tiling Engine for Unified DiTiler

Handles DiT models where position information is stored as explicit per-token
position IDs (e.g. Krea2's SingleStreamDiT, Flux's DiT). Tiling works by
modifying the img_ids via post_input patches.

Model-agnostic: works with any model using explicit position IDs for RoPE.
"""

import random
import math
import torch
from torch import Tensor
from typing import List, Tuple

from .tiling_core import BBox, split_bboxes, gaussian_weight, repeat_to_batch_size, ceildiv, print_run_info

# ---------------------------------------------------------------------------
# Position axes helper
# ---------------------------------------------------------------------------

def _match_position_axes(ids: Tensor, target_axes: int) -> Tensor:
    """Pad or trim the last (position-axes) dimension of ids to target_axes."""
    cur = ids.shape[-1]
    if cur == target_axes:
        return ids
    if cur < target_axes:
        pad_shape = list(ids.shape)
        pad_shape[-1] = target_axes - cur
        pad = torch.zeros(pad_shape, device=ids.device, dtype=ids.dtype)
        return torch.cat([ids, pad], dim=-1)
    return ids[..., :target_axes]


# ---------------------------------------------------------------------------
# post_input patches
# ---------------------------------------------------------------------------

def make_tile_offset_patch(offsets: List[Tuple[int, int]]):
    def post_input_patch(io_dict):
        imgpos = io_dict.get("img_ids")
        if imgpos is None:
            return io_dict
        b = imgpos.shape[0]
        if b != len(offsets):
            return io_dict
        new_imgpos = imgpos.clone()
        for i, (row_off, col_off) in enumerate(offsets):
            if row_off == 0 and col_off == 0:
                continue
            new_imgpos[i, :, 1] += row_off
            new_imgpos[i, :, 2] += col_off
        io_dict["img_ids"] = new_imgpos
        return io_dict
    return post_input_patch


def make_extra_tokens_patch(extra_tokens: Tensor, extra_ids: Tensor):
    def post_input_patch(io_dict):
        img = io_dict.get("img")
        imgpos = io_dict.get("img_ids")
        if img is None or imgpos is None or extra_tokens is None:
            return io_dict
        b = img.shape[0]
        ctx = extra_tokens.to(device=img.device, dtype=img.dtype)
        ctx_ids = extra_ids.to(device=imgpos.device, dtype=imgpos.dtype)
        if ctx.shape[0] != b:
            ctx = repeat_to_batch_size(ctx, b)
        if ctx_ids.shape[0] != b:
            ctx_ids = repeat_to_batch_size(ctx_ids, b)

        # Ensure position-axes dimension matches the model's img_ids.
        ctx_ids = _match_position_axes(ctx_ids, imgpos.shape[-1])

        io_dict["img"] = torch.cat([img, ctx], dim=1)
        io_dict["img_ids"] = torch.cat([imgpos, ctx_ids], dim=1)
        return io_dict
    return post_input_patch


def make_debug_sequence_patch(state, bbox, row_off, col_off, ctx_token_count,
                              batch_item_idx, patch_size):
    """One-shot diagnostic: prints the position-ID layout of one tile's sequence."""
    def _fmt(ids_2d):
        n = ids_2d.shape[0]
        if n == 0:
            return "(0 tokens) [empty]"

        def one(v):
            return f"[{float(v[0]):.2f}, {float(v[1]):.2f}, {float(v[2]):.2f}]"

        if n <= 10:
            return f"({n} tokens) [" + ", ".join(one(v) for v in ids_2d) + "]"
        first = ", ".join(one(v) for v in ids_2d[:5])
        last = ", ".join(one(v) for v in ids_2d[-5:])
        return f"({n} tokens) first5=[{first}] last5=[{last}]"

    def post_input_patch(io_dict):
        if state.get("printed"):
            return io_dict
        img_ids = io_dict.get("img_ids")
        txt_ids = io_dict.get("txt_ids")
        if img_ids is None:
            return io_dict
        state["printed"] = True

        total_img = img_ids.shape[1]
        tile_count = total_img - ctx_token_count
        bi = batch_item_idx
        tile_ids = img_ids[bi, :tile_count].detach().cpu()
        ctx_ids = img_ids[bi, tile_count:tile_count + ctx_token_count].detach().cpu()

        print("=" * 95)
        print(f"[DiTiler][SEQ DEBUG] chosen tile bbox(x={bbox.x}, y={bbox.y}, "
              f"w={bbox.w}, h={bbox.h})  offset(row={row_off}, col={col_off})  patch={patch_size}")
        if txt_ids is not None and txt_ids.shape[0] > bi:
            print(f"  text tokens       {_fmt(txt_ids[bi].detach().cpu())}")
        print(f"  tile img tokens   {_fmt(tile_ids)}")
        if ctx_token_count > 0:
            print(f"  context tokens    {_fmt(ctx_ids)}")
        print(f"  TOTAL img-seq len = {total_img}  (tile {tile_count} + ctx {ctx_token_count})")
        print("=" * 95)
        return io_dict
    return post_input_patch


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class IdsRopeTilingImpl:
    """Tiling engine for models with explicit position IDs (img_ids).
    Modifies img_ids via post_input patches to give each tile its true position."""

    def __init__(
        self,
        tile_width,
        tile_height,
        overlap_x,
        overlap_y,
        tile_batch_size,
        patch_size,
        use_cross_tile_context=False,
        cross_tile_context=0.0,
        cross_tile_context_index=0,
        first_layer=None,
        source_latent=None,
        debug=False,
        tile_contexts=None,
        adapter=None,
        context_dim=0,
        position_dims=3,
        vae_scale=8,
    ):
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y
        self.tile_batch_size = tile_batch_size
        self.patch_size = patch_size
        self.debug = debug
        self.use_cross_tile_context = use_cross_tile_context
        self.cross_tile_context = cross_tile_context
        self.cross_tile_context_index = cross_tile_context_index
        self.first_layer = first_layer
        self._warned_no_context = False
        self._warned_context_capped = False
        self._cross_tile_ctx_h = self._cross_tile_ctx_w = None
        self._printed_context_info = False

        # Clean source latent for cross-tile overview.
        self.source_latent = source_latent

        # Model adapter and config.
        self.adapter = adapter
        self.context_dim = context_dim
        self.position_dims = position_dims
        self.vae_scale = vae_scale

        self._debug_state = {"printed": False}
        self._debug_seq_state = {"printed": False}

        # Per-tile visual conditioning support.
        self.tile_contexts = tile_contexts
        self._warned_tile_contexts = False
        self._printed_tile_context_replace = False

        self.h = self.w = self.ndim = None
        self.bboxes: List[BBox] = []
        self.weight_map: Tensor = None

    # ------------------------------------------------------------------
    # Per-tile conditioning replacement
    # ------------------------------------------------------------------

    def _apply_tile_contexts(self, c_tile, bboxes, batch_idx, N, device):
        if self.tile_contexts is None:
            return

        if len(self.tile_contexts) != len(self.bboxes):
            if not self._warned_tile_contexts:
                print(f"[DiTiler] tile_contexts count {len(self.tile_contexts)} "
                      f"does not match tile count {len(self.bboxes)}. Using incoming context.")
                self._warned_tile_contexts = True
            return

        start = batch_idx * self.tile_batch_size
        end = start + len(bboxes)
        ctxs = self.tile_contexts[start:end]

        if len(ctxs) != len(bboxes):
            return

        context_key = None
        if self.adapter is not None and self.adapter.context_key in c_tile:
            context_key = self.adapter.context_key
        elif "context" in c_tile:
            context_key = "context"
        else:
            for k, v in c_tile.items():
                if isinstance(v, torch.Tensor) and v.dim() in (2, 3) and v.shape[-1] == self.context_dim:
                    context_key = k
                    break

        if context_key is None:
            return

        original_context = c_tile.get(context_key)
        dtype = original_context.dtype if isinstance(original_context, torch.Tensor) else None

        batched = []
        for ctx in ctxs:
            t = ctx
            if t.dim() == 1:
                t = t.unsqueeze(0)
            if t.dim() == 2:
                t = t.unsqueeze(0)
            elif t.dim() == 3:
                if t.shape[0] != 1:
                    t = t[:1]
            else:
                return

            if dtype is not None:
                t = t.to(device=device, dtype=dtype)
            else:
                t = t.to(device=device)

            if t.shape[0] != N:
                t = repeat_to_batch_size(t, N)
            batched.append(t)

        try:
            c_tile[context_key] = torch.cat(batched, dim=0)
        except RuntimeError:
            return

        if not self._printed_tile_context_replace:
            print(f"[DiTiler] Per-tile context active: replacing "
                  f"'{context_key}' with shape {tuple(c_tile[context_key].shape)}")
            self._printed_tile_context_replace = True

    # ------------------------------------------------------------------
    # Cross-tile context helpers
    # ------------------------------------------------------------------

    def _strip_overlapping_context_tokens(self, tokens, ids, bboxes, p):
        """Remove cross-tile context tokens whose scaled positions fall inside any tile."""
        if tokens is None or ids is None:
            return None, None

        row_positions = ids[0, :, 1].detach().float().cpu()
        col_positions = ids[0, :, 2].detach().float().cpu()

        keep_mask = torch.ones(row_positions.shape[0], dtype=torch.bool, device="cpu")

        for bbox in bboxes:
            row_off = bbox.y // p
            col_off = bbox.x // p
            tile_h_tokens = bbox.h // p
            tile_w_tokens = bbox.w // p

            row_in_tile = (row_positions >= row_off) & (row_positions < row_off + tile_h_tokens)
            col_in_tile = (col_positions >= col_off) & (col_positions < col_off + tile_w_tokens)
            keep_mask &= ~(row_in_tile & col_in_tile)

        if keep_mask.sum().item() == 0:
            return None, None

        return (
            tokens[:, keep_mask.to(device=tokens.device), :],
            ids[:, keep_mask.to(device=ids.device), :],
        )

    def _build_cross_tile_context_tokens(self, x_in: Tensor):
        if self.first_layer is None:
            if not self._warned_no_context:
                print("[DiTiler] cross_tile_context requested but the model's "
                      "embedding layer wasn't found -- skipping.")
                self._warned_no_context = True
            return None, None

        # Ensure geometry is initialized (defensive guard)
        if self._cross_tile_ctx_h is None or self._cross_tile_ctx_w is None:
            self._init_cross_tile_context_geometry(self.h, self.w)

        p = self.patch_size
        ctx_h, ctx_w = self._cross_tile_ctx_h, self._cross_tile_ctx_w

        # Use clean source latent if available, otherwise the working latent.
        if self.source_latent is not None:
            x = self.source_latent
        else:
            x = x_in

        temporal = x.ndim == 5
        if temporal:
            b, c, t, h, w = x.shape
            x = x.reshape(b * t, c, h, w)

        target_h, target_w = ctx_h * p, ctx_w * p
        x_small = torch.nn.functional.adaptive_avg_pool2d(x, (target_h, target_w))

        bN, C = x_small.shape[0], x_small.shape[1]
        patched = x_small.reshape(bN, C, ctx_h, p, ctx_w, p)
        patched = patched.permute(0, 2, 4, 1, 3, 5).reshape(bN, ctx_h * ctx_w, C * p * p)

        weight_dtype = next(self.first_layer.parameters()).dtype
        weight_device = next(self.first_layer.parameters()).device
        tokens = self.first_layer(patched.to(device=weight_device, dtype=weight_dtype))

        # Positions stay in the WORKING latent's coordinate space.
        full_h, full_w = self.h // p, self.w // p
        row_centers = (
            torch.arange(ctx_h, device=tokens.device, dtype=torch.float32) + 0.5
        ) * (full_h / ctx_h)
        col_centers = (
            torch.arange(ctx_w, device=tokens.device, dtype=torch.float32) + 0.5
        ) * (full_w / ctx_w)

        ids = torch.zeros(ctx_h, ctx_w, self.position_dims, device=tokens.device, dtype=torch.float32)
        ids[..., 0] = self.cross_tile_context_index
        ids[..., 1] = row_centers[:, None]
        ids[..., 2] = col_centers[None, :]
        ids = ids.reshape(1, ctx_h * ctx_w, self.position_dims).repeat(tokens.shape[0], 1, 1)

        return tokens, ids

    # ------------------------------------------------------------------
    # Init / geometry
    # ------------------------------------------------------------------

    def _maybe_init(self, x_in: Tensor):
        H, W = x_in.shape[-2], x_in.shape[-1]
        ndim = x_in.ndim
        if self.h == H and self.w == W and self.ndim == ndim and self.bboxes:
            return
        self.h, self.w, self.ndim = H, W, ndim
        device = x_in.device
        self.bboxes, self.weight_map = split_bboxes(
            W, H, self.tile_width, self.tile_height,
            self.overlap_x, self.overlap_y,
            device, ndim=ndim,
        )

        # Init context geometry BEFORE printing so token counts are available
        if self.use_cross_tile_context:
            self._init_cross_tile_context_geometry(H, W)

        ctx_tokens = 0
        if self.use_cross_tile_context and self._cross_tile_ctx_h is not None:
            ctx_tokens = self._cross_tile_ctx_h * self._cross_tile_ctx_w

        print_run_info(
            H, W, self.vae_scale,
            self.bboxes[0].w, self.bboxes[0].h,
            len(self.bboxes), self.patch_size,
            overlap_x=self.overlap_x,
            overlap_y=self.overlap_y,
            ctx_tokens=ctx_tokens,
            cond_text_tokens=getattr(self, '_cond_text_tokens', None),
            cond_img_tokens=getattr(self, '_cond_img_tokens', None),
        )

    def _init_cross_tile_context_geometry(self, H, W):
        p = self.patch_size
        full_h, full_w = H // p, W // p

        # Area-based, aspect-ratio-preserving context grid calculation
        max_tokens = full_h * full_w
        target_tokens = self.cross_tile_context * max_tokens

        aspect = full_h / full_w
        ctx_w = max(2, round(math.sqrt(target_tokens / aspect)))
        ctx_h = max(2, round(ctx_w * aspect))

        ctx_h = min(ctx_h, full_h)
        ctx_w = min(ctx_w, full_w)

        source_desc = "working latent (noised)"
        capped = False

        if self.source_latent is not None:
            ref_H, ref_W = self.source_latent.shape[-2], self.source_latent.shape[-1]
            ref_full_h = max(1, ref_H // p)
            ref_full_w = max(1, ref_W // p)
            source_desc = (
                f"source_latent (clean, {ref_H}x{ref_W} latent, "
                f"native {ref_full_h}x{ref_full_w} tokens)"
            )
            if ctx_h > ref_full_h or ctx_w > ref_full_w:
                capped = True
                if not self._warned_context_capped:
                    print(
                        f"[DiTiler] WARNING: cross_tile_context={self.cross_tile_context:.2f} "
                        f"requests a {ctx_h}x{ctx_w} overview grid, but the connected "
                        f"source latent only contains a {ref_full_h}x{ref_full_w} native token "
                        f"grid. Capping overview to the source latent resolution."
                    )
                    self._warned_context_capped = True
                ctx_h = max(1, min(ctx_h, ref_full_h))
                ctx_w = max(1, min(ctx_w, ref_full_w))

        self._cross_tile_ctx_h, self._cross_tile_ctx_w = ctx_h, ctx_w   

    # ------------------------------------------------------------------
    # Main wrapper entry point
    # ------------------------------------------------------------------

    def __call__(self, model_function, args):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]

        self._maybe_init(x_in)

        N = x_in.shape[0]
        x_buffer = torch.zeros_like(x_in)

        context_tokens = context_ids = None
        if self.use_cross_tile_context:
            context_tokens, context_ids = self._build_cross_tile_context_tokens(x_in)

        num_batches = ceildiv(len(self.bboxes), self.tile_batch_size)
        batches = [
            self.bboxes[i * self.tile_batch_size:(i + 1) * self.tile_batch_size]
            for i in range(num_batches)
        ]

        p = self.patch_size

        # Pick ONE random tile for the sequence debug, only on the first forward pass.
        chosen_batch_num = None
        chosen_local_idx = None
        if self.debug and not self._debug_seq_state["printed"]:
            chosen_global_idx = random.randint(0, len(self.bboxes) - 1)
            chosen_batch_num = chosen_global_idx // self.tile_batch_size
            chosen_local_idx = chosen_global_idx % self.tile_batch_size

        for batch_idx, bboxes in enumerate(batches):
            x_tile = torch.cat([x_in[bbox.slicer] for bbox in bboxes], dim=0)
            t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])

            c_tile = {}
            for k, v in c_in.items():
                if k == "transformer_options":
                    continue
                if isinstance(v, torch.Tensor):
                    if self.adapter is not None:
                        v = self.adapter.normalize_conditioning_tensor(v)
                    if v.dim() >= 1 and v.shape[0] != x_tile.shape[0]:
                        v = repeat_to_batch_size(v, x_tile.shape[0])
                c_tile[k] = v

            transformer_options = dict(c_in.get("transformer_options", {}))
            patches = dict(transformer_options.get("patches", {}))
            post_input = list(patches.get("post_input", []))

            offsets_per_batch_item = []
            for bbox in bboxes:
                row_off = bbox.y // p
                col_off = bbox.x // p
                offsets_per_batch_item.extend([(row_off, col_off)] * N)

            post_input.append(make_tile_offset_patch(offsets_per_batch_item))

            ctx_count = 0
            if context_tokens is not None:
                ctx_t, ctx_ids_stripped = self._strip_overlapping_context_tokens(
                    context_tokens,
                    context_ids,
                    bboxes,
                    p,
                )
                if ctx_t is not None:
                    post_input.append(make_extra_tokens_patch(ctx_t, ctx_ids_stripped))
                    ctx_count = ctx_t.shape[1]

            # Sequence debug: register LAST so it sees the fully assembled sequence.
            if (
                self.debug
                and not self._debug_seq_state["printed"]
                and batch_idx == chosen_batch_num
            ):
                chosen_bbox = bboxes[chosen_local_idx]
                c_row_off = chosen_bbox.y // p
                c_col_off = chosen_bbox.x // p
                batch_item_idx = chosen_local_idx * N
                post_input.append(
                    make_debug_sequence_patch(
                        self._debug_seq_state,
                        chosen_bbox,
                        c_row_off,
                        c_col_off,
                        ctx_count,
                        batch_item_idx,
                        p,
                    )
                )

            patches["post_input"] = post_input
            transformer_options["patches"] = patches
            c_tile["transformer_options"] = transformer_options

            # Per-tile visual conditioning replacement.
            if getattr(self, "tile_contexts", None) is not None:
                self._apply_tile_contexts(c_tile, bboxes, batch_idx, N, x_tile.device)

            x_tile_out = model_function(x_tile, t_tile, **c_tile)

            for i, bbox in enumerate(bboxes):
                w = gaussian_weight(bbox.w, bbox.h, x_in.device, ndim=x_in.ndim)
                x_buffer[bbox.slicer] += x_tile_out[i * N:(i + 1) * N] * w

            del x_tile, x_tile_out, c_tile

        x_buffer = x_buffer / self.weight_map.clamp(min=1e-6)
        return x_buffer