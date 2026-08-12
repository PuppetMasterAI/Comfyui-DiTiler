"""
Unified DiTiler

RoPE-aware tiled diffusion for DiT image models in ComfyUI.

Model-agnostic: auto-detects the loaded model's architecture and dispatches
to the appropriate tiling engine:
  - ids_rope engine (Krea2, Flux): explicit position IDs modified via post_input
  - matrix_rope engine (Anima/Cosmos): precomputed rotation matrices reconstructed
    via attn1_patch + spatial tensor extended via post_input

Simplified tile controls:
  tile_size    : maximum tile side in pixels (drives tile count, same for all models)
  tile_overlap : fraction of tile side used as overlap (0.01 - 0.5)

Cross-tile context:
  cross_tile_context : fraction of native resolution per axis (0.0 - 1.0)
  1.0 = full native grid, 0.2 = 20% of information tokens. Same logic for all models.
"""

import torch

from .model import detect_adapter

from .tiling_core import compute_tile_grid
from .ids_rope_tiling import IdsRopeTilingImpl
from .matrix_rope_tiling import MatrixRopeTilingImpl
from .visual_conditioning import (
    prepare_image,
    prepare_image_exact,
    crop_image_to_bbox,
    extract_conditioning_tensor,
)


MAX_RESOLUTION = 8192


class UnifiedDiTiler:
    """
    Unified tiled detailer node. https://github.com/PuppetMasterAI/Comfyui-DiTiler

    Wraps a DiT model so the KSampler runs it tile-by-tile with RoPE-aware
    position offsets, Gaussian-weighted blending, and optional context conditioning.
    Tile geometry is derived from tile_size (max tile side in pixels) and tile_overlap
    (fraction). Model architecture is auto-detected.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "upscaled_image": ("IMAGE",),
                "source_latent": ("LATENT", {
                    "tooltip": (
                        "latent (source) — the clean pre-upscale latent. "
                        "Used as the clean cross-tile overview source (ids_rope engine). "
                        "Not used by the matrix_rope engine."
                    ),
                }),
                "upscaled_latent": ("LATENT", {
                    "tooltip": (
                        "latent (upscaled) — the upscaled working latent. Defines the tile "
                        "grid and is forwarded to the detail KSampler."
                    ),
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "dynamic_prompts": True,
                    "default": "Preserve the visible content and enhance fine detail.",
                }),

                "tile_size": ("INT", {
                    "default": 1280,
                    "min": 512,
                    "max": MAX_RESOLUTION,
                    "step": 16,
                    "tooltip": (
                        "Maximum tile side in pixels. The node computes the tile grid and "
                        "aligned tile size automatically. Tiles will not exceed this size. "
                        "Same behaviour for all models."
                    ),
                }),
                "tile_overlap": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 0.4,
                    "step": 0.01,
                    "tooltip": (
                        "Fraction of the tile side used as overlap between adjacent tiles. "
                        "Higher values improve seam blending but may be adapted down if the "
                        "tile count cannot support the requested overlap."
                    ),
                }),
                "cross_tile_context": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Fraction of the maximum available overview tokens used for whole-image "
                        "awareness. 1.0 = full native resolution per axis. 0.2 = 20% of "
                        "information tokens. 0.0 = disabled. Same logic for all models."
                    ),
                }),
                "visual_conditioning": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Strength of per-tile visual conditioning for multimodal models (Krea2). "
                        "1.0 = full visual encoding at tile_size resolution. "
                        "0.5 = half-resolution encoding (fewer vision tokens, weaker constraint). "
                        "0.0 = disabled (text-only conditioning). "
                        "Automatically disabled for text-only models (Flux, Anima)."
                    ),
                }),
            },
            "optional": {
                "tile_batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": (
                        "Number of tiles processed per model call. Higher = faster but more VRAM. "
                        "Keep at 1 for matrix_rope models (shared RoPE offset state)."
                    ),
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Prints verbose per-run info. For matrix_rope models, includes an "
                        "automatic RoPE reconstruction self-test."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "positive", "negative", "upscaled_latent")
    FUNCTION = "apply"
    CATEGORY = "DiTiler"

    def apply(
        self,
        model,
        clip,
        upscaled_image,
        source_latent,
        upscaled_latent,
        prompt,
        tile_size,
        tile_overlap,
        visual_conditioning,
        cross_tile_context,
        tile_batch_size=1,
        debug=False,
    ):
        # ------------------------------------------------------------------
        # 0. Auto-detect model adapter.
        # ------------------------------------------------------------------
        diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
        if diffusion_model is None:
            raise ValueError("[DiTiler] Could not access model.model.diffusion_model")

        adapter = detect_adapter(diffusion_model)
        adapter.debug = debug 
        compression = adapter.vae_compression

        print(
            f"[DiTiler] Using model adapter: {adapter.display_name} "
            f"(engine={adapter.tiling_engine}, vae_compression={compression}, "
            f"packing_size={adapter.packing_size})"
        )

        # ------------------------------------------------------------------
        # 1. Read working latent size and validate VAE compression.
        # ------------------------------------------------------------------
        post_samples = upscaled_latent["samples"]
        if post_samples.ndim == 5:
            _, _, _, H, W = post_samples.shape
        else:
            _, _, H, W = post_samples.shape

        img_h, img_w = upscaled_image.shape[1], upscaled_image.shape[2]
        actual_comp_h = img_h / H
        actual_comp_w = img_w / W
        if abs(actual_comp_h - compression) > 1.0 or abs(actual_comp_w - compression) > 1.0:
            print(
                f"[DiTiler] WARNING: Expected VAE compression {compression}x "
                f"but detected {actual_comp_h:.1f}x(H) / {actual_comp_w:.1f}x(W). "
                f"Image: {img_h}x{img_w}px, Latent: {H}x{W}. "
                f"Tile calculations may be incorrect."
            )

        src_samples = source_latent["samples"] if source_latent is not None else None
        if src_samples is not None and src_samples.shape[1] != post_samples.shape[1]:
            print(
                f"[DiTiler] WARNING: source_latent has {src_samples.shape[1]} channels "
                f"but upscaled_latent has {post_samples.shape[1]} channels. "
                f"They may come from different VAEs."
            )

        # ------------------------------------------------------------------
        # 2. Compute tile grid (coverage-based, same for all models).
        # ------------------------------------------------------------------
        align_px = adapter.vae_compression * adapter.packing_size
        tile_w, tile_h, overlap_x, overlap_y, cols, rows = compute_tile_grid(
            canvas_w_latent=W,
            canvas_h_latent=H,
            tile_size_px=tile_size,
            overlap_fraction=tile_overlap,
            align_px=align_px,
            min_tile_px=512,
            compression=compression,
        )

        # ------------------------------------------------------------------
        # 3. Build global positive/negative conditioning via adapter.
        # ------------------------------------------------------------------
        positive = adapter.encode_positive_conditioning(
            clip, prompt, upscaled_image, tile_size,
            visual_strength=visual_conditioning, 
        )
        negative = adapter.encode_negative_conditioning(clip, positive)

        # Extract conditioning token info for the debug print
        cond_text_tokens = None
        cond_img_tokens = None
        for entry in positive:
            if isinstance(entry, (list, tuple)) and len(entry) > 0:
                tensor = entry[0]
                if isinstance(tensor, torch.Tensor):
                    total = tensor.shape[0] if tensor.dim() == 2 else tensor.shape[1]
                    if adapter.supports_image_conditioning and visual_conditioning > 0.0:
                        # Multimodal: estimate image tokens from adapter's last encode info
                        info = getattr(adapter, 'last_conditioning_info', None)
                        if info:
                            cond_text_tokens = info.get("text_tokens", total)
                            cond_img_tokens = info.get("img_tokens", 0)
                        else:
                            cond_text_tokens = total
                            cond_img_tokens = 0
                    else:
                        cond_text_tokens = total
                        cond_img_tokens = 0
                    break
        
        # ------------------------------------------------------------------
        # 4. Encode per-tile visual conditioning (multimodal models only).
        # ------------------------------------------------------------------
        tile_conditioning_active = (visual_conditioning > 0.0) and adapter.supports_image_conditioning
        if visual_conditioning > 0.0 and not adapter.supports_image_conditioning:
            print(f"[DiTiler] {adapter.display_name} does not support image conditioning -- "
                f"per-tile visual conditioning disabled.")

        tile_contexts = None
        if tile_conditioning_active:
            from .tiling_core import split_bboxes, BBox
            bboxes, _ = split_bboxes(
                W, H,
                tile_w, tile_h,
                overlap_x, overlap_y,                  # ← both overlaps
                torch.device("cpu"),
                ndim=4,
            )

            tile_contexts = []
            first_tile_size = None
            for bbox in bboxes:
                crop = crop_image_to_bbox(
                    upscaled_image,
                    bbox,
                    latent_w=W,
                    latent_h=H,
                    compression=compression,
                )

                effective_tile_resolution = max(256, int(tile_size * visual_conditioning))
                if first_tile_size is None:
                    prepared = prepare_image(crop, effective_tile_resolution)
                    first_tile_size = (prepared.shape[1], prepared.shape[2])
                else:
                    prepared = prepare_image_exact(
                        crop,
                        target_h=first_tile_size[0],
                        target_w=first_tile_size[1],
                    )

                tile_cond = adapter.encode_tile_conditioning(clip, prompt, prepared)
                if tile_cond is not None:
                    tile_ctx = extract_conditioning_tensor(tile_cond)
                    tile_ctx = tile_ctx.detach().cpu()
                    tile_contexts.append(tile_ctx)

        # ------------------------------------------------------------------
        # 5. Prepare model wrapper dependencies.
        # ------------------------------------------------------------------
        patch_size = adapter.get_patch_size(diffusion_model)
        use_cross_tile_context = cross_tile_context > 0.0

        source_tensor = None
        if source_latent is not None:
            source_tensor = source_latent["samples"]

        # ------------------------------------------------------------------
        # 6. Construct the appropriate tiling engine and patch the model.
        # ------------------------------------------------------------------
        model = model.clone()

        if adapter.tiling_engine == "ids_rope":
            first_layer = None
            if use_cross_tile_context:
                first_layer = adapter.get_embedding_layer(diffusion_model)
                if first_layer is None:
                    print(
                        f"[DiTiler] cross_tile_context requested but couldn't find "
                        f"the {adapter.display_name} embedding layer -- disabling."
                    )
                    use_cross_tile_context = False

            impl = IdsRopeTilingImpl(
                tile_width=tile_w,
                tile_height=tile_h,
                overlap_x=overlap_x,
                overlap_y=overlap_y,
                tile_batch_size=tile_batch_size,
                patch_size=patch_size,
                use_cross_tile_context=use_cross_tile_context,
                cross_tile_context=cross_tile_context,
                cross_tile_context_index=adapter.cross_tile_context_index,
                first_layer=first_layer,
                source_latent=source_tensor,
                debug=debug,
                tile_contexts=tile_contexts if tile_conditioning_active else None,
                adapter=adapter,
                context_dim=adapter.context_dim,
                position_dims=adapter.position_dims,
                vae_scale=compression,
            )

            impl._cond_text_tokens = cond_text_tokens
            impl._cond_img_tokens = cond_img_tokens
            model.set_model_unet_function_wrapper(impl)

        elif adapter.tiling_engine == "matrix_rope":
            pos_embedder = adapter.get_pos_embedder(diffusion_model)
            x_embedder = adapter.get_embedding_layer(diffusion_model)
            concat_padding_mask = adapter.get_concat_padding_mask(diffusion_model)
            patch_temporal = adapter.get_patch_temporal(diffusion_model)

            use_ctx = use_cross_tile_context
            if use_ctx and x_embedder is None:
                print(
                    f"[DiTiler] cross_tile_context requested but x_embedder "
                    f"not found -- disabling."
                )
                use_ctx = False
            if use_ctx and adapter.has_extra_pos_embedder(diffusion_model):
                print(
                    f"[DiTiler] checkpoint has extra_pos_embedder -- "
                    f"disabling cross_tile_context to avoid shape mismatch."
                )
                use_ctx = False

            impl = MatrixRopeTilingImpl(
                tile_width=tile_w,
                tile_height=tile_h,
                tile_overlap=min(overlap_x, overlap_y),
                tile_batch_size=tile_batch_size,
                patch_spatial=patch_size,
                patch_temporal=patch_temporal,
                pos_embedder=pos_embedder,
                x_embedder=x_embedder,
                use_cross_tile_context=use_ctx,
                cross_tile_context=cross_tile_context,
                cross_tile_context_index=adapter.cross_tile_context_index,
                debug=debug,
                concat_padding_mask=concat_padding_mask,
                vae_scale=compression,
                thumb_weight=getattr(adapter, 'thumb_weight', 1.0),  
                strip_weight=getattr(adapter, 'strip_weight', 1.0), 
                source_latent=source_tensor,
                context_jitter=getattr(adapter, 'context_jitter', 0.0),
            )

            impl._cond_text_tokens = cond_text_tokens
            impl._cond_img_tokens = cond_img_tokens
            model.set_model_unet_function_wrapper(impl)
            if pos_embedder is not None:
                model.set_model_attn1_patch(impl.attn1_patch)
            if use_ctx:
                model.set_model_post_input_patch(impl.post_input)

        else:
            raise ValueError(
                f"[DiTiler] Unknown tiling_engine: {adapter.tiling_engine!r}"
            )

        model.model_options["unified_ditiler"] = True

        return (model, positive, negative, upscaled_latent)


NODE_CLASS_MAPPINGS = {
    "UnifiedDiTiler": UnifiedDiTiler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnifiedDiTiler": "DiTiler (Tiled detailer for DiT)",
}