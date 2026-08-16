"""
Anima model adapter.

Anima is Cosmos Predict2's MiniTrainDIT + LLMAdapter text preprocessor.
Uses VideoRopePosition3DEmb (precomputed rotation matrices) for position encoding,
which requires the matrix_rope tiling engine.
Text encoder: Qwen3 0.6B (text-only, no vision).
"""

import torch

from .base import ModelAdapterBase

# Anima's prompt template, applied by the workflow's prompt node
_ANIMA_PROMPT_PREFIX = (
    "You are an assistant designed to generate high quality anime images "
    "based on textual prompts. <Prompt Start> "
)

_ANIMA_NEGATIVE_PREFIX = (
    "You are an assistant designed to generate low-quality images "
    "based on textual prompts <Prompt Start> "
)

class AnimaAdapter(ModelAdapterBase):
    name = "anima"
    display_name = "Anima"
    tiling_engine = "matrix_rope"

    supports_image_conditioning = False   # Qwen3 0.6B text-only
    use_prompt_prefix = False
    context_key = "context"
    context_dim = 1024
    position_dims = 3
    vae_compression = 8
    packing_size = 2  # patch_spatial
    cross_tile_context_index = 0

    # ---- AnimaLLLite structural conditioning (internal config) ----
    lllite_grayscale = True          # convert control image to grayscale
    lllite_strength = 1.0            # FiLM modulation strength
    # Sigma window where LLLite is ACTIVE. AnimaLLLitePatch self-gates on these.
    lllite_start_percent = 0.0     # turns ON at 10% denoise if 0.1
    lllite_end_percent = 0.3     # turns OFF at 90% denoise if 0.9

    # ---- L-shape cross-tile context sigma gating (internal config) ----
    # Active in the EARLY denoise phase (cross-tile consistency), staggered so
    # it never overlaps LLLite (avoids the sequence-length mismatch).
    context_start_percent = 0.4     # context ON at 10% denoise if 0.1
    context_end_percent = 0.6      # context OFF at 85% denoise if 0.85
    # L-shape context component weights — tune here to isolate effects.
    thumb_weight = 1.0     # 2D whole-image thumbnail <- no impact
    strip_weight = 1.0    # horizontal strip + vertical strip
    context_jitter = 1.0    # RoPE position jitter # normalized: 1.0 = one full context-cell of jitter, 0.2 = 20%

    def detect(self, diffusion_model) -> bool:
        cls_name = type(diffusion_model).__name__
        if cls_name in ("MiniTrainDIT", "GeneralDIT"):
            return True
        return (
            hasattr(diffusion_model, "pos_embedder")
            and hasattr(diffusion_model, "x_embedder")
            and hasattr(diffusion_model, "blocks")
            and hasattr(diffusion_model, "final_layer")
            and hasattr(diffusion_model.pos_embedder, "dim_spatial_range")
        )

    def get_patch_size(self, diffusion_model) -> int:
        return getattr(diffusion_model, "patch_spatial", 2)

    def get_embedding_layer(self, diffusion_model):
        return getattr(diffusion_model, "x_embedder", None)

    def _encode_anima(self, clip, text):
        """Encode via Anima's CLIP with return_dict=True to get T5-XXL metadata,
        then restructure into ComfyUI's [[tensor, dict], ...] format."""
        tokens = clip.tokenize(text)
        cond = clip.encode_from_tokens(tokens, return_dict=True)
        conditioning = [[
            cond["cond"],
            {k: v for k, v in cond.items() if k != "cond"}
        ]]
        return conditioning

    def encode_positive_conditioning(self, clip, prompt, upscaled_image, image_resolution,
                                      visual_strength=1.0):
        prompt_text = str(prompt or "").strip()
        if self.use_prompt_prefix:
            full_prompt = _ANIMA_PROMPT_PREFIX + prompt_text
        else:
            full_prompt = prompt_text
        return self._encode_anima(clip, full_prompt)

    def encode_negative_conditioning(self, clip, positive_conditioning):
        if self.use_prompt_prefix:
            return self._encode_anima(clip, _ANIMA_NEGATIVE_PREFIX)
        else:
            return self._encode_anima(clip, "")

    def encode_tile_conditioning(self, clip, prompt, prepared_image):
        return None  # text-only, no per-tile visual conditioning

    # Anima-specific accessors used by the matrix_rope engine
    def get_pos_embedder(self, diffusion_model):
        return getattr(diffusion_model, "pos_embedder", None)

    def get_concat_padding_mask(self, diffusion_model) -> bool:
        return getattr(diffusion_model, "concat_padding_mask", True)

    def get_patch_temporal(self, diffusion_model) -> int:
        return getattr(diffusion_model, "patch_temporal", 1)

    def has_extra_pos_embedder(self, diffusion_model) -> bool:
        return getattr(diffusion_model, "extra_pos_embedder", None) is not None