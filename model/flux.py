"""
Flux model adapter.

Flux uses:
  - Flux DiT architecture (DoubleStreamBlock + SingleStreamBlock)
  - T5-based text encoder (text-only, no image conditioning)
  - patch_size attribute for patch size
  - 'img_in' attribute for the patch-embedding layer
  - 'c_crossattn' key for cross-attention context in the conditioning dict

Per-tile visual conditioning is disabled because Flux's text encoder
is text-only (T5) and cannot process images.
"""

from typing import Optional
import torch

from .base import ModelAdapterBase


class FluxAdapter(ModelAdapterBase):
    """Adapter for Flux DiT models (Flux.1, Flux2, etc.)."""

    name = "flux"
    display_name = "Flux"
    tiling_engine = "ids_rope"  
    supports_image_conditioning = False
    context_key = "c_crossattn"

    vae_compression = 16      # VAE: pixels/16 → latent
    packing_size = 1          # no additional packing (patch_size=1)
    context_dim = 12288
    position_dims = 4

    cross_tile_context_index = 0
    
    def detect(self, diffusion_model) -> bool:
        """Detect Flux by class name or characteristic attributes."""
        cls_name = type(diffusion_model).__name__
        if cls_name == "Flux":
            return True
        return (
            hasattr(diffusion_model, "img_in")
            and hasattr(diffusion_model, "txt_in")
            and hasattr(diffusion_model, "double_blocks")
            and hasattr(diffusion_model, "single_blocks")
        )

    def get_patch_size(self, diffusion_model) -> int:
        return getattr(diffusion_model, "patch_size", 2)

    def get_embedding_layer(self, diffusion_model):
        return getattr(diffusion_model, "img_in", None)

    def encode_positive_conditioning(self, clip, prompt, upscaled_image, image_resolution,
                                    visual_strength=1.0):
        """
        Encode positive conditioning using standard CLIP text encoding.
        Flux's text encoder (T5) is text-only — the upscaled_image is ignored.
        Returns the raw ComfyUI conditioning format from clip.encode_from_tokens
        WITHOUT any reshaping.
        """
        prompt_text = str(prompt or "").strip()
        tokens = clip.tokenize(prompt_text)
        return self.ensure_cond_format(clip.encode_from_tokens(tokens))

    def encode_negative_conditioning(self, clip, positive_conditioning):
        """
        Negative conditioning is an empty-prompt encode (Flux convention).
        Returns the raw ComfyUI conditioning format WITHOUT any reshaping.
        """
        tokens = clip.tokenize("")
        return self.ensure_cond_format(clip.encode_from_tokens(tokens))

    def encode_tile_conditioning(self, clip, prompt, prepared_image):
        """
        Per-tile conditioning is not supported for Flux (text-only encoder).
        Returns None to indicate the tiler should skip per-tile conditioning.
        """
        return None

    def get_tile_system_prompt(self) -> str:
        return ""

    def get_global_system_prompt(self) -> str:
        return ""

    def normalize_conditioning_tensor(self, tensor):
        if tensor.dim() == 2:
            return tensor.unsqueeze(1)  # (1, features) -> (1, 1, features)
        return tensor