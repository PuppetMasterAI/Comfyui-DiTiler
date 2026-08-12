"""
Krea2 model adapter.

Krea2 uses:
  - SingleStreamDiT architecture
  - Qwen3-VL multimodal text encoder (supports image+text conditioning)
  - patch attribute for patch size
  - 'first' attribute for the patch-embedding layer
  - 'context' key for cross-attention context in the conditioning dict
"""

from typing import Optional
import torch

from .base import ModelAdapterBase
from ..visual_conditioning import (
    prepare_image,
    ensure_conditioning_format,
    extract_conditioning_tensor,
    zero_conditioning,
)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_GLOBAL = (
    "The connected image shows the current state of the image being refined. "
    "Use it as the visual foundation and spatial context. The text prompt describes "
    "the desired refinement or additional detail to apply. Preserve all existing "
    "content, subjects, composition, and spatial relationships while applying the "
    "described refinement."
)

SYSTEM_PROMPT_TILE = (
    "The connected image shows one crop of the image being refined. "
    "Use it as the local visual foundation. Preserve all visible content, subjects, "
    "composition, and spatial relationships inside the crop while enhancing fine detail."
)

# Qwen3-VL chat template.
# The first {} is filled with the system prompt.
# The resulting {} in the user slot is filled by clip.tokenize with the actual
# tokenized user content (vision placeholder + prompt text).
LLAMA_TEMPLATE = (
    "<|im_start|>system\n{} <|im_end|>\n"
    "<|im_start|>user\n{{}} <|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Vision placeholder prefix for one reference image.
IMAGE_PAD = "<|vision_start|><|image_pad|><|vision_end|>\n"

# ---------------------------------------------------------------------------
# Qwen3-VL encode helper (Krea2-specific)
# ---------------------------------------------------------------------------

def encode_visual_conditioning(clip, prepared_image, prompt: str, system_prompt: str):
    """
    Encode an image + prompt with the multimodal CLIP/Qwen3-VL encoder.
    Returns the raw CLIP conditioning output.
    """
    prompt_text = str(prompt or "").strip()
    full_prompt = IMAGE_PAD + prompt_text
    llama_template = LLAMA_TEMPLATE.format(system_prompt)
    tokens = clip.tokenize(
        full_prompt,
        images=[prepared_image],
        llama_template=llama_template,
    )
    return clip.encode_from_tokens(tokens)

class Krea2Adapter(ModelAdapterBase):
    """Adapter for Krea2's SingleStreamDiT with Qwen3-VL encoder."""

    name = "krea2"
    display_name = "Krea2"
    tiling_engine = "ids_rope"  
    supports_image_conditioning = True
    context_key = "context"

    vae_compression = 8       # VAE: pixels/8 → latent
    packing_size = 2          # 2×2 packing: latent/2 → tokens
    context_dim = 30720 # txtlayers × txtdim 
    position_dims = 3

    cross_tile_context_index = 0
    
    def detect(self, diffusion_model) -> bool:
        """Detect Krea2 by class name or characteristic attributes."""
        cls_name = type(diffusion_model).__name__
        if cls_name == "SingleStreamDiT":
            return True
        return (
            hasattr(diffusion_model, "first")
            and hasattr(diffusion_model, "txtfusion")
            and hasattr(diffusion_model, "txtmlp")
        )

    def get_patch_size(self, diffusion_model) -> int:
        return getattr(diffusion_model, "patch", 2)

    def get_embedding_layer(self, diffusion_model):
        return getattr(diffusion_model, "first", None)
    def encode_positive_conditioning(self, clip, prompt, upscaled_image, image_resolution,
                                    visual_strength=1.0):
        """Encode positive conditioning via Qwen3-VL multimodal encoding.
        visual_strength scales the encoding resolution: 1.0 = full, 0.0 = text-only."""
        if visual_strength <= 0.0:
            prompt_text = str(prompt or "").strip()
            tokens = clip.tokenize(prompt_text)
            cond = clip.encode_from_tokens(tokens)
            # Track token info
            tensor = cond[0][0] if isinstance(cond[0][0], torch.Tensor) else None
            total = self._get_seq_len(tensor) if tensor is not None else 0
            self.last_conditioning_info = {"text_tokens": total, "img_tokens": 0}
            return self.ensure_cond_format(cond)

        effective_resolution = max(256, int(image_resolution * visual_strength))
        image = prepare_image(upscaled_image, effective_resolution)
        raw = encode_visual_conditioning(clip, image, prompt, SYSTEM_PROMPT_GLOBAL)

        # Track token info: get sequence length from the correct dimension
        tensor = raw[0][0] if isinstance(raw[0][0], torch.Tensor) else None
        total = self._get_seq_len(tensor) if tensor is not None else 0

        # Qwen3-VL vision tokens: estimate from image resolution, but never exceed total
        img_tokens_estimate = (effective_resolution // 28) ** 2
        img_tokens = min(img_tokens_estimate, total)
        text_tokens = max(0, total - img_tokens)

        self.last_conditioning_info = {"text_tokens": text_tokens, "img_tokens": img_tokens}
        return self.ensure_cond_format(raw)


    def encode_negative_conditioning(self, clip, positive_conditioning):
        """Negative conditioning is zeroed positive (Krea2 convention)."""
        from ..visual_conditioning import zero_conditioning
        return self.ensure_cond_format(zero_conditioning(positive_conditioning))

    def encode_tile_conditioning(self, clip, prompt, prepared_image):
        """Encode per-tile conditioning using Qwen3-VL with the tile crop."""
        raw = encode_visual_conditioning(
            clip, prepared_image, prompt, self.get_tile_system_prompt()
        )
        return extract_conditioning_tensor(raw)

    def normalize_conditioning_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Krea2 conditioning tensors from Qwen3-VL are already in the expected
        (seq, features) format. No normalization needed.
        """
        return tensor

    def get_tile_system_prompt(self) -> str:
        return SYSTEM_PROMPT_TILE

    def get_global_system_prompt(self) -> str:
        return SYSTEM_PROMPT_GLOBAL

    @staticmethod
    def _get_seq_len(tensor: torch.Tensor) -> int:
        """Extract sequence length from a conditioning tensor, handling both
        2D (seq, features) and 3D (batch, seq, features) formats."""
        if tensor is None:
            return 0
        if tensor.dim() == 3:
            return tensor.shape[1]  # (batch, seq_len, features)
        elif tensor.dim() == 2:
            return tensor.shape[0]  # (seq_len, features)
        return 0