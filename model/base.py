"""
Abstract base class for model adapters.

Each adapter encapsulates the model-specific details needed by UnifiedDiTiler:
  - Patch size and embedding layer discovery
  - Conditioning encoding strategy (positive, negative, per-tile)
  - Whether the model's text encoder supports image conditioning
"""

from typing import Optional
import torch


class ModelAdapterBase:
    """Abstract base for model-specific adapters."""

    # Unique name used in the registry and model_type dropdown.
    name: str = "base"
    # Human-readable name for console output.
    display_name: str = "Base Model"
    tiling_engine: str = "ids_rope" # "ids_rope" or "matrix_rope"
    # Whether the model's text encoder supports multimodal (image+text) encoding.
    # If False, per-tile visual conditioning is disabled.
    supports_image_conditioning: bool = False
    # The key in the conditioning dict that holds the cross-attention context.
    context_key: str = "context"
    debug: bool = False  # 
    # VAE / patchification config
    vae_compression: int = 8      # pixels → latent spatial downscale factor
    packing_size: int = 1         # latent → token packing factor (patch_size)
    # dims
    context_dim: int = 0  # 0 = unknown
    position_dims: int = 3

    cross_tile_context_index: int = -1 # specifc to Krea2 atm
    
    def detect(self, diffusion_model) -> bool:
        """Return True if this adapter matches the given diffusion model."""
        raise NotImplementedError

    def get_patch_size(self, diffusion_model) -> int:
        """Return the spatial patch size used by the model."""
        raise NotImplementedError

    def get_embedding_layer(self, diffusion_model):
        """
        Return the patch-embedding layer that converts patchified latents
        into token-space embeddings.
        """
        raise NotImplementedError

    def encode_positive_conditioning(self, clip, prompt, upscaled_image, image_resolution,
                                      visual_strength=1.0):
        """
        Encode the positive conditioning for the full image.
        Returns ComfyUI conditioning format: [[tensor, dict], ...]
        """
        raise NotImplementedError

    def encode_negative_conditioning(self, clip, positive_conditioning):
        """
        Encode the negative conditioning.
        Returns ComfyUI conditioning format: [[tensor, dict], ...]
        """
        raise NotImplementedError

    def encode_tile_conditioning(self, clip, prompt, prepared_image):
        """
        Encode per-tile conditioning from a prepared tile crop image.
        Returns the raw conditioning tensor, or None if not supported.
        Only called when supports_image_conditioning is True.
        """
        raise NotImplementedError

    def get_tile_system_prompt(self) -> str:
        """Return the system prompt used for per-tile conditioning encoding."""
        raise NotImplementedError

    def get_global_system_prompt(self) -> str:
        """Return the system prompt used for global conditioning encoding."""
        raise NotImplementedError

    def get_position_dims_count(self, diffusion_model) -> int:
        """Return the number of RoPE position-ID axes the model expects."""
        params = getattr(diffusion_model, 'params', None)
        if params is not None and hasattr(params, 'axes_dim'):
            return len(params.axes_dim)
        return 3  # safe default
    def normalize_conditioning_tensor(self, tensor):
        return tensor  # default: no-op

    def ensure_cond_format(self, cond):
        """
        Guarantee conditioning is in ComfyUI format: [[tensor, dict], ...]
        Does NOT modify tensor dimensions — only fixes the outer list structure.
        """
        if isinstance(cond, torch.Tensor):
            return [[cond, {}]]
        if isinstance(cond, (list, tuple)):
            if len(cond) == 0:
                return []
            # Already [[tensor, dict], ...]
            if all(
                isinstance(entry, (list, tuple))
                and len(entry) > 0
                and isinstance(entry[0], torch.Tensor)
                for entry in cond
            ):
                return [
                    [entry[0], entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}]
                    for entry in cond
                ]
            # Flat [tensor, dict] — missing outer list
            if isinstance(cond[0], torch.Tensor):
                meta = cond[1] if len(cond) > 1 and isinstance(cond[1], dict) else {}
                return [[cond[0], meta]]
        return cond