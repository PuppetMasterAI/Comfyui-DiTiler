"""
AnimaLLLite tiled conditioning handler.
Manages per-tile control-image cropping and the AnimaLLLite patch objects.
Registered patches mirror AnimaLLLiteApply, but the DiTiler swaps the control
image per tile instead of using a single full-image control.
"""
import torch
from comfy.ldm.anima.lllite import (
    AnimaLLLitePatch, AnimaLLLiteAttentionPatch, AnimaLLLiteMLPPatch,
)


class AnimaLLLiteTiling:
    def __init__(self, model_patch, control_image, adapter,
                 strength=1.0, start_percent=0.0, end_percent=1.0):
        self.model_patch = model_patch
        self.lllite_model = model_patch.model
        self.adapter = adapter
        self.base_strength = strength
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.control_image = self._prepare_control(control_image)
        # Sigma window always-active; the tiling engine gates via patch.strength
        self.patch = AnimaLLLitePatch(
            model_patch, self.control_image, None,
            strength, float('inf'), 0.0,
        )
        self.attn_patch = AnimaLLLiteAttentionPatch(
            self.patch, {"q": "self_attn_q_proj", "k": "self_attn_k_proj", "v": "self_attn_v_proj"})
        self.attn2_patch = AnimaLLLiteAttentionPatch(self.patch, {"q": "cross_attn_q_proj"})
        self.mlp_patch = AnimaLLLiteMLPPatch(self.patch)

    def _prepare_control(self, image):
        """Convert the control image to the channel layout the checkpoint expects.
        Grayscale = luminance replicated across channels (R=G=B), NOT collapsed to 1."""
        cond_in_channels = self.lllite_model.cond_in_channels

        if getattr(self.adapter, "lllite_grayscale", True):
            # Compute luminance
            if image.shape[-1] >= 3:
                gray = image[..., :3].mean(dim=-1, keepdim=True)   # (B, H, W, 1)
            else:
                gray = image[..., :1]
            # Replicate to match the checkpoint's expected channel count.
            # For cond_in_channels=3 this gives R=G=B=gray (colorless but 3-channel).
            return gray.repeat(1, 1, 1, cond_in_channels)          # (B, H, W, cond_in_channels)

        # Grayscale disabled: just match the checkpoint's channel count
        return image[..., :cond_in_channels]

    def set_tile_image(self, bbox, vae_scale):
        """Crop the control image to the tile's bbox; AnimaLLLitePatch will
        upscale it to the tile's pixel size during the forward pass."""
        px, py = bbox.x * vae_scale, bbox.y * vae_scale
        pw, ph = bbox.w * vae_scale, bbox.h * vae_scale
        self.patch.image = self.control_image[:, py:py + ph, px:px + pw, :]