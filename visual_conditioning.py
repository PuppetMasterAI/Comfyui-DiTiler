"""
Visual conditioning helpers for Unified DiTiler.

This module contains no ComfyUI node registration.
It only provides helper functions used by the unified detailer node.
"""

import torch
import comfy.utils


# ---------------------------------------------------------------------------
# Conditioning normalization helpers
# ---------------------------------------------------------------------------

def _to_2d_conditioning_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Normalize a conditioning tensor to:

        (seq, features)

    Accepts:
        (features,)
        (seq, features)
        (1, seq, features)
        (B, seq, features) -> takes first batch item
    """
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor conditioning, got {type(t)}")

    if t.dim() == 1:
        return t.unsqueeze(0)

    if t.dim() == 2:
        return t

    if t.dim() == 3:
        if t.shape[0] == 1:
            return t.squeeze(0)
        return t[0]

    raise ValueError(f"Unsupported conditioning tensor shape {tuple(t.shape)}")


def extract_conditioning_tensor(cond):
    """
    Extract the raw conditioning tensor from common ComfyUI/CLIP return formats.

    Handles:
        tensor
        [tensor]
        [tensor, dict]
        [[tensor, dict], ...]
        tuple(tensor, dict)
        list of token tensors
    """
    if isinstance(cond, torch.Tensor):
        return _to_2d_conditioning_tensor(cond)

    if isinstance(cond, (list, tuple)):
        if len(cond) == 0:
            raise ValueError("Empty conditioning returned by CLIP encoder.")

        first = cond[0]

        # Already ComfyUI conditioning entries: [[tensor, dict], ...]
        if isinstance(first, (list, tuple)):
            if len(first) == 0:
                raise ValueError("Empty conditioning entry returned by CLIP encoder.")
            return _to_2d_conditioning_tensor(first[0])

        # Raw tensor output: [tensor] / [tensor, dict] / (tensor, dict)
        if isinstance(first, torch.Tensor):
            if len(cond) > 1 and isinstance(cond[1], dict):
                return _to_2d_conditioning_tensor(first)

            if len(cond) == 1:
                return _to_2d_conditioning_tensor(first)

            # Rare fallback: list of token tensors.
            if all(isinstance(x, torch.Tensor) for x in cond):
                if all(x.dim() == 1 for x in cond):
                    return torch.stack(cond, dim=0)
                if all(x.dim() == 2 for x in cond):
                    return torch.cat(cond, dim=0)

            return _to_2d_conditioning_tensor(first)

    raise ValueError(f"Unsupported conditioning type: {type(cond)}")


def ensure_conditioning_format(cond):
    """
    Normalize CLIP output into ComfyUI conditioning format:

        [
            [tensor, dict],
            ...
        ]
    """
    if isinstance(cond, torch.Tensor):
        return [[_to_2d_conditioning_tensor(cond), {}]]

    if isinstance(cond, (list, tuple)):
        if len(cond) == 0:
            return []

        # Already valid ComfyUI conditioning entries.
        if all(
            isinstance(entry, (list, tuple))
            and len(entry) > 0
            and isinstance(entry[0], torch.Tensor)
            for entry in cond
        ):
            normalized = []
            for entry in cond:
                tensor = _to_2d_conditioning_tensor(entry[0])
                meta = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
                normalized.append([tensor, dict(meta) if isinstance(meta, dict) else {}])
            return normalized

        # Raw encoder output: [tensor] / [tensor, dict] / (tensor, dict)
        if isinstance(cond[0], torch.Tensor):
            tensor = _to_2d_conditioning_tensor(cond[0])
            meta = cond[1] if len(cond) > 1 and isinstance(cond[1], dict) else {}
            return [[tensor, dict(meta) if isinstance(meta, dict) else {}]]

    # Fallback.
    return [[extract_conditioning_tensor(cond), {}]]


def zero_conditioning(conditioning):
    """
    Zero conditioning tensors for negative conditioning.

    Metadata dicts are copied, and tensor values inside dicts are zeroed too.
    """
    zeroed = []

    for entry in conditioning:
        cond = entry[0]

        rest = []
        for item in entry[1:]:
            if isinstance(item, dict):
                new_item = {}
                for k, v in item.items():
                    if isinstance(v, torch.Tensor):
                        new_item[k] = torch.zeros_like(v)
                    else:
                        new_item[k] = v
                rest.append(new_item)
            else:
                rest.append(item)

        zeroed.append([torch.zeros_like(cond)] + rest)

    return zeroed


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def prepare_image(image: torch.Tensor, resolution: int) -> torch.Tensor:
    """
    Resize the image to fit within resolution, keep aspect ratio, ensure 3 channels.

    Input/output: ComfyUI IMAGE format (B, H, W, C).
    """
    image = image[:1]

    samples = image.movedim(-1, 1)  # BHWC -> BCHW
    h, w = samples.shape[2], samples.shape[3]

    if h <= 0 or w <= 0:
        raise ValueError("Cannot prepare an empty image for conditioning.")

    scale = resolution / max(h, w)
    new_h = max(16, round(h * scale))
    new_w = max(16, round(w * scale))

    samples = comfy.utils.common_upscale(samples, new_w, new_h, "area", "disabled")

    if samples.shape[1] == 1:
        samples = samples.repeat(1, 3, 1, 1)
    elif samples.shape[1] > 3:
        samples = samples[:, :3]

    return samples.movedim(1, -1)  # BCHW -> BHWC


def prepare_image_exact(image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Resize image to an exact target size and ensure 3 channels.

    Used for tile crops after the first tile has defined the conditioning image size,
    so all tile conditionings have the same sequence length.
    """
    image = image[:1]

    target_h = max(16, int(target_h))
    target_w = max(16, int(target_w))

    samples = image.movedim(-1, 1)  # BHWC -> BCHW
    samples = comfy.utils.common_upscale(samples, target_w, target_h, "area", "disabled")

    if samples.shape[1] == 1:
        samples = samples.repeat(1, 3, 1, 1)
    elif samples.shape[1] > 3:
        samples = samples[:, :3]

    return samples.movedim(1, -1)  # BCHW -> BHWC


def crop_image_to_bbox(
    image: torch.Tensor,
    bbox,
    latent_w: int,
    latent_h: int,
    compression: int,
) -> torch.Tensor:
    """
    Crop the upscaled image using a latent-space bbox.

    Uses the actual image/latent size ratio, falling back to the VAE compression
    factor if needed.
    """
    image = image[:1]

    img_h, img_w = image.shape[1], image.shape[2]

    scale_x = img_w / float(latent_w) if latent_w > 0 else float(compression)
    scale_y = img_h / float(latent_h) if latent_h > 0 else float(compression)

    x0 = int(round(bbox.x * scale_x))
    y0 = int(round(bbox.y * scale_y))
    x1 = int(round((bbox.x + bbox.w) * scale_x))
    y1 = int(round((bbox.y + bbox.h) * scale_y))

    x0 = max(0, min(x0, img_w - 1))
    y0 = max(0, min(y0, img_h - 1))

    x1 = max(x0 + 1, min(x1, img_w))
    y1 = max(y0 + 1, min(y1, img_h))

    return image[:, y0:y1, x0:x1, :]
