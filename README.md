# ComfyUI Unified DiTiler

RoPE-aware tiled diffusion and visual conditioning for DiT image models in ComfyUI.

This package solves a specific problem: modern Diffusion Transformers (DiTs) encode spatial position via RoPE. Unlike UNet-style models, you cannot naively split a large latent into
tiles and denoise each one independently — every tile would "think" it's the top-left corner of its own small image, producing repetition and seam artifacts. These nodes give each tile its true position and feed it surrounding context so tiles blend into one coherent image.

---

## Installation

1. git clone https://github.com/PuppetMasterAI/Comfyui-DiTiler in `ComfyUI/custom_nodes/`.
2. Restart ComfyUI. You should see in the console: "[DiTiler] Loaded nodes: UnifiedDiTiler"

---

## Supported Models

| Model | Engine | Position Mechanism | Text Encoder | CFG |
|---|---|---|---|---|
| Krea2 | ids_rope | Explicit img_ids modified via post_input | Qwen3-VL (multimodal) | 1.0 (turbo) |
| Flux2 | ids_rope | Explicit img_ids modified via post_input | T5-XXL (text-only) | > 1.0 |
| Anima | matrix_rope | Precomputed rotation matrices reconstructed via attn1_patch | Qwen3 0.6B + LLMAdapter | > 1.0 |

## Contents

- [Architecture](#architecture)
- [Node: Unified DiTiler](#node-unified-ditiler)
- [File structure](#file-structure)
- [Installation](#installation)
- [Example workflow](#example-workflow)
- [Recommended settings](#recommended-settings)
- [Model-specific notes](#model-specific-notes)
- [Technical appendix](#technical-appendix)
- [Known limitations](#known-limitations)

## Architecture

### Two tiling engines

The Unified DiTiler dispatches to one of two tiling engines based on the detected model:

ids_rope engine (Krea2, Flux): Position is stored as explicit per-token img_ids.
Tiling works by modifying the IDs via post_input patches — adding the tile's true
(row, col) offset. Context tokens are appended to the flat sequence with arbitrary
position IDs.

matrix_rope engine (Anima/Cosmos): Position is baked into precomputed rotation
matrices (VideoRopePosition3DEmb). Tiling works by reconstructing the matrices via
attn1_patch with the offset baked in. Context is an L-shaped spatial extension
(horizontal strip + vertical strip + 2D thumbnail) concatenated via post_input.

### Model adapters

Each model has an adapter (model/krea2.py, model/flux.py, model/anima.py) that provides:
- Model detection logic
- Patch size, VAE compression, position dimensions
- Conditioning encoding (positive and negative)
- Embedding layer access for cross-tile context

The adapter's tiling_engine attribute determines which engine the Unified DiTiler uses.

## Node: Unified DiTiler

Category: DiTiler
Outputs: MODEL, CONDITIONING, CONDITIONING, LATENT

Wraps a DiT model so the KSampler runs it tile-by-tile with RoPE-aware position offsets,
Gaussian-weighted blending, and optional cross-tile context. Model architecture is
auto-detected.

### Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| model | MODEL | — | The DiT model to patch (auto-detected). |
| clip | CLIP | — | The model's CLIP encoder. |
| upscaled_image | IMAGE | — | The upscaled image for visual conditioning. |
| source_latent | LATENT | — | Clean pre-upscale latent (overview source for ids_rope). |
| upscaled_latent | LATENT | — | The upscaled working latent (defines tile grid). |
| prompt | STRING | — | The detail prompt. |
| tile_size | INT | 1280 | Max tile side in pixels. |
| tile_overlap | FLOAT | 0.1 | Fraction of tile side used as overlap. |
| visual_conditioning | FLOAT | 0.15 | Strength of per-tile visual conditioning (Krea2 only). 0.0 = disabled. |
| cross_tile_context | FLOAT | 0.5 | Fraction of native tokens for whole-image awareness. 0.0 = disabled. |
| tile_batch_size | INT | 1 | Tiles per model call. |
| debug | BOOLEAN | False | Verbose logging + RoPE self-test (matrix_rope). |

### Outputs

| Output | Type | Description |
|---|---|---|
| model | MODEL | Patched model. Wire to KSampler model. |
| positive | CONDITIONING | Encoded positive conditioning. Wire to KSampler positive. |
| negative | CONDITIONING | Encoded negative conditioning. Wire to KSampler negative. |
| upscaled_latent | LATENT | Passthrough. Wire to KSampler latent_image. |

### Cross-tile context

When cross_tile_context > 0, each tile receives whole-image awareness:

- ids_rope engine: A 2D overview grid of the full image is embedded and appended as
  tokens with scaled positions. The area-based calculation preserves aspect ratio
  (0.5 = 50% of total tokens).

- matrix_rope engine: An L-shaped context is concatenated spatially:
  - Horizontal strip (tile's rows at full width)
  - Vertical strip (full height at tile's columns)
  - 2D thumbnail (coarse whole-image overview)

### Visual conditioning (Krea2 only)

When visual_conditioning > 0 on a multimodal model, each tile receives its own visual
conditioning from its image crop encoded through Qwen3-VL. The float controls the encoding
resolution: 1.0 = full tile_size resolution, 0.5 = half resolution (fewer vision tokens,
weaker constraint). Automatically disabled for text-only models (Flux, Anima).

## File structure

Comfyui-DiTiler/
├── __init__.py                  # Node registration
├── ditiler.py                   # UnifiedDiTiler node (orchestration + dispatch)
├── tiling_core.py               # Shared: BBox, split_bboxes, gaussian_weight, compute_tile_grid
├── ids_rope_tiling.py           # Engine: explicit position IDs (Krea2, Flux)
├── matrix_rope_tiling.py        # Engine: precomputed rotation matrices (Anima, Cosmos)
├── visual_conditioning.py       # Krea2 visual conditioning helpers (not a registered node)
└── model/
    ├── __init__.py              # Registry + detect_adapter
    ├── base.py                  # Abstract adapter
    ├── krea2.py                 # Krea2 adapter (ids_rope, multimodal)
    ├── flux.py                  # Flux adapter (ids_rope, text-only)
    └── anima.py                 # Anima adapter (matrix_rope, text-only + LLMAdapter)

## Example workflow

KSampler #1 (base generation, 1024×1024)
     │
     ├──────────────────────────────────────────────┐  source_latent
     ▼                                              │
 VAEDecode → Image → Upscale (4×) ──► upscaled_image (4096×4096)
     │                                              │
     ▼                                              │
 VAEEncode → upscaled_latent (512×512)              │
     │                                              │
     ▼                                              ▼
 ┌────────────────────────────────────────────────────────────────┐
 │ Unified DiTiler                                                 │
 │   model + clip + upscaled_image + source_latent                 │
 │   + upscaled_latent + prompt                                    │
 │   → patched MODEL, positive, negative, upscaled_latent          │
 └────────────────────────────────────────────────────────────────┘
     │ model    │ positive  │ negative  │ upscaled_latent
     ▼          ▼           ▼           ▼
 KSampler #2 (denoise ~0.30, steps 20-30)
     │
     ▼
 VAEDecode → final detailed image

## Recommended settings

| Parameter | Krea2 (turbo) | Flux | Anima |
|---|---|---|---|
| tile_size | 1280 | 1280 | 1280 |
| tile_overlap | 0.1 | 0.1 | 0.1 |
| cross_tile_context | 0.25 | 0.25 | 0.5 |
| visual_conditioning | 0.15 | 0.0 (auto) | 0.0 (auto) |
| denoise | 0.25–0.35 | 0.25–0.35 | 0.35–0.45 |
| cfg | 1.0 | 3.5–5.0 | 4.0–7.0 |
| steps | 12–16 | 20–30 | 20–30 |

## Model-specific notes

### Krea2
- Text encoder: Qwen3-VL (multimodal — text + image encoded jointly)
- Visual conditioning supported via visual_conditioning float
- Turbo model: CFG=1.0, negative conditioning is zeroed (unused)
- post_input hook required in comfy/ldm/krea2/model.py

### Flux
- Text encoder: T5-XXL (text-only)
- Visual conditioning not supported (auto-disabled)
- Position IDs may have 4 axes (handled by _match_position_axes)
- CFG > 1.0: negative conditioning is a zeroed version of positive

### Anima
- Text encoder: Qwen3 0.6B with LLMAdapter preprocessing
- Prompt templates required:
  - Positive: "You are an assistant designed to generate high quality anime images based on textual prompts. <Prompt Start> "
  - Negative: "You are an assistant designed to generate low-quality images based on textual prompts <Prompt Start> "
- return_dict=True encoding: Anima's CLIP returns a dict with t5xxl_ids and
  t5xxl_weights metadata required by the LLMAdapter. The adapter restructures this into
  ComfyUI's [[tensor, dict], ...] format.
- CFG > 1.0: negative uses the low-quality template (NOT zeros)
- concat_padding_mask: the model adds a mask channel before x_embedder; context
  embedding replicates this
- No visual conditioning (text-only encoder)
- Turbo variant: no prompt prefix, CFG=1.0

## Technical appendix

### ids_rope: 3-axis position IDs

Krea2/Flux use explicit img_ids with (index, row, col) per token:
- Axis 0: frame/index (0 = main image, 1+ = references/context)
- Axis 1: row position
- Axis 2: column position

Tiling adds the tile's true offset to axes 1 and 2. Context tokens get scaled positions
spanning the full native grid.

### matrix_rope: precomputed rotation matrices

Anima/Cosmos uses VideoRopePosition3DEmb which precomputes rotation matrices from
local (0-based) T/H/W. The attn1_patch hook reconstructs these matrices with the
tile's offset baked in, reading frequency buffers directly from the model's pos_embedder.

The L-shaped context satisfies the separable grid constraint: one H position per row,
one W position per column. The 2D thumbnail in the corner gets positions from the
cross-product of the vertical strip's H centers and horizontal strip's W centers.

### Tiling algorithm

Coverage-based tile count (ceildiv(canvas, tile)) with stride-based overlap guarantee.
When tiles perfectly divide the canvas, overlap is clamped to 0 (use a tile_size that
doesn't perfectly divide to get actual overlap). Gaussian-weighted blending normalizes
by coverage for seamless results.

## Known limitations

- Tile size should be a multiple of vae_compression × patch_size (16 for Krea2/Anima,
  16 for Flux) to land on clean token boundaries.
- tile_batch_size > 1 is not supported for matrix_rope models (shared RoPE offset state).
- No ControlNet-tile support.
- Anima's extra_pos_embedder checkpoints disable cross_tile_context (shape mismatch).
- Visual conditioning is Krea2-only (requires multimodal encoder).