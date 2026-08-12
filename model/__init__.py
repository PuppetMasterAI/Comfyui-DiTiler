"""
Model adapter registry for Unified DiTiler.
"""

from .base import ModelAdapterBase
from .krea2 import Krea2Adapter
from .flux import FluxAdapter
from .anima import AnimaAdapter 

# Registry of all available adapters, keyed by name.
ADAPTERS = {
    "krea2": Krea2Adapter,
    "flux": FluxAdapter,
    "anima": AnimaAdapter, 
}

def get_adapter_by_name(name: str) -> ModelAdapterBase:
    """Return an adapter instance by its registered name."""
    adapter_cls = ADAPTERS.get(name)
    if adapter_cls is None:
        raise ValueError(f"Unknown model adapter: {name!r}. Available: {list(ADAPTERS.keys())}")
    return adapter_cls()


def detect_adapter(diffusion_model) -> ModelAdapterBase:
    """Auto-detect the correct adapter from the diffusion model instance."""
    for adapter_cls in ADAPTERS.values():
        adapter = adapter_cls()
        if adapter.detect(diffusion_model):
            return adapter
    raise ValueError(
        f"Could not auto-detect model adapter for {type(diffusion_model).__name__}. "
        f"Please select a model_type explicitly. Available: {list(ADAPTERS.keys())}"
    )

