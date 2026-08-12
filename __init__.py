"""
Unified DiTiler custom nodes for ComfyUI.
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _load(module_name):
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    try:
        import importlib
        module = importlib.import_module(f".{module_name}", package=__package__)
        NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))
        return True
    except Exception as e:
        print(f"[DiTiler] Failed to load {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

_loaded = _load("ditiler")

if _loaded:
    print(f"[DiTiler] Loaded nodes: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
else:
    print("[DiTiler] No nodes loaded -- check the errors above.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]