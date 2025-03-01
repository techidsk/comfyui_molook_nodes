"""Top-level package for comfyui_molook_nodes."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """ComfyUI-molook-nodes"""
__email__ = "ecpknymt@gmail.com"
__version__ = "0.1.0"

from .src.molook_nodes.nodes import NODE_CLASS_MAPPINGS
from .src.molook_nodes.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
