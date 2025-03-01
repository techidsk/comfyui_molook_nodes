import importlib
import os
import pkgutil
from typing import Any, Dict

# Initialize empty dictionaries for our mappings
NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

# Get the current package directory
package_dir = os.path.dirname(__file__)

# Iterate through all modules in the current package
for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
    # Skip the current module to avoid circular imports
    if module_name == "nodes":
        continue

    # Import the module
    module = importlib.import_module(
        f".{module_name}", package=__name__.rsplit(".", 1)[0]
    )

    # Check if the module has the required mappings
    if hasattr(module, "NODE_CLASS_MAPPINGS"):
        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)

    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
