import importlib
import sys
from pathlib import Path

def import_module(module_path):
    try:
        module = importlib.import_module(module_path)
        return module
    except ImportError as e:
        raise ImportError(f"Failed to import {module_path}: {e}") from e