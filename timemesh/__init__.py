# timemesh/__init__.py
from __future__ import annotations  # For forward reference compatibility

from .data_loader import DataLoader
from .preprocessing import Normalizer

__all__ = [
    "DataLoader",
    "Normalizer",
]

# Optional but recommended for type hinting clarity
__version__ = "0.1.0"  # Should match pyproject.toml version
