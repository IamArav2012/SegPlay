# lung_segmentator/__init__.py

from . import config
from . import layers
from .fine_tuning import load_dataset
from .analysis import manual_evaluate, sample_from_dataset

"""
lung_segmentator package initializer.

This file sets up the package namespace by importing key modules.

Note to users:
This repo is designed as a learning sandbox. You’re encouraged to
directly modify the source files like config.py, fine_tuning.py, layers.py, etc.
The __init__.py is primarily for organizing imports and should rarely
need changes unless you want to customize package-level imports.
"""