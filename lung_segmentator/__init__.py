# lung_segmentator/__init__.py

from . import config
from . import layers
from .fine_tuning import load_dataset
from .analysis import manual_evaluate, sample_from_dataset

"""
This file contains the global configuration settings for the lung segmentator.
To modify values for customizing behavior of the model, use "config.variable="
Examples of variables that can be modified include IMG_SIZE, batch size, and the path variables.
"""