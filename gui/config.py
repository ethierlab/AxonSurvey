import json
import sys
from pathlib import Path

# Load config from json
config_path = Path(__file__).parent / "config.json"
try:
    with open(config_path, "r") as f:
        _config_data = json.load(f)
except FileNotFoundError:
    print(f"ERROR: Configuration file not found at {config_path}", file=sys.stderr)
    _config_data = {}

class Config:
    """Configuration constants for the application"""
    # Directory paths
    DATA_DIR = Path(_config_data.get("DATA_DIR", "./data/project_scans"))
    CACHE_DIR = Path(_config_data.get("CACHE_DIR", "./data/feature_seg_cache"))
    
    TRAINING_DIR = Path(_config_data.get("TRAINING_DIR", "./data/manual_tracings/2025-05-31_rat301_train"))
    TESTING_DIR = Path(_config_data.get("TESTING_DIR", "./data/manual_tracings/2025-05-31_rat301_test"))
    EXPERIMENTS_DIR = Path(_config_data.get("EXPERIMENTS_DIR", "./data/experiments"))
    FIGURES_DIR = Path(_config_data.get("FIGURES_DIR", "./data/figures"))

    USED_SEGMENTATION_MODEL_NAME = _config_data.get("USED_SEGMENTATION_MODEL_NAME", "DLTr_GIGAAug")
    USED_SEGMENTATION_MODEL_PATH = Path(_config_data.get("USED_SEGMENTATION_MODEL_PATH", "./data/trained_models/default_model.pth"))

    STATIC_DIR = Path(_config_data.get("STATIC_DIR", "static"))
    TEMPLATES_DIR = Path(_config_data.get("TEMPLATES_DIR", "templates"))
    
    MAX_DISPLAY_SIZE = tuple(_config_data.get("MAX_DISPLAY_SIZE", [800, 600]))
    DEFAULT_COLORMAP = _config_data.get("DEFAULT_COLORMAP", "viridis")

    # Not sure if the following are useful: 
    IMAGE_PATTERNS = _config_data.get("IMAGE_PATTERNS", ["image.tif", "image.tiff", "*.tif", "*.tiff"])
    SEGMENTATION_SUFFIX = _config_data.get("SEGMENTATION_SUFFIX", "_seg.TIF")
    FEATURE_MAP_SUFFIX = _config_data.get("FEATURE_MAP_SUFFIX", "_features.TIF")
    METADATA_FILE = _config_data.get("METADATA_FILE", "meta.json")

# Simple debugging checks
for path_attr, path_val in [
    ("DATA_DIR", Config.DATA_DIR),
    ("CACHE_DIR", Config.CACHE_DIR),
    ("TRAINING_DIR", Config.TRAINING_DIR),
    ("TESTING_DIR", Config.TESTING_DIR),
    ("USED_SEGMENTATION_MODEL_PATH", Config.USED_SEGMENTATION_MODEL_PATH)
]:
    if not path_val.exists():
        print(f"WARNING: {path_attr} path does not exist: {path_val}", file=sys.stderr)
    