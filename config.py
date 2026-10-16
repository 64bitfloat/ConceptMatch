"""
Config module for CLIP-based Concept Matcher application.
Defines default settings, thresholds, model identifiers, and assets paths.
"""
from dataclasses import dataclass
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

# Ensure assets directory exists
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Default Model Configuration
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"

# Default Prompt / Concept
DEFAULT_CONCEPT_PROMPT = "a photo of a cat"

# Default Threshold (%)
DEFAULT_THRESHOLD = 80.0

# Default Images
DEFAULT_IMAGE_A_PATH = str(ASSETS_DIR / "image_a.png")
DEFAULT_IMAGE_B_PATH = str(ASSETS_DIR / "image_b.png")

# Scaling / Calibration defaults
# CLIP raw cosine similarities typically fall between 0.15 and 0.35 for matching concepts
DEFAULT_SCALING_METHOD = "min_max"  # Options: "min_max", "sigmoid", "raw"
DEFAULT_MIN_COSINE = 0.15
DEFAULT_MAX_COSINE = 0.30
DEFAULT_SIGMOID_MIDPOINT = 0.23
DEFAULT_SIGMOID_TEMPERATURE = 24.0

@dataclass
class AppConfig:
    """Application runtime configuration parameters."""
    concept_prompt: str = DEFAULT_CONCEPT_PROMPT
    threshold: float = DEFAULT_THRESHOLD
    image_a_source: str = DEFAULT_IMAGE_A_PATH
    image_b_source: str = DEFAULT_IMAGE_B_PATH
    scaling_method: str = DEFAULT_SCALING_METHOD
    min_cosine: float = DEFAULT_MIN_COSINE
    max_cosine: float = DEFAULT_MAX_COSINE
    sigmoid_midpoint: float = DEFAULT_SIGMOID_MIDPOINT
    sigmoid_temperature: float = DEFAULT_SIGMOID_TEMPERATURE
