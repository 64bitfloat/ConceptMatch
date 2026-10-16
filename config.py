"""
Configuration module for 'Dog, Cat, or Penis' classifier app.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"

# Result images corresponding to the three categories
RESULT_IMAGES = {
    "dog": str(ASSETS_DIR / "result_dog.jpg"),
    "cat": str(ASSETS_DIR / "result_cat.jpg"),
    "chinpo": str(ASSETS_DIR / "result_chinpo.webp"),
}

CANDIDATES = [
    {
        "key": "dog",
        "name": "犬",
        "prompt": "a dog",
        "image_path": RESULT_IMAGES["dog"]
    },
    {
        "key": "cat",
        "name": "猫",
        "prompt": "a cat",
        "image_path": RESULT_IMAGES["cat"]
    },
    {
        "key": "chinpo",
        "name": "ちんぽ",
        "prompt": "a penis",
        "image_path": RESULT_IMAGES["chinpo"]
    }
]
