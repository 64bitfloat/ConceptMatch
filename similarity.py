"""
CLIP classification engine for 3-way recognition.
"""
from typing import Tuple, Dict, Any, List
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from config import DEFAULT_MODEL_NAME, CANDIDATES


def get_device() -> torch.device:
    """Determine best available computing device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_clip_model(model_name: str = DEFAULT_MODEL_NAME) -> Tuple[CLIPModel, CLIPProcessor, torch.device]:
    """Load and return cached CLIP model, processor and device."""
    device = get_device()
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def classify_three_way(
    image: Image.Image,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    candidates: List[Dict[str, Any]] = CANDIDATES
) -> Dict[str, Any]:
    """
    Classify input image into one of the 3 categories (Dog, Cat, or Penis)
    using zero-shot image-text similarity.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    prompts = [c["prompt"] for c in candidates]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        best_index = int(torch.argmax(probs).item())

    return candidates[best_index]
