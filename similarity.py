"""
Similarity calculation and calibration module using Hugging Face CLIP.
Provides cosine similarity calculation and conversion to intuitive 0-100% scale.
"""
import math
from typing import Tuple, Dict, Any
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

from config import DEFAULT_MODEL_NAME


def get_device() -> torch.device:
    """Determine the optimal computing device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_clip_model(model_name: str = DEFAULT_MODEL_NAME) -> Tuple[CLIPModel, CLIPProcessor, torch.device]:
    """
    Load CLIP model and processor.
    Designed to be cached with @st.cache_resource in Streamlit.
    """
    device = get_device()
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def scale_cosine_to_percentage(
    cosine_sim: float,
    method: str = "min_max",
    min_val: float = 0.15,
    max_val: float = 0.30,
    midpoint: float = 0.23,
    temperature: float = 24.0
) -> float:
    """
    Maps raw CLIP cosine similarity to an intuitive 0.0 - 100.0% scale.

    CLIP cosine similarity for image-text pairs usually clusters in [0.15, 0.35].
    Simple cosine * 100 would cap realistic matches around 30%.
    This function provides adjusted scaling so that:
      - Dissimilar concepts approach 0%
      - Highly matching concepts approach 100%

    Args:
        cosine_sim: Raw cosine similarity (-1.0 to 1.0)
        method: Scaling algorithm ('min_max', 'sigmoid', or 'raw')
        min_val: Baseline similarity mapped to 0% (in min_max mode)
        max_val: Target similarity mapped to 100% (in min_max mode)
        midpoint: Sigmoid inflection point (in sigmoid mode)
        temperature: Sigmoid steepness factor (in sigmoid mode)

    Returns:
        float: Scaled similarity percentage between 0.0 and 100.0
    """
    if method == "min_max":
        # Linear normalization between min_val and max_val with clamping
        if max_val <= min_val:
            scaled = 0.0
        else:
            scaled = (cosine_sim - min_val) / (max_val - min_val) * 100.0
        return max(0.0, min(100.0, float(scaled)))

    elif method == "sigmoid":
        # Smooth S-curve calibration centered at midpoint
        exponent = -temperature * (cosine_sim - midpoint)
        # Prevent math overflow
        exponent = max(-50.0, min(50.0, exponent))
        scaled = 100.0 / (1.0 + math.exp(exponent))
        return max(0.0, min(100.0, float(scaled)))

    elif method == "raw":
        # Pure cosine mapped to 0-100% without clustering calibration
        return max(0.0, min(100.0, cosine_sim * 100.0))

    else:
        # Fallback to min_max
        return scale_cosine_to_percentage(cosine_sim, "min_max", min_val, max_val)


def _extract_tensor(features_output: Any) -> torch.Tensor:
    """Safely extract torch.Tensor from either Tensor or HuggingFace output objects."""
    if isinstance(features_output, torch.Tensor):
        return features_output
    if hasattr(features_output, "pooler_output") and features_output.pooler_output is not None:
        return features_output.pooler_output
    if hasattr(features_output, "last_hidden_state") and features_output.last_hidden_state is not None:
        return features_output.last_hidden_state[:, 0, :]
    if isinstance(features_output, (tuple, list)) and len(features_output) > 0:
        return _extract_tensor(features_output[0])
    raise TypeError(f"Cannot extract tensor from type: {type(features_output)}")


def calculate_similarity(
    image: Image.Image,
    concept_prompt: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    scaling_method: str = "min_max",
    min_cosine: float = 0.15,
    max_cosine: float = 0.30,
    sigmoid_midpoint: float = 0.23,
    sigmoid_temperature: float = 24.0
) -> Dict[str, Any]:
    """
    Compute raw cosine similarity and calibrated percentage between an image and concept text.

    Args:
        image: PIL Image object
        concept_prompt: Text description of the concept
        model: CLIPModel
        processor: CLIPProcessor
        device: torch.device
        ...scaling parameters

    Returns:
        dict: Contains 'raw_cosine', 'percentage', and 'prompt'
    """
    # Ensure RGB format for Pillow image
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Preprocess inputs
    inputs = processor(
        text=[concept_prompt],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            image_features = outputs.image_embeds
        else:
            image_features = _extract_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))

        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            text_features = outputs.text_embeds
        else:
            text_features = _extract_tensor(model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask")
            ))

        # L2 Normalization
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity is the dot product of normalized vectors
        cosine_sim = torch.sum(image_features * text_features, dim=-1).item()

    # Scale to percentage
    percentage = scale_cosine_to_percentage(
        cosine_sim=cosine_sim,
        method=scaling_method,
        min_val=min_cosine,
        max_val=max_cosine,
        midpoint=sigmoid_midpoint,
        temperature=sigmoid_temperature
    )

    return {
        "raw_cosine": float(cosine_sim),
        "percentage": float(percentage),
        "prompt": concept_prompt
    }
