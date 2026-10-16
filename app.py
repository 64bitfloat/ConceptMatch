"""
Dog, Cat, or Penis Classifier App.
Ultra-minimalist, sleek UI, self-contained for cloud stability.
"""
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List
from PIL import Image
import streamlit as st

# Ensure root directory is always on python sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Assets and candidates setup
ASSETS_DIR = BASE_DIR / "assets"
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"

RESULT_IMAGES = {
    "dog": ASSETS_DIR / "result_dog.jpg",
    "cat": ASSETS_DIR / "result_cat.jpg",
    "chinpo": ASSETS_DIR / "result_chinpo.webp",
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

# Page configuration
st.set_page_config(
    page_title="犬か猫かちんぽか",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom sleek styling
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .app-title {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        color: #1e293b;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .app-subtitle {
        font-size: 1.05rem;
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    [data-testid="stFileUploadDropzone"] {
        border-radius: 16px;
        padding: 2.5rem 1rem;
        border: 2px dashed #cbd5e1;
        background-color: #f8fafc;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #94a3b8;
        background-color: #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_clip_resources():
    """Load PyTorch CLIP model and processor."""
    import torch
    from transformers import CLIPProcessor, CLIPModel

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = CLIPModel.from_pretrained(DEFAULT_MODEL_NAME).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(DEFAULT_MODEL_NAME)
    return model, processor, device


def classify_image(
    image: Image.Image,
    model,
    processor,
    device,
    candidates: List[Dict[str, Any]] = CANDIDATES
) -> Dict[str, Any]:
    """Zero-shot classification across candidates."""
    import torch

    if image.mode != "RGB":
        image = image.convert("RGB")

    prompts = [c["prompt"] for c in candidates]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        best_index = int(torch.argmax(probs).item())

    return candidates[best_index]


def main():
    st.markdown('<div class="app-title">犬か猫かちんぽか</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">アップした画像が「犬」「猫」「ちんぽ」のどれに一番近いか判定します。</div>', unsafe_allow_html=True)

    # Preload model in background
    with st.spinner("AIモデルを準備中..."):
        model, processor, device = load_clip_resources()

    # Minimal drag & drop upload area
    uploaded_file = st.file_uploader(
        "画像をアップロード",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            input_image = Image.open(uploaded_file)
            
            with st.spinner("判定中..."):
                winner = classify_image(
                    image=input_image,
                    model=model,
                    processor=processor,
                    device=device
                )

            # Display the designated resulting image
            res_image_path = Path(winner["image_path"])
            if res_image_path.exists():
                res_img = Image.open(res_image_path)
                st.image(res_img, use_container_width=True)
            else:
                st.error("指定画像が見つかりません。")

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
