"""
Dog, Cat, or Penis Classifier App.
Ultra-minimalist, sleek UI.
"""
from pathlib import Path
from PIL import Image
import streamlit as st

from similarity import load_clip_model, classify_three_way
from config import DEFAULT_MODEL_NAME

# Page setup - minimalist & centered
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
    
    .result-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 2rem;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
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
def get_model():
    """Load and cache CLIP model."""
    return load_clip_model(DEFAULT_MODEL_NAME)


def main():
    st.markdown('<div class="app-title">犬か猫かちんぽか</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">アップした画像が「犬」「猫」「ちんぽ」のどれに一番近いか判定します。</div>', unsafe_allow_html=True)

    # Preload model in background
    model, processor, device = get_model()

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
                winner = classify_three_way(
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
