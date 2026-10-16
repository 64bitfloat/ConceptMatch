"""
CLIP Concept Matcher Web Application
Built with Streamlit and Hugging Face Transformers.
"""
from pathlib import Path
from typing import Optional
import io
import requests
from PIL import Image
import streamlit as st

from config import (
    ASSETS_DIR,
    DEFAULT_CONCEPT_PROMPT,
    DEFAULT_THRESHOLD,
    DEFAULT_IMAGE_A_PATH,
    DEFAULT_IMAGE_B_PATH,
    DEFAULT_SCALING_METHOD,
    DEFAULT_MIN_COSINE,
    DEFAULT_MAX_COSINE,
    DEFAULT_SIGMOID_MIDPOINT,
    DEFAULT_SIGMOID_TEMPERATURE,
    DEFAULT_MODEL_NAME,
)
from similarity import load_clip_model, calculate_similarity

# Page configuration
st.set_page_config(
    page_title="CLIP Concept Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .score-container {
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    .score-number {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        margin: 10px 0;
    }
    .score-success {
        color: #059669;
    }
    .score-fail {
        color: #e11d48;
    }
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-success {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .badge-fail {
        background-color: #ffe4e6;
        color: #9f1239;
        border: 1px solid #fecdd3;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_clip_model():
    """Load and cache the CLIP model and processor to prevent reload overhead."""
    return load_clip_model(DEFAULT_MODEL_NAME)


def load_image_from_source(source: str) -> Optional[Image.Image]:
    """
    Load an image from a local file path or HTTP/HTTPS URL.
    Returns None if loading fails.
    """
    if not source:
        return None
    source = source.strip()
    try:
        if source.startswith(("http://", "https://")):
            response = requests.get(source, timeout=8)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
        else:
            path = Path(source)
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            if not path.exists():
                return None
            img = Image.open(path)
        return img.convert("RGB")
    except Exception as e:
        st.sidebar.error(f"画像読み込みエラー ({source[:30]}...): {e}")
        return None


def render_sidebar():
    """Render the configuration settings in the sidebar."""
    st.sidebar.header("⚙️ 設定 (Configuration)")

    # 1. Concept Prompt
    # Initialize prompt in session_state if not present
    if "prompt_input" not in st.session_state:
        st.session_state["prompt_input"] = DEFAULT_CONCEPT_PROMPT

    def on_preset_change():
        chosen = st.session_state.get("prompt_preset_select")
        if chosen and chosen != "(現在の入力を使用)":
            st.session_state["prompt_input"] = chosen

    prompt_presets = [
        "(現在の入力を使用)",
        "a photo of a cat",
        "an illustration of a cat",
        "a photo of a dog",
        "a photograph of a sports car",
        "a cup of coffee",
    ]

    st.sidebar.selectbox(
        "プリセットから選択",
        prompt_presets,
        index=0,
        key="prompt_preset_select",
        on_change=on_preset_change
    )

    prompt = st.sidebar.text_input(
        "比較テキスト (Prompt)",
        key="prompt_input",
        help="アップロードされた画像がこの概念にどれくらい似ているかをCLIPが判定します。"
    )

    # 2. Threshold
    st.sidebar.subheader("2. 判定閾値 (%)")
    threshold = st.sidebar.slider(
        "合格閾値",
        min_value=1.0,
        max_value=99.0,
        value=float(DEFAULT_THRESHOLD),
        step=1.0,
        format="%.0f%%",
        help="類似度パーセンテージがこの値以上のときに画像A、未満のときに画像Bが表示されます。"
    )

    # 3. Destination Images (Image A & B)
    st.sidebar.subheader("3. 表示画像の設定")

    with st.sidebar.expander("画像A (閾値以上で表示)", expanded=False):
        img_a_src = st.text_input(
            "パスまたはURL",
            value=DEFAULT_IMAGE_A_PATH,
            key="img_a_src"
        )
        custom_a = st.file_uploader("手動アップロードで差し替え (画像A)", type=["png", "jpg", "jpeg"], key="upload_a")
        if custom_a is not None:
            img_a_loaded = Image.open(custom_a).convert("RGB")
        else:
            img_a_loaded = load_image_from_source(img_a_src)
        if img_a_loaded:
            st.image(img_a_loaded, caption="画像Aプレビュー", use_container_width=True)

    with st.sidebar.expander("画像B (閾値未満で表示)", expanded=False):
        img_b_src = st.text_input(
            "パスまたはURL",
            value=DEFAULT_IMAGE_B_PATH,
            key="img_b_src"
        )
        custom_b = st.file_uploader("手動アップロードで差し替え (画像B)", type=["png", "jpg", "jpeg"], key="upload_b")
        if custom_b is not None:
            img_b_loaded = Image.open(custom_b).convert("RGB")
        else:
            img_b_loaded = load_image_from_source(img_b_src)
        if img_b_loaded:
            st.image(img_b_loaded, caption="画像Bプレビュー", use_container_width=True)

    # 4. Calibration & Scaling Settings
    with st.sidebar.expander("🛠️ CLIP類似度スケーリング調整", expanded=False):
        st.caption(
            "CLIPの生のコサイン類似度は通常0.15〜0.35に集中します。"
            "直感的な0〜100%にマッピングするための補正設定です。"
        )
        scaling_method = st.selectbox(
            "スケーリング方式",
            ["min_max", "sigmoid", "raw"],
            format_func=lambda x: {
                "min_max": "Min-Max 線形スケーリング (推奨)",
                "sigmoid": "シグモイド補正 (滑らかなS字)",
                "raw": "生コサイン × 100% (未補正)"
            }[x]
        )

        min_cosine = DEFAULT_MIN_COSINE
        max_cosine = DEFAULT_MAX_COSINE
        sigmoid_midpoint = DEFAULT_SIGMOID_MIDPOINT
        sigmoid_temperature = DEFAULT_SIGMOID_TEMPERATURE

        if scaling_method == "min_max":
            min_cosine = st.number_input("0%基準コサイン値 (Min)", value=DEFAULT_MIN_COSINE, step=0.01, format="%.2f")
            max_cosine = st.number_input("100%基準コサイン値 (Max)", value=DEFAULT_MAX_COSINE, step=0.01, format="%.2f")
        elif scaling_method == "sigmoid":
            sigmoid_midpoint = st.number_input("中心点 (Midpoint 50%)", value=DEFAULT_SIGMOID_MIDPOINT, step=0.01, format="%.2f")
            sigmoid_temperature = st.number_input("感度 (Temperature)", value=DEFAULT_SIGMOID_TEMPERATURE, step=1.0)

    return {
        "prompt": prompt,
        "threshold": threshold,
        "image_a": img_a_loaded,
        "image_b": img_b_loaded,
        "scaling_method": scaling_method,
        "min_cosine": min_cosine,
        "max_cosine": max_cosine,
        "sigmoid_midpoint": sigmoid_midpoint,
        "sigmoid_temperature": sigmoid_temperature,
    }


def main():
    # Title & Subtitle
    st.markdown('<div class="main-header">🎯 CLIP 概念類似度チェッカー</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'アップロードした画像が指定した概念にどれくらい似ているかをCLIP AIモデルで判定し、'
        'スコアに応じて画像Aまたは画像Bを表示します。'
        '</div>',
        unsafe_allow_html=True
    )

    # Render Sidebar and retrieve settings
    config = render_sidebar()

    # Pre-load CLIP model
    with st.spinner("🤖 AIモデル (CLIP) を読み込んでいます... 初回のみダウンロードに数十秒かかります"):
        model, processor, device = get_cached_clip_model()

    # Layout: Top Input Section
    st.markdown("### 📤 判定する画像をアップロード")

    col_input1, col_input2 = st.columns([2, 1])

    uploaded_image: Optional[Image.Image] = None

    with col_input1:
        uploaded_file = st.file_uploader(
            "画像をドラッグ＆ドロップ または ファイルを選択 (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="main_uploader"
        )
        if uploaded_file is not None:
            try:
                uploaded_image = Image.open(uploaded_file).convert("RGB")
            except Exception as e:
                st.error(f"画像を開けませんでした: {e}")

    with col_input2:
        st.write("または **サンプル画像で試す**:")
        sample_col1, sample_col2 = st.columns(2)
        
        sample_cat_path = ASSETS_DIR / "sample_cat.png"
        sample_dog_path = ASSETS_DIR / "sample_dog.png"

        with sample_col1:
            if st.button("🐱 猫のサンプル", use_container_width=True):
                if sample_cat_path.exists():
                    st.session_state["sample_selected"] = sample_cat_path
        with sample_col2:
            if st.button("🐶 犬のサンプル", use_container_width=True):
                if sample_dog_path.exists():
                    st.session_state["sample_selected"] = sample_dog_path

    # If sample button was pressed and no file uploaded
    if uploaded_file is None and "sample_selected" in st.session_state:
        selected_path = st.session_state["sample_selected"]
        if Path(selected_path).exists():
            uploaded_image = Image.open(selected_path).convert("RGB")
            st.info(f"サンプル画像を読み込みました: `{Path(selected_path).name}`")

    # Separator
    st.divider()

    # If image is ready, run similarity check
    if uploaded_image is not None:
        with st.spinner(f"「{config['prompt']}」との類似度を計算中..."):
            result = calculate_similarity(
                image=uploaded_image,
                concept_prompt=config["prompt"],
                model=model,
                processor=processor,
                device=device,
                scaling_method=config["scaling_method"],
                min_cosine=config["min_cosine"],
                max_cosine=config["max_cosine"],
                sigmoid_midpoint=config["sigmoid_midpoint"],
                sigmoid_temperature=config["sigmoid_temperature"]
            )

        score = result["percentage"]
        raw_cosine = result["raw_cosine"]
        threshold = config["threshold"]
        is_passed = score >= threshold

        # Display Section: 2 Columns (Left: Input Image, Right: Result & Destination Image)
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown('<div class="card-title">📷 入力された画像</div>', unsafe_allow_html=True)
            st.image(uploaded_image, caption=f"解像度: {uploaded_image.width} × {uploaded_image.height}", use_container_width=True)

        with col_right:
            st.markdown('<div class="card-title">📊 判定スコア & 結果</div>', unsafe_allow_html=True)

            score_class = "score-success" if is_passed else "score-fail"
            badge_class = "badge-success" if is_passed else "badge-fail"
            badge_text = f"判定: {'達成 ( >= ' + str(int(threshold)) + '% )' if is_passed else '未達成 ( < ' + str(int(threshold)) + '% )'}"

            st.markdown(
                f"""
                <div class="score-container">
                    <span class="badge {badge_class}">{badge_text}</span>
                    <div class="score-number {score_class}">{score:.1f}%</div>
                    <div style="color: #64748b; font-size: 0.95rem;">
                        対象概念: <strong>"{config['prompt']}"</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Progress bar
            st.progress(min(1.0, max(0.0, score / 100.0)))

            st.caption(
                f"ℹ️ 生のコサイン類似度: `{raw_cosine:.4f}` | 判定閾値: `{threshold:.1f}%` | スケーリング: `{config['scaling_method']}`"
            )

            st.write("")

            # Conditional Image Display
            if is_passed:
                st.success(f"🎉 判定成功: 類似度 {score:.1f}% が閾値 {threshold:.0f}% 以上を達成しました！【画像A】を表示します。")
                if config["image_a"]:
                    st.image(config["image_a"], caption="【画像A】達成時の表示画像", use_container_width=True)
                else:
                    st.warning("画像Aが読み込めませんでした。サイドバーで画像URLまたはパスを確認してください。")
            else:
                st.warning(f"⚠️ 判定未達: 類似度 {score:.1f}% は閾値 {threshold:.0f}% 未満です。【画像B】を表示します。")
                if config["image_b"]:
                    st.image(config["image_b"], caption="【画像B】未達成時の表示画像", use_container_width=True)
                else:
                    st.warning("画像Bが読み込めませんでした。サイドバーで画像URLまたはパスを確認してください。")

    else:
        # Placeholder when no image is uploaded
        st.info("👆 上のエリアから画像（JPG, PNG）をアップロードするか、「サンプル画像で試す」ボタンをクリックしてください。")
        
        # Show configuration preview
        st.markdown("---")
        st.markdown("#### ⚙️ 現在の設定一覧")
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            st.metric(label="判定対象の概念", value=config["prompt"])
        with pcol2:
            st.metric(label="条件分岐の閾値", value=f"{config['threshold']:.0f} %")
        with pcol3:
            st.metric(label="スケーリング方式", value=config["scaling_method"])


if __name__ == "__main__":
    main()
