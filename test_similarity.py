"""
Verification test script for CLIP Concept Matcher.
Tests model loading and similarity calculation on generated sample images.
"""
from PIL import Image
from config import ASSETS_DIR, DEFAULT_CONCEPT_PROMPT
from similarity import load_clip_model, calculate_similarity

def main():
    print("Testing CLIP model loading...")
    model, processor, device = load_clip_model()
    print(f"Model loaded successfully on device: {device}")

    cat_path = ASSETS_DIR / "sample_cat.png"
    dog_path = ASSETS_DIR / "sample_dog.png"

    cat_img = Image.open(cat_path)
    dog_img = Image.open(dog_path)

    prompt = DEFAULT_CONCEPT_PROMPT
    print(f"\nEvaluating with prompt: '{prompt}'")

    cat_result = calculate_similarity(cat_img, prompt, model, processor, device)
    dog_result = calculate_similarity(dog_img, prompt, model, processor, device)

    print(f"Cat Image -> Raw Cosine: {cat_result['raw_cosine']:.4f}, Scaled Score: {cat_result['percentage']:.2f}%")
    print(f"Dog Image -> Raw Cosine: {dog_result['raw_cosine']:.4f}, Scaled Score: {dog_result['percentage']:.2f}%")

    # Also test with dog prompt
    prompt_dog = "a photo of a dog"
    print(f"\nEvaluating with prompt: '{prompt_dog}'")
    cat_result2 = calculate_similarity(cat_img, prompt_dog, model, processor, device)
    dog_result2 = calculate_similarity(dog_img, prompt_dog, model, processor, device)

    print(f"Cat Image -> Raw Cosine: {cat_result2['raw_cosine']:.4f}, Scaled Score: {cat_result2['percentage']:.2f}%")
    print(f"Dog Image -> Raw Cosine: {dog_result2['raw_cosine']:.4f}, Scaled Score: {dog_result2['percentage']:.2f}%")

    print("\n[SUCCESS] Verification completed!")

if __name__ == "__main__":
    main()
