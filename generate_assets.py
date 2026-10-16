"""
Script to generate test assets (Image A, Image B, and sample input images).
Generates standalone PNG images with Pillow without external dependencies.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def get_default_font(size: int = 24):
    """Attempt to load a reasonable TrueType font or fallback to default."""
    try:
        # Standard Windows fonts
        for font_name in ["segoeui.ttf", "arial.ttf", "tahoma.ttf"]:
            try:
                return ImageFont.truetype(font_name, size)
            except Exception:
                continue
    except Exception:
        pass
    return ImageFont.load_default()


def create_image_a():
    """Generate Image A (Score >= 80% success banner)."""
    width, height = 600, 400
    img = Image.new("RGBA", (width, height), (16, 185, 129, 255)) # Emerald Green
    draw = ImageDraw.Draw(img)

    # Decorative background card
    margin = 20
    draw.rounded_rectangle(
        [margin, margin, width - margin, height - margin],
        radius=25,
        fill=(5, 150, 105, 255),
        outline=(167, 243, 208, 255),
        width=4
    )

    # Center circle for icon
    center_x, center_y = width // 2, height // 2 - 40
    radius = 65
    draw.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        fill=(255, 255, 255, 240),
        outline=(209, 250, 229, 255),
        width=4
    )

    # Checkmark inside circle
    check_coords = [
        (center_x - 30, center_y - 2),
        (center_x - 10, center_y + 20),
        (center_x + 35, center_y - 25)
    ]
    draw.line(check_coords, fill=(5, 150, 105, 255), width=10, joint="curve")

    # Titles
    font_large = get_default_font(34)
    font_sub = get_default_font(20)

    title_text = "IMAGE A: MATCH 80%+ PASS!"
    sub_text = "Target Concept Confirmed (Score >= 80%)"

    draw.text((center_x, height - 120), title_text, fill=(255, 255, 255), font=font_large, anchor="mm")
    draw.text((center_x, height - 70), sub_text, fill=(209, 250, 229), font=font_sub, anchor="mm")

    out_path = ASSETS_DIR / "image_a.png"
    img.save(out_path)
    print(f"Created: {out_path}")


def create_image_b():
    """Generate Image B (Score < 80% notice banner)."""
    width, height = 600, 400
    img = Image.new("RGBA", (width, height), (244, 63, 94, 255)) # Rose Red
    draw = ImageDraw.Draw(img)

    # Decorative background card
    margin = 20
    draw.rounded_rectangle(
        [margin, margin, width - margin, height - margin],
        radius=25,
        fill=(225, 29, 72, 255),
        outline=(254, 205, 211, 255),
        width=4
    )

    # Center circle for icon
    center_x, center_y = width // 2, height // 2 - 40
    radius = 65
    draw.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        fill=(255, 255, 255, 240),
        outline=(255, 228, 230, 255),
        width=4
    )

    # Exclamation mark or info mark inside circle
    draw.line([(center_x, center_y - 30), (center_x, center_y + 10)], fill=(225, 29, 72, 255), width=10)
    draw.ellipse([center_x - 5, center_y + 22, center_x + 5, center_y + 32], fill=(225, 29, 72, 255))

    # Titles
    font_large = get_default_font(34)
    font_sub = get_default_font(20)

    title_text = "IMAGE B: BELOW 80%"
    sub_text = "Similarity did not meet threshold (< 80%)"

    draw.text((center_x, height - 120), title_text, fill=(255, 255, 255), font=font_large, anchor="mm")
    draw.text((center_x, height - 70), sub_text, fill=(255, 228, 230), font=font_sub, anchor="mm")

    out_path = ASSETS_DIR / "image_b.png"
    img.save(out_path)
    print(f"Created: {out_path}")


def create_sample_cat():
    """Generate a clean visual depiction of a cat face for testing."""
    width, height = 512, 512
    img = Image.new("RGB", (width, height), (248, 250, 252)) # Light slate
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2 + 30

    # Cat Ears (triangles)
    ear_left = [(cx - 150, cy - 80), (cx - 190, cy - 220), (cx - 70, cy - 130)]
    ear_right = [(cx + 150, cy - 80), (cx + 190, cy - 220), (cx + 70, cy - 130)]
    draw.polygon(ear_left, fill=(245, 158, 11), outline=(217, 119, 6)) # Amber
    draw.polygon(ear_right, fill=(245, 158, 11), outline=(217, 119, 6))

    # Inner ears (pink)
    inner_l = [(cx - 140, cy - 90), (cx - 170, cy - 190), (cx - 80, cy - 130)]
    inner_r = [(cx + 140, cy - 90), (cx + 170, cy - 190), (cx + 80, cy - 130)]
    draw.polygon(inner_l, fill=(253, 164, 175))
    draw.polygon(inner_r, fill=(253, 164, 175))

    # Cat Head
    draw.ellipse([cx - 160, cy - 140, cx + 160, cy + 140], fill=(251, 191, 36), outline=(217, 119, 6), width=4)

    # Eyes
    eye_y = cy - 20
    draw.ellipse([cx - 95, eye_y - 25, cx - 45, eye_y + 25], fill=(30, 41, 59))
    draw.ellipse([cx + 45, eye_y - 25, cx + 95, eye_y + 25], fill=(30, 41, 59))
    # Eye sparkle
    draw.ellipse([cx - 75, eye_y - 18, cx - 60, eye_y - 3], fill=(255, 255, 255))
    draw.ellipse([cx + 65, eye_y - 18, cx + 80, eye_y - 3], fill=(255, 255, 255))

    # Nose
    nose_y = cy + 25
    nose_tri = [(cx, nose_y + 15), (cx - 16, nose_y - 10), (cx + 16, nose_y - 10)]
    draw.polygon(nose_tri, fill=(244, 63, 94))

    # Mouth
    draw.arc([cx - 35, nose_y + 5, cx, nose_y + 35], start=0, end=180, fill=(71, 85, 105), width=4)
    draw.arc([cx, nose_y + 5, cx + 35, nose_y + 35], start=0, end=180, fill=(71, 85, 105), width=4)

    # Whiskers
    draw.line([(cx - 70, cy + 30), (cx - 180, cy + 15)], fill=(71, 85, 105), width=3)
    draw.line([(cx - 70, cy + 45), (cx - 180, cy + 50)], fill=(71, 85, 105), width=3)
    draw.line([(cx + 70, cy + 30), (cx + 180, cy + 15)], fill=(71, 85, 105), width=3)
    draw.line([(cx + 70, cy + 45), (cx + 180, cy + 50)], fill=(71, 85, 105), width=3)

    # Label
    font = get_default_font(28)
    draw.text((cx, 40), "Sample Test Image: CAT", fill=(30, 41, 59), font=font, anchor="mm")

    out_path = ASSETS_DIR / "sample_cat.png"
    img.save(out_path)
    print(f"Created: {out_path}")


def create_sample_dog():
    """Generate a clean visual depiction of a dog for testing."""
    width, height = 512, 512
    img = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2 + 20

    # Dog floppy ears
    draw.ellipse([cx - 200, cy - 80, cx - 110, cy + 110], fill=(120, 53, 15))
    draw.ellipse([cx + 110, cy - 80, cx + 200, cy + 110], fill=(120, 53, 15))

    # Dog Head
    draw.ellipse([cx - 140, cy - 130, cx + 140, cy + 130], fill=(180, 83, 9), outline=(120, 53, 15), width=4)

    # Snout
    draw.ellipse([cx - 80, cy + 10, cx + 80, cy + 110], fill=(217, 119, 6))

    # Eyes
    eye_y = cy - 30
    draw.ellipse([cx - 80, eye_y - 20, cx - 40, eye_y + 20], fill=(30, 41, 59))
    draw.ellipse([cx + 40, eye_y - 20, cx + 80, eye_y + 20], fill=(30, 41, 59))

    # Big black dog nose
    draw.ellipse([cx - 30, cy + 25, cx + 30, cy + 65], fill=(15, 23, 42))

    # Tongue
    draw.chord([cx - 25, cy + 80, cx + 25, cy + 130], start=0, end=180, fill=(244, 63, 94))

    # Label
    font = get_default_font(28)
    draw.text((cx, 40), "Sample Test Image: DOG", fill=(30, 41, 59), font=font, anchor="mm")

    out_path = ASSETS_DIR / "sample_dog.png"
    img.save(out_path)
    print(f"Created: {out_path}")


if __name__ == "__main__":
    create_image_a()
    create_image_b()
    create_sample_cat()
    create_sample_dog()
