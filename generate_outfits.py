"""
ToneFit ML — Gemini Outfit Image Generator
Generates 72 outfit images (12 sub-types × 6 styles)
using Google Gemini Imagen API.

Outfits are unisex / gender-neutral — no age or gender personalization.
Color palette accuracy is the sole personalization axis (personal color season).

Requirements:
    pip install google-generativeai pillow

Usage:
    1. Set your Gemini API key below
    2. Place your reference image as 'reference.jpg' in the same folder
    3. Run: python generate_outfits.py
    4. Images saved to: outfits/[subtype_key]/[style].jpg
"""

import google.generativeai as genai
import os
import time
from pathlib import Path

# ── SET YOUR API KEY ──────────────────────────────────────────
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
# ─────────────────────────────────────────────────────────────

genai.configure(api_key=GEMINI_API_KEY)

# ── LAYOUT DESCRIPTION (based on your reference photo) ────────
# Change this to match whatever reference photo you use
LAYOUT = (
    "Unisex fashion outfit flat lay on a clean white background. "
    "Gender-neutral, age-agnostic clothing items neatly arranged: "
    "top folded at the upper area, bottom folded below it, "
    "shoes placed at the bottom, and a bag on the side. "
    "Minimal, editorial, clean styling. "
    "No person, no face, no model. Soft studio lighting. "
    "High quality fashion photography style."
)

# ── STYLE DESCRIPTIONS ────────────────────────────────────────
STYLE_DESC = {
    "casual":       "unisex casual everyday wear — relaxed, comfortable, and effortless",
    "smart_casual": "unisex smart casual — polished but relaxed, between casual and formal",
    "classic":      "unisex classic timeless style — elegant, structured, and refined",
    "streetwear":   "unisex streetwear — urban, trendy, bold, and contemporary",
    "retro":        "unisex retro vintage style — nostalgic, playful, and characterful",
    "formal":       "unisex formal occasion wear — sophisticated, elegant, and polished",
}

# ── ALL 12 SUB-TYPES WITH THEIR BEST COLORS ───────────────────
SUBTYPES = {
    "spring_warm": {
        "display_name": "Warm Spring",
        "undertone": "warm, golden, and clear",
        "colors": "peach, coral, golden yellow, camel, warm ivory, salmon, warm turquoise, apricot",
    },
    "spring_light": {
        "display_name": "Light Spring",
        "undertone": "light, warm, and delicate",
        "colors": "light peach, blush pink, vanilla, soft coral, warm white, light apricot, soft aqua",
    },
    "spring_bright": {
        "display_name": "Bright Spring",
        "undertone": "warm, vivid, and clear",
        "colors": "bright coral, warm yellow, poppy red, bright turquoise, vivid orange, ivory, clear green",
    },
    "summer_cool": {
        "display_name": "Cool Summer",
        "undertone": "cool, muted, and soft",
        "colors": "dusty rose, powder blue, lavender, mauve, soft gray, steel blue, periwinkle, soft sage",
    },
    "summer_light": {
        "display_name": "Light Summer",
        "undertone": "light, cool, and soft",
        "colors": "soft lavender, baby blue, pale pink, cool white, light gray, icy periwinkle, pale mauve",
    },
    "summer_soft": {
        "display_name": "Soft Summer",
        "undertone": "cool, muted, and blended",
        "colors": "dusty mauve, grayed blue, soft sage, muted rose, stone gray, faded denim, dusty lavender",
    },
    "autumn_warm": {
        "display_name": "Warm Autumn",
        "undertone": "warm, earthy, and golden",
        "colors": "rust, olive green, camel, burnt orange, chocolate brown, mustard, terracotta, copper",
    },
    "autumn_soft": {
        "display_name": "Soft Autumn",
        "undertone": "warm, muted, and gentle",
        "colors": "soft rust, muted olive, warm taupe, dusty orange, warm gray, muted gold, soft terracotta",
    },
    "autumn_deep": {
        "display_name": "Deep Autumn",
        "undertone": "deep, warm, and rich",
        "colors": "dark chocolate, deep rust, forest green, deep olive, burgundy, deep teal, dark camel",
    },
    "winter_cool": {
        "display_name": "Cool Winter",
        "undertone": "cool, clear, and high-contrast",
        "colors": "true red, cobalt blue, fuchsia, hot pink, emerald green, navy, black, crisp white, burgundy",
    },
    "winter_deep": {
        "display_name": "Deep Winter",
        "undertone": "deep, cool, and dramatic",
        "colors": "true black, deep navy, dark burgundy, deep plum, dark forest green, charcoal, graphite",
    },
    "winter_bright": {
        "display_name": "Bright Winter",
        "undertone": "cool, vivid, and high-contrast",
        "colors": "true white, black, bright cobalt, true red, hot pink, electric blue, emerald, fuchsia",
    },
}


def build_prompt(subtype_key: str, style_key: str) -> str:
    """Build the full prompt for a given sub-type and style."""
    st = SUBTYPES[subtype_key]
    style_desc = STYLE_DESC[style_key]

    prompt = (
        f"{LAYOUT} "
        f"The outfit is for a {st['display_name']} personal color season person. "
        f"This is a {style_desc} outfit. "
        f"Use ONLY these colors from the {st['display_name']} palette: {st['colors']}. "
        f"The undertone is {st['undertone']}. "
        f"All clothing items, accessories, and shoes must use colors from this palette. "
        f"Do not use any warm colors for cool sub-types or cool colors for warm sub-types. "
        f"The result should feel cohesive, stylish, and true to the {st['display_name']} color season."
    )
    return prompt


def generate_outfit_image(prompt: str, output_path: str) -> bool:
    """Generate one outfit image using Gemini Imagen."""
    try:
        model = genai.ImageGenerationModel("imagen-3.0-generate-002")
        result = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1",
            safety_filter_level="block_only_high",
        )

        if result.images:
            result.images[0].save(output_path)
            return True
        else:
            print(f"  ⚠ No image generated for {output_path}")
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def generate_all():
    """Generate all 72 outfit images."""
    print("=" * 60)
    print("  ToneFit ML — Outfit Image Generator")
    print("  12 sub-types × 6 styles = 72 images total")
    print("=" * 60)

    total = 0
    success = 0
    failed = []

    for subtype_key, subtype_data in SUBTYPES.items():
        # Create output folder
        folder = Path(f"outfits/{subtype_key}")
        folder.mkdir(parents=True, exist_ok=True)

        print(f"\n📁 {subtype_data['display_name']}")

        for style_key in STYLE_DESC.keys():
            output_path = str(folder / f"{style_key}.jpg")

            # Skip if already generated
            if os.path.exists(output_path):
                print(f"  ⏭  {style_key} — already exists, skipping")
                success += 1
                total += 1
                continue

            print(f"  🎨 Generating {style_key}...", end=" ", flush=True)

            prompt = build_prompt(subtype_key, style_key)
            ok = generate_outfit_image(prompt, output_path)

            if ok:
                print("✅")
                success += 1
            else:
                print("❌")
                failed.append(f"{subtype_key}/{style_key}")

            total += 1

            # Rate limit: wait 2 seconds between requests
            time.sleep(2)

    print("\n" + "=" * 60)
    print(f"  Done: {success}/{total} images generated successfully")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for f in failed:
            print(f"    - {f}")
    print("=" * 60)
    print(f"\n  Images saved to: outfits/[subtype]/[style].jpg")
    print(f"  Use these paths in tonefit_data_complete.json")


def print_all_prompts():
    """Print all 72 prompts without generating images — for review."""
    print("=" * 60)
    print("  All 72 Prompts")
    print("=" * 60)
    n = 1
    for subtype_key in SUBTYPES:
        for style_key in STYLE_DESC:
            print(f"\n[{n}] {SUBTYPES[subtype_key]['display_name']} — {style_key}")
            print(f"Output: outfits/{subtype_key}/{style_key}.jpg")
            print(f"Prompt: {build_prompt(subtype_key, style_key)}")
            print("-" * 60)
            n += 1


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--print-prompts":
        # Just print all prompts without generating
        print_all_prompts()
    else:
        # Generate all images
        generate_all()
