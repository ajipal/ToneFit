"""
ToneFit ML — Prediction Demo (Step 7)
=======================================
Takes a face photo and predicts the personal color season using all 3 models.

What this script does:
  1. Detects and crops the face from the input image (OpenCV Haar Cascade)
  2. Extracts CIELab + HSV color features (same as preprocess.py)
  3. Runs prediction through SVM, Random Forest, and MobileNetV2
  4. Shows a combined result display:
       - Predicted season per model with confidence bar
       - Majority vote (ensemble) across all 3 models
       - Color palette swatches for the predicted season
       - 2-3 outfit sample photos from outfits/[season]/

Usage:
  python predict.py path/to/photo.jpg

  # Example:
  python predict.py test_photo.jpg

  # To skip the display and just print results:
  python predict.py photo.jpg --no-display

Requirements:
  pip install opencv-python scikit-learn scikit-image tensorflow numpy matplotlib Pillow

Outfit images:
  Place sample outfit photos in:
    outfits/spring/   (2-3 .jpg or .png images)
    outfits/summer/
    outfits/autumn/
    outfits/winter/
  These are pre-curated reference photos — not virtual try-on.
"""

import os
import sys
import pickle
import logging
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
from skimage import color as skcolor
import tensorflow as tf

# ─── CONFIG ───────────────────────────────────────────────────────────────────

MODELS_DIR  = "models"
OUTFITS_DIR = "outfits"
VALID_EXT   = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SEASONS     = ["spring", "summer", "autumn", "winter"]
MIN_FACE_SIZE = 60

# Season color palettes — hex codes for display swatches
# Based on Korean personal color analysis best-color recommendations
SEASON_PALETTES = {
    "spring": {
        "label": "Spring — Warm, Light, Clear",
        "colors": ["#F4A261", "#E76F51", "#FFD166", "#90E0EF", "#FFDDD2",
                   "#C9B99A", "#F7C59F", "#E8A598"],
        "description": (
            "Warm undertone · Light contrast · Clear and bright\n"
            "Best colors: peach, coral, warm turquoise, golden yellow, ivory"
        ),
    },
    "summer": {
        "label": "Summer — Cool, Light, Muted",
        "colors": ["#90B4CE", "#C9ADA7", "#B5C0D0", "#D4B4C8", "#A8DADC",
                   "#F2E9E4", "#CDB4DB", "#BDE0FE"],
        "description": (
            "Cool undertone · Light contrast · Soft and muted\n"
            "Best colors: lavender, dusty rose, powder blue, soft mauve, rose beige"
        ),
    },
    "autumn": {
        "label": "Autumn — Warm, Deep, Muted",
        "colors": ["#A0522D", "#C68642", "#8B4513", "#D2691E", "#BC8A5F",
                   "#6B8E23", "#8D6E4B", "#CD853F"],
        "description": (
            "Warm undertone · Deep contrast · Muted and earthy\n"
            "Best colors: rust, camel, olive green, burnt orange, warm brown"
        ),
    },
    "winter": {
        "label": "Winter — Cool, Deep, Clear",
        "colors": ["#2B2D42", "#FFFFFF", "#1A1A2E", "#5B5EA6", "#C1121F",
                   "#0077B6", "#6A0572", "#023E8A"],
        "description": (
            "Cool undertone · Deep contrast · Clear and vivid\n"
            "Best colors: black, white, royal blue, jewel tones, true red"
        ),
    },
}

# Season accent colors for UI
SEASON_ACCENT = {
    "spring": "#F4A261",
    "summer": "#90B4CE",
    "autumn": "#A0522D",
    "winter": "#5B5EA6",
}

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── FACE DETECTION ───────────────────────────────────────────────────────────

def detect_and_crop_face(img_path):
    """
    Detect the largest face in the image and return a 224x224 crop.
    Returns (face_bgr, face_rgb) or raises RuntimeError if no face found.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
    )

    if len(faces) == 0:
        raise RuntimeError(
            "No face detected in the image.\n"
            "Tips: use a clear, well-lit, front-facing photo."
        )

    # Pick the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # 20% padding
    pad = int(0.20 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad)
    y2 = min(img_bgr.shape[0], y + h + pad)

    face_bgr = cv2.resize(img_bgr[y1:y2, x1:x2], (224, 224))
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    log.info(f"  Face detected at x={x}, y={y}, w={w}, h={h}")
    return face_bgr, face_rgb


# ─── FEATURE EXTRACTION ───────────────────────────────────────────────────────

def extract_features(face_bgr):
    """
    Extract the same 10 color features used during preprocessing.
    Returns a (1, 10) numpy array ready for SVM/RF prediction.
    """
    img_rgb      = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_norm = img_rgb.astype(np.float32) / 255.0
    img_lab      = skcolor.rgb2lab(img_rgb_norm)

    L = img_lab[:, :, 0]
    a = img_lab[:, :, 1]
    b = img_lab[:, :, 2]

    L_mean = float(np.mean(L))
    a_mean = float(np.mean(a))
    b_mean = float(np.mean(b))
    L_std  = float(np.std(L))
    a_std  = float(np.std(a))
    b_std  = float(np.std(b))
    ITA    = float(np.degrees(np.arctan((L_mean - 50.0) / b_mean))) if abs(b_mean) > 1e-6 else 0.0

    img_hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H_mean  = float(np.mean(img_hsv[:, :, 0] / 180.0))
    S_mean  = float(np.mean(img_hsv[:, :, 1] / 255.0))
    V_mean  = float(np.mean(img_hsv[:, :, 2] / 255.0))

    features = np.array([[L_mean, a_mean, b_mean, L_std, a_std, b_std,
                          ITA, H_mean, S_mean, V_mean]], dtype=np.float32)

    log.info(f"  L*={L_mean:.1f}  a*={a_mean:.1f}  b*={b_mean:.1f}  ITA={ITA:.1f}")
    return features


# ─── MODEL LOADING ────────────────────────────────────────────────────────────

def load_models():
    """Load all available trained models. Skips any that are missing."""
    models_loaded = {}

    # Scaler (required for SVM and RF)
    scaler = None
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        log.warning("  scaler.pkl not found — SVM and RF will use unscaled features.")

    # SVM
    svm_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    if os.path.exists(svm_path):
        with open(svm_path, "rb") as f:
            models_loaded["SVM"] = pickle.load(f)
        log.info("  Loaded: SVM")
    else:
        log.warning(f"  {svm_path} not found — skipping.")

    # Random Forest
    rf_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    if os.path.exists(rf_path):
        with open(rf_path, "rb") as f:
            models_loaded["Random Forest"] = pickle.load(f)
        log.info("  Loaded: Random Forest")
    else:
        log.warning(f"  {rf_path} not found — skipping.")

    # MobileNetV2
    dl_path = os.path.join(MODELS_DIR, "mobilenetv2_model.h5")
    if os.path.exists(dl_path):
        models_loaded["MobileNetV2"] = tf.keras.models.load_model(dl_path)
        log.info("  Loaded: MobileNetV2")
    else:
        log.warning(f"  {dl_path} not found — skipping.")

    return models_loaded, scaler


# ─── PREDICTION ───────────────────────────────────────────────────────────────

def predict_all(models_loaded, scaler, features_raw, face_bgr):
    """
    Run predictions through all loaded models.
    Returns a dict: {model_name: {"season": str, "confidence": float, "probs": array}}
    """
    predictions = {}

    # Scale features for traditional ML models
    features_scaled = scaler.transform(features_raw) if scaler is not None else features_raw

    for name, model in models_loaded.items():

        if name in ("SVM", "Random Forest"):
            probs      = model.predict_proba(features_scaled)[0]   # shape: (4,)
            label_idx  = int(np.argmax(probs))
            confidence = float(probs[label_idx])

        elif name == "MobileNetV2":
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            face_rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            img_array = preprocess_input(face_rgb.astype(np.float32))
            img_array = np.expand_dims(img_array, axis=0)           # (1, 224, 224, 3)
            probs      = model.predict(img_array, verbose=0)[0]     # shape: (4,)
            label_idx  = int(np.argmax(probs))
            confidence = float(probs[label_idx])

        else:
            continue

        predictions[name] = {
            "season":     SEASONS[label_idx],
            "confidence": confidence,
            "probs":      probs,
        }
        log.info(f"  {name:<20}: {SEASONS[label_idx].upper():<8} ({confidence*100:.1f}%)")

    return predictions


def ensemble_vote(predictions):
    """
    Majority vote across all models.
    If tied, use the season with the highest average probability.
    """
    if not predictions:
        return None

    votes = {}
    avg_probs = np.zeros(4)

    for pred in predictions.values():
        season = pred["season"]
        votes[season] = votes.get(season, 0) + 1
        avg_probs += pred["probs"]

    avg_probs /= len(predictions)
    max_votes  = max(votes.values())
    top_seasons = [s for s, v in votes.items() if v == max_votes]

    if len(top_seasons) == 1:
        winner = top_seasons[0]
    else:
        # Tiebreak by average probability
        winner = SEASONS[int(np.argmax(avg_probs))]

    return {
        "season":     winner,
        "confidence": float(avg_probs[SEASONS.index(winner)]),
        "votes":      votes,
        "avg_probs":  avg_probs,
    }


# ─── RESULT DISPLAY ───────────────────────────────────────────────────────────

def load_outfit_images(season, max_outfits=3):
    """Load pre-curated outfit sample images for the predicted season."""
    outfit_dir = os.path.join(OUTFITS_DIR, season)
    if not os.path.isdir(outfit_dir):
        return []

    files = sorted([
        f for f in os.listdir(outfit_dir)
        if f.lower().endswith(VALID_EXT)
    ])[:max_outfits]

    outfits = []
    for fname in files:
        path = os.path.join(outfit_dir, fname)
        img  = cv2.imread(path)
        if img is not None:
            outfits.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return outfits


def display_results(face_rgb, predictions, ensemble, show=True, save_path=None):
    """
    Build and show the full result display:
      Row 1: Face crop | Model predictions with confidence bars
      Row 2: Color palette swatches
      Row 3: Outfit sample photos (if available)
    """
    season   = ensemble["season"]
    palette  = SEASON_PALETTES[season]
    accent   = SEASON_ACCENT[season]
    outfits  = load_outfit_images(season)
    n_models = len(predictions)

    # ── Layout ────────────────────────────────────────────────────────────
    has_outfits   = len(outfits) > 0
    n_outfit_cols = len(outfits) if has_outfits else 0
    fig_height    = 10 + (3 if has_outfits else 0)

    fig = plt.figure(figsize=(14, fig_height), facecolor="#F8F8F8")

    if has_outfits:
        gs = GridSpec(3, max(n_models + 1, n_outfit_cols),
                      figure=fig, hspace=0.45, wspace=0.3)
    else:
        gs = GridSpec(2, n_models + 1,
                      figure=fig, hspace=0.45, wspace=0.3)

    # ── Title ─────────────────────────────────────────────────────────────
    fig.suptitle(
        f"ToneFit ML — Personal Color Result: {season.upper()}",
        fontsize=16, fontweight="bold", color=accent, y=0.98
    )

    # ── Face crop ─────────────────────────────────────────────────────────
    ax_face = fig.add_subplot(gs[0, 0])
    ax_face.imshow(face_rgb)
    ax_face.set_title("Detected Face", fontsize=11, fontweight="bold")
    ax_face.axis("off")
    for spine in ax_face.spines.values():
        spine.set_edgecolor(accent)
        spine.set_linewidth(2)

    # ── Per-model confidence bars ──────────────────────────────────────────
    for col, (model_name, pred) in enumerate(predictions.items(), start=1):
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(model_name, fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(SEASONS) - 0.5)
        ax.axis("off")

        for i, (season_name, prob) in enumerate(zip(SEASONS, pred["probs"])):
            is_predicted = season_name == pred["season"]
            bar_color    = SEASON_ACCENT[season_name] if is_predicted else "#DDDDDD"
            text_weight  = "bold" if is_predicted else "normal"

            ax.barh(i, prob, color=bar_color, height=0.6, edgecolor="white")
            ax.text(prob + 0.02, i, f"{prob*100:.1f}%",
                    va="center", fontsize=8.5, fontweight=text_weight)
            ax.text(-0.02, i, season_name.capitalize(),
                    va="center", ha="right", fontsize=9, fontweight=text_weight)

        ax.text(0.5, -0.45,
                f"→ {pred['season'].upper()} ({pred['confidence']*100:.0f}%)",
                ha="center", va="top", fontsize=9, fontweight="bold",
                color=SEASON_ACCENT[pred["season"]],
                transform=ax.transAxes)

    # ── Ensemble result + color palette ───────────────────────────────────
    ax_palette = fig.add_subplot(gs[1, :])
    ax_palette.set_xlim(0, 1)
    ax_palette.set_ylim(0, 1)
    ax_palette.axis("off")

    # Ensemble label
    vote_str = "  |  ".join([f"{s.capitalize()}: {v}" for s, v in ensemble["votes"].items()])
    ax_palette.text(
        0.5, 0.97,
        f"Ensemble Result: {season.upper()}   (Votes: {vote_str})",
        ha="center", va="top", fontsize=13, fontweight="bold",
        color=accent, transform=ax_palette.transAxes,
    )
    ax_palette.text(
        0.5, 0.80,
        palette["description"],
        ha="center", va="top", fontsize=10, color="#444444",
        transform=ax_palette.transAxes,
    )

    # Color swatches
    swatch_colors = palette["colors"]
    n_swatches    = len(swatch_colors)
    swatch_width  = 0.85 / n_swatches
    swatch_start  = (1 - 0.85) / 2

    for i, hex_color in enumerate(swatch_colors):
        x = swatch_start + i * swatch_width
        rect = mpatches.FancyBboxPatch(
            (x + 0.005, 0.08),
            swatch_width - 0.01, 0.42,
            boxstyle="round,pad=0.01",
            facecolor=hex_color,
            edgecolor="white",
            linewidth=1.5,
            transform=ax_palette.transAxes,
            clip_on=False,
        )
        ax_palette.add_patch(rect)
        ax_palette.text(
            x + swatch_width / 2, 0.04,
            hex_color.upper(),
            ha="center", va="top",
            fontsize=6.5, color="#555555",
            transform=ax_palette.transAxes,
        )

    # ── Outfit samples ─────────────────────────────────────────────────────
    if has_outfits:
        for col, outfit_img in enumerate(outfits):
            ax_out = fig.add_subplot(gs[2, col])
            ax_out.imshow(outfit_img)
            ax_out.axis("off")
            ax_out.set_title(f"Outfit {col+1}", fontsize=9, color="#666666")
            if col == 0:
                ax_out.set_ylabel(
                    f"{season.capitalize()} Outfits",
                    fontsize=10, fontweight="bold", color=accent
                )
    else:
        # Show a message if no outfit images are available
        ax_msg = fig.add_subplot(gs[1, :]) if not has_outfits else None
        if ax_msg is None:
            pass
        else:
            ax_msg.text(
                0.5, 0.1,
                f"Add outfit reference photos to outfits/{season}/ to see them here.",
                ha="center", va="center", fontsize=9, color="#999999",
                transform=ax_msg.transAxes, style="italic"
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#F8F8F8")
        log.info(f"  Result saved: {save_path}")

    if show:
        plt.show()

    plt.close()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run(image_path, show_display=True):
    log.info("=" * 60)
    log.info("  ToneFit ML — Prediction Demo")
    log.info("=" * 60)
    log.info(f"\n  Input image: {image_path}")

    # ── Face detection ─────────────────────────────────────────────────────
    log.info("\n[Step 1] Detecting face...")
    face_bgr, face_rgb = detect_and_crop_face(image_path)

    # ── Feature extraction ─────────────────────────────────────────────────
    log.info("\n[Step 2] Extracting color features...")
    features_raw = extract_features(face_bgr)

    # ── Load models ────────────────────────────────────────────────────────
    log.info("\n[Step 3] Loading models...")
    models_loaded, scaler = load_models()

    if not models_loaded:
        log.error("No trained models found. Run train_traditional.py and train_deeplearning.py first.")
        return

    # ── Run predictions ────────────────────────────────────────────────────
    log.info("\n[Step 4] Running predictions...")
    predictions = predict_all(models_loaded, scaler, features_raw, face_bgr)

    # ── Ensemble vote ──────────────────────────────────────────────────────
    ensemble = ensemble_vote(predictions)
    log.info(f"\n  Ensemble result: {ensemble['season'].upper()} ({ensemble['confidence']*100:.1f}%)")

    # ── Display ────────────────────────────────────────────────────────────
    img_name  = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join("results", f"prediction_{img_name}.png")
    os.makedirs("results", exist_ok=True)

    log.info("\n[Step 5] Building result display...")
    display_results(face_rgb, predictions, ensemble,
                    show=show_display, save_path=save_path)

    # ── Print final summary ────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  PREDICTION RESULT")
    log.info("=" * 60)
    log.info(f"  Personal Color Season : {ensemble['season'].upper()}")
    log.info(f"  {SEASON_PALETTES[ensemble['season']]['label']}")
    log.info(f"\n  Individual model predictions:")
    for model_name, pred in predictions.items():
        log.info(f"    {model_name:<20}: {pred['season'].upper():<8} ({pred['confidence']*100:.1f}%)")
    log.info(f"\n  Best colors for you:")
    for hex_color in SEASON_PALETTES[ensemble["season"]]["colors"][:4]:
        log.info(f"    {hex_color}")
    log.info("=" * 60)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ToneFit ML — Predict personal color season from a face photo."
    )
    parser.add_argument(
        "image",
        help="Path to the input face photo (jpg, png, etc.)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip the matplotlib display window (still saves the result image)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        log.error(f"Image not found: {args.image}")
        sys.exit(1)

    run(args.image, show_display=not args.no_display)
