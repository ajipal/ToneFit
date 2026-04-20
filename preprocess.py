"""
ToneFit ML — Preprocessing Script (Step 2)
============================================
Processes raw face crops into a clean feature dataset ready for ML training.

What this script does:
  1. Removes duplicate images using perceptual hashing
  2. Converts each face image to CIELab and HSV color spaces
  3. Extracts color features from the face region:
       - L*, a*, b* mean and std (CIELab)
       - ITA score (Individual Typology Angle) — key skin tone metric
       - H, S, V mean (HSV)
  4. Saves all features to features.csv
  5. Encodes season labels as integers (spring=0, summer=1, autumn=2, winter=3)
  6. Normalizes features using MinMaxScaler
  7. Splits into 80% train / 20% test (stratified, random_state=42)
  8. Saves: X_train.npy, X_test.npy, y_train.npy, y_test.npy

Requirements:
  pip install opencv-python scikit-learn scikit-image pandas numpy imagehash Pillow

Usage:
  python preprocess.py

Outputs:
  features.csv         — full feature table with labels
  X_train.npy          — training features (normalized)
  X_test.npy           — test features (normalized)
  y_train.npy          — training labels (integers)
  y_test.npy           — test labels (integers)
  models/scaler.pkl    — fitted MinMaxScaler (save for use during prediction)
"""

import os
import logging
import pickle

import cv2
import numpy as np
import pandas as pd
import imagehash
from PIL import Image
from skimage import color as skcolor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATASET_DIR  = "dataset"
SEASONS      = ["spring", "summer", "autumn", "winter"]
LABEL_MAP    = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}
VALID_EXT    = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

FEATURES_CSV = "features.csv"
MODELS_DIR   = "models"
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")

HASH_THRESHOLD = 5   # Hamming distance — images closer than this are duplicates
RANDOM_STATE   = 42
TEST_SIZE      = 0.20

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── SETUP ────────────────────────────────────────────────────────────────────

os.makedirs(MODELS_DIR, exist_ok=True)

# ─── STEP 1: DUPLICATE REMOVAL ────────────────────────────────────────────────

def remove_duplicates(season_dir):
    """
    Find and delete visually duplicate images using perceptual hashing.

    Perceptual hashing (pHash) compares image content — not file bytes.
    Two images with a Hamming distance below HASH_THRESHOLD are considered
    duplicates. The second occurrence is deleted.

    Returns the number of duplicates removed.
    """
    files = sorted([
        f for f in os.listdir(season_dir)
        if f.lower().endswith(VALID_EXT)
    ])

    hashes  = {}
    removed = 0

    for fname in files:
        path = os.path.join(season_dir, fname)
        try:
            img  = Image.open(path)
            phash = imagehash.phash(img)
        except Exception as e:
            log.warning(f"    Could not hash {fname}: {e}")
            continue

        duplicate = False
        for existing_hash in hashes:
            if phash - existing_hash < HASH_THRESHOLD:
                log.info(f"    Duplicate removed: {fname}")
                os.remove(path)
                removed += 1
                duplicate = True
                break

        if not duplicate:
            hashes[phash] = fname

    return removed


# ─── STEP 2 & 3: FEATURE EXTRACTION ──────────────────────────────────────────

def extract_features(img_path):
    """
    Extract CIELab and HSV color features from a single face image.

    Features extracted:
      CIELab space:
        L_mean  — mean Lightness (0=black, 100=white)
        a_mean  — mean red-green axis (positive = reddish)
        b_mean  — mean yellow-blue axis (positive = yellowish, warmer)
        L_std   — standard deviation of Lightness (contrast measure)
        a_std   — standard deviation of a*
        b_std   — standard deviation of b*
        ITA     — Individual Typology Angle
                  Formula: arctan((L* - 50) / b*) × (180/π)
                  Higher ITA → lighter/cooler skin (Summer/Winter)
                  Lower ITA  → darker/warmer skin (Spring/Autumn)

      HSV space:
        H_mean  — mean Hue (0–360°)
        S_mean  — mean Saturation (0–1)
        V_mean  — mean Value / brightness (0–1)

    Returns a dict of features, or None if the image cannot be read.
    """
    # Read image (OpenCV loads as BGR)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        log.warning(f"    Cannot read image: {img_path}")
        return None

    # ── CIELab conversion ──────────────────────────────────────────────────
    # Convert BGR → RGB → CIELab
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_norm = img_rgb.astype(np.float32) / 255.0  # skimage expects [0,1]
    img_lab = skcolor.rgb2lab(img_rgb_norm)             # shape: (H, W, 3)

    L = img_lab[:, :, 0]   # Lightness channel
    a = img_lab[:, :, 1]   # red-green channel
    b = img_lab[:, :, 2]   # yellow-blue channel

    L_mean = float(np.mean(L))
    a_mean = float(np.mean(a))
    b_mean = float(np.mean(b))
    L_std  = float(np.std(L))
    a_std  = float(np.std(a))
    b_std  = float(np.std(b))

    # ITA score — avoid division by zero
    # Higher ITA = lighter and cooler skin tone
    if abs(b_mean) < 1e-6:
        ITA = 0.0
    else:
        ITA = float(np.degrees(np.arctan((L_mean - 50.0) / b_mean)))

    # ── HSV conversion ─────────────────────────────────────────────────────
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # OpenCV HSV range: H=[0,180], S=[0,255], V=[0,255] — normalize all to [0,1]
    H = img_hsv[:, :, 0] / 180.0
    S = img_hsv[:, :, 1] / 255.0
    V = img_hsv[:, :, 2] / 255.0

    H_mean = float(np.mean(H))
    S_mean = float(np.mean(S))
    V_mean = float(np.mean(V))

    return {
        "L_mean": L_mean,
        "a_mean": a_mean,
        "b_mean": b_mean,
        "L_std":  L_std,
        "a_std":  a_std,
        "b_std":  b_std,
        "ITA":    ITA,
        "H_mean": H_mean,
        "S_mean": S_mean,
        "V_mean": V_mean,
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info("  ToneFit ML — Preprocessing")
    log.info("=" * 60)

    # ── Step 1: Remove duplicates ──────────────────────────────────────────
    log.info("\n[Step 1] Removing duplicate images...")
    total_dupes = 0
    for season in SEASONS:
        season_dir = os.path.join(DATASET_DIR, season)
        if not os.path.isdir(season_dir):
            log.warning(f"  Season folder not found: {season_dir}  (skipping)")
            continue
        dupes = remove_duplicates(season_dir)
        log.info(f"  {season:<8}: {dupes} duplicate(s) removed")
        total_dupes += dupes
    log.info(f"  Total duplicates removed: {total_dupes}")

    # ── Step 2 & 3: Extract features ──────────────────────────────────────
    log.info("\n[Step 2] Extracting color features from face images...")
    log.info("  (This may take a few minutes depending on dataset size)")

    rows = []

    for season in SEASONS:
        season_dir = os.path.join(DATASET_DIR, season)
        if not os.path.isdir(season_dir):
            log.warning(f"  Season folder not found: {season_dir}  (skipping)")
            continue

        files = sorted([
            f for f in os.listdir(season_dir)
            if f.lower().endswith(VALID_EXT)
        ])

        log.info(f"  {season.upper():<8}: {len(files)} images")

        for fname in files:
            path     = os.path.join(season_dir, fname)
            features = extract_features(path)

            if features is None:
                continue  # skip unreadable images

            row = {
                "filename": fname,
                "season":   season,
                "label":    LABEL_MAP[season],
            }
            row.update(features)
            rows.append(row)

    if len(rows) == 0:
        log.error("\nNo images processed. Check that dataset/ folders contain images.")
        log.error("Run collect_data.py first, then re-run this script.")
        return

    # ── Step 4: Save features.csv ──────────────────────────────────────────
    log.info("\n[Step 3] Saving feature table...")

    df = pd.DataFrame(rows, columns=[
        "filename", "season", "label",
        "L_mean", "a_mean", "b_mean",
        "L_std",  "a_std",  "b_std",
        "ITA",
        "H_mean", "S_mean", "V_mean",
    ])

    df.to_csv(FEATURES_CSV, index=False)
    log.info(f"  Saved: {FEATURES_CSV}  ({len(df)} rows)")

    # Print class distribution
    log.info("\n  Class distribution:")
    for season in SEASONS:
        count = len(df[df["season"] == season])
        bar   = "█" * (count // 3)
        log.info(f"    {season:<8}: {count:>4} images  {bar}")

    # ── Step 5: Normalize features ─────────────────────────────────────────
    log.info("\n[Step 4] Normalizing features with MinMaxScaler...")

    feature_cols = ["L_mean", "a_mean", "b_mean", "L_std", "a_std", "b_std",
                    "ITA", "H_mean", "S_mean", "V_mean"]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler for use during prediction
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    log.info(f"  Scaler saved: {SCALER_PATH}")

    # ── Step 6: Train/test split ───────────────────────────────────────────
    log.info("\n[Step 5] Splitting into train (80%) / test (20%)...")
    log.info("  (stratified split — preserves class balance in both sets)")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    log.info(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
    log.info(f"  X_test : {X_test.shape}    y_test : {y_test.shape}")

    # Print per-class split counts
    for season, label in LABEL_MAP.items():
        tr = int((y_train == label).sum())
        te = int((y_test  == label).sum())
        log.info(f"    {season:<8}: {tr} train  /  {te} test")

    # ── Step 7: Save numpy arrays ──────────────────────────────────────────
    log.info("\n[Step 6] Saving numpy arrays...")

    np.save("X_train.npy", X_train)
    np.save("X_test.npy",  X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy",  y_test)

    log.info("  Saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")

    # ── Summary ────────────────────────────────────────────────────────────
    log.info(f"\n{'=' * 60}")
    log.info("  DONE — Preprocessing complete")
    log.info(f"{'=' * 60}")
    log.info(f"  Total images processed : {len(df)}")
    log.info(f"  Features per image     : {len(feature_cols)}")
    log.info(f"  Training samples       : {len(X_train)}")
    log.info(f"  Test samples           : {len(X_test)}")
    log.info(f"\n  Output files:")
    log.info(f"    {FEATURES_CSV}")
    log.info(f"    X_train.npy, X_test.npy")
    log.info(f"    y_train.npy, y_test.npy")
    log.info(f"    {SCALER_PATH}")
    log.info(f"\n  Next step → run: python train_traditional.py")
    log.info("=" * 60)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
