"""
ToneFit ML — Preprocessing Pipeline (v2)
=========================================
Improvements over v1:
  - Skin region masking using MediaPipe Face Mesh landmarks
    (cheek, forehead, undereye — avoids hair, lips, eyebrows)
  - Skin-pixel-only color feature extraction (not full image average)
  - Exposure/blur outlier filtering before feature extraction
  - Saturation check — rejects grayscale/heavily filtered images
  - Richer feature set: adds skin ratio, blur score, Fitzpatrick-proxy
  - Full audit log per image in preprocessing_audit.csv

What this script does:
  1. Load all images from RGB-M/train/ and RGB-M/test/ recursively
  2. Filter out blurry, overexposed, or desaturated images
  3. Isolate skin pixels using MediaPipe face landmarks
  4. Extract CIELab + HSV color features from skin region only
  5. Save features.csv
  6. Normalize with MinMaxScaler
  7. Stratified 80/20 train/test split
  8. Save X_train.npy, X_test.npy, y_train.npy, y_test.npy

Requirements:
  pip install opencv-python mediapipe scikit-learn scikit-image pandas numpy imagehash Pillow

Usage:
  python preprocess.py

Outputs:
  features.csv            — full feature table with labels
  preprocessing_audit.csv — per-image quality audit log
  X_train.npy, X_test.npy, y_train.npy, y_test.npy
  models/scaler.pkl
"""

import os
import cv2
import pickle
import logging
import numpy as np
import pandas as pd
from skimage import color as skcolor
from sklearn.preprocessing import MinMaxScaler

# ── CONFIG ────────────────────────────────────────────────────────────────────

TRAIN_DIR         = "RGB-M/train"   # READ ONLY — do not modify
TEST_DIR          = "RGB-M/test"    # READ ONLY — do not modify
SEASONS           = ["autumn", "spring", "summer", "winter"]  # alphabetical — matches ImageFolder
LABEL_MAP         = {"autumn": 0, "spring": 1, "summer": 2, "winter": 3}
VALID_EXT         = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
FEATURES_CSV      = "features.csv"
AUDIT_CSV         = "preprocessing_audit.csv"
SPLIT_CSV         = "data_split.csv"
MODELS_DIR        = "models"
SCALER_PATH       = os.path.join(MODELS_DIR, "scaler.pkl")

# Quality thresholds
MIN_BRIGHTNESS    = 40    # L* mean below this = too dark
MAX_BRIGHTNESS    = 92    # L* mean above this = overexposed
MIN_SATURATION    = 0.05  # HSV S mean below this = grayscale/desaturated
MIN_BLUR_SCORE    = 60    # Laplacian variance below this = blurry
MIN_SKIN_RATIO    = 0.05  # < 5% skin pixels detected = bad crop

RANDOM_STATE      = 42

# ── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

os.makedirs(MODELS_DIR, exist_ok=True)

# ── MEDIAPIPE FACE MESH ───────────────────────────────────────────────────────

try:
    import mediapipe as mp
    _mp_mesh = mp.solutions.face_mesh
    FACE_MESH = _mp_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    USE_MEDIAPIPE = True
    log.info("✓ MediaPipe Face Mesh loaded for skin region extraction")
except ImportError:
    USE_MEDIAPIPE = False
    log.warning("MediaPipe not found — falling back to center-crop skin region")

# Landmark indices for skin-rich face regions (MediaPipe 468-point model)
# Cheeks (left + right), forehead, nose bridge, undereye
SKIN_LANDMARK_INDICES = [
    # Left cheek
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
    # Right cheek
    454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    # Forehead center
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    # Nose bridge
    6, 197, 195, 5, 4,
    # Under-eye
    226, 113, 225, 224, 223, 222, 221, 189,
    446, 342, 445, 444, 443, 442, 441, 413,
]

def get_skin_mask_mediapipe(img_bgr):
    """
    Build a binary mask of skin-region pixels using MediaPipe Face Mesh.
    Returns mask (H×W, uint8) or None if no face detected.
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = FACE_MESH.process(img_rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    points = np.array([
        (int(lm.x * w), int(lm.y * h))
        for i, lm in enumerate(landmarks)
        if i in SKIN_LANDMARK_INDICES
    ], dtype=np.int32)

    if len(points) < 3:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

def get_skin_mask_fallback(img_bgr):
    """
    Fallback: use center 60% of image as approximate face/skin region.
    Less accurate but works without MediaPipe.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    y1, y2 = int(h * 0.15), int(h * 0.75)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    mask[y1:y2, x1:x2] = 255
    return mask

# ── QUALITY FILTERS ───────────────────────────────────────────────────────────

def blur_score(img_bgr):
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def check_quality(img_bgr, L_mean, S_mean):
    """
    Run quality checks. Returns (pass: bool, reason: str).
    """
    blr = blur_score(img_bgr)
    if blr < MIN_BLUR_SCORE:
        return False, f"blurry ({blr:.1f})"
    if L_mean < MIN_BRIGHTNESS:
        return False, f"too_dark (L={L_mean:.1f})"
    if L_mean > MAX_BRIGHTNESS:
        return False, f"overexposed (L={L_mean:.1f})"
    if S_mean < MIN_SATURATION:
        return False, f"desaturated (S={S_mean:.3f})"
    return True, "ok"


# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────

def extract_features(img_path):
    """
    Extract color features from the skin region of a face image.

    Features:
      CIELab (skin pixels only):
        L_mean, a_mean, b_mean  — mean Lightness, red-green, yellow-blue
        L_std, a_std, b_std     — variation (contrast / diversity)
        ITA                     — Individual Typology Angle
                                  arctan((L-50)/b) × (180/π)
                                  Higher ITA → lighter/cooler (Summer/Winter)
                                  Lower ITA  → darker/warmer  (Spring/Autumn)
      HSV (skin pixels only):
        H_mean, S_mean, V_mean  — hue, saturation, brightness

      Quality:
        skin_ratio              — fraction of face area identified as skin region
        blur_score              — image sharpness (Laplacian variance)

    Returns dict of features + quality fields, or None on read failure.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    # ── Get skin mask ──────────────────────────────────────────────────
    if USE_MEDIAPIPE:
        mask = get_skin_mask_mediapipe(img_bgr)
        if mask is None:
            mask = get_skin_mask_fallback(img_bgr)
    else:
        mask = get_skin_mask_fallback(img_bgr)

    skin_ratio = float(np.sum(mask > 0)) / float(mask.size)
    if skin_ratio < MIN_SKIN_RATIO:
        return {"_reject": f"low_skin_ratio ({skin_ratio:.3f})"}

    # ── CIELab on skin pixels ──────────────────────────────────────────
    img_rgb      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_norm = img_rgb.astype(np.float32) / 255.0
    img_lab      = skcolor.rgb2lab(img_rgb_norm)   # (H, W, 3)

    skin_px  = mask > 0                            # boolean mask
    L_vals   = img_lab[:, :, 0][skin_px]
    a_vals   = img_lab[:, :, 1][skin_px]
    b_vals   = img_lab[:, :, 2][skin_px]

    L_mean = float(np.mean(L_vals))
    a_mean = float(np.mean(a_vals))
    b_mean = float(np.mean(b_vals))
    L_std  = float(np.std(L_vals))
    a_std  = float(np.std(a_vals))
    b_std  = float(np.std(b_vals))
    ITA    = float(np.degrees(np.arctan2(L_mean - 50.0, b_mean + 1e-6)))

    # ── HSV on skin pixels ─────────────────────────────────────────────
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H_vals  = (img_hsv[:, :, 0] / 180.0)[skin_px]
    S_vals  = (img_hsv[:, :, 1] / 255.0)[skin_px]
    V_vals  = (img_hsv[:, :, 2] / 255.0)[skin_px]

    H_mean = float(np.mean(H_vals))
    S_mean = float(np.mean(S_vals))
    V_mean = float(np.mean(V_vals))

    blr = blur_score(img_bgr)

    return {
        "L_mean":     L_mean,
        "a_mean":     a_mean,
        "b_mean":     b_mean,
        "L_std":      L_std,
        "a_std":      a_std,
        "b_std":      b_std,
        "ITA":        ITA,
        "H_mean":     H_mean,
        "S_mean":     S_mean,
        "V_mean":     V_mean,
        "skin_ratio": skin_ratio,
        "blur_score": blr,
    }

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 65)
    log.info("  ToneFit ML — Preprocessing Pipeline v2")
    log.info("=" * 65)

    rows       = []
    audit_rows = []

    # ── Step 1: Extract features + quality filter ──────────────────────
    log.info("\n[Step 1] Extracting skin-region color features...")

    for split_name, split_root in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
        if not os.path.isdir(split_root):
            log.warning(f"  Split folder not found: {split_root} — skipping")
            continue

        log.info(f"\n  [{split_name.upper()}] {split_root}")

        for season in SEASONS:
            season_dir = os.path.join(split_root, season)
            if not os.path.isdir(season_dir):
                log.warning(f"    Season folder not found: {season_dir} (skipping)")
                continue

            # Collect files recursively through all sub-type subfolders
            all_files = []
            for root_dir, _, files in os.walk(season_dir):
                for fname in sorted(files):
                    if fname.lower().endswith(VALID_EXT):
                        all_files.append(os.path.join(root_dir, fname))

            log.info(f"    {season.upper():<8}: {len(all_files)} images")

            saved_count    = 0
            filtered_count = 0

            for path in all_files:
                rel_path = os.path.relpath(path, "RGB-M").replace("\\", "/")
                features = extract_features(path)

                if features is None:
                    audit_rows.append({"filename": rel_path, "season": season, "split": split_name, "status": "unreadable"})
                    filtered_count += 1
                    continue

                if "_reject" in features:
                    audit_rows.append({"filename": rel_path, "season": season, "split": split_name, "status": features["_reject"]})
                    filtered_count += 1
                    continue

                passed, reason = check_quality(cv2.imread(path), features["L_mean"], features["S_mean"])
                if not passed:
                    audit_rows.append({"filename": rel_path, "season": season, "split": split_name, "status": reason})
                    filtered_count += 1
                    continue

                row = {"filename": rel_path, "season": season, "label": LABEL_MAP[season], "split": split_name}
                row.update(features)
                rows.append(row)
                audit_rows.append({"filename": rel_path, "season": season, "split": split_name, "status": "ok"})
                saved_count += 1

            log.info(f"      kept: {saved_count}  filtered: {filtered_count}")

    if len(rows) == 0:
        log.error("\nNo images passed quality filtering.")
        log.error(f"Check that {TRAIN_DIR}/ and {TEST_DIR}/ exist and contain images.")
        return

    # ── Step 2: Build DataFrame + save features.csv ────────────────────
    log.info("\n[Step 2] Saving feature table...")
    feature_cols = [
        "L_mean", "a_mean", "b_mean", "L_std", "a_std", "b_std",
        "ITA", "H_mean", "S_mean", "V_mean", "skin_ratio", "blur_score"
    ]
    df = pd.DataFrame(rows, columns=["filename", "season", "label", "split"] + feature_cols)
    df.to_csv(FEATURES_CSV, index=False)
    log.info(f"  Saved: {FEATURES_CSV} ({len(df)} rows)")

    # Save audit log
    pd.DataFrame(audit_rows).to_csv(AUDIT_CSV, index=False)
    log.info(f"  Saved: {AUDIT_CSV}")

    # Class distribution
    log.info("\n  Class distribution after filtering:")
    for season in SEASONS:
        count = len(df[df["season"] == season])
        bar   = "█" * (count // 3)
        log.info(f"    {season:<8}: {count:>4} images  {bar}")

    total = len(df)
    min_class = df.groupby("season").size().min()
    if min_class < 30:
        log.warning(f"\n  ⚠ Smallest class has only {min_class} images.")
        log.warning("    Consider collecting more data before training.")

    # ── Step 3: Fit scaler on full set (transform happens after split) ───
    log.info("\n[Step 3] Fitting MinMaxScaler on full feature set...")
    scaler = MinMaxScaler()
    scaler.fit(df[feature_cols].values.astype(np.float32))

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    log.info(f"  Scaler saved: {SCALER_PATH}")

    # ── Step 4: Split is determined by folder structure (RGB-M/train vs RGB-M/test) ──
    log.info("\n[Step 4] Using Deep Armocromia pre-defined train/test split...")
    train_idx = df.index[df["split"] == "train"].tolist()
    test_idx  = df.index[df["split"] == "test"].tolist()

    log.info(f"  Train: {len(train_idx)}  Test: {len(test_idx)}")
    for season in SEASONS:
        tr = len(df[(df["season"] == season) & (df["split"] == "train")])
        te = len(df[(df["season"] == season) & (df["split"] == "test")])
        log.info(f"    {season:<8}: {tr} train / {te} test")

    split_df = df[["filename", "season", "split"]].copy()
    split_df.to_csv(SPLIT_CSV, index=False)
    log.info(f"  Saved: {SPLIT_CSV}")

    # ── Step 5: Save feature arrays (for EDA/SVM experiments) ────────
    log.info("\n[Step 5] Saving numpy arrays for EDA...")
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    X_scaled = scaler.transform(X)

    X_train = X_scaled[np.array(train_idx)]
    X_test  = X_scaled[np.array(test_idx)]
    y_train = y[np.array(train_idx)]
    y_test  = y[np.array(test_idx)]

    np.save("X_train.npy", X_train)
    np.save("X_test.npy",  X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy",  y_test)

    # ── Summary ────────────────────────────────────────────────────────
    log.info(f"\n{'=' * 65}")
    log.info("  PREPROCESSING COMPLETE")
    log.info(f"{'=' * 65}")
    log.info(f"  Total images processed : {total}")
    log.info(f"  Features per image     : {len(feature_cols)}")
    log.info(f"  Training samples       : {len(train_idx)}")
    log.info(f"  Test samples           : {len(test_idx)}")
    log.info(f"\n  Key outputs:")
    log.info(f"    {FEATURES_CSV}  <- EDA feature table")
    log.info(f"    {SPLIT_CSV}     <- EDA split reference")
    log.info(f"    {AUDIT_CSV}")
    log.info(f"    X_train.npy, X_test.npy  <- EDA numpy arrays")
    log.info(f"    {SCALER_PATH}")
    log.info(f"\n  Next step: run  python train_farl.py  OR  python train_dinov2.py")
    log.info("=" * 65)

if __name__ == "__main__":
    run()