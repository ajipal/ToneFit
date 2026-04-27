"""
ToneFit ML — Data Collection Pipeline (v2)
===========================================
Improvements over v1:
  - MediaPipe face detection (replaces weak Haar cascade)
  - Automatic white balance correction before saving crops
  - Per-celebrity duplicate detection (MD5 + pHash)
  - Metadata manifest (dataset_manifest.csv) — tracks celeb, season, nationality
  - Confidence filtering — rejects low-confidence face detections
  - Consistent 224x224 crops with 25% padding

Output structure:
  dataset/
    spring/   ← labeled face crops
    summer/
    autumn/
    winter/
    raw/      ← original downloads (kept for audit)
    rejected/ ← no face detected or low confidence
  dataset_manifest.csv  ← full audit trail

Requirements:
  pip install icrawler opencv-python mediapipe imagehash Pillow pandas

Usage:
  python collectdata.py
  python collectdata.py --seasons spring summer   ← run specific seasons only
"""

import os
import cv2
import shutil
import hashlib
import logging
import argparse
import imagehash
import pandas as pd
import numpy as np
from PIL import Image
from icrawler.builtin import BingImageCrawler

# ── CONFIG ────────────────────────────────────────────────────────────────────

IMAGES_PER_CELEBRITY = 20      # Download more, filter down to best (need ~5–7 saved per celeb)
MIN_FACE_CONFIDENCE  = 0.75    # MediaPipe detection confidence threshold
MIN_FACE_SIZE_PX     = 80      # Minimum face bounding box (pixels)
CROP_SIZE            = 224     # Output image size
PHASH_THRESHOLD      = 8       # Hamming distance for near-duplicate detection
BASE_DIR             = "dataset"
RAW_DIR              = os.path.join(BASE_DIR, "raw")
REJECTED_DIR         = os.path.join(BASE_DIR, "rejected")
MANIFEST_CSV         = "dataset_manifest.csv"

# ── CELEBRITY LIST ────────────────────────────────────────────────────────────
# Labels sourced from publicly documented professional color analyses.
# Format: (search_query, celebrity_name, nationality)

CELEBRITIES = {
    "spring": [
        # Filipino
        ("Anne Curtis actress Philippines", "Anne Curtis", "Filipino"),
        ("Julia Barretto actress Philippines", "Julia Barretto", "Filipino"),
        ("Marian Rivera actress Philippines", "Marian Rivera", "Filipino"),
        ("James Reid actor Philippines", "James Reid", "Filipino"),
        # Korean
        ("IU Korean singer kpop", "IU", "Korean"),
        ("Yoona SNSD Korean singer", "Yoona", "Korean"),
        ("Kim Chaewon Le Sserafim kpop", "Kim Chaewon", "Korean"),
        ("Kim Soo-hyun Korean actor", "Kim Soo-hyun", "Korean"),
        ("Jung Hae-in Korean actor", "Jung Hae-in", "Korean"),
        ("Felix Stray Kids kpop", "Felix", "Korean"),
        # Western
        ("Chris Hemsworth actor", "Chris Hemsworth", "Western"),
        ("Pedro Pascal actor", "Pedro Pascal", "Western"),
        ("Sterling K Brown actor", "Sterling K Brown", "Western"),
        ("Emma Stone actress", "Emma Stone", "Western"),
        ("Ariana Grande singer", "Ariana Grande", "Western"),
    ],
    "summer": [
        # Filipino
        ("Jodi Sta Maria actress Philippines", "Jodi Sta. Maria", "Filipino"),
        ("Janine Gutierrez actress Philippines", "Janine Gutierrez", "Filipino"),
        ("Shaina Magdayao actress Philippines", "Shaina Magdayao", "Filipino"),
        ("Richard Gutierrez actor Philippines", "Richard Gutierrez", "Filipino"),
        ("Matteo Guidicelli actor Philippines", "Matteo Guidicelli", "Filipino"),
        # Korean
        ("Son Ye-jin Korean actress", "Son Ye-jin", "Korean"),
        ("Irene Red Velvet kpop", "Irene", "Korean"),
        ("Jang Wonyoung IVE kpop", "Jang Wonyoung", "Korean"),
        ("Taeyeon SNSD Korean singer", "Taeyeon", "Korean"),
        ("Cha Eunwoo ASTRO kpop", "Cha Eunwoo", "Korean"),
        ("Jimin BTS kpop", "Jimin", "Korean"),
        # Western
        ("Taylor Swift singer", "Taylor Swift", "Western"),
        ("Nicole Kidman actress", "Nicole Kidman", "Western"),
        ("Robert Pattinson actor", "Robert Pattinson", "Western"),
        ("Timothee Chalamet actor", "Timothée Chalamet", "Western"),
    ],
    "autumn": [
        # Filipino (primary focus — keeping 10)
        ("Kathryn Bernardo actress Philippines", "Kathryn Bernardo", "Filipino"),
        ("Nadine Lustre actress Philippines", "Nadine Lustre", "Filipino"),
        ("Gabbi Garcia actress Philippines", "Gabbi Garcia", "Filipino"),
        ("Coleen Garcia actress Philippines", "Coleen Garcia", "Filipino"),
        ("Liza Soberano actress Philippines", "Liza Soberano", "Filipino"),
        ("Piolo Pascual actor Philippines", "Piolo Pascual", "Filipino"),
        ("Daniel Padilla actor Philippines", "Daniel Padilla", "Filipino"),
        ("Enrique Gil actor Philippines", "Enrique Gil", "Filipino"),
        ("Coco Martin actor Philippines", "Coco Martin", "Filipino"),
        ("Joshua Garcia actor Philippines", "Joshua Garcia", "Filipino"),
        # Korean
        ("Jennie Blackpink kpop", "Jennie", "Korean"),
        ("Jaehyun NCT kpop", "Jaehyun", "Korean"),
        ("V BTS Taehyung kpop", "V (BTS)", "Korean"),
        # Western
        ("Beyonce singer", "Beyoncé", "Western"),
        ("Oscar Isaac actor", "Oscar Isaac", "Western"),
    ],
    "winter": [
        # Filipino
        ("Heart Evangelista actress Philippines", "Heart Evangelista", "Filipino"),
        ("Pia Wurtzbach Philippines", "Pia Wurtzbach", "Filipino"),
        ("Alden Richards actor Philippines", "Alden Richards", "Filipino"),
        ("Dingdong Dantes actor Philippines", "Dingdong Dantes", "Filipino"),
        ("Paulo Avelino actor Philippines", "Paulo Avelino", "Filipino"),
        ("Xian Lim actor Philippines", "Xian Lim", "Filipino"),
        ("Donny Pangilinan actor Philippines", "Donny Pangilinan", "Filipino"),
        # Korean
        ("Jisoo Blackpink kpop", "Jisoo", "Korean"),
        ("Suga BTS Yoongi kpop", "Suga (BTS)", "Korean"),
        ("Jin BTS kpop", "Jin (BTS)", "Korean"),
        ("Hyun Bin Korean actor", "Hyun Bin", "Korean"),
        ("Song Hye-kyo Korean actress", "Song Hye-kyo", "Korean"),
        # Western
        ("Megan Fox actress", "Megan Fox", "Western"),
        ("Dua Lipa singer", "Dua Lipa", "Western"),
        ("Keanu Reeves actor", "Keanu Reeves", "Western"),
    ],
}

# ── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── SETUP DIRS ────────────────────────────────────────────────────────────────

for season in CELEBRITIES:
    os.makedirs(os.path.join(BASE_DIR, season), exist_ok=True)
    os.makedirs(os.path.join(RAW_DIR, season), exist_ok=True)
os.makedirs(REJECTED_DIR, exist_ok=True)

# ── FACE DETECTOR (MediaPipe) ─────────────────────────────────────────────────

try:
    import mediapipe as mp
    _mp_face = mp.solutions.face_detection
    FACE_DETECTOR = _mp_face.FaceDetection(
        model_selection=1,              # model 1 = better for non-frontal / varied distances
        min_detection_confidence=MIN_FACE_CONFIDENCE
    )
    USE_MEDIAPIPE = True
    log.info("✓ Using MediaPipe face detector")
except (ImportError, AttributeError):
    log.warning("MediaPipe not available — falling back to Haar cascade")
    CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    USE_MEDIAPIPE = False

# ── WHITE BALANCE ─────────────────────────────────────────────────────────────

def white_balance_grayworld(img_bgr):
    """
    Gray World white balance correction.
    Assumes the average color of a natural scene is gray.
    Corrects for warm/cool lighting bias — critical for accurate skin tone analysis.
    """
    img = img_bgr.astype(np.float32)
    mean_b = np.mean(img[:, :, 0])
    mean_g = np.mean(img[:, :, 1])
    mean_r = np.mean(img[:, :, 2])
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    img[:, :, 0] = np.clip(img[:, :, 0] * (mean_gray / (mean_b + 1e-6)), 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * (mean_gray / (mean_g + 1e-6)), 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * (mean_gray / (mean_r + 1e-6)), 0, 255)
    return img.astype(np.uint8)

# ── FACE CROP ─────────────────────────────────────────────────────────────────

def detect_and_crop_face(img_path, output_path):
    """
    Detect the best (largest, most confident) face in the image.
    Applies white balance correction before saving.
    Returns True if a face was found and saved, False otherwise.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return False

    h_img, w_img = img_bgr.shape[:2]

    if USE_MEDIAPIPE:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = FACE_DETECTOR.process(img_rgb)

        if not result.detections:
            return False

        # Pick the detection with highest confidence
        best = max(result.detections, key=lambda d: d.score[0])
        bbox = best.location_data.relative_bounding_box

        # Convert relative → absolute pixels
        x = int(bbox.xmin * w_img)
        y = int(bbox.ymin * h_img)
        w = int(bbox.width * w_img)
        h = int(bbox.height * h_img)

        # Reject tiny faces
        if w < MIN_FACE_SIZE_PX or h < MIN_FACE_SIZE_PX:
            return False

    else:
        # Haar cascade fallback
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(MIN_FACE_SIZE_PX, MIN_FACE_SIZE_PX))
        if len(faces) == 0:
            return False
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add 25% padding around face
    pad = int(0.25 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)

    face_crop = img_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return False

    # Apply white balance correction
    face_crop = white_balance_grayworld(face_crop)

    # Resize to 224x224
    face_crop = cv2.resize(face_crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, face_crop)
    return True

# ── DUPLICATE CHECK ───────────────────────────────────────────────────────────

def md5_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def is_duplicate(img_path, seen_hashes_md5, seen_hashes_phash):
    """Check both exact (MD5) and near-duplicate (pHash) against seen sets."""
    md5 = md5_hash(img_path)
    if md5 in seen_hashes_md5:
        return True
    seen_hashes_md5.add(md5)

    try:
        ph = imagehash.phash(Image.open(img_path))
        for existing in seen_hashes_phash:
            if ph - existing < PHASH_THRESHOLD:
                return True
        seen_hashes_phash.add(ph)
    except Exception:
        pass

    return False

# ── DOWNLOAD ──────────────────────────────────────────────────────────────────

def download_celebrity(query, save_dir, max_num):
    """Download images from Bing Images for a celebrity query."""
    try:
        crawler = BingImageCrawler(
            storage={"root_dir": save_dir},
            log_level=logging.WARNING,
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=2,
        )
        crawler.crawl(
            keyword=query,
            max_num=max_num,
            filters={"type": "photo"},
        )
    except Exception as e:
        log.warning(f"  Download error for '{query}': {e}")

# ── PROCESS ONE CELEBRITY ─────────────────────────────────────────────────────

def process_celebrity(query, celeb_name, season, nationality, manifest_rows):
    """Download, crop, deduplicate, and save face images for one celebrity."""
    raw_celeb_dir = os.path.join(RAW_DIR, season, celeb_name.replace(" ", "_"))
    season_dir    = os.path.join(BASE_DIR, season)
    os.makedirs(raw_celeb_dir, exist_ok=True)

    # Skip if already collected (resume-safe re-runs)
    celeb_slug    = celeb_name.replace(" ", "_")
    already_saved = [
        f for f in os.listdir(season_dir)
        if f.startswith(f"{season}_{celeb_slug}_") and f.lower().endswith(".jpg")
    ]
    if len(already_saved) >= 3:
        log.info(f"          → skipping (already have {len(already_saved)} images)")
        for fname in already_saved:
            manifest_rows.append({
                "filename":    fname,
                "season":      season,
                "celebrity":   celeb_name,
                "nationality": nationality,
                "status":      "saved",
                "source_raw":  "pre-existing",
            })
        return len(already_saved), 0

    # 1. Download
    download_celebrity(query, raw_celeb_dir, IMAGES_PER_CELEBRITY)

    valid_ext = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    files = [
        os.path.join(raw_celeb_dir, f)
        for f in os.listdir(raw_celeb_dir)
        if f.lower().endswith(valid_ext)
    ]

    seen_md5   = set()
    seen_phash = set()
    saved      = 0
    rejected   = 0

    for src_path in files:
        # Duplicate check on raw image
        if is_duplicate(src_path, seen_md5, seen_phash):
            rejected += 1
            continue

        # Build output filename
        base_name  = f"{season}_{celeb_name.replace(' ', '_')}_{saved+1:03d}.jpg"
        dst_path   = os.path.join(season_dir, base_name)
        rej_path   = os.path.join(REJECTED_DIR, base_name)

        success = detect_and_crop_face(src_path, dst_path)

        if success:
            saved += 1
            manifest_rows.append({
                "filename":    base_name,
                "season":      season,
                "celebrity":   celeb_name,
                "nationality": nationality,
                "status":      "saved",
                "source_raw":  src_path,
            })
        else:
            shutil.copy2(src_path, rej_path)
            rejected += 1
            manifest_rows.append({
                "filename":    base_name,
                "season":      season,
                "celebrity":   celeb_name,
                "nationality": nationality,
                "status":      "rejected_no_face",
                "source_raw":  src_path,
            })

    return saved, rejected

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(seasons_to_run=None):
    if seasons_to_run is None:
        seasons_to_run = list(CELEBRITIES.keys())

    log.info("=" * 65)
    log.info("  ToneFit ML — Data Collection Pipeline v2")
    log.info("=" * 65)
    log.info(f"  Seasons      : {seasons_to_run}")
    log.info(f"  Photos/celeb : {IMAGES_PER_CELEBRITY} (downloads, then filters)")
    log.info(f"  Face detector: {'MediaPipe' if USE_MEDIAPIPE else 'Haar (fallback)'}")
    log.info(f"  White balance: Gray World correction ✓")
    log.info("=" * 65)

    manifest_rows  = []
    total_saved    = 0
    total_rejected = 0

    for season in seasons_to_run:
        celebrities = CELEBRITIES[season]
        log.info(f"\n{'─' * 65}")
        log.info(f"  SEASON: {season.upper()} — {len(celebrities)} celebrities")
        log.info(f"{'─' * 65}")

        for i, (query, celeb_name, nationality) in enumerate(celebrities, 1):
            log.info(f"  [{i:02}/{len(celebrities)}] {celeb_name} ({nationality})")
            saved, rejected = process_celebrity(
                query, celeb_name, season, nationality, manifest_rows
            )
            log.info(f"          → saved: {saved}  rejected: {rejected}")
            total_saved    += saved
            total_rejected += rejected

    # Save manifest
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(MANIFEST_CSV, index=False)
    log.info(f"\n  Manifest saved: {MANIFEST_CSV}")

    # Summary
    log.info(f"\n{'=' * 65}")
    log.info("  COLLECTION COMPLETE")
    log.info(f"{'=' * 65}")
    log.info(f"  Total saved    : {total_saved}")
    log.info(f"  Total rejected : {total_rejected}")
    log.info(f"\n  Per-season breakdown:")

    grand_total = 0
    for season in CELEBRITIES:
        folder = os.path.join(BASE_DIR, season)
        if not os.path.exists(folder):
            continue
        valid_ext = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        count = len([f for f in os.listdir(folder) if f.lower().endswith(valid_ext)])
        grand_total += count
        bar = "█" * (count // 3)
        log.info(f"    {season:<8}: {count:>4} images  {bar}")

    log.info(f"\n  TOTAL: {grand_total} labeled face images")
    log.info(f"\n  Next step → run: python preprocess.py")
    log.info("=" * 65)

# ── RUN ───────────────────────────────────────────────────────────────────────
#
#  HOW TO SPLIT AMONG YOUR GROUP:
#  ────────────────────────────────
#  Person 1 → RUN_SEASONS = ["spring"]
#  Person 2 → RUN_SEASONS = ["summer"]
#  Person 3 → RUN_SEASONS = ["autumn"]
#  Person 4 → RUN_SEASONS = ["winter"]
#
#  Then merge all dataset/ folders + dataset_manifest.csv rows together.
#
#  Run all at once:
#  → RUN_SEASONS = None

RUN_SEASONS = None  # Change to e.g. ["spring"] to run one season

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", choices=list(CELEBRITIES.keys()),
                        help="Seasons to run (default: all)")
    args = parser.parse_args()
    run(seasons_to_run=args.seasons or RUN_SEASONS)