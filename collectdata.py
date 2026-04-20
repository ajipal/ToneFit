"""
ToneFit ML — Data Collection Script
=====================================
Downloads face images of verified celebrities per personal color season.

Seasons:
  spring  → Warm, Light, Clear
  summer  → Cool, Light, Muted
  autumn  → Warm, Deep, Muted
  winter  → Cool, Deep, Clear

Steps:
  1. Searches Google Images for each celebrity by name
  2. Downloads up to 5 images per celebrity into raw folder
  3. Detects and crops the largest face from each image
  4. Saves cropped faces (224x224) into labeled season folders
  5. Rejected images (no face found) go into rejected folder

Output structure:
  dataset/
    spring/     → labeled spring face crops
    summer/     → labeled summer face crops
    autumn/     → labeled autumn face crops
    winter/     → labeled winter face crops
    raw/        → original downloaded images
    rejected/   → images where no face was detected

Requirements:
  pip install icrawler opencv-python

Usage:
  python collect_data.py

Split among group of 4:
  Edit RUN_SEASONS at the bottom of this file
  Person 1 → RUN_SEASONS = ["spring"]
  Person 2 → RUN_SEASONS = ["summer"]
  Person 3 → RUN_SEASONS = ["autumn"]
  Person 4 → RUN_SEASONS = ["winter"]
  Then merge all dataset/ folders together after.
"""

import os
import cv2
import shutil
import hashlib
import logging
from icrawler.builtin import GoogleImageCrawler

# ─── CONFIG ───────────────────────────────────────────────────────────────────

IMAGES_PER_CELEBRITY = 5
MIN_FACE_SIZE        = 60
BASE_DIR             = "dataset"
RAW_DIR              = os.path.join(BASE_DIR, "raw")
REJECTED_DIR         = os.path.join(BASE_DIR, "rejected")

# ─── VERIFIED CELEBRITY LIST ──────────────────────────────────────────────────
# Labels sourced from publicly documented professional color analyses
# Sources: Korean beauty blogs, color analysis communities, certified analysts

CELEBRITIES = {

    "spring": [
        # Filipino (4)
        "Anne Curtis actress Philippines face",
        "Julia Barretto actress Philippines face",
        "Marian Rivera actress Philippines face",
        "James Reid actor Philippines face",
        # Korean (6)
        "IU Korean singer face",
        "Yoona SNSD Korean singer face",
        "Kim Chaewon Le Sserafim kpop face",
        "Kim Soo-hyun Korean actor face",
        "Jung Hae-in Korean actor face",
        "Felix Stray Kids kpop face",
        # Western (5)
        "Chris Hemsworth actor face",
        "Pedro Pascal actor face",
        "Sterling K Brown actor face",
        "Emma Stone actress face",
        "Ariana Grande singer face",
    ],

    "summer": [
        # Filipino (5)
        "Jodi Sta Maria actress Philippines face",
        "Janine Gutierrez actress Philippines face",
        "Shaina Magdayao actress Philippines face",
        "Richard Gutierrez actor Philippines face",
        "Matteo Guidicelli actor Philippines face",
        # Korean (6)
        "Son Ye-jin Korean actress face",
        "Irene Red Velvet kpop face",
        "Jang Wonyoung IVE kpop face",
        "Taeyeon SNSD Korean singer face",
        "Cha Eunwoo ASTRO kpop face",
        "Jimin BTS kpop face",
        # Western (4)
        "Taylor Swift singer face",
        "Nicole Kidman actress face",
        "Robert Pattinson actor face",
        "Timothee Chalamet actor face",
    ],

    "autumn": [
        # Filipino (10)
        "Kathryn Bernardo actress Philippines face",
        "Nadine Lustre actress Philippines face",
        "Gabbi Garcia actress Philippines face",
        "Coleen Garcia actress Philippines face",
        "Liza Soberano actress Philippines face",
        "Piolo Pascual actor Philippines face",
        "Daniel Padilla actor Philippines face",
        "Enrique Gil actor Philippines face",
        "Coco Martin actor Philippines face",
        "Joshua Garcia actor Philippines face",
        # Korean (3)
        "Jennie Blackpink kpop face",
        "Jaehyun NCT kpop face",
        "V BTS Taehyung face",
        # Western (2)
        "Beyonce singer face",
        "Oscar Isaac actor face",
    ],

    "winter": [
        # Filipino (7)
        "Heart Evangelista actress Philippines face",
        "Pia Wurtzbach Philippines face",
        "Alden Richards actor Philippines face",
        "Dingdong Dantes actor Philippines face",
        "Paulo Avelino actor Philippines face",
        "Xian Lim actor Philippines face",
        "Donny Pangilinan actor Philippines face",
        # Korean (5)
        "Jisoo Blackpink kpop face",
        "Suga BTS Yoongi face",
        "Jin BTS face",
        "Hyun Bin Korean actor face",
        "Song Hye-kyo Korean actress face",
        # Western (3)
        "Megan Fox actress face",
        "Dua Lipa singer face",
        "Keanu Reeves actor face",
    ],
}

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ─── SETUP DIRECTORIES ────────────────────────────────────────────────────────

for season in CELEBRITIES:
    os.makedirs(os.path.join(BASE_DIR, season), exist_ok=True)
    os.makedirs(os.path.join(RAW_DIR, season), exist_ok=True)
os.makedirs(REJECTED_DIR, exist_ok=True)

# ─── FACE DETECTOR ────────────────────────────────────────────────────────────

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def file_hash(path):
    """MD5 hash for duplicate detection."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def download_celebrity(query, save_dir, max_num):
    """Download images from Google Images for a celebrity query."""
    log.info(f"    Downloading: {query}")
    try:
        crawler = GoogleImageCrawler(
            storage={"root_dir": save_dir},
            log_level=logging.WARNING,
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=2,
        )
        crawler.crawl(
            keyword=query,
            max_num=max_num,
            filters={"type": "face"},
        )
    except Exception as e:
        log.warning(f"    Download error for '{query}': {e}")


def crop_face(img_path, output_path, min_size=MIN_FACE_SIZE):
    """
    Detect and crop the largest face from an image.
    Adds 20% padding. Saves 224x224 crop.
    Returns True if face found, False otherwise.
    """
    img = cv2.imread(img_path)
    if img is None:
        return False

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size),
    )

    if len(faces) == 0:
        return False

    # Pick largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add 20% padding
    pad = int(0.20 * max(w, h))
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(img.shape[1], x + w + pad)
    y2  = min(img.shape[0], y + h + pad)

    face_crop = img[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (224, 224))
    cv2.imwrite(output_path, face_crop)
    return True


def process_season_raw(season):
    """
    Process all raw images for a season.
    Returns (saved_count, rejected_count)
    """
    raw_dir    = os.path.join(RAW_DIR, season)
    season_dir = os.path.join(BASE_DIR, season)
    valid_ext  = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    files = [
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(valid_ext)
    ]

    saved    = 0
    rejected = 0

    for fname in files:
        src  = os.path.join(raw_dir, fname)
        base = f"{season}_{fname}"
        dst  = os.path.join(season_dir, base)
        rej  = os.path.join(REJECTED_DIR, base)

        if os.path.exists(dst):
            saved += 1
            continue

        success = crop_face(src, dst)
        if success:
            saved += 1
        else:
            try:
                shutil.copy2(src, rej)
            except Exception:
                pass
            rejected += 1

    return saved, rejected


def count_images(folder):
    valid_ext = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    if not os.path.exists(folder):
        return 0
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith(valid_ext)
    ])


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run(seasons_to_run=None):
    if seasons_to_run is None:
        seasons_to_run = list(CELEBRITIES.keys())

    log.info("=" * 60)
    log.info("  ToneFit ML — Data Collection")
    log.info("=" * 60)
    log.info(f"  Seasons      : {seasons_to_run}")
    log.info(f"  Photos/celeb : {IMAGES_PER_CELEBRITY}")
    log.info(f"  Total targets: {sum(len(CELEBRITIES[s]) for s in seasons_to_run)} celebrities")
    log.info("=" * 60)

    total_saved    = 0
    total_rejected = 0

    for season in seasons_to_run:
        celebrities = CELEBRITIES[season]
        raw_dir     = os.path.join(RAW_DIR, season)

        log.info(f"\n{'─' * 60}")
        log.info(f"  SEASON: {season.upper()}  ({len(celebrities)} celebrities)")
        log.info(f"{'─' * 60}")

        # Step 1: Download
        for i, query in enumerate(celebrities, 1):
            log.info(f"  [{i:02}/{len(celebrities)}] {query}")
            download_celebrity(query, raw_dir, IMAGES_PER_CELEBRITY)

        # Step 2: Crop faces
        log.info(f"\n  Cropping faces from raw images...")
        saved, rejected = process_season_raw(season)

        log.info(f"  ✓ Saved    : {saved}")
        log.info(f"  ✗ Rejected : {rejected}")
        total_saved    += saved
        total_rejected += rejected

    # Summary
    log.info(f"\n{'=' * 60}")
    log.info("  DONE")
    log.info(f"{'=' * 60}")
    log.info(f"  Total saved    : {total_saved}")
    log.info(f"  Total rejected : {total_rejected}")
    log.info(f"\n  Final dataset:")

    grand_total = 0
    for season in CELEBRITIES:
        folder = os.path.join(BASE_DIR, season)
        count  = count_images(folder)
        grand_total += count
        bar    = "█" * (count // 3)
        log.info(f"    {season:<8} : {count:>4} images  {bar}")

    log.info(f"\n    TOTAL    : {grand_total} labeled face images")
    log.info(f"\n  Saved to: ./{BASE_DIR}/")
    log.info("=" * 60)


# ─── RUN ──────────────────────────────────────────────────────────────────────
#
#  HOW TO SPLIT AMONG YOUR GROUP:
#  ───────────────────────────────
#  Each person edits RUN_SEASONS below, runs on their own laptop,
#  then copies their season folder into the shared dataset/ folder.
#
#  Person 1 → RUN_SEASONS = ["spring"]
#  Person 2 → RUN_SEASONS = ["summer"]
#  Person 3 → RUN_SEASONS = ["autumn"]
#  Person 4 → RUN_SEASONS = ["winter"]
#
#  To run all at once (one person, takes ~30-40 mins):
#  → RUN_SEASONS = None

RUN_SEASONS = None  # Change to ["spring"] etc. to run only one season

if __name__ == "__main__":
    run(seasons_to_run=RUN_SEASONS)