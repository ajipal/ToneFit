"""
Creates a clean zip of the ToneFit project scripts for Kaggle upload.
Includes all training scripts (classic and color-fusion variants), preprocess,
evaluate, and app. Does NOT include RGB-M dataset files.

Usage:
    python create_kaggle_zip.py

Output:
    tonefit-scripts-kaggle.zip  — upload to Kaggle as a dataset or notebook input
"""

import re
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Scripts to include in the Kaggle zip
INCLUDE_FILES = [
    "preprocess.py",
    "train_farl.py",
    "train_dinov2.py",
    "train_farl_color.py",
    "train_dinov2_color.py",
    "train_svm.py",
    "evaluate.py",
    "app.py",
    "requirements.txt",
    "ToneFit_Kaggle_FaRL_SVM.ipynb",
]

OUT_ZIP = SCRIPT_DIR / "tonefit-scripts-kaggle.zip"

print(f"Zipping ToneFit scripts → {OUT_ZIP} ...")
count = 0

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in INCLUDE_FILES:
        fpath = SCRIPT_DIR / fname
        if fpath.exists():
            zf.write(fpath, fname)
            print(f"  + {fname}")
            count += 1
        else:
            print(f"  ! MISSING: {fname} (skipped)")

    # Also zip RGB-M dataset if present (separate section for dataset upload)
    rgb_m_candidates = [SCRIPT_DIR / "RGB-M", SCRIPT_DIR.parent / "RGB-M"]
    for candidate in rgb_m_candidates:
        if candidate.exists():
            print(f"\nFound RGB-M at: {candidate} — adding dataset files...")
            img_count = 0
            for file_path in sorted(candidate.rglob("*")):
                if file_path.is_file():
                    clean_parts = [re.sub(r'\s+(\.[^.]+)$', r'\1', p.strip()) for p in file_path.parts]
                    arcname = str(Path(*clean_parts))
                    zf.write(file_path, arcname)
                    img_count += 1
                    if img_count % 500 == 0:
                        print(f"  {img_count} dataset files zipped...")
            print(f"  + RGB-M ({img_count} files)")
            count += img_count
            break

size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
print(f"\n✅ Done: {OUT_ZIP} ({size_mb:.1f} MB, {count} total files)")
print("Upload to Kaggle → New Dataset → Upload.")
