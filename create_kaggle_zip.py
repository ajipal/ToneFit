"""
Creates a clean zip of RGB-M for Kaggle upload.
Strips trailing whitespace from filenames without modifying RGB-M.
"""

import re
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Check current dir, then parent dir (RGB-M may sit next to ToneFit/)
for candidate in [SCRIPT_DIR / "RGB-M", SCRIPT_DIR.parent / "RGB-M"]:
    if candidate.exists():
        RGB_M = candidate
        break
else:
    raise FileNotFoundError(
        "RGB-M folder not found. Check that it exists next to or inside ToneFit/."
    )

OUT_ZIP = SCRIPT_DIR / "RGB-M-kaggle.zip"
print(f"Found RGB-M at: {RGB_M}")

print(f"Zipping {RGB_M} → {OUT_ZIP} ...")
count = 0

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for file_path in sorted(RGB_M.rglob("*")):
        if file_path.is_file():
            clean_parts = [re.sub(r'\s+(\.[^.]+)$', r'\1', p.strip()) for p in file_path.parts]
            arcname = str(Path(*clean_parts))
            zf.write(file_path, arcname)
            count += 1
            if count % 500 == 0:
                print(f"  {count} files zipped...")

size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
print(f"\n✅ Done: {OUT_ZIP} ({size_mb:.0f} MB, {count} files)")
print("Upload RGB-M-kaggle.zip to Kaggle → New Dataset → Upload.")
