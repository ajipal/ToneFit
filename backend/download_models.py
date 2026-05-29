"""
ToneFit — Model file downloader
Downloads model files from Hugging Face Hub at startup if not already present.

Required env var:
  HF_REPO_ID  — your HF model repo, e.g. "your-username/tonefit-models"

Optional env var:
  HF_TOKEN    — for private repos (not needed if repo is public)
  MODELS_DIR  — override model directory path
"""

import os
import pathlib

MODELS_DIR = pathlib.Path(
    os.environ.get("MODELS_DIR", str(pathlib.Path(__file__).parent.parent / "models"))
)

MODEL_FILES = [
    "farl_model.pth",
    "farl_weights.pth",
    "svm_model.pkl",
    "scaler.pkl",
]


def download_models():
    repo_id = os.environ.get("HF_REPO_ID")
    if not repo_id:
        print("[models] HF_REPO_ID not set — using local model files only")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[models] huggingface_hub not installed — skipping download")
        return

    token = os.environ.get("HF_TOKEN")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for filename in MODEL_FILES:
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"[models] {filename} — already exists, skipping")
            continue
        try:
            print(f"[models] Downloading {filename} from {repo_id}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(MODELS_DIR),
                token=token,
            )
            print(f"[models] {filename} — done")
        except Exception as e:
            print(f"[models] Warning: could not download {filename}: {e}")


if __name__ == "__main__":
    download_models()
