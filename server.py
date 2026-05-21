"""
ToneFit FastAPI ML inference server.
Separate from the Streamlit demo (app.py) — this is what the Next.js frontend calls.

Start locally:
    uvicorn server:app --host 0.0.0.0 --port 8000

Deploy (Railway / Render / any Python host):
    Set PORT env variable; update PYTHON_API_URL in Vercel to point to your deployed URL.
"""

import io

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from hierarchical_head import (
    DeepArmocromiaHierarchical,
    SEASON_CLASSES,
    SUBTYPE_CLASSES,
)

# ── Image transform (must match train.py exactly) ──────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(title="ToneFit ML API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Load model once at startup ─────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepArmocromiaHierarchical(checkpoint_path="models/farl_weights.pth")

ckpt = torch.load("results/best_hierarchical.pth", map_location=device)
model.head.load_state_dict(ckpt["head_state"])
model.eval()
model.to(device)

print(f"[ToneFit API] Loaded hierarchical model on {device}")
print(f"[ToneFit API] Best checkpoint from epoch {ckpt.get('epoch', '?')}")

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    tensor = TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        season_logits, subtype_logits = model(tensor, hard_season=True)

    season_probs  = F.softmax(season_logits,  dim=1)[0]
    subtype_probs = F.softmax(subtype_logits, dim=1)[0]

    season_idx  = int(season_probs.argmax())
    subtype_idx = int(subtype_probs.argmax())

    top3 = sorted(
        [
            {"subtype": SUBTYPE_CLASSES[i], "confidence": float(subtype_probs[i])}
            for i in range(len(SUBTYPE_CLASSES))
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )[:3]

    return {
        "season":              SEASON_CLASSES[season_idx],
        "subtype":             SUBTYPE_CLASSES[subtype_idx],
        "season_confidence":   float(season_probs[season_idx]),
        "subtype_confidence":  float(subtype_probs[subtype_idx]),
        "top3":                top3,
    }
