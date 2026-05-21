"""
ToneFit FastAPI ML inference server.
Serves the trained FaRL flat baseline (Model A).

Start locally:
    uvicorn server:app --host 0.0.0.0 --port 8000

Architecture loaded from models/farl_model.pth:
    CLIP ViT-B/16 + FaRL weights (frozen, 512-d features)
    → shared FC(512→256) → ReLU → Dropout(0.5)
    → season_head  FC(256→4)
    → subtype_head FC(256→12)
"""

import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Label definitions (must match train_farl.py) ───────────────────────────────

SEASON_CLASSES = ["autumn", "spring", "summer", "winter"]
SUBTYPE_CLASSES = [
    "autumn_deep", "autumn_soft", "autumn_warm",
    "spring_bright", "spring_light", "spring_warm",
    "summer_cool", "summer_light", "summer_soft",
    "winter_bright", "winter_cool", "winter_deep",
]

# ── Model definition (must match train_farl.py FaRLModel) ─────────────────────

FEATURE_DIM = 512
SHARED_DIM  = FEATURE_DIM // 2  # 256

class FaRLInferenceModel(nn.Module):
    def __init__(self, visual_backbone):
        super().__init__()
        self.backbone = visual_backbone
        self.shared = nn.Sequential(
            nn.Linear(FEATURE_DIM, SHARED_DIM),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.season_head  = nn.Linear(SHARED_DIM, 4)
        self.subtype_head = nn.Linear(SHARED_DIM, 12)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        shared   = self.shared(features)
        return self.season_head(shared), self.subtype_head(shared)


# ── Image transform (CLIP normalization — must match train_farl.py) ────────────

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std= [0.26862954, 0.26130258, 0.27577711],
    ),
])

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(title="ToneFit ML API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Load model once at startup ─────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FARL_WEIGHTS = Path("models/farl_weights.pth")
MODEL_CKPT   = Path("models/farl_model.pth")

if not FARL_WEIGHTS.exists():
    raise FileNotFoundError(
        f"FaRL pretrained weights not found at {FARL_WEIGHTS}. "
        "Download from: https://github.com/FacePerceiver/FaRL/releases"
    )

if not MODEL_CKPT.exists():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_CKPT}. "
        "Run train_farl.py on Colab/Kaggle and copy models/farl_model.pth here."
    )

try:
    import clip
except ImportError:
    raise ImportError(
        "CLIP is required. Install with:\n"
        "  pip install git+https://github.com/openai/CLIP.git"
    )

print(f"[ToneFit API] Loading trained checkpoint from {MODEL_CKPT}...")
ckpt = torch.load(str(MODEL_CKPT), map_location="cpu", weights_only=False)

# Build CLIP ViT-B/16 architecture (no internet download — weights come from checkpoint)
clip_model, _ = clip.load("ViT-B/16", device="cpu")
vit = clip_model.visual.float()

# Load backbone weights saved by train_farl.py (avoids re-applying FaRL weights)
if "backbone" in ckpt:
    vit.load_state_dict(ckpt["backbone"], strict=False)
    print(f"[ToneFit API] Backbone loaded from checkpoint")
else:
    # Fallback: apply FaRL pretrained weights manually
    print(f"[ToneFit API] No backbone in checkpoint — applying FaRL weights from {FARL_WEIGHTS}")
    farl_state = torch.load(str(FARL_WEIGHTS), map_location="cpu", weights_only=False)
    state_dict = farl_state.get("state_dict", farl_state)
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    clip_model.load_state_dict(cleaned, strict=False)
    vit = clip_model.visual.float()

for param in vit.parameters():
    param.requires_grad = False

model = FaRLInferenceModel(vit)
model.shared.load_state_dict(ckpt["shared"])
model.season_head.load_state_dict(ckpt["season_head"])
model.subtype_head.load_state_dict(ckpt["subtype_head"])
print(f"[ToneFit API] Head loaded — shared, season_head, subtype_head")

model.eval()
model.to(device)

print(f"[ToneFit API] Ready on {device}")

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
        season_logits, subtype_logits = model(tensor)

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
        "season":             SEASON_CLASSES[season_idx],
        "subtype":            SUBTYPE_CLASSES[subtype_idx],
        "season_confidence":  float(season_probs[season_idx]),
        "subtype_confidence": float(subtype_probs[subtype_idx]),
        "top3":               top3,
    }
