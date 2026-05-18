"""
ToneFit — FastAPI Backend
Models: FaRL (ViT-B/16) + SVM (RBF on CIELab/HSV features)

Endpoints:
  GET  /health            → model load status
  POST /predict           → image → season prediction + confidences
  GET  /seasons/{season}  → full recommendation data for a season
"""

import io
import pickle
import pathlib
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image
from skimage import color as skcolor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import joblib

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
HAAR_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

FARL_HEAD_PATH    = MODELS_DIR / "farl_model.pth"
FARL_WEIGHTS_PATH = MODELS_DIR / "farl_weights.pth"
SVM_PATH          = MODELS_DIR / "svm_model.pkl"
SCALER_PATH       = MODELS_DIR / "scaler.pkl"

SEASONS     = ["autumn", "spring", "summer", "winter"]
NUM_CLASSES = 4

# SVM uses 7 color features in this order
SVM_FEATURES = ["L_mean", "a_mean", "b_mean", "ITA", "H_mean", "S_mean", "V_mean"]
# Full feature order that the scaler was fit on (from preprocess.py)
ALL_FEATURES = ["L_mean", "a_mean", "b_mean", "L_std", "a_std", "b_std",
                "ITA", "H_mean", "S_mean", "V_mean", "skin_ratio", "blur_score"]
SVM_COL_INDICES = [ALL_FEATURES.index(c) for c in SVM_FEATURES]

# ─────────────────────────────────────────────
#  Season metadata
# ─────────────────────────────────────────────
SEASON_DATA = {
    "spring": {
        "emoji": "🌸",
        "undertone": "Warm · Light · Clear",
        "best_colors": ["Peach", "Coral", "Warm Turquoise", "Gold", "Ivory", "Light Salmon"],
        "best_hex": ["#FFDAB9", "#FF7F50", "#40E0D0", "#FFD700", "#FFFFF0", "#FA8072"],
        "avoid_colors": ["Cool gray", "Icy blue", "Stark black", "Cool purple"],
        "metals": "Gold, Rose Gold",
        "outfits": {
            "casual": "Light floral prints, peach tones, warm whites",
            "smart_casual": "Coral blouse with cream trousers, warm beige blazer",
            "formal": "Champagne gold gown, warm ivory suit",
        },
        "accessories": {
            "bags": "Tan leather, warm beige, camel tote",
            "sunglasses": "Gold frames, warm tortoiseshell, brown lens",
            "jewelry": "Gold chains, pearl accents, rose gold rings",
            "scarves": "Warm peach silk, ivory floral print",
        },
        "makeup": {
            "foundation": "Warm/yellow undertone foundation",
            "lips": "Peach, coral, warm pink, salmon",
            "blush": "Peach, apricot, warm rose",
            "eyeshadow": "Warm brown, bronze, champagne gold",
        },
    },
    "summer": {
        "emoji": "☁️",
        "undertone": "Cool · Light · Muted",
        "best_colors": ["Lavender", "Dusty Rose", "Powder Blue", "Sage Green", "Mauve", "Soft Gray"],
        "best_hex": ["#E6E6FA", "#DCB4B4", "#B0C4DE", "#B2C2B0", "#E0B0C8", "#D3D3D3"],
        "avoid_colors": ["Warm orange", "Mustard yellow", "Olive green", "Bright neon"],
        "metals": "Silver, White Gold, Platinum",
        "outfits": {
            "casual": "Dusty rose top, soft gray jeans, powder blue linen",
            "smart_casual": "Mauve blazer with light gray trousers, lavender blouse",
            "formal": "Dusty blue gown, soft lavender suit",
        },
        "accessories": {
            "bags": "Soft gray leather, dusty rose, pale blue",
            "sunglasses": "Silver frames, gray-blue lens, cool tortoiseshell",
            "jewelry": "Silver, white gold, moonstone, pearl",
            "scarves": "Lavender silk, soft gray cashmere",
        },
        "makeup": {
            "foundation": "Cool/pink undertone foundation",
            "lips": "Mauve, berry, cool pink, rose",
            "blush": "Cool pink, soft rose, berry",
            "eyeshadow": "Cool taupe, mauve, soft lavender, gray",
        },
    },
    "autumn": {
        "emoji": "🍂",
        "undertone": "Warm · Deep · Muted",
        "best_colors": ["Rust", "Olive Green", "Camel", "Burnt Orange", "Chocolate Brown", "Mustard"],
        "best_hex": ["#B7410E", "#708238", "#C19A6B", "#CC5500", "#7B3F00", "#E1AD01"],
        "avoid_colors": ["Icy blue", "Cool pink", "Stark white", "Electric neon"],
        "metals": "Gold, Bronze, Antique Gold",
        "outfits": {
            "casual": "Olive green tee, rust jeans, camel jacket",
            "smart_casual": "Burnt orange blouse with chocolate brown trousers",
            "formal": "Deep emerald gown, rich burgundy suit",
        },
        "accessories": {
            "bags": "Camel leather, tan suede, chocolate brown tote",
            "sunglasses": "Tortoiseshell, warm brown frames, amber lens",
            "jewelry": "Gold, bronze, amber stones, wooden accents",
            "scarves": "Rust wool, olive plaid, warm terracotta",
        },
        "makeup": {
            "foundation": "Warm/golden undertone foundation",
            "lips": "Brick red, terracotta, warm brown, deep rust",
            "blush": "Warm peach, terracotta, copper",
            "eyeshadow": "Warm brown, bronze, copper, olive gold",
        },
    },
    "winter": {
        "emoji": "❄️",
        "undertone": "Cool · Deep · Clear",
        "best_colors": ["True White", "Black", "Royal Blue", "Crimson", "Emerald", "Dark Magenta"],
        "best_hex": ["#FFFFFF", "#000000", "#4169E1", "#DC143C", "#50C878", "#8B008B"],
        "avoid_colors": ["Warm brown", "Mustard", "Dusty pastels", "Warm beige"],
        "metals": "Silver, Platinum, Cool White Gold",
        "outfits": {
            "casual": "Black tee, crisp white jeans, royal blue jacket",
            "smart_casual": "Emerald green top with black trousers, cobalt blazer",
            "formal": "Stark white or black gown, jewel-tone suit",
        },
        "accessories": {
            "bags": "Black leather, crisp white, deep navy",
            "sunglasses": "Black frames, silver hardware, dark lens",
            "jewelry": "Silver, platinum, sapphire, diamond",
            "scarves": "Black cashmere, white silk, royal blue wool",
        },
        "makeup": {
            "foundation": "Cool/neutral undertone foundation",
            "lips": "True red, berry, deep plum, cool pink",
            "blush": "Cool pink, berry, soft rose",
            "eyeshadow": "Cool gray, navy, silver, deep plum",
        },
    },
}

# ─────────────────────────────────────────────
#  FaRL image transform
# ─────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
#  FaRL model definition
# ─────────────────────────────────────────────

class FaRLHead(nn.Module):
    """Classifier head only — matches train_farl.py checkpoint keys."""

    def __init__(self, feature_dim: int = 512):
        super().__init__()
        shared_dim = feature_dim // 2
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.season_head  = nn.Linear(shared_dim, 4)
        self.subtype_head = nn.Linear(shared_dim, 12)

    def forward(self, features):
        shared = self.shared(features)
        return self.season_head(shared)


# ─────────────────────────────────────────────
#  Model loading (once at startup)
# ─────────────────────────────────────────────

def _load_farl():
    if not FARL_HEAD_PATH.exists():
        return None, None, f"Head not found: {FARL_HEAD_PATH}"
    if not FARL_WEIGHTS_PATH.exists():
        return None, None, f"Backbone weights not found: {FARL_WEIGHTS_PATH}"
    try:
        import clip
    except ImportError:
        return None, None, "CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP backbone + apply FaRL weights
        clip_model, preprocess_clip = clip.load("ViT-B/16", device="cpu")
        farl_state = torch.load(str(FARL_WEIGHTS_PATH), map_location="cpu", weights_only=False)
        state_dict = farl_state.get("state_dict", farl_state)
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        clip_model.load_state_dict(cleaned, strict=False)
        backbone = clip_model.visual.float().to(device)
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()

        # Load classifier head
        ckpt = torch.load(FARL_HEAD_PATH, map_location=device)
        feature_dim = ckpt["shared"]["0.weight"].shape[1]
        head = FaRLHead(feature_dim=feature_dim)
        head.shared.load_state_dict(ckpt["shared"])
        head.season_head.load_state_dict(ckpt["season_head"])
        head.subtype_head.load_state_dict(ckpt["subtype_head"])
        head.to(device).eval()

        return backbone, head, None
    except Exception as exc:
        return None, None, str(exc)


def _load_svm():
    if not SVM_PATH.exists():
        return None, None, f"File not found: {SVM_PATH}"
    if not SCALER_PATH.exists():
        return None, None, f"Scaler not found: {SCALER_PATH}. Run preprocess.py first."
    try:
        svm = joblib.load(SVM_PATH)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return svm, scaler, None
    except Exception as exc:
        return None, None, str(exc)


_farl_backbone, _farl_head, _farl_error = _load_farl()
_svm_model, _scaler, _svm_error         = _load_svm()

# ─────────────────────────────────────────────
#  Face detection
# ─────────────────────────────────────────────

def detect_and_crop_face(pil_image: Image.Image):
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces   = cv2.CascadeClassifier(HAAR_PATH).detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_x, pad_y = int(w * 0.15), int(h * 0.15)
    x1 = max(0, x - pad_x);  y1 = max(0, y - pad_y)
    x2 = min(img_bgr.shape[1], x + w + pad_x)
    y2 = min(img_bgr.shape[0], y + h + pad_y)
    return pil_image.convert("RGB").crop((x1, y1, x2, y2))

# ─────────────────────────────────────────────
#  FaRL inference
# ─────────────────────────────────────────────

def farl_inference(backbone: nn.Module, head: nn.Module, pil_image: Image.Image) -> dict:
    device = next(backbone.parameters()).device
    tensor = TRANSFORM(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = backbone(tensor)
        logits   = head(features)
        probs    = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    return {season: round(prob, 4) for season, prob in zip(SEASONS, probs)}

# ─────────────────────────────────────────────
#  SVM feature extraction + inference
# ─────────────────────────────────────────────

def _extract_color_features(pil_image: Image.Image) -> np.ndarray:
    """Extract the 12 color features preprocess.py produces, return as 1-D array."""
    img_rgb = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0

    # CIELab
    lab = skcolor.rgb2lab(img_rgb)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    L_mean, a_mean, b_mean = float(L.mean()), float(a.mean()), float(b.mean())
    L_std,  a_std,  b_std  = float(L.std()),  float(a.std()),  float(b.std())

    # ITA — Individual Typology Angle
    ita = float(np.degrees(np.arctan2(L_mean - 50, b_mean + 1e-6)))

    # HSV
    img_bgr = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H_mean  = float(hsv[:, :, 0].mean())
    S_mean  = float(hsv[:, :, 1].mean() / 255.0)
    V_mean  = float(hsv[:, :, 2].mean() / 255.0)

    # Placeholder values for skin_ratio and blur_score
    # (SVM only uses the first 7 features; these are needed for scaler alignment)
    skin_ratio  = 0.5
    blur_score  = 100.0

    return np.array([L_mean, a_mean, b_mean, L_std, a_std, b_std,
                     ita, H_mean, S_mean, V_mean, skin_ratio, blur_score],
                    dtype=np.float32)


def svm_inference(svm, scaler, pil_image: Image.Image) -> dict:
    full_features = _extract_color_features(pil_image).reshape(1, -1)
    scaled        = scaler.transform(full_features)
    svm_features  = scaled[:, SVM_COL_INDICES]
    proba         = svm.predict_proba(svm_features)[0]
    return {season: round(float(p), 4) for season, p in zip(SEASONS, proba)}

# ─────────────────────────────────────────────
#  Response schemas
# ─────────────────────────────────────────────

class ModelStatus(BaseModel):
    loaded: bool
    error: str | None = None


class PredictionResponse(BaseModel):
    season: str
    confidence: float
    farl: dict | None = None
    svm: dict | None = None
    face_detected: bool

# ─────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────

app = FastAPI(
    title="ToneFit API",
    description="Personal color season prediction — FaRL + SVM.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── GET /health ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "farl": ModelStatus(loaded=_farl_head is not None, error=_farl_error),
            "svm":  ModelStatus(loaded=_svm_model is not None, error=_svm_error),
        },
    }


# ── POST /predict ──────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if _farl_head is None and _svm_model is None:
        raise HTTPException(status_code=503, detail="No models are loaded.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    face         = detect_and_crop_face(image)
    input_image  = face if face is not None else image

    farl_conf, svm_conf = None, None

    if _farl_head is not None:
        try:
            farl_conf = farl_inference(_farl_backbone, _farl_head, input_image)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"FaRL inference error: {exc}")

    if _svm_model is not None:
        try:
            svm_conf = svm_inference(_svm_model, _scaler, input_image)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"SVM inference error: {exc}")

    available = [c for c in (farl_conf, svm_conf) if c is not None]
    averaged  = {
        s: sum(c[s] for c in available) / len(available)
        for s in SEASONS
    }
    top_season = max(averaged, key=averaged.get)

    return PredictionResponse(
        season=top_season,
        confidence=round(averaged[top_season], 4),
        farl=farl_conf,
        svm=svm_conf,
        face_detected=face is not None,
    )


# ── GET /seasons/{season} ──────────────────────────────────────────────────────

@app.get("/seasons/{season}")
def get_season(season: str):
    season = season.lower()
    if season not in SEASON_DATA:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown season '{season}'. Valid: {SEASONS}",
        )
    return SEASON_DATA[season]
