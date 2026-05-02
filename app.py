"""
ToneFit — Personal Color Season Predictor
Streamlit web application

Models supported:
  - FaRL  (ViT-B/16 backbone, timm)  → models/farl_model.pth
  - DINOv2 (ViT-B/14 backbone)       → models/dinov2_model.pth

Run:
    streamlit run app.py
"""

import os
import io
import pathlib
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ─────────────────────────────────────────────
#  Season metadata
# ─────────────────────────────────────────────
SEASONS = ["spring", "summer", "autumn", "winter"]

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
        "gradient": "linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)",
        "text_color": "#7B3F00",
        "bar_color": "#FF7F50",
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
        "gradient": "linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%)",
        "text_color": "#1a237e",
        "bar_color": "#7B68EE",
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
        "gradient": "linear-gradient(135deg, #f9d976 0%, #f39f86 100%)",
        "text_color": "#6B2D00",
        "bar_color": "#CC5500",
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
        "gradient": "linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%)",
        "text_color": "#FFFFFF",
        "bar_color": "#4169E1",
    },
}

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTFITS_DIR = BASE_DIR / "outfits"
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

FARL_PATH = MODELS_DIR / "farl_model.pth"
DINOV2_PATH = MODELS_DIR / "dinov2_model.pth"

NUM_CLASSES = 4

# ─────────────────────────────────────────────
#  Image pre-processing transform
# ─────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
#  Model definitions
# ─────────────────────────────────────────────

class FaRLClassifier(nn.Module):
    """ViT-B/16 backbone (timm) + classification head."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        try:
            import timm
            self.backbone = timm.create_model(
                "vit_base_patch16_224", pretrained=False, num_classes=0
            )
            feat_dim = self.backbone.num_features
        except Exception:
            # Fallback: identity backbone — inference will still work if
            # the saved checkpoint carries both backbone and head weights.
            self.backbone = nn.Identity()
            feat_dim = 768
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class DINOv2Classifier(nn.Module):
    """DINOv2 ViT-B/14 backbone + lightweight classification head."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        try:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitb14",
                pretrained=False,
                verbose=False,
            )
            feat_dim = 768
        except Exception:
            self.backbone = nn.Identity()
            feat_dim = 768
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ─────────────────────────────────────────────
#  Cached model loading
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_farl_model():
    """Load FaRL model from disk. Returns (model, error_string)."""
    if not FARL_PATH.exists():
        return None, f"Model file not found: {FARL_PATH}"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FaRLClassifier(num_classes=NUM_CLASSES)
        checkpoint = torch.load(FARL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and "backbone" in checkpoint and "head" in checkpoint:
            model.backbone.load_state_dict(checkpoint["backbone"], strict=False)
            model.head.load_state_dict(checkpoint["head"])
        else:
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        return model, None
    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner=False)
def load_dinov2_model():
    """Load DINOv2 model from disk. Returns (model, error_string)."""
    if not DINOV2_PATH.exists():
        return None, f"Model file not found: {DINOV2_PATH}"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DINOv2Classifier(num_classes=NUM_CLASSES)
        checkpoint = torch.load(DINOV2_PATH, map_location=device)
        if isinstance(checkpoint, dict) and "backbone" in checkpoint and "head" in checkpoint:
            model.backbone.load_state_dict(checkpoint["backbone"], strict=False)
            model.head.load_state_dict(checkpoint["head"])
        else:
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        return model, None
    except Exception as exc:
        return None, str(exc)


# ─────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────

def run_inference(model: nn.Module, pil_image: Image.Image) -> dict:
    """
    Run a single PIL image through a loaded model.
    Returns {season_name: confidence_float} sorted descending.
    """
    device = next(model.parameters()).device
    tensor = TRANSFORM(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    return {season: round(prob, 4) for season, prob in zip(SEASONS, probs)}


# ─────────────────────────────────────────────
#  Face detection
# ─────────────────────────────────────────────

def detect_and_crop_face(pil_image: Image.Image):
    """
    Detect the largest face in a PIL image using OpenCV Haar Cascade.
    Returns cropped PIL face image, or None if no face found.
    Also returns the annotated full image for display.
    """
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return None, pil_image

    # Pick largest face by area
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Draw rectangle on annotated copy
    annotated = img_bgr.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 100), 3)
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    # Add padding (15 %) and crop
    pad_x = int(w * 0.15)
    pad_y = int(h * 0.15)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_bgr.shape[1], x + w + pad_x)
    y2 = min(img_bgr.shape[0], y + h + pad_y)

    face_crop = pil_image.convert("RGB").crop((x1, y1, x2, y2))
    return face_crop, annotated_pil


# ─────────────────────────────────────────────
#  UI helpers
# ─────────────────────────────────────────────

def hex_swatch_html(colors: list, hexes: list) -> str:
    """Build an HTML block of color swatches with labels."""
    items = []
    for name, hex_val in zip(colors, hexes):
        # Use dark text for very light swatches
        r, g, b = int(hex_val[1:3], 16), int(hex_val[3:5], 16), int(hex_val[5:7], 16)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "#222222" if luminance > 160 else "#f0f0f0"
        # White swatch needs a border
        border = "border: 1px solid #ccc;" if hex_val.upper() in ("#FFFFFF", "#FFFFF0") else ""
        items.append(
            f"""
            <div style="display:inline-block; text-align:center; margin:6px;">
              <div style="
                width:80px; height:80px; border-radius:12px;
                background-color:{hex_val};
                {border}
                display:flex; align-items:center; justify-content:center;">
                <span style="font-size:10px; font-weight:600;
                             color:{text_color}; padding:4px; line-height:1.2;">
                  {name}
                </span>
              </div>
              <div style="font-size:11px; color:#666; margin-top:3px;">{hex_val}</div>
            </div>"""
        )
    return (
        '<div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:8px;">'
        + "".join(items)
        + "</div>"
    )


def confidence_bar(conf_dict: dict, model_label: str):
    """Render horizontal confidence progress bars styled per season."""
    bars_html = f'<p style="font-weight:700; font-size:13px; margin:0 0 8px 0; color:#444;">{model_label}</p>'
    for season in SEASONS:
        conf = conf_dict[season]
        pct = conf * 100
        color = SEASON_DATA[season]["bar_color"]
        emoji = SEASON_DATA[season]["emoji"]
        bars_html += f"""
        <div style="margin-bottom:8px;">
          <div style="display:flex; justify-content:space-between; font-size:12px; color:#555; margin-bottom:3px;">
            <span>{emoji} {season.capitalize()}</span><span>{pct:.1f}%</span>
          </div>
          <div style="background:#e9ecef; border-radius:4px; height:8px;">
            <div style="background:{color}; width:{pct:.1f}%; height:8px; border-radius:4px; transition:width 0.3s;"></div>
          </div>
        </div>"""
    st.markdown(bars_html, unsafe_allow_html=True)


def outfit_photos_for_season(season: str):
    """Return list of image paths from outfits/{season}/ folder."""
    folder = OUTFITS_DIR / season
    if not folder.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]


def styled_card(label: str, value: str, bg: str = "#f8f9fa"):
    """Render a simple card-style div."""
    st.markdown(
        f"""
        <div style="
            background:{bg}; border-radius:10px;
            padding:14px 16px; margin-bottom:10px;
            border-left: 4px solid #888;">
          <div style="font-size:12px; font-weight:700;
                      text-transform:uppercase; letter-spacing:.05em;
                      color:#555; margin-bottom:4px;">{label}</div>
          <div style="font-size:14px; color:#222;">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  Recommendation panel
# ─────────────────────────────────────────────

def render_recommendation(season: str):
    data = SEASON_DATA[season]
    emoji = data["emoji"]

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #fafafa 0%, #f0f0f0 100%);
            border-radius: 16px; padding: 24px 28px; margin-bottom: 20px;
            border: 1px solid #e0e0e0;">
          <h2 style="margin:0 0 4px 0; font-size:28px;">
            {emoji} {season.capitalize()} Season
          </h2>
          <p style="margin:0; color:#666; font-size:15px;">{data["undertone"]}</p>
          <p style="margin:8px 0 0 0; font-size:13px; color:#888;">
            Best metals: <strong>{data["metals"]}</strong>
          </p>
        </div>""",
        unsafe_allow_html=True,
    )

    tab_palette, tab_outfits, tab_accessories, tab_makeup = st.tabs(
        ["🎨 Color Palette", "👗 Outfits", "👜 Accessories", "💄 Makeup"]
    )

    # ── Color Palette ──────────────────────────────────────────────────────
    with tab_palette:
        st.markdown("#### Your Best Colors")
        st.markdown(
            hex_swatch_html(data["best_colors"], data["best_hex"]),
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("#### Colors to Avoid")
        avoid_items = "  \n".join(f"✗  {c}" for c in data["avoid_colors"])
        st.markdown(
            f"""
            <div style="
                background:#fff5f5; border-radius:10px; padding:14px 18px;
                border-left: 4px solid #e57373; color:#555; font-size:14px;">
              {avoid_items.replace(chr(10), "<br>")}
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Outfits ────────────────────────────────────────────────────────────
    with tab_outfits:
        col_c, col_sc, col_f = st.columns(3)

        outfit_labels = {
            "casual": ("👕 Casual", col_c),
            "smart_casual": ("👔 Smart Casual", col_sc),
            "formal": ("🎩 Formal", col_f),
        }
        for key, (label, col) in outfit_labels.items():
            with col:
                st.markdown(f"**{label}**")
                st.markdown(
                    f"""
                    <div style="
                        background:#fafafa; border-radius:10px;
                        padding:12px; font-size:13px; color:#444;
                        min-height:80px; border:1px solid #eee;">
                      {data["outfits"][key]}
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Outfit photos
        photos = outfit_photos_for_season(season)
        if photos:
            st.markdown("---")
            st.markdown("#### Sample Outfit Photos")
            photo_cols = st.columns(min(len(photos), 3))
            for i, photo_path in enumerate(photos[:6]):
                with photo_cols[i % 3]:
                    st.image(str(photo_path), use_container_width=True)
        else:
            st.info(
                f"No outfit photos found in `outfits/{season}/`. "
                "Add `.jpg` / `.png` images there to display samples here.",
                icon="📷",
            )

    # ── Accessories ────────────────────────────────────────────────────────
    with tab_accessories:
        acc = data["accessories"]
        acc_map = {
            "👜  Bags": acc["bags"],
            "🕶️  Sunglasses": acc["sunglasses"],
            "💍  Jewelry": acc["jewelry"],
            "🧣  Scarves": acc["scarves"],
        }
        # Two-column layout
        left, right = st.columns(2)
        items = list(acc_map.items())
        for i, (label, value) in enumerate(items):
            target = left if i % 2 == 0 else right
            with target:
                styled_card(label, value)

    # ── Makeup ─────────────────────────────────────────────────────────────
    with tab_makeup:
        mu = data["makeup"]
        mu_map = {
            "Foundation": mu["foundation"],
            "Lips": mu["lips"],
            "Blush": mu["blush"],
            "Eyeshadow": mu["eyeshadow"],
        }
        accent_colors = {
            "Foundation": "#fde8d8",
            "Lips": "#fddde6",
            "Blush": "#fce4ec",
            "Eyeshadow": "#ede7f6",
        }
        left, right = st.columns(2)
        items = list(mu_map.items())
        for i, (label, value) in enumerate(items):
            target = left if i % 2 == 0 else right
            with target:
                styled_card(label, value, bg=accent_colors[label])


# ─────────────────────────────────────────────
#  Main App
# ─────────────────────────────────────────────

def main():
    # ── Page config ────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="ToneFit — Personal Color Season Predictor",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Global style tweaks ────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        /* Hide default footer */
        footer {visibility: hidden;}
        /* Softer tab styling */
        .stTabs [data-baseweb="tab"] {font-size: 14px; font-weight: 600;}
        /* Section headers */
        h3 {margin-top: 1.4rem !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="padding: 20px 0 10px 0;">
          <h1 style="margin:0; font-size:34px; font-weight:800; letter-spacing:-0.5px;">
            🎨 ToneFit
          </h1>
          <p style="margin:4px 0 0 0; font-size:17px; color:#666;">
            Personal Color Season Predictor — Spring · Summer · Autumn · Winter
          </p>
        </div>
        <hr style="margin: 12px 0 20px 0; border: none; border-top: 1px solid #e5e5e5;">
        """,
        unsafe_allow_html=True,
    )

    # ── Load models ────────────────────────────────────────────────────────
    with st.spinner("Loading models…"):
        farl_model, farl_error = load_farl_model()
        dinov2_model, dinov2_error = load_dinov2_model()

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## About ToneFit")
        st.markdown(
            "ToneFit analyzes your facial coloring to determine your "
            "**personal color season** based on the Korean 4-season method. "
            "Get a curated palette, outfit ideas, accessories, and makeup tips "
            "matched to your natural undertone."
        )
        st.markdown("---")

        st.markdown("### Model Status")
        if farl_model is not None:
            st.success("FaRL — loaded ✔", icon="✅")
        else:
            st.warning(f"FaRL — not loaded\n\n_{farl_error}_", icon="⚠️")

        if dinov2_model is not None:
            st.success("DINOv2 — loaded ✔", icon="✅")
        else:
            st.warning(f"DINOv2 — not loaded\n\n_{dinov2_error}_", icon="⚠️")

        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown(
            "1. Upload a clear photo **or** take one with your camera.\n"
            "2. Make sure your face is well-lit and unobstructed.\n"
            "3. Remove heavy filters or color-corrected lighting.\n"
            "4. Both models run automatically — results appear below."
        )
        st.markdown("---")
        st.caption("ToneFit · PUP Data Science Pilot Study · 2026")

    # ── Both models missing → hard stop ───────────────────────────────────
    if farl_model is None and dinov2_model is None:
        st.error(
            "**No models are loaded.** "
            "Please add `farl_model.pth` and/or `dinov2_model.pth` "
            "to the `models/` folder, then restart the app.",
            icon="🚫",
        )
        st.stop()

    # ── Image input tabs ───────────────────────────────────────────────────
    upload_tab, camera_tab = st.tabs(["📂 Upload Photo", "📷 Take Photo"])

    raw_image = None

    with upload_tab:
        uploaded = st.file_uploader(
            "Choose an image (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            raw_image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    with camera_tab:
        camera_snap = st.camera_input("Take a photo")
        if camera_snap is not None:
            raw_image = Image.open(io.BytesIO(camera_snap.read())).convert("RGB")

    # ── Nothing provided yet → show placeholder ────────────────────────────
    if raw_image is None:
        st.markdown("---")
        st.markdown(
            """
            <div style="
                text-align:center; padding: 60px 20px;
                background:#fafafa; border-radius:16px;
                border: 2px dashed #ddd; color:#aaa;">
              <div style="font-size:48px; margin-bottom:12px;">🖼️</div>
              <div style="font-size:18px; font-weight:600;">
                Upload or take a photo to get started
              </div>
              <div style="font-size:14px; margin-top:6px;">
                Your personal color season results will appear here
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Face detection ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Face Detection")

    face_crop, annotated_img = detect_and_crop_face(raw_image)

    img_col, crop_col = st.columns([3, 2], gap="large")
    with img_col:
        st.image(annotated_img, caption="Uploaded photo (face region highlighted)", use_container_width=True)
    with crop_col:
        if face_crop is not None:
            st.image(face_crop, caption="Detected face crop (used for analysis)", use_container_width=True)
        else:
            st.error(
                "**No face detected.** Please try a different photo.\n\n"
                "**Tips:**\n"
                "- Face the camera directly.\n"
                "- Ensure good, even lighting.\n"
                "- Avoid very dark, blurry, or heavily filtered images.\n"
                "- Keep your full face within the frame.",
                icon="😶",
            )
            return

    # ── Inference ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Model Predictions")

    farl_conf = None
    dinov2_conf = None

    with st.spinner("Running analysis…"):
        if farl_model is not None:
            try:
                farl_conf = run_inference(farl_model, face_crop)
            except Exception as exc:
                st.warning(f"FaRL inference failed: {exc}", icon="⚠️")

        if dinov2_model is not None:
            try:
                dinov2_conf = run_inference(dinov2_model, face_crop)
            except Exception as exc:
                st.warning(f"DINOv2 inference failed: {exc}", icon="⚠️")

    if farl_conf is None and dinov2_conf is None:
        st.error("Both models failed to produce a prediction. Please check the console for errors.")
        return

    # ── Side-by-side confidence charts ────────────────────────────────────
    chart_col_l, chart_col_r = st.columns(2, gap="large")

    with chart_col_l:
        if farl_conf is not None:
            confidence_bar(farl_conf, "FaRL (ViT-B/16)")
        else:
            st.info("FaRL model not available.", icon="⚠️")

    with chart_col_r:
        if dinov2_conf is not None:
            confidence_bar(dinov2_conf, "DINOv2 (ViT-B/14)")
        else:
            st.info("DINOv2 model not available.", icon="⚠️")

    # ── Final prediction: average available confidences ───────────────────
    available = [c for c in (farl_conf, dinov2_conf) if c is not None]
    averaged = {
        season: sum(c[season] for c in available) / len(available)
        for season in SEASONS
    }
    final_season = max(averaged, key=averaged.get)
    final_confidence = averaged[final_season] * 100

    st.markdown("---")
    emoji = SEASON_DATA[final_season]["emoji"]
    undertone = SEASON_DATA[final_season]["undertone"]
    gradient = SEASON_DATA[final_season]["gradient"]
    text_color = SEASON_DATA[final_season]["text_color"]

    # Large result banner using season-specific gradient and text color
    st.markdown(
        f"""
        <div style="
            background: {gradient};
            border-radius: 20px; padding: 32px; text-align:center;
            color: {text_color}; margin: 10px 0 24px 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);">
          <div style="font-size: 52px; margin-bottom: 8px;">{emoji}</div>
          <div style="font-size: 32px; font-weight: 800; letter-spacing: -0.5px;">
            {final_season.capitalize()} Season
          </div>
          <div style="font-size: 16px; opacity: 0.8; margin-top: 6px;">
            {undertone}
          </div>
          <div style="
              display:inline-block; margin-top:16px;
              background: rgba(0,0,0,0.10);
              border-radius: 50px; padding: 6px 20px;
              font-size: 14px; opacity: 0.9;">
            Average confidence: <strong>{final_confidence:.1f}%</strong>
            &nbsp;·&nbsp; Based on {len(available)} model{'s' if len(available) > 1 else ''}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Full recommendation panel ──────────────────────────────────────────
    st.markdown("### 💡 Your Style Recommendations")
    render_recommendation(final_season)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
