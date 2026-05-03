# CLAUDE.md — ToneFit ML Project Brief

Read this fully before doing anything. This is the single source of truth.

---

## What This Project Is

ToneFit ML is a pilot study for a Data Science / Current Trends in IT class at PUP Manila.

Goals:
1. Compare FaRL (face-specialized Vision Transformer) vs DINOv2 (general self-supervised ViT) for personal color season classification
2. Evaluate both models using accuracy, precision, recall, F1-score, confusion matrix
3. Build a working Streamlit app where a user takes a photo and gets season prediction + clothing/accessory/makeup recommendations

---

## Personal Color Season System

| Season | Undertone | Features |
|--------|-----------|----------|
| Spring | Warm | Light, clear, peachy |
| Summer | Cool | Light, muted, ashy |
| Autumn | Warm | Deep, muted, earthy |
| Winter | Cool | Deep, clear, high contrast |

Rule: Spring + Autumn = Warm. Summer + Winter = Cool. No Winter Warm exists.

---

## Final Confirmed Decisions

- Dataset: Deep Armocromia (Stacchio et al., ECCV 2024) — already downloaded, already split
- Model A: FaRL (Face Representation Learning — Microsoft, face-specialized ViT)
- Model B: DINOv2 (Meta AI, general self-supervised ViT)
- Comparison: Both Vision Transformers — same category, fair comparison
- Classifier head: FC(dim//2) → ReLU → Dropout(0.5) → FC(4) — same for both
- Training: AdamW(lr=1e-3, weight_decay=1e-5), CosineAnnealingWarmRestarts(T_0=10, eta_min=1e-5), 50 epochs, batch_size=64
- Fallback: If FaRL setup fails → use ResNeXt50 from torchvision
- Deployment: Streamlit web app on Streamlit Cloud

---

## What is NOT in scope

- SVM, Random Forest, or any traditional ML models
- Virtual try-on or generative image editing
- 12-tone sub-season classification
- Custom CNN from scratch
- Celebrity scraped dataset (replaced by Deep Armocromia)

---

## Dataset Structure (READ ONLY — DO NOT MODIFY)

The Deep Armocromia dataset is already downloaded and split.
Location: RGB-M/ folder in the workspace root.

```
RGB-M/
  train/
    autumn/
      deep/     ← sub-type (treat all as class: autumn)
      soft/
      warm/
    spring/
      bright/
      light/
      warm/
    summer/
      cool/
      light/
      soft/
    winter/
      bright/
      cool/
      deep/
  test/
    autumn/
      deep/
      soft/
      warm/
    spring/
      bright/
      light/
      warm/
    summer/
      cool/
      light/
      soft/
    winter/
      bright/
      cool/
      deep/
```

**CRITICAL RULES for dataset:**
1. RGB-M is READ ONLY — never move, copy, delete, or modify any files inside
2. Use RGB-M only
3. Train/test split is already done — use it exactly as-is, do NOT re-split
4. When loading images, read RECURSIVELY through sub-type subfolders (deep, soft, warm, etc.)
5. Treat ALL images under a season folder as the same class label — ignore sub-type distinctions
6. Do NOT create merge_dataset.py or any script that modifies dataset folders

**Loading images correctly:**
```python
# CORRECT — recursive loading, treats sub-types as same class
from torchvision.datasets import ImageFolder
from torchvision import transforms

train_dataset = ImageFolder(
    root='RGB-M/train',
    transform=train_transforms
)
# ImageFolder automatically reads recursively
# autumn/deep/, autumn/soft/, autumn/warm/ → all labeled as "autumn"
```

**Class label mapping (ImageFolder alphabetical order):**
- autumn = 0
- spring = 1
- summer = 2
- winter = 3

---

## Full Pipeline

### Step 1 — Preprocessing (preprocess.py) ✅ EXISTS — UPDATE PATHS
- Load images from RGB-M/train/ recursively
- Extract CIELab/HSV features for EDA ONLY (not model input)
- Features: L_mean, a_mean, b_mean, L_std, a_std, b_std, ITA, H_mean, S_mean, V_mean
- ITA formula: arctan((L_mean - 50) / b_mean) × (180/π)
- Save to features.csv
- Save data_split.csv (filename, season, split) based on RGB-M train/test folders
- DO NOT copy or move any images

### Step 2 — EDA (eda.ipynb)
- Class distribution per season
- Sample face images per season
- CIELab/HSV feature distributions per season
- ITA score boxplot per season
- PCA scatter plot
- Correlation heatmap

### Step 3 — Model A: FaRL (train_farl.py) ✅ EXISTS — UPDATE PATHS
Load FaRL as frozen feature extractor:
```python
import timm
import torch

model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
checkpoint = torch.load('models/farl_weights.pth', map_location='cpu')
state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model.eval()
for param in model.parameters():
    param.requires_grad = False
feature_dim = 768
```

Classifier head:
```python
classifier = nn.Sequential(
    nn.Linear(768, 384),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(384, 4)
)
```

Training:
- Load training images from: RGB-M/train/ (recursive)
- Test on: RGB-M/test/ (final evaluation)
- AdamW(lr=1e-3, weight_decay=1e-5)
- CosineAnnealingWarmRestarts(T_0=10, eta_min=1e-5)
- 50 epochs, batch_size=64
- CrossEntropyLoss
- Data augmentation: random crop, horizontal flip (p=0.5), color jitter (brightness=0.4, contrast=0.2, saturation=0.2), random sharpness (factor=2, p=0.2)
- Save: models/farl_model.pth
- Save history: results/farl_history.json
- Plot: results/farl_training.png

FALLBACK if FaRL weights not found:
```python
from torchvision import models
backbone = models.resnext50_32x4d(pretrained=True)
feature_dim = 2048
```

### Step 4 — Model B: DINOv2 (train_dinov2.py) ✅ EXISTS — UPDATE PATHS
Load DINOv2 as frozen feature extractor:
```python
import torch
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
for param in backbone.parameters():
    param.requires_grad = False
feature_dim = 768
```

Same classifier head, same training settings as FaRL.
- Load from: RGB-M/train/ (recursive)
- Test on: RGB-M/test/
- Save: models/dinov2_model.pth
- Save history: results/dinov2_history.json
- Plot: results/dinov2_training.png

### Step 5 — Evaluation (evaluate.py) ✅ EXISTS — UPDATE PATHS
- Load test images from RGB-M/test/ (recursive)
- Load both models
- Per model: accuracy, weighted precision, recall, F1
- Per-class recall (especially autumn)
- Top-2 accuracy
- Confusion matrix heatmap
- Training time comparison
- Output comparison table including paper baselines:

| Model | Source | Accuracy | F1 |
|-------|--------|---------|-----|
| FaRL16 | Stacchio et al. (2024) | 0.525 | 0.516 |
| FaRL64 | Stacchio et al. (2024) | 0.554 | 0.548 |
| ResNeXt50 | Stacchio et al. (2024) | 0.513 | 0.502 |
| FaRL (ours) | This study | TBD | TBD |
| DINOv2 (ours) | This study | TBD | TBD |

- Save: results/comparison_table.csv
- Save: results/confusion_farl.png, results/confusion_dinov2.png
- Save: results/model_comparison.png

### Step 6 — Streamlit App (app.py) ✅ EXISTS
- st.camera_input OR st.file_uploader
- Face detection (OpenCV Haar Cascade)
- Load best performing model
- Show confidence scores from BOTH models
- Display per predicted season:
  - Season name + undertone description
  - Color palette swatches (best + avoid)
  - Outfit reference photos from outfits/{season}/
  - Accessory recommendations
  - Makeup tips

---

## Season Recommendation Data

```python
SEASON_DATA = {
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
            "formal": "Deep emerald gown, rich burgundy suit"
        },
        "accessories": {
            "bags": "Camel leather, tan suede, chocolate brown tote",
            "sunglasses": "Tortoiseshell, warm brown frames, amber lens",
            "jewelry": "Gold, bronze, amber stones, wooden accents",
            "scarves": "Rust wool, olive plaid, warm terracotta"
        },
        "makeup": {
            "foundation": "Warm/golden undertone foundation",
            "lips": "Brick red, terracotta, warm brown, deep rust",
            "blush": "Warm peach, terracotta, copper",
            "eyeshadow": "Warm brown, bronze, copper, olive gold"
        }
    },
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
            "formal": "Champagne gold gown, warm ivory suit"
        },
        "accessories": {
            "bags": "Tan leather, warm beige, camel tote",
            "sunglasses": "Gold frames, warm tortoiseshell, brown lens",
            "jewelry": "Gold chains, pearl accents, rose gold rings",
            "scarves": "Warm peach silk, ivory floral print"
        },
        "makeup": {
            "foundation": "Warm/yellow undertone foundation",
            "lips": "Peach, coral, warm pink, salmon",
            "blush": "Peach, apricot, warm rose",
            "eyeshadow": "Warm brown, bronze, champagne gold"
        }
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
            "formal": "Dusty blue gown, soft lavender suit"
        },
        "accessories": {
            "bags": "Soft gray leather, dusty rose, pale blue",
            "sunglasses": "Silver frames, gray-blue lens, cool tortoiseshell",
            "jewelry": "Silver, white gold, moonstone, pearl",
            "scarves": "Lavender silk, soft gray cashmere"
        },
        "makeup": {
            "foundation": "Cool/pink undertone foundation",
            "lips": "Mauve, berry, cool pink, rose",
            "blush": "Cool pink, soft rose, berry",
            "eyeshadow": "Cool taupe, mauve, soft lavender, gray"
        }
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
            "formal": "Stark white or black gown, jewel-tone suit"
        },
        "accessories": {
            "bags": "Black leather, crisp white, deep navy",
            "sunglasses": "Black frames, silver hardware, dark lens",
            "jewelry": "Silver, platinum, sapphire, diamond",
            "scarves": "Black cashmere, white silk, royal blue wool"
        },
        "makeup": {
            "foundation": "Cool/neutral undertone foundation",
            "lips": "True red, berry, deep plum, cool pink",
            "blush": "Cool pink, berry, soft rose",
            "eyeshadow": "Cool gray, navy, silver, deep plum"
        }
    }
}
```

---

## File Structure

```
ToneFit/                     ← main project folder
├── app.py                   ← Streamlit app
├── collectdata.py           ← supplemental data collection (not used now)
├── preprocess.py            ← feature extraction + EDA prep
├── train_farl.py            ← Model A training
├── train_dinov2.py          ← Model B training
├── evaluate.py              ← model comparison
├── eda.ipynb                ← EDA notebook
├── ToneFit_Colab.ipynb      ← Colab notebook
├── requirements.txt
├── README.md
├── CLAUDE.md
├── models/                  ← saved model weights (created at runtime)
│   ├── farl_weights.pth     ← FaRL pretrained weights (download separately)
│   ├── farl_model.pth       ← trained FaRL classifier
│   └── dinov2_model.pth     ← trained DINOv2 classifier
├── results/                 ← evaluation outputs (created at runtime)
└── outfits/                 ← curated outfit photos
    ├── autumn/
    ├── spring/
    ├── summer/
    └── winter/

RGB-M/                       ← dataset (READ ONLY)
  train/
    autumn/  summer/  winter/  spring/
    (each season has sub-type subfolders — treat all as same class)
  test/
    autumn/  summer/  winter/  spring/
```

---

## requirements.txt

```
streamlit
torch
torchvision
timm>=0.9.0
opencv-python-headless
scikit-image
numpy
pillow
scikit-learn
matplotlib
seaborn
pandas
imagehash
```

---

## Evaluation Paper Baselines

From Stacchio et al. (2024) Table 2 — use these as reference in evaluate.py output:
- FaRL16: accuracy=0.525, F1=0.516, Top-2=0.815
- FaRL64: accuracy=0.554, F1=0.548, Top-2=0.808
- ResNeXt50: accuracy=0.513, F1=0.502, Top-2=0.789

Expected autumn confusion: 80 samples misclassified as winter (per paper confusion matrix)

---

## Academic Notes

- Gap: No study has evaluated DINOv2 on Deep Armocromia or compared it against FaRL
- Novelty: First FaRL vs DINOv2 comparison on Deep Armocromia dataset
- Autumn: Expected hardest class — track separately, cite paper confusion matrix
- Do NOT mention: AWS AI Stylist, Filipino focus as primary, dataset validation problems
- Cite Deep Armocromia paper for all baseline numbers

---

## Current Status

- [x] FaRL setup complete
- [x] DINOv2 setup complete
- [x] preprocess.py written
- [x] train_farl.py written
- [x] train_dinov2.py written
- [x] evaluate.py written
- [x] app.py written
- [x] requirements.txt created
- [x] Dataset downloaded (RGB-M folder)
- [x] Season folders renamed to English
- [ ] Update all scripts to use RGB-M/train/ and RGB-M/test/ paths
- [ ] Run preprocess.py for EDA
- [ ] Train both models on Colab T4 GPU
- [ ] Evaluate and compare results
- [ ] Deploy app on Streamlit Cloud
- [ ] Write final paper

---

## What Claude Code Should Do Right Now

Update these 4 scripts to correctly use the RGB-M dataset:

1. preprocess.py — read from RGB-M/train/ recursively, extract CIELab/HSV features to features.csv
2. train_farl.py — load training images from RGB-M/train/, test images from RGB-M/test/
3. train_dinov2.py — same as train_farl.py
4. evaluate.py — load test images from RGB-M/test/, output comparison table with paper baselines

DO NOT run any scripts.
DO NOT modify any files inside RGB-M/.