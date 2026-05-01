# ToneFit ML

**Machine Learning-Based Personal Color Season Prediction and Clothing Color Recommendation**

A pilot study applying machine learning to classify personal color seasons (Spring, Summer, Autumn, Winter) from facial images — with a focus on Filipino skin tone representation. Grounded in Korean seasonal color analysis methodology.

---

## Project Overview

ToneFit ML predicts a user's personal color season from a facial image and recommends suitable clothing color palettes. The system compares two Vision Transformer model approaches:

- **Model A** — FaRL (Face Representation Learning) ViT-Base, Microsoft, pretrained on LAION-Face
- **Model B** — DINOv2 ViT-Base, Meta AI, self-supervised general-purpose ViT

Both models are benchmarked against baselines from the Deep Armocromia paper (Stacchio et al., ECCV 2024).

**Output:** Predicted season + confidence scores + clothing color palette + curated outfit sample photos

---

## Personal Color Seasons

| Season | Undertone | Characteristics |
|--------|-----------|-----------------|
| Spring | Warm | Light, clear, peachy |
| Summer | Cool | Light, muted, ashy |
| Autumn | Warm | Deep, muted, earthy |
| Winter | Cool | Deep, clear, high contrast |

---

## Project Structure

```
ToneFit/
├── collectdata.py         # Step 1: Download and crop celebrity face images
├── preprocess.py          # Step 2: Extract CIELab/HSV features, generate data_split.csv
├── eda.ipynb              # Step 3: Exploratory data analysis
├── train_farl.py          # Step 4: Train FaRL ViT-Base model
├── train_dinov2.py        # Step 5: Train DINOv2 ViT-Base model
├── evaluate.py            # Step 6: Compare model performance vs paper baselines
├── app.py                 # Step 7: Streamlit web app
├── dataset/               # Collected and labeled face images
│   ├── spring/
│   ├── summer/
│   ├── autumn/
│   └── winter/
├── models/                # Saved trained models (.pth files)
├── results/               # Evaluation outputs, confusion matrices, comparison charts
├── outfits/               # Curated outfit reference photos per season
│   ├── spring/
│   ├── summer/
│   ├── autumn/
│   └── winter/
├── data_split.csv         # Train/test split manifest (filename, season, split)
├── features.csv           # Extracted CIELab/HSV features per image
├── CLAUDE.md              # Full project brief for AI assistance
└── README.md
```

---

## Dataset

Self-collected dataset of verified celebrity face images labeled by personal color season.

**Sources:**
- Filipino celebrities (primary focus) — labels from documented professional diagnoses
- Korean celebrities — from K-beauty color analysis communities
- Western celebrities — from international color analysis resources
- Deep Armocromia Dataset (Stacchio et al., ECCV 2024) — ~4,920 expert-labeled images

**Current size:** ~952 raw images collected; ~570 usable after manual cleaning

| Season | Filipino | Korean | Western | Total Celebrities |
|--------|----------|--------|---------|-------------------|
| Spring | 4 | 6 | 5 | 15 |
| Summer | 5 | 6 | 4 | 15 |
| Autumn | 10 | 3 | 2 | 15 |
| Winter | 7 | 5 | 3 | 15 |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/ajipal/ToneFit.git
cd ToneFit

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Collect dataset
```bash
python collectdata.py
```

To split among group members, edit `RUN_SEASONS` at the bottom of the file:
```python
RUN_SEASONS = ["spring"]   # or "summer", "autumn", "winter"
```

### Step 2 — Manual cleaning
Review each season folder in Windows Explorer. Delete images where the face does not belong to the correct celebrity.

### Step 3 — Preprocess & extract features
```bash
python preprocess.py
```
Outputs: `features.csv`, `data_split.csv`, `X_train.npy`, `X_test.npy`, etc.

### Step 4 — Train models (run on Colab with T4 GPU)
```bash
python train_farl.py      # FaRL ViT-Base (Model A)
python train_dinov2.py    # DINOv2 ViT-Base (Model B)
```

### Step 5 — Evaluate
```bash
python evaluate.py
```
Outputs confusion matrices and comparison chart vs Deep Armocromia paper baselines.

### Step 6 — Run Streamlit app
```bash
streamlit run app.py
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch, timm (FaRL), torch.hub (DINOv2) |
| Image Processing | OpenCV |
| Color Analysis | scikit-image (CIELab), NumPy |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |

---

## Model Architecture

Both models share the same classifier head following the Deep Armocromia paper:

```
Backbone (FaRL ViT-Base or DINOv2 ViT-Base, feature_dim=768)
  → Linear(768, 384)
  → ReLU
  → Dropout(0.5)
  → Linear(384, 4)
  → Softmax
```

Training: AdamW (lr=1e-3, weight_decay=1e-5), CosineAnnealingWarmRestarts (T_0=10), 50 epochs, batch_size=64

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score (macro + weighted)
- Confusion Matrix per model
- Top-2 Accuracy (for borderline season cases)
- Comparison against paper baselines:
  - FaRL16: 0.525 | FaRL64: 0.554 | ResNeXt50: 0.513

---

## References

- Stacchio, L. et al. (2024). Deep Armocromia. ECCV 2024. github.com/lorenzo-stacchio/Deep-Armocromia
- Kye & Lee (2022). Skin color classification of Koreans using clustering. PMC9907718.
- PSY222 et al. (2023). ColorInsight. github.com/PSY222/Colorinsight
- KIISE (2024). Learning-based Model Comparison for Personal Color Diagnosis.
- Wang et al. (2022). FaRL: General Facial Representation Learning. CVPR 2022.
- Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. Meta AI.

---

## Team

Polytechnic University of the Philippines — Data Science / Current Trends in IT Course
Group of 4 students, 2025
