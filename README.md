# ToneFit ML

**Machine Learning-Based Personal Color Season Prediction and Clothing Color Recommendation**

A pilot study applying machine learning to automate personal color season classification (Spring, Summer, Autumn, Winter) from facial images, using the Deep Armocromia dataset and comparing FaRL against DINOv2.

---

## Introduction

### Context and Frame of Reference

Personal color analysis, known in Italian as Armocromia, is a methodology that identifies the most flattering color palette for an individual based on their natural physical features — specifically skin tone, undertone, hair color, and eye color. The system classifies individuals into four seasonal categories: Spring (Warm, Light), Summer (Cool, Light), Autumn (Warm, Deep), and Winter (Cool, Deep). Each season corresponds to a recommended set of clothing colors, accessories, and makeup that harmonize with the individual's natural coloring, while contrasting colors that clash with their undertone are identified as unflattering.

Korea has established itself as the global center of personal color analysis practice, with the methodology gaining widespread adoption in mainstream fashion and beauty culture. Academic and technology sectors have responded by developing machine learning approaches to automate what has historically been a subjective, expert-driven process. The publication of the Deep Armocromia dataset at ECCV 2024 (Stacchio et al., 2024) — the first large-scale dataset of face images labeled by certified Armocromia professionals — has created a foundation for rigorous, reproducible machine learning research in this domain.

Stacchio et al. (2024) evaluated FaRL (Face Representation Learning) and ResNeXt50 on this dataset, finding that face-specialized Vision Transformers marginally outperform CNN-based models for season classification. However, their study did not evaluate DINOv2, a powerful general-purpose self-supervised Vision Transformer released by Meta AI in 2023, which has demonstrated strong performance across diverse visual classification tasks including facial analysis. ToneFit ML investigates whether DINOv2 can match or exceed FaRL on the Armocromia classification task, and delivers a working application that makes personal color analysis accessible to any user through a standard web browser.

### Motivation from Previous Works

The integration of machine learning into the fashion industry has expanded the possibilities for objective and automated styling systems. Prior research has shown that computer vision can effectively translate human visual perception into quantifiable digital representations. In the context of color analysis, Stacchio et al. (2024) demonstrated the use of deep learning models for Armocromia classification, where architectures such as FaRL64, FaRL16, and ResNeXt50 achieved accuracies of 0.554, 0.525, and 0.513 respectively on a 4-season classification task. Their findings highlight that face-specialized pretraining, such as FaRL trained on LAION-Face, provides a measurable advantage over general-purpose models pretrained on ImageNet. However, the study also emphasized persistent challenges in Armocromia classification, particularly the high confusion between visually similar categories such as Autumn and Winter, where significant misclassification rates were observed.

In parallel, advancements in self-supervised learning have introduced more generalized yet powerful visual representation models. Oquab et al. (2023) proposed DINOv2, a Vision Transformer trained on a large-scale curated dataset of 142 million images, which has shown strong transferability across various downstream vision tasks. Subsequent studies (Zhao et al., 2025) have reported that DINOv2 can outperform both convolutional neural networks and supervised transformer models in fine-grained facial classification problems. This suggests that despite not being explicitly face-pretrained, DINOv2 may offer competitive or even superior performance compared to face-specialized models such as FaRL in Armocromia classification tasks, making it a relevant candidate for further comparative evaluation.

### The Gap, Problem, or Opportunity

The Deep Armocromia paper evaluated FaRL and ResNeXt50 but did not include DINOv2 or any other general-purpose self-supervised transformer. This creates a specific research gap: it is unknown whether DINOv2's powerful general visual representations can match or exceed FaRL's face-specialized representations for personal color season classification. Answering this question has practical implications — DINOv2 is significantly easier to deploy than FaRL, which requires specialized setup and custom pretrained weights. If DINOv2 achieves comparable accuracy, it represents a more accessible alternative for future personal color classification systems.

Additionally, no working, publicly accessible application currently exists that combines ML-based personal color season prediction with comprehensive clothing, accessory, and makeup recommendations in a single deployable tool.

---

## Personal Color Seasons

| Season | Undertone | Characteristics |
|--------|-----------|-----------------|
| Spring | Warm, Light | Clear, peachy, bright |
| Summer | Cool, Light | Muted, ashy, soft |
| Autumn | Warm, Deep | Earthy, muted, rich |
| Winter | Cool, Deep | High contrast, clear, bold |

---

## Project Structure

```
ToneFit/
├── collectdata.py         # Data collection script (reference only)
├── preprocess.py          # Step 1: Extract CIELab/HSV features, generate data_split.csv
├── eda.ipynb              # Step 2: Exploratory data analysis
├── train_farl.py          # Step 3: Train FaRL ViT-Base model (Model A)
├── train_dinov2.py        # Step 4: Train DINOv2 ViT-Base model (Model B)
├── evaluate.py            # Step 5: Compare model performance vs paper baselines
├── app.py                 # Step 6: Streamlit web app
├── audit_dataset.py       # Optional: dataset quality check after preprocess.py
├── dataset/               # Deep Armocromia images (not tracked in git)
│   ├── spring/
│   ├── summer/
│   ├── autumn/
│   └── winter/
├── models/                # Saved model weights (not tracked in git)
│   └── farl_weights.pth   # FaRL ep64 pretrained weights — download separately
├── results/               # Evaluation outputs (not tracked in git)
├── outfits/               # Curated outfit reference photos per season
├── data_split.csv         # Train/test manifest — generated by preprocess.py
├── features.csv           # CIELab/HSV features — generated by preprocess.py
├── CLAUDE.md              # Full project brief
└── README.md
```

---

## Dataset

This project uses the **Deep Armocromia dataset** (Stacchio et al., ECCV 2024) — the first publicly available large-scale dataset of face images labeled by certified Armocromia professionals.

- ~4,920 expert-labeled face images
- 4 classes: Spring, Autumn, Summer, Winter
- Pre-defined train/test split included in `annotations.csv`
- Download requires filling the request form (see below)

**Dataset access:**
1. Fill the request form: [https://forms.gle/icac2opCYqF79RyE9](https://forms.gle/icac2opCYqF79RyE9)
2. Download from Google Drive (link provided after form submission)
3. Extract into `dataset/spring/`, `dataset/summer/`, `dataset/autumn/`, `dataset/winter/`

> Note: Italian season names in the zip map as follows — `primavera` → spring, `estate` → summer, `autunno` → autumn, `inverno` → winter

---

## Setup

```bash
git clone https://github.com/ajipal/ToneFit.git
cd ToneFit
pip install -r requirements.txt
```

**FaRL pretrained weights** (download once, place in `models/`):
```bash
# ~622 MB — FaRL ViT-Base trained on LAION-Face for 64 epochs
wget https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth -O models/farl_weights.pth
```

---

## Usage

### Step 1 — Preprocess dataset
```bash
python preprocess.py
```
Outputs: `features.csv`, `data_split.csv`

### Step 2 — Train models (Colab with T4 GPU recommended)
```bash
python train_farl.py      # Model A — FaRL ViT-Base
python train_dinov2.py    # Model B — DINOv2 ViT-B/14
```

### Step 3 — Evaluate
```bash
python evaluate.py
```
Generates confusion matrices and comparison chart vs Deep Armocromia paper baselines.

### Step 4 — Run the web app
```bash
streamlit run app.py
```

---

## Model Architecture

Both models share the same classifier head following the Deep Armocromia paper:

```
Backbone (FaRL ViT-Base or DINOv2 ViT-B/14, feature_dim=768, frozen)
  → Linear(768, 384) → ReLU → Dropout(0.5) → Linear(384, 4)
```

**Training settings:** AdamW (lr=1e-3, weight_decay=1e-5), CosineAnnealingWarmRestarts (T_0=10), 50 epochs, batch_size=64

---

## Paper Baselines (Deep Armocromia, Stacchio et al. 2024)

| Model | Accuracy | F1 (macro) |
|-------|----------|------------|
| FaRL16 | 0.525 | 0.519 |
| FaRL64 | 0.554 | 0.548 |
| ResNeXt50 | 0.513 | 0.502 |

ToneFit ML evaluates FaRL64 and DINOv2 against these baselines.

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

## References

- Stacchio, L., Paolanti, M., Spigarelli, F., & Frontoni, E. (2025). Deep Armocromia: A novel dataset for face seasonal color analysis and classification. In *Computer Vision – ECCV 2024 Workshops* (pp. 352–367). Springer Nature Switzerland. https://doi.org/10.1007/978-3-031-91569-7_22
- Wang, W., et al. (2022). FaRL: General facial representation learning in a frozen state. *CVPR 2022*.
- Oquab, M., et al. (2023). DINOv2: Learning robust visual features without supervision. *Meta AI*. arXiv:2304.07193
- Zhao, et al. (2025). Fine-grained facial classification with self-supervised transformers.
- Kye & Lee (2022). Skin color classification of Koreans using clustering. PMC9907718.

---

## Team

Polytechnic University of the Philippines — Data Science / Current Trends in IT Course
Group of 4 students, 2025
