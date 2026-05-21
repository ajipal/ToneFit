# ToneFit ML

**Personal Color Season Classification — FaRL vs SVM**

A pilot study for the Data Science / Current Trends in IT course at Polytechnic University of the Philippines (PUP Manila).

Compares a deep learning model (FaRL) against a classical computer vision baseline (SVM + CIELab features) for Armocromia classification, directly addressing Recommendation 3 of Stacchio et al. (ECCV 2024).

---

## Research Question

> Do deep learning models (FaRL) provide an advantage over classical computer vision (SVM + CIELab) for personal color season classification?

---

## Models

### Model A — FaRL (Deep Learning)
- Backbone: CLIP ViT-Base/16 with FaRL pretrained weights (frozen)
- Classifier: FC(512→256) → ReLU → Dropout(0.5) → FC(256→4)
- Script: `train_farl.py`
- Result: **Season Accuracy: 0.5537**

### Model B — SVM (Classical CV)
- Features: 10-dim CIELab / HSV / ITA extracted from face images
- Kernel: RBF, GridSearchCV (C=[0.1,1,10,100], gamma=[scale,auto])
- Class weighting: balanced, 5-fold cross-validation
- Script: `train_svm.py`
- Result: TBD

---

## Dataset

**Deep Armocromia** (Stacchio et al., ECCV 2024) — the first large-scale dataset of face images labeled by certified Armocromia professionals.

- ~4,920 expert-labeled face images
- 4 seasons: Spring, Summer, Autumn, Winter
- Pre-defined train/test split: `RGB-M/train/` and `RGB-M/test/`
- Request access: [github.com/lorenzo-stacchio/Deep-Armocromia](https://github.com/lorenzo-stacchio/Deep-Armocromia)
- Place `RGB-M/` in the project root — **never modify its contents**

---

## Paper Baselines (Stacchio et al. 2024)

| Model | Accuracy | F1 (macro) |
|-------|----------|------------|
| FaRL-16 | 0.525 | 0.519 |
| FaRL-64 | 0.554 | 0.548 |
| ResNeXt50 | 0.513 | 0.502 |

---

## Project Structure

```
ToneFit/
├── train_farl.py             ← Model A: FaRL deep learning (DONE ✅)
├── train_svm.py              ← Model B: SVM classical CV
├── preprocess.py             ← Feature extraction (CIELab/HSV/ITA)
├── evaluate.py               ← Compare Model A vs Model B vs paper
├── backend/
│   └── main.py               ← FastAPI ML inference server
├── server.py                 ← FastAPI server for Next.js frontend
├── app.py                    ← Streamlit demo
├── annotations.csv           ← Dataset labels (Stacchio et al.)
├── tonefit_data_complete.json ← App content: all 12 sub-types data
├── requirements.txt
├── ToneFit_Colab.ipynb       ← Main training pipeline (Colab T4)
├── ToneFit_Kaggle_FaRL_SVM.ipynb ← Kaggle alternative
├── eda.ipynb                 ← Exploratory data analysis
├── configs/                  ← Training configs
├── models/                   ← Saved weights (gitignored)
│   ├── farl_weights.pth      ← FaRL-64 pretrained (652 MB)
│   ├── farl_model.pth        ← Trained FaRL head
│   ├── svm_model.pkl         ← Trained SVM
│   └── scaler.pkl            ← Feature scaler
├── results/                  ← Evaluation outputs (gitignored)
├── RGB-M/                    ← Dataset — READ ONLY
├── archive/                  ← Old/unused scripts
└── web/                      ← Next.js frontend (Vercel)
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Frontend | Next.js (React), Tailwind CSS, TypeScript |
| Backend | Node.js (Next.js API routes) |
| ML Service | Python, FastAPI |
| Deep Learning | PyTorch, CLIP, timm |
| Classical CV | scikit-learn, scikit-image, OpenCV |
| Features | CIELab, HSV, ITA (Individual Typology Angle) |
| Training | Google Colab (T4 GPU) / Kaggle |

---

## Setup

```bash
git clone https://github.com/ajipal/ToneFit.git
cd ToneFit
pip install -r requirements.txt
```

**FaRL pretrained weights** (~652 MB, place in `models/`):
```bash
wget https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth -O models/farl_weights.pth
```

**Next.js frontend:**
```bash
cd web && npm install
```

---

## Running Locally

**ML inference server:**
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

**Next.js frontend:**
```bash
cd web && npm run dev
```

---

## References

- Stacchio, L., Paolanti, M., Spigarelli, F., & Frontoni, E. (2025). Deep Armocromia: A novel dataset for face seasonal color analysis and classification. *ECCV 2024 Workshops* (pp. 352–367). Springer. https://doi.org/10.1007/978-3-031-91569-7_22
- Wang, W., et al. (2022). FaRL: General facial representation learning in a frozen state. *CVPR 2022*.

---

## Team

Polytechnic University of the Philippines — Data Science / Current Trends in IT
2025–2026
