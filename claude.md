# ToneFit ML — CLAUDE.md

## CRITICAL — READ FIRST

This project compares two models ONLY:
- Model A: FaRL (Deep Learning) — DONE ✅ accuracy 0.5537
- Model B: SVM (Classical CV) — train_svm.py needed ⏳ (check first if this already exists)

tonefit_data_complete.json — DO NOT DELETE OR ARCHIVE EVER
This is the app content database for all 12 color sub-types.

---

## Project

Personal color season classification using Deep Armocromia dataset
(Stacchio et al., ECCV 2024). 4,920 RGB-M face images, 4 seasons.

Research question: Do deep learning models (FaRL) provide an
advantage over classical computer vision (SVM + CIELab) for
Armocromia classification? — Directly answering Stacchio et al.
(2024) Recommendation 3.

---

## Tech Stack

- Frontend: Next.js (React)
- Backend: Node.js (Next.js API routes)
- ML service: Python FastAPI (backend/main.py + server.py)
- App data: tonefit_data_complete.json

---

## Models

### Model A — FaRL (DONE ✅)
- Script: train_farl.py
- Backbone: CLIP ViT-Base/16 with FaRL pretrained weights, frozen
- Classifier: FC(512→256) → ReLU → Dropout(0.5) → FC(256→4)
- Trained weights: models/farl_model.pth
- Accuracy: 0.5537

### Model B — SVM (check first if train_svm.py already exists)
- Script: train_svm.py
- Kernel: RBF, GridSearchCV C=[0.1,1,10,100] gamma=[scale,auto]
- Input: 10-dim CIELab/HSV/ITA features from features.csv (preprocess.py)
- class_weight=balanced, 5-fold CV
- Save: models/svm_model.pkl, models/scaler.pkl

---

## Dataset Structure (READ ONLY — DO NOT MODIFY)

Location: RGB-M/ folder in workspace root.

```
RGB-M/
  train/
    autumn/ spring/ summer/ winter/
  test/
    autumn/ spring/ summer/ winter/
```

CRITICAL RULES:
1. RGB-M is READ ONLY — never move, copy, delete, or modify anything inside
2. Train/test split already done — use as-is

---

## Paper Baselines (Stacchio et al. 2024)

| Model      | Season Acc | F1    |
|------------|-----------|-------|
| FaRL-16    | 0.525     | 0.516 |
| FaRL-64    | 0.554     | 0.548 |
| ResNeXt50  | 0.513     | 0.502 |

Target: Beat FaRL-64 (0.554) with Model A, show comparison with Model B (SVM).

---

## File Structure

```
ToneFit/
├── train_farl.py              ← Model A training (DONE)
├── train_svm.py               ← Model B training
├── preprocess.py              ← Feature extraction — do not modify
├── evaluate.py                ← Compare both models vs paper baselines
├── backend/main.py            ← FastAPI ML server (FaRL + SVM inference)
├── server.py                  ← FastAPI server for Next.js frontend
├── app.py                     ← Streamlit demo
├── tonefit_data_complete.json ← App content DB — NEVER DELETE
├── annotations.csv            ← Dataset labels
├── ToneFit_Colab.ipynb        ← Main training pipeline
├── ToneFit_Kaggle_FaRL_SVM.ipynb ← Kaggle alternative
├── eda.ipynb                  ← EDA
├── models/farl_weights.pth    ← FaRL-64 pretrained (gitignored)
├── models/farl_model.pth      ← Trained FaRL head (gitignored)
├── results/                   ← Training outputs (gitignored)
├── RGB-M/                     ← Dataset — READ ONLY
├── archive/                   ← Old scripts — keep entire folder
└── web/                       ← Next.js frontend
```

---

## DO NOT

- Use MobileNetV2, DINOv2, ResNet, or any other model
- Delete or archive tonefit_data_complete.json
- Use Streamlit as the main deployment target
- Reference Filipino celebrity dataset
- Add Random Forest as a primary model
- Modify RGB-M/ in any way
