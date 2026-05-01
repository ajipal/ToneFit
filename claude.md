# CLAUDE.md — ToneFit ML Project Brief

This file briefs Claude Code on everything about this project.
Read this fully before doing anything.

---

## What This Project Is

ToneFit ML is a **pilot study** for a Data Science class at PUP (Polytechnic University of the Philippines).

The goal is to build and compare machine learning models that:
1. Take a facial image as input
2. Predict the user's **personal color season** (Spring / Summer / Autumn / Winter)
3. Output a **clothing color palette** that suits that season
4. Show **sample outfit photos** for that season

This is based on **Korean personal color analysis** — a widely practiced beauty methodology that classifies people into 4 seasons based on skin tone, undertone, and contrast level.

---

## Personal Color Season System

| Season | Undertone | Features | Best Colors |
|--------|-----------|----------|-------------|
| Spring | Warm | Light, clear, peachy skin | Peach, coral, warm turquoise, gold |
| Summer | Cool | Light, muted, ashy skin | Lavender, dusty rose, powder blue |
| Autumn | Warm | Deep, muted, earthy skin | Rust, olive, camel, burnt orange |
| Winter | Cool | Deep, clear, high contrast | Black, white, jewel tones, royal blue |

**Key rule:** Spring and Autumn = Warm undertone. Summer and Winter = Cool undertone. There is no "Winter Warm" in the 4-season system.

---

## Project Scope (What We Are and Are NOT Building)

### ✅ We ARE building:
- Automated face detection from uploaded photo
- CIELab + HSV color feature extraction from face region (for EDA)
- FaRL (Face Representation Learning) ViT-Base classifier (Model A)
- DINOv2 ViT-Base classifier (Model B)
- Season prediction with confidence score from both models (ensemble)
- Color palette display per season
- Side-by-side outfit sample photo display (pre-curated, not virtual try-on)
- Streamlit web app (deployable on Streamlit Cloud)

### ❌ We are NOT building:
- SVM / Random Forest traditional ML models
- Virtual try-on or body warping
- 12-tone sub-season classification
- Real-time video analysis
- A deployed production app (prototype only)

---

## Dataset

### Self-Collected Celebrity Dataset
Face images of celebrities with publicly documented personal color seasons.
Labels come from professional color analyst diagnoses published online — NOT guessed by the team.

**Current status:** ~952 raw images collected (spring=254, summer=211, autumn=244, winter=243).
Expect ~570 usable images after manual cleaning (40% loss estimate).

**Structure:**
```
dataset/
  spring/    → face images (224x224 px, cropped)
  summer/    → face images
  autumn/    → face images
  winter/    → face images
```

**Celebrity breakdown per season:**

Spring (15 celebrities):
- Filipino: Anne Curtis, Julia Barretto, Marian Rivera, James Reid
- Korean: IU, Yoona, Kim Chaewon, Kim Soo-hyun, Jung Hae-in, Felix (Stray Kids)
- Western: Chris Hemsworth, Pedro Pascal, Sterling K. Brown, Emma Stone, Ariana Grande

Summer (15 celebrities):
- Filipino: Jodi Sta. Maria, Janine Gutierrez, Shaina Magdayao, Richard Gutierrez, Matteo Guidicelli
- Korean: Son Ye-jin, Irene (Red Velvet), Jang Wonyoung, Taeyeon, Cha Eunwoo, Jimin (BTS)
- Western: Taylor Swift, Nicole Kidman, Robert Pattinson, Timothée Chalamet

Autumn (15 celebrities):
- Filipino: Kathryn Bernardo, Nadine Lustre, Gabbi Garcia, Coleen Garcia, Liza Soberano, Piolo Pascual, Daniel Padilla, Enrique Gil, Coco Martin, Joshua Garcia
- Korean: Jennie (BLACKPINK), Jaehyun (NCT), V/Taehyung (BTS)
- Western: Beyoncé, Oscar Isaac

Winter (15 celebrities):
- Filipino: Heart Evangelista, Pia Wurtzbach, Alden Richards, Dingdong Dantes, Paulo Avelino, Xian Lim, Donny Pangilinan
- Korean: Jisoo (BLACKPINK), Suga/Yoongi (BTS), Jin (BTS), Hyun Bin, Song Hye-kyo
- Western: Megan Fox, Dua Lipa, Keanu Reeves

### Deep Armocromia Dataset (Primary / Supplement)
Download from: https://github.com/lorenzo-stacchio/Deep-Armocromia
~4,920 expert-labeled face images (Spring/Summer/Autumn/Winter), ECCV 2024 paper dataset.
Add to the corresponding season folders after downloading.

---

## Full Pipeline (Steps in Order)

### Step 1 — Data Collection (`collectdata.py`) ✅ DONE
- Uses BingImageCrawler to scrape images for each celebrity
- Applies OpenCV Haar Cascade face detector
- Crops and saves 224x224 face images per season folder
- Result: ~952 images across 4 seasons

### Step 2 — Preprocessing (`preprocess.py`) ✅ DONE
- Remove duplicates (perceptual hashing)
- Convert images to CIELab and HSV color spaces
- Extract features per image:
  - L* mean, a* mean, b* mean (CIELab)
  - L* std, a* std, b* std
  - ITA score: arctan((L* - 50) / b*) × (180/π)
  - Hue mean, Saturation mean, Value mean (HSV)
- Save features to CSV: `features.csv`
- Save train/test split manifest: `data_split.csv` (columns: filename, season, split)
- Split: 80% train / 20% test (stratified, random_state=42) — follows Deep Armocromia paper
- Save: `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy` (EDA use only)

### Step 3 — EDA (`eda.ipynb`) 🔲 TODO
- Class distribution bar chart
- Sample face images per season (grid)
- Feature distributions (boxplots of L*, a*, b* per season)
- Correlation heatmap of features
- PCA visualization (2D scatter, colored by season)

### Step 4 — FaRL Training (`train_farl.py`) 🔲 TODO
- Load ViT-Base via timm, apply FaRL pretrained weights from `models/farl_weights.pth`
- Fallback: ResNeXt50 from torchvision if FaRL weights not found
- Classifier head: FC(384) → ReLU → Dropout(0.5) → FC(4)
- Optimizer: AdamW(lr=1e-3, weight_decay=1e-5)
- Scheduler: CosineAnnealingWarmRestarts(T_0=10, eta_min=1e-5)
- 50 epochs, batch_size=64, early stopping
- Save: `models/farl_model.pth`, `results/farl_history.json`, `results/farl_training.png`

### Step 5 — DINOv2 Training (`train_dinov2.py`) 🔲 TODO
- Load via torch.hub: `facebookresearch/dinov2`, `dinov2_vitb14` (feature_dim=768)
- Same classifier head, optimizer, scheduler, and epoch settings as FaRL
- Save: `models/dinov2_model.pth`, `results/dinov2_history.json`, `results/dinov2_training.png`

### Step 6 — Evaluation (`evaluate.py`) 🔲 TODO
- Load both FaRL and DINOv2 models (gracefully skip missing)
- Compute: accuracy, top-2 accuracy, macro/weighted F1, per-season recall
- Generate confusion matrices per model
- Compare against paper baselines:
  - FaRL16: 0.525 accuracy | FaRL64: 0.554 accuracy | ResNeXt50: 0.513 accuracy
- Save: `results/comparison_table.csv`, `results/model_comparison.png`

### Step 7 — Streamlit App (`app.py`) 🔲 TODO
- st.file_uploader + st.camera_input tabs
- OpenCV Haar Cascade face detection
- Load both FaRL and DINOv2 with @st.cache_resource
- Final prediction = argmax of averaged confidences from both models
- Recommendation tabs: Color Palette, Outfits, Accessories, Makeup
- Load outfit photos from `outfits/{season}/` if present

---

## Professor's Required Outline

The final paper must follow this structure:
1. Introduction (context, motivation, gap/problem)
2. Objectives & Research Questions
3. Significance
4. Scope and Delimitation
5. Related Works
6. Methodology
   - Data Acquisition (source, collection steps, tools)
   - Data Preprocessing (cleaning, feature extraction, transformation)
   - Feature Engineering (binning, encoding, selection)
   - Data Split
   - Model Development (classifiers, validation)
   - Model Performance Metrics
   - Model Testing
   - Model Evaluation
7. Results and Discussion
8. Conclusion and Recommendation
9. References (APA format)

---

## Important Academic Notes

- **Labeling methodology:** Season labels sourced from publicly documented celebrity color analysis diagnoses. Not self-assigned. Cite as: "ground truth labels derived from professionally documented personal color diagnoses available in public Korean beauty resources and color analysis communities."
- **Labeling limitation:** Labels rely on existing analyst consensus, not independent expert verification. Acknowledge in paper.
- **Class imbalance:** Filipino celebrities skew heavily toward Autumn and Winter. Address using class_weight='balanced' in SVM/RF and class_weight in Keras.
- **ITA score:** Key feature — Individual Typology Angle. Formula: `ITA = arctan((L* - 50) / b*) × (180/π)`. Higher ITA = lighter/cooler skin. Lower ITA = darker/warmer skin.
- **CIELab color space** is the primary feature space — consistent with Korean skin tone classification literature (Kye & Lee, 2022; Soonchunhyang University, 2023).
- **Autumn is historically the hardest class to predict** in existing studies (ColorInsight, 2023). Track Autumn recall separately.
- **Paper baselines (Deep Armocromia, Stacchio et al. ECCV 2024):** FaRL64=0.554, ResNeXt50=0.513 — these are the benchmarks to compare against.

---

## Tech Stack

```
Python 3.10+
torch / torchvision      # FaRL, DINOv2, deep learning
timm>=0.9.0              # ViT-Base model loading for FaRL
opencv-python            # face detection, image processing
scikit-learn             # preprocessing, metrics
scikit-image             # CIELab color conversion
pandas                   # data handling
numpy                    # array operations
matplotlib               # plotting
seaborn                  # heatmaps, visualizations
streamlit                # web app interface
imagehash                # duplicate detection
pillow                   # image I/O
icrawler                 # Bing Image scraping (data collection only)
```

---

## File Naming Conventions

- Processed face crops: `dataset/[season]/[season]_[celebrity_name]_[number].jpg`
- Data split manifest: `data_split.csv` — columns: filename, season, split
- Feature CSV: `features.csv` — columns: filename, season, label, L_mean, a_mean, b_mean, L_std, a_std, b_std, ITA, H_mean, S_mean, V_mean
- Models: `models/farl_model.pth`, `models/dinov2_model.pth`
- Scaler: `models/scaler.pkl`
- Results: `results/comparison_table.csv`, `results/confusion_matrix_[model].png`

---

## Current Status

- [x] Project proposal written
- [x] Celebrity list verified (60 celebrities across 4 seasons)
- [x] Data collection script written (`collectdata.py`)
- [x] Dataset collected (~952 images)
- [ ] Manual dataset cleaning (team reviews each season folder, deletes wrong-person images)
- [ ] Deep Armocromia dataset downloaded and merged
- [x] Preprocessing script written (`preprocess.py`)
- [ ] Preprocessing run (run after cleaning)
- [ ] EDA notebook written (`eda.ipynb`)
- [x] FaRL training script written (`train_farl.py`)
- [x] DINOv2 training script written (`train_dinov2.py`)
- [x] Evaluation script written (`evaluate.py`)
- [x] Streamlit app written (`app.py`)
- [ ] Models trained on Colab with T4 GPU
- [ ] Final paper written

---

## When Helping With This Project

1. Always follow the pipeline order above
2. The next steps are: manual data cleaning → run preprocess.py → train on Colab
3. Keep code clean, well-commented, and beginner-friendly — the team is not highly technical
4. Use Google Colab-compatible code where possible (the team will likely run on Colab)
5. Always save intermediate outputs (CSV, .npy files, model files) so steps can be run independently
6. When writing evaluation code, always compare both models side by side against Deep Armocromia paper baselines
