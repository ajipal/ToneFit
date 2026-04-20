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
- CIELab + HSV color feature extraction from face region
- SVM and Random Forest classifiers (Model A)
- MobileNetV2 deep learning classifier (Model B)
- Season prediction with confidence score
- Color palette display per season
- Side-by-side outfit sample photo display (pre-curated, not virtual try-on)
- Simple prototype interface (Jupyter notebook or basic web UI)

### ❌ We are NOT building:
- Virtual try-on or body warping
- 12-tone sub-season classification
- Real-time video analysis
- A deployed production app

---

## Dataset

### Self-Collected Celebrity Dataset
Face images of celebrities with publicly documented personal color seasons.
Labels come from professional color analyst diagnoses published online — NOT guessed by the team.

**Structure:**
```
dataset/
  spring/    → ~75 face images (224x224 px, cropped)
  summer/    → ~75 face images
  autumn/    → ~75 face images
  winter/    → ~75 face images
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

### CapstoneA Supplement
Download from: https://universe.roboflow.com/capstonea-9fv4r/personal-color
230 labeled images (Spring/Summer/Autumn/Winter), CC BY 4.0
Add to the corresponding season folders after downloading.

---

## Full Pipeline (Steps in Order)

### Step 1 — Data Collection (`collect_data.py`) ✅ DONE
- Uses icrawler to scrape Google Images for each celebrity
- Applies OpenCV Haar Cascade face detector
- Crops and saves 224x224 face images per season folder

### Step 2 — Preprocessing (`preprocess.py`) 🔲 TODO
- Remove duplicates (perceptual hashing)
- Manual quality check (delete bad crops)
- Convert images to CIELab and HSV color spaces
- Extract features per image:
  - L* mean, a* mean, b* mean (CIELab)
  - L* std, a* std, b* std
  - ITA score: arctan((L* - 50) / b*) × (180/π)
  - Hue mean, Saturation mean, Value mean (HSV)
- Save features to CSV: `features.csv` with columns [filename, season, L_mean, a_mean, b_mean, L_std, a_std, b_std, ITA, H_mean, S_mean, V_mean]
- Encode labels: spring=0, summer=1, autumn=2, winter=3
- Normalize features using MinMaxScaler
- Split: 80% train / 20% test (stratified, random_state=42)
- Save: `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`

### Step 3 — EDA (`eda.ipynb`) 🔲 TODO
- Class distribution bar chart
- Sample face images per season (grid)
- Feature distributions (boxplots of L*, a*, b* per season)
- Correlation heatmap of features
- PCA visualization (2D scatter, colored by season)

### Step 4 — Traditional ML Training (`train_traditional.py`) 🔲 TODO
- Load X_train, y_train
- Train SVM (RBF kernel, GridSearchCV for C and gamma)
- Train Random Forest (100 estimators, evaluate feature importance)
- 5-fold stratified cross-validation on training set
- Save models: `models/svm_model.pkl`, `models/rf_model.pkl`
- Save scaler: `models/scaler.pkl`

### Step 5 — Deep Learning Training (`train_deeplearning.py`) 🔲 TODO
- Load images from dataset/ folders using ImageDataGenerator
- MobileNetV2 base (pretrained ImageNet, frozen initially)
- Add: GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(4, Softmax)
- Compile: Adam(lr=0.0001), categorical_crossentropy
- Train: 30 epochs, batch_size=32, early stopping (patience=5)
- Unfreeze top layers for fine-tuning (second pass)
- Save model: `models/mobilenetv2_model.h5`
- Plot training/validation loss and accuracy curves

### Step 6 — Evaluation (`evaluate.py`) 🔲 TODO
- Load all 3 models (SVM, RF, MobileNetV2)
- Run predictions on X_test / test image folder
- Generate for each model:
  - Classification report (accuracy, precision, recall, F1)
  - Confusion matrix (heatmap)
  - Top-2 accuracy
- Save results table to `results/comparison_table.csv`
- Print final comparison table

### Step 7 — Prediction Demo (`predict.py`) 🔲 TODO
- Accept image path as argument
- Detect and crop face using OpenCV
- Extract features
- Run all 3 models, show predictions with confidence
- Display season name + color palette swatches
- Show 2-3 outfit sample images from `outfits/[season]/`

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

---

## Tech Stack

```
Python 3.10+
opencv-python        # face detection, image processing
scikit-learn         # SVM, Random Forest, metrics, preprocessing
tensorflow           # MobileNetV2, deep learning
pandas               # data handling
numpy                # array operations
matplotlib           # plotting
seaborn              # heatmaps, visualizations
scikit-image         # CIELab color conversion
icrawler             # Google Image scraping (data collection only)
imagehash            # duplicate detection
```

---

## File Naming Conventions

- Raw downloaded images: `dataset/raw/[season]/[number].jpg`
- Processed face crops: `dataset/[season]/[season]_[celebrity_name]_[number].jpg`
- Feature CSV: `features.csv` — columns: filename, season, label, L_mean, a_mean, b_mean, L_std, a_std, b_std, ITA, H_mean, S_mean, V_mean
- Models: `models/svm_model.pkl`, `models/rf_model.pkl`, `models/mobilenetv2_model.h5`
- Scaler: `models/scaler.pkl`
- Results: `results/comparison_table.csv`, `results/confusion_matrix_[model].png`

---

## Current Status

- [x] Project proposal written
- [x] Celebrity list verified (60 celebrities across 4 seasons)
- [x] Data collection script written (`collect_data.py`)
- [ ] Dataset collected (run `collect_data.py`)
- [ ] CapstoneA dataset downloaded and merged
- [ ] Preprocessing script written
- [ ] EDA notebook written
- [ ] Traditional ML training script written
- [ ] Deep learning training script written
- [ ] Evaluation script written
- [ ] Prediction demo script written
- [ ] Final paper written

---

## When Helping With This Project

1. Always follow the pipeline order above
2. The next step to work on is **Step 2 — Preprocessing** (`preprocess.py`)
3. Keep code clean, well-commented, and beginner-friendly — the team is not highly technical
4. Use Google Colab-compatible code where possible (the team will likely run on Colab)
5. Always save intermediate outputs (CSV, .npy files, model files) so steps can be run independently
6. When writing evaluation code, always compare all 3 models side by side