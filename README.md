# ToneFit ML 🎨

**Machine Learning-Based Personal Color Season Prediction and Clothing Color Recommendation**

A pilot study applying machine learning to classify personal color seasons (Spring, Summer, Autumn, Winter) from facial images — with a focus on Filipino skin tone representation. Grounded in Korean seasonal color analysis methodology.

---

## 📌 Project Overview

ToneFit ML predicts a user's personal color season from a facial image and recommends suitable clothing color palettes. The system compares two model approaches:

- **Model A** — Traditional ML (SVM, Random Forest) using CIELab/HSV color features
- **Model B** — Deep Learning (MobileNetV2) using raw face images

**Output:** Predicted season + clothing color palette + curated outfit sample photos

---

## 🌸 Personal Color Seasons

| Season | Undertone | Characteristics |
|--------|-----------|-----------------|
| Spring | Warm | Light, clear, peachy |
| Summer | Cool | Light, muted, ashy |
| Autumn | Warm | Deep, muted, earthy |
| Winter | Cool | Deep, clear, high contrast |

---

## 📁 Project Structure

```
ToneFit/
├── collect_data.py        # Step 1: Download and crop celebrity face images
├── preprocess.py          # Step 2: Extract CIELab/HSV features + clean data
├── train_traditional.py   # Step 3: Train SVM and Random Forest models
├── train_deeplearning.py  # Step 4: Train MobileNetV2 model
├── evaluate.py            # Step 5: Compare model performance
├── predict.py             # Step 6: Run prediction on new image
├── dataset/               # Collected and labeled face images
│   ├── spring/
│   ├── summer/
│   ├── autumn/
│   └── winter/
├── models/                # Saved trained models
├── results/               # Evaluation outputs, confusion matrices
├── outfits/               # Curated outfit reference photos per season
│   ├── spring/
│   ├── summer/
│   ├── autumn/
│   └── winter/
├── CLAUDE.md              # Full project brief for AI assistance
└── README.md
```

---

## 🗂️ Dataset

Self-collected dataset of verified celebrity face images labeled by personal color season.

**Sources:**
- Filipino celebrities (primary focus) — labels from documented professional diagnoses
- Korean celebrities — from K-beauty color analysis communities
- Western celebrities — from international color analysis resources
- CapstoneA Personal Color Dataset (Roboflow Universe, 230 images, CC BY 4.0)

**Target size:** ~530 labeled face images (75 per season self-collected + 230 CapstoneA)

| Season | Filipino | Korean | Western | Total Celebrities |
|--------|----------|--------|---------|-------------------|
| Spring | 4 | 6 | 5 | 15 |
| Summer | 5 | 6 | 4 | 15 |
| Autumn | 10 | 3 | 2 | 15 |
| Winter | 7 | 5 | 3 | 15 |

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/ajipal/ToneFit.git
cd ToneFit

# Install dependencies
pip install icrawler opencv-python scikit-learn tensorflow pandas numpy matplotlib seaborn
```

---

## 🚀 Usage

### Step 1 — Collect dataset
```bash
python collect_data.py
```

To split among group members, edit `RUN_SEASONS` at the bottom of the file:
```python
RUN_SEASONS = ["spring"]   # or "summer", "autumn", "winter"
```

### Step 2 — Preprocess & extract features
```bash
python preprocess.py
```

### Step 3 — Train models
```bash
python train_traditional.py   # SVM + Random Forest
python train_deeplearning.py  # MobileNetV2
```

### Step 4 — Evaluate
```bash
python evaluate.py
```

### Step 5 — Predict on new image
```bash
python predict.py --image path/to/photo.jpg
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras (MobileNetV2) |
| Traditional ML | Scikit-learn (SVM, Random Forest) |
| Image Processing | OpenCV, MediaPipe |
| Color Analysis | scikit-image (CIELab), colormath |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## 📊 Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score (macro + weighted)
- Confusion Matrix
- Top-2 Accuracy (for borderline season cases)
- Training vs. Validation loss curves (deep learning)

---

## 📚 References

- Groh et al. (2021). Fitzpatrick17k Dataset. CVPR Workshop.
- Kye & Lee (2022). Skin color classification of Koreans using clustering. PMC9907718.
- PSY222 et al. (2023). ColorInsight. github.com/PSY222/Colorinsight
- KIISE (2024). Learning-based Model Comparison for Personal Color Diagnosis.
- Capstonea (2022). Personal Color Dataset. Roboflow Universe.

---

## 👥 Team

Polytechnic University of the Philippines — Data Science Course
Group of 4 students, 2025