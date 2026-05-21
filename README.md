# ToneFit AI

**AI-Powered Personal Color Season Classification (Armocromia)**

A pilot study for the Data Science / Current Trends in IT course at Polytechnic University of the Philippines (PUP Manila). Compares FaRL-64 baseline vs. a novel two-stage hierarchical FaRL-64 head for 4-season and 12-subtype Armocromia classification, with a full Next.js web app for real-world deployment.

---

## What This Project Does

ToneFit predicts a user's personal color season from a facial photo — not just the broad 4-season category (Spring / Summer / Autumn / Winter), but one of **12 sub-types** (e.g., Deep Autumn, Bright Winter) — and generates a complete style guide covering color palettes, clothing, accessories, and makeup.

The main academic contribution is the **two-stage hierarchical head**: Stage 2 (sub-type) is explicitly conditioned on Stage 1 (season) logits, which is novel for this domain.

---

## Personal Color Season System

| Season | Sub-Type | Label |
|--------|----------|-------|
| Spring | Warm | `spring_warm` |
| Spring | Light | `spring_light` |
| Spring | Bright | `spring_bright` |
| Summer | Cool | `summer_cool` |
| Summer | Light | `summer_light` |
| Summer | Soft | `summer_soft` |
| Autumn | Warm | `autumn_warm` |
| Autumn | Soft | `autumn_soft` |
| Autumn | Deep | `autumn_deep` |
| Winter | Cool | `winter_cool` |
| Winter | Deep | `winter_deep` |
| Winter | Bright | `winter_bright` |

4-season rule: Spring + Autumn = Warm. Summer + Winter = Cool.

---

## Models

### Model A — FaRL-64 Flat Two-Head Baseline (`train_farl.py`)
- Frozen FaRL-64 backbone (CLIP ViT-B/16, 512-d output)
- Shared FC(512→256, ReLU, Dropout 0.5) → season head (256→4) + subtype head (256→12)
- Joint training: combined CE loss

### Model B — FaRL-64 12-Class Single Head (`train_farl_12class.py`)
- Same frozen backbone, single 12-class head only
- Season derived from predicted sub-type
- Adds Top-3 accuracy metric (paper standard)

### Hierarchical Model — Novel Contribution (`train.py`)
- Same frozen backbone
- **Stage 1:** FC(768→384) → FC(384→4) — season classification
- **Stage 2:** FC(384+4→12) — sub-type, conditioned on Stage 1 softmax
- Stage 2 input = concat(shared features [384-d], season softmax [4-d]) = 388-d
- This lets the sub-type head explicitly condition on the predicted season

### Model C — FaRL-64 + LLRD + Unfreeze (`train_farl_improved.py`)
- Created only if Model B achieves ≥ 0.30 sub-type accuracy
- Unfreezes last 4 transformer blocks with layer-wise learning rate decay (LLRD)

---

## Results

| Model | Season Acc | F1 | SubType Acc | Top-3 | Status |
|-------|-----------|-----|-------------|-------|--------|
| FaRL-16 (paper) | 0.525 | 0.516 | 0.318 | 0.663 | Baseline |
| FaRL-64 (paper) | 0.554 | 0.548 | 0.313 | 0.651 | Baseline |
| ResNeXt50 (paper) | 0.513 | 0.502 | 0.281 | 0.614 | Baseline |
| Model A — FaRL-64 Flat | TBD | TBD | TBD | TBD | Needs rerun |
| Model B — FaRL-64 12-class | TBD | TBD | TBD | TBD | Pending |
| **Hierarchical FaRL-64** | **0.5636** | TBD | **0.3213** | TBD | Done ✓ |
| Model C — FaRL-64 + LLRD | TBD | TBD | TBD | TBD | Pending |

---

## Project Structure

```
ToneFit/
├── train.py                  ← Hierarchical model training (DONE)
├── train_farl.py             ← Model A: FaRL-64 flat two-head baseline
├── train_farl_12class.py     ← Model B: FaRL-64 12-class single head
├── train_farl_improved.py    ← Model C: FaRL-64 + LLRD (create after Model B)
├── hierarchical_head.py      ← Hierarchical head architecture
├── evaluate.py               ← Compare all models vs paper baselines
├── server.py                 ← FastAPI ML inference server (for Next.js frontend)
├── app.py                    ← Streamlit demo app
├── preprocess.py             ← EDA feature extraction
├── annotations.csv           ← Original dataset annotations (Stacchio et al.)
├── requirements.txt
├── configs/
│   └── hierarchical.yaml     ← Training config for hierarchical model
├── models/
│   └── farl_weights.pth      ← FaRL-64 pretrained weights (652 MB, gitignored)
├── results/                  ← Training outputs: .pth checkpoints, history JSON, reports (gitignored)
├── RGB-M/                    ← Deep Armocromia dataset — READ ONLY, do not modify
│   ├── train/
│   └── test/
├── outfits/                  ← Curated outfit reference photos per season
├── ToneFit_Colab.ipynb       ← Main training pipeline (run on Colab T4)
├── eda.ipynb                 ← Exploratory data analysis
├── CLAUDE.md                 ← Full project brief (source of truth)
└── web/                      ← Next.js frontend (deployed on Vercel)
    ├── src/app/
    │   ├── page.tsx           ← Homepage
    │   ├── onboarding/        ← 4-step onboarding (name, style, age, photo)
    │   ├── processing/        ← Analysis loading screen + real API call
    │   ├── results/           ← Season result + color palette + style guide
    │   └── api/analyze/       ← Next.js proxy → Python ML server
    ├── src/components/Nav.tsx
    ├── src/lib/season-data.ts ← All 12-season style data
    └── .env.local             ← PYTHON_API_URL (points to server.py)
```

---

## Dataset

**Deep Armocromia** (Stacchio et al., ECCV 2024) — the first large-scale dataset of face images labeled by certified Armocromia professionals.

- ~4,920 expert-labeled face images across 12 sub-types
- Pre-defined train/test split, nested as `RGB-M/train/<season>/<subtype>/`
- Request access at: [github.com/lorenzo-stacchio/Deep-Armocromia](https://github.com/lorenzo-stacchio/Deep-Armocromia)
- Place the `RGB-M/` folder in the project root — **never modify its contents**

---

## Setup

### Python (ML training + inference server)

```bash
git clone https://github.com/ajipal/ToneFit.git
cd ToneFit
pip install -r requirements.txt
```

**FaRL pretrained weights** (~652 MB, place in `models/`):
```bash
wget https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth -O models/farl_weights.pth
```

### Next.js frontend

```bash
cd web
npm install
```

Create `web/.env.local`:
```
PYTHON_API_URL=http://localhost:8000
```

---

## Running Locally

**Terminal 1 — ML inference server:**
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Next.js frontend:**
```bash
cd web
npm run dev
```

Open `http://localhost:3000`.

---

## Training Pipeline (Google Colab)

Open `ToneFit_Colab.ipynb` on Colab (T4 GPU recommended) and run cells in order:

| Step | Description |
|------|-------------|
| 0 | Check GPU |
| 1 | Mount Google Drive |
| 2 | Clone repo + install dependencies |
| 3 | Link RGB-M dataset from Drive |
| 4–5 | Preprocessing + EDA |
| 6 | Train Model A + Model B |
| 7 | Train Hierarchical model |
| 8 | Compare all models (`evaluate.py`) |
| 9 | Save results + download zip |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| ML Framework | PyTorch, timm |
| Backbone | FaRL-64 (CLIP ViT-B/16, face-pretrained) |
| Image Processing | Pillow, OpenCV, torchvision |
| Data / Metrics | NumPy, scikit-learn, pandas |
| Visualization | Matplotlib, Seaborn |
| ML API Server | FastAPI, Uvicorn |
| Frontend | Next.js 15, React 19, Tailwind CSS, TypeScript |
| Deployment | Vercel (frontend) + Railway/Render (Python server) |
| Training | Google Colab (T4 GPU) |

---

## Architecture Diagram

```
User photo (browser)
      │
      ▼
Next.js /api/analyze          ← Vercel
      │  POST /predict
      ▼
FastAPI server.py             ← Railway / Render
  FaRLBackbone (frozen)
      │  768-d features
      ▼
  HierarchicalArmocromiaHead
  ├── Stage 1: FC(384→4)  → season logits
  └── Stage 2: FC(388→12) → subtype logits (conditioned on Stage 1 softmax)
      │
      ▼
{ season, subtype, confidence, top3 }
      │
      ▼
Results page (color palette, clothes, makeup)
```

---

## Paper Baselines

Stacchio, L., Paolanti, M., Spigarelli, F., & Frontoni, E. (2025). Deep Armocromia: A novel dataset for face seasonal color analysis and classification. *ECCV 2024 Workshops* (pp. 352–367). Springer. https://doi.org/10.1007/978-3-031-91569-7_22

---

## References

- Wang, W., et al. (2022). FaRL: General facial representation learning in a frozen state. *CVPR 2022*.
- Stacchio et al. (2024). Deep Armocromia dataset. *ECCV 2024 Workshops*.

---

## Team

Polytechnic University of the Philippines — Data Science / Current Trends in IT  
Group study, 2025–2026
