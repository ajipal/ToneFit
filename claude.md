# CLAUDE.md — ToneFit ML Project Brief

Read this fully before doing anything. This is the single source of truth.

---

## What This Project Is

ToneFit ML is a pilot study for a Data Science / Current Trends in IT class at PUP Manila.

Goals:
1. Compare FaRL-64 baseline vs FaRL-64 Hierarchical (novel contribution) for Armocromia classification
2. Evaluate on both 4-season AND 12-subtype classification tasks
3. Build a working Streamlit app where a user takes a photo and gets season prediction + clothing/accessory/makeup recommendations

---

## Personal Color Season System (12 Sub-types)

| Season | Sub-type | Full Label |
|--------|----------|------------|
| Spring | Warm | spring_warm |
| Spring | Light | spring_light |
| Spring | Bright | spring_bright |
| Summer | Cool | summer_cool |
| Summer | Light | summer_light |
| Summer | Soft | summer_soft |
| Autumn | Warm | autumn_warm |
| Autumn | Soft | autumn_soft |
| Autumn | Deep | autumn_deep |
| Winter | Cool | winter_cool |
| Winter | Deep | winter_deep |
| Winter | Bright | winter_bright |

4-Season rule: Spring + Autumn = Warm. Summer + Winter = Cool.

---

## Final Confirmed Decisions

- Dataset: Deep Armocromia (Stacchio et al., ECCV 2024) — already downloaded
- Model A: FaRL-64 Flat Two-Head Baseline (4-season + 12-subtype, jointly trained, frozen backbone — matches paper)
- Model B: FaRL-64 12-class Sub-Type only (single head, derives season from prediction, Top-3 metric)
- Model C: FaRL-64 + LLRD + unfreeze last 4 layers — only if Model B achieves ~0.30+ sub-type accuracy
- Hierarchical Model: FaRL-64 two-stage head (Stage 2 conditioned on Stage 1 logits — novel contribution)
- Comparison: All models vs each other and vs Stacchio et al. 2024 paper baselines
- Deployment: Streamlit web app on Streamlit Cloud
- Active branch: twelve-season

---

## What is NOT in scope

- SVM, Random Forest traditional ML
- Virtual try-on or generative image editing
- Custom CNN from scratch
- Celebrity scraped dataset
- DINOv2 (was a fallback model; not actively developed)

---

## Dataset Structure (READ ONLY — DO NOT MODIFY)

Location: RGB-M/ folder in workspace root.

```
RGB-M/
  train/
    autumn/
      deep/     <- sub-type label: autumn_deep
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
    (same structure as train)
```

CRITICAL RULES:
1. RGB-M is READ ONLY — never move, copy, delete, or modify anything inside
2. Train/test split already done — use as-is
3. DO NOT create any script that modifies dataset folders

---

## Model Architecture Overview

### Model A — FaRL-64 Flat Two-Head Baseline (train_farl.py) DONE
- Backbone: CLIP ViT-B/16 + FaRL pretrained weights (frozen, 512-d output)
- Head: backbone -> shared FC(512->256, ReLU, Dropout 0.5) -> season_head(256->4) + subtype_head(256->12)
- Joint training: combined CE loss (season + subtype simultaneously)
- Feature caching: backbone runs once, features saved to results/features_train_farl.pt
- Saves: models/farl_model.pth, results/farl_history.json, results/farl_test_report.txt

### Model B — FaRL-64 12-Class Sub-Type (train_farl_12class.py) CREATED
- Same frozen FaRL-64 backbone
- Head: FC(512->256, ReLU, Dropout 0.5) -> FC(256->12) — single 12-class head only
- Season derived afterward: "autumn_deep".split("_")[0] -> "autumn"
- Adds Top-3 accuracy metric (paper standard for 12-class)
- Feature caching: results/features_train_12class.pt
- Saves: models/farl_12class_model.pth, results/farl_12class_history.json, results/farl_12class_report.txt

### Hierarchical Model — FaRL-64 Two-Stage Head (train.py) DONE
- Same frozen FaRL-64 backbone
- Head: shared FC(512->256) -> Stage 1: FC(256->4) season -> Stage 2: FC(256+4->12) subtype conditioned on Stage 1 softmax
- Novel contribution: Stage 2 input = concat(shared features [256-d], season softmax [4-d]) = 260-d
- Config-driven via configs/hierarchical.yaml
- Supports resume from checkpoint (saves per-epoch checkpoint)
- Syncs to Google Drive every 5 epochs
- Saves: results/best_hierarchical.pth, results/hierarchical_history.json, results/hierarchical_test_report.txt

### Model C — FaRL-64 + LLRD + Unfreeze (train_farl_improved.py) NOT YET CREATED
Only create IF Model B achieves ~0.30+ sub-type accuracy.

Improvements over baseline:
1. Unfreeze last 4 transformer blocks
2. Layer-wise Learning Rate Decay: lr = base_lr * (0.75 ^ (num_layers - layer_idx))
3. Classifier head gets full base_lr

```python
def get_llrd_params(backbone, classifier, base_lr=1e-3, decay=0.75):
    params = []
    blocks = list(backbone.transformer.resblocks)
    num_blocks = len(blocks)
    for i, block in enumerate(blocks):
        if i < num_blocks - 4:
            for param in block.parameters():
                param.requires_grad = False
        else:
            lr = base_lr * (decay ** (num_blocks - i))
            params.append({"params": block.parameters(), "lr": lr})
    params.append({"params": classifier.parameters(), "lr": base_lr})
    return params
```

- Save: models/farl_improved_model.pth, results/farl_improved_history.json

---

## Training Settings (all models)

- AdamW(lr=1e-3, weight_decay=1e-5)
- CosineAnnealingWarmRestarts(T_0=10, eta_min=1e-5)
- 50 epochs, batch_size=64
- Balanced class weights (sklearn compute_class_weight)
- argparse uses parse_known_args() to avoid Jupyter/Colab crash

---

## Per-Epoch Output Format (all models — consistent)

```
------------------------------------------------------------------------
 Epoch      Loss   Tr Szn   Tr Sub   Val Szn   Val Sub          LR    Time
------------------------------------------------------------------------
     1    1.8432   0.3124   0.1021    0.3210    0.0987    1.00e-03   12.3s <-- best
[INFO] Estimated total time: XX.X min
...
------------------------------------------------------------------------
[INFO] Training complete.
[INFO] Best val season accuracy: 0.XXXX at epoch XX
[INFO] Sub-type accuracy at best epoch: 0.XXXX
[INFO] Best model saved to: models/...
```

Final section (all models): Season (4-class) classification report -> confusion matrix -> autumn recall -> Sub-Type (12-class) report -> confusion matrix.

---

## Evaluation (evaluate.py)

Reads all result JSON/TXT files and produces a comparison table.
Loaders: load_model_a(), load_model_b(), load_model_c(), load_hierarchical(), load_dinov2() (optional).
Outputs: results/evaluation_summary.csv, results/evaluation_summary.txt

Paper baselines included for reference:

| Model | Season Acc | F1 | SubType Acc | Top-3 |
|-------|-----------|-----|-------------|-------|
| FaRL-16 (paper) | 0.525 | 0.516 | 0.318 | 0.663 |
| FaRL-64 (paper) | 0.554 | 0.548 | 0.313 | 0.651 |
| ResNeXt50 (paper) | 0.513 | 0.502 | 0.281 | 0.614 |

---

## Colab Pipeline (ToneFit_Colab.ipynb) — branch: twelve-season

- Step 0: Check GPU
- Step 1: Mount Google Drive
- Step 2: Clone repo (twelve-season branch) + install dependencies
- Step 3: Link RGB-M dataset from Drive
- Step 4: Preprocessing (EDA features via preprocess.py)
- Step 5: EDA plots
- Step 6: Train Model A + Model B (sequential, shared setup)
  - 6.1 git pull + install CLIP + clear module cache
  - 6.2 Get FaRL weights (Drive first, then download)
  - 6.3 Train Model A (train_farl.main())
  - 6.4 Show Model A curves
  - 6.5 Train Model B (train_farl_12class.main())
  - 6.6 Show Model B curves
  - 6.7 Save all to Drive
- Step 7: Train Hierarchical (train.py via subprocess, supports resume)
- Step 8: Compare all models (evaluate.main())
- Step 9: Save all results + download zip

---

## App (app.py)

### Onboarding Questions (first-time users only)
Saved to st.session_state.profile — no database needed.

Q0: First name (text input)
Q1: Gender — Male / Female / Non-binary / Prefer not to say
Q2: Style preference (multi-select) — Casual / Smart Casual / Formal / Streetwear / Minimalist / Bohemian / Sporty
Q3: Age range — Under 18 / 18-24 / 25-34 / 35-44 / 45+
Q4: Budget — Budget-friendly / Mid-range / Premium / Luxury
Q5: Makeup preference — Natural / Bold / No makeup

### Season Display Names
```python
SEASON_DISPLAY = {
    "autumn": "Warm Autumn", "spring": "Warm Spring",
    "summer": "Cool Summer", "winter": "Cool Winter"
}
SUBTYPE_DISPLAY = {
    "autumn_deep": "Deep Autumn", "autumn_soft": "Soft Autumn", "autumn_warm": "Warm Autumn",
    "spring_bright": "Bright Spring", "spring_light": "Light Spring", "spring_warm": "Warm Spring",
    "summer_cool": "Cool Summer", "summer_light": "Light Summer", "summer_soft": "Soft Summer",
    "winter_bright": "Bright Winter", "winter_cool": "Cool Winter", "winter_deep": "Deep Winter",
}
```

---

## File Structure (current state)

```
ToneFit/
|-- app.py                      <- Streamlit app
|-- preprocess.py               <- EDA feature extraction
|-- train_farl.py               <- Model A: FaRL-64 flat two-head DONE
|-- train_farl_12class.py       <- Model B: FaRL-64 12-class CREATED
|-- train_farl_improved.py      <- Model C: FaRL-64 LLRD+unfreeze (create after Model B)
|-- train.py                    <- Hierarchical FaRL-64 DONE
|-- hierarchical_head.py        <- Hierarchical head architecture (used by train.py)
|-- evaluate.py                 <- Compare all models
|-- eda.ipynb                   <- EDA notebook
|-- ToneFit_Colab.ipynb         <- Main Colab training pipeline (twelve-season branch)
|-- ToneFit_Kaggle.ipynb        <- Kaggle alternative
|-- requirements.txt
|-- README.md
|-- CLAUDE.md
|-- annotations.csv             <- Original dataset annotations (Stacchio et al.)
|-- configs/
|   `-- hierarchical.yaml       <- Hierarchical model config
|-- models/
|   `-- farl_weights.pth        <- FaRL-64 pretrained (652MB, gitignored)
|-- results/                    <- Training outputs (gitignored)
`-- outfits/
    |-- autumn/
    |-- spring/
    |-- summer/
    `-- winter/

RGB-M/                          <- READ ONLY
  train/ test/
```

---

## Training Results

| Model | Season Acc | F1 | SubType Acc | Status |
|-------|-----------|-----|-------------|--------|
| FaRL-64 Baseline (Model A) | 0.5537* | 0.5442* | TBD | Script updated — needs Colab rerun |
| FaRL-64 12-class (Model B) | TBD | TBD | TBD | Script ready — run Step 6 in Colab |
| Hierarchical FaRL-64 | 0.5636 | TBD | 0.3213 | Done |
| FaRL-64 Improved (Model C) | TBD | TBD | TBD | Create after Model B reaches 0.30+ |

*Previous run with old single-head script. New two-head script needs a fresh run.

---

## Academic Notes

- Gap: No study has applied LLRD + layer unfreezing to FaRL for Armocromia classification
- Novelty: Two-stage hierarchical head where sub-type prediction is conditioned on season prediction
- Autumn is historically the hardest class — track recall separately in every report
- Sub-type accuracy target: ~0.30+ (matching paper's FaRL-16 baseline of 0.318)
- Season accuracy target: >0.554 (beating paper's FaRL-64 best)
- Do NOT mention: AWS AI Stylist, Filipino demographic focus

---

## What Claude Code Should Do Next

1. Run Step 6 in Colab to get fresh Model A and Model B results
2. IF Model B achieves ~0.30+ sub-type accuracy: CREATE train_farl_improved.py (Model C)
3. Update app.py to load the new two-head Model A checkpoint format (models/farl_model.pth now saves shared + season_head + subtype_head state dicts)
4. DO NOT modify train_farl.py, train.py, evaluate.py, or RGB-M folder
