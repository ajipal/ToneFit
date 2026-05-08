# Deep Armocromia — Hierarchical FaRL-64 Classifier

## Project goal
Extend the Deep Armocromia paper (ECCV 2024) by replacing the flat 2-FC classifier
head with a two-stage hierarchical head on top of a frozen FaRL-64 backbone.
The goal is to beat FaRL-64's Season accuracy (0.554) AND FaRL-16's Sub-Type
accuracy (0.318) in a single unified model.

## Repository layout
```
deep_armocromia/
├── CLAUDE.md                  ← this file
├── data/
│   ├── raw/                   ← original Deep Armocromia images + masks
│   └── processed/             ← 224×224 RGB-masked crops (ready for training)
├── models/
│   ├── farl_backbone.py       ← frozen FaRL-64 feature extractor wrapper
│   ├── hierarchical_head.py   ← NEW: two-stage hierarchical classifier
│   └── baseline.py            ← original flat 2-FC head (for comparison)
├── train.py                   ← training loop (shared for both models)
├── evaluate.py                ← evaluation + confusion matrix generation
├── configs/
│   ├── baseline.yaml          ← original paper hyperparams
│   └── hierarchical.yaml      ← new model hyperparams
├── results/                   ← saved checkpoints + metrics JSON
└── requirements.txt
```

## Architecture — hierarchical head (the novel contribution)
```
FaRL-64 backbone (frozen, ViT-B/16)
    └── 768-d feature vector
         └── Shared FC: 768 → 384, ReLU, Dropout(0.5)
              ├── Stage 1 head: FC 384 → 4  (Season: Spring/Summer/Autumn/Winter)
              └── Stage 2 head: FC (384 + 4) → 12  (Sub-Type, conditioned on season logits)
```
Stage 2 receives the concatenation of the shared 384-d features AND the 4-d
softmax output from Stage 1. This is the key architectural difference from the
original paper's flat head.

## Targets to beat (from original paper Tables 2 & 3)
| Task | Original best | Our target |
|---|---|---|
| Season (4-class) | FaRL-64: 0.554 accuracy | > 0.554 |
| Sub-Type (12-class) | FaRL-16: 0.318 accuracy | > 0.318 |

## Original paper setup (replicate exactly for fair comparison)
- Backbone: FaRL-Base-Patch16, pretrained on LAION-Face, **weights frozen**
- Input: RGB-masked face crops, 3×224×224 (hair + skin + eyes only, from Facer masks)
- Train/test split: 80/20, ~4000 train / ~920 test, no identity overlap
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-5)
- Scheduler: CosineAnnealingWarmRestarts (T_0=10, eta_min=1e-5)
- Epochs: 50, batch size: 64
- Data augmentation: random crop, horizontal flip (p=0.5), color jitter
  (brightness=0.4, contrast=0.2, saturation=0.2), random sharpness (factor=2, p=0.2)
- Metrics: accuracy, precision, recall, F1, top-2 (season) / top-3 (sub-type)

## Training strategy for hierarchical model
- Train both stages jointly with a combined loss:
  `total_loss = loss_season + lambda * loss_subtype`
  Start with lambda=1.0, tune if needed.
- Use the same AdamW + cosine schedule as above.
- Save best checkpoint by season accuracy (primary metric).

## Environment
- Python 3.10+
- PyTorch 2.3.0 + CUDA 12.1
- timm 0.8.3 (for FaRL backbone loading via pytorch-image-models)
- torchmetrics 1.4.0
- GPU: NVIDIA RTX 4070 (8 GB VRAM)

## Key conventions
- Always load FaRL-64 with `model.eval()` and `torch.no_grad()` during feature extraction.
- FaRL checkpoint key: `farl-base-patch16-64ep` from the official Microsoft repo.
- All metrics computed with torchmetrics, macro-averaged.
- Confusion matrices saved as PNG to results/.
- Do not modify the data preprocessing pipeline — use the existing Facer masks.

## Paper motivation (cite in writing)
The original paper conclusion states: *"we will explore those paradigms involving
hierarchical and ordinal learning, given that such nature matches the inner nature
of the Armocromia classes."* This work directly implements that suggestion.