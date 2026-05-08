# Deep Armocromia — Hierarchical FaRL-64 Classifier

## Project goal
Extend the Deep Armocromia paper (ECCV 2024) by replacing the flat 2-FC classifier
head with a two-stage hierarchical head on top of a frozen FaRL-64 backbone.
The goal is to beat FaRL-64's Season accuracy (0.554) AND FaRL-16's Sub-Type
accuracy (0.318) in a single unified model.

## Repository layout
```
ToneFit/
├── CLAUDE.md                      ← this file
├── hierarchical_head.py           ← FaRLBackbone + HierarchicalArmocromiaHead +
│                                     HierarchicalLoss + DeepArmocromiaHierarchical
├── train.py                       ← hierarchical model training loop
│                                     (--resume, --drive_dir, early stopping)
├── train_farl.py                  ← FaRL-64 baseline training (season, 4-class)
├── evaluate_baseline.py           ← FaRL-64 baseline evaluation (season metrics)
├── evaluate_hierarchical.py       ← hierarchical model evaluation
│                                     (season + sub-type metrics, confusion matrices)
├── models/
│   └── baseline.py                ← flat 2-FC head class (for ablation)
├── configs/
│   └── hierarchical.yaml          ← hyperparameters for hierarchical training
├── RGB-M/
│   ├── train/                     ← 4,008 images  (season/subtype/ hierarchy)
│   └── test/                      ← 912 images
├── results/                       ← checkpoints, metrics JSON, confusion matrix PNGs
├── ToneFit_Colab.ipynb            ← end-to-end Colab pipeline
└── requirements.txt
```

## Models

### Model A — FaRL-64 Baseline (flat head, replication)
```
FaRL-64 backbone (frozen, ViT-B/16)
    └── 768-d feature vector
         └── FC 768 → 384, ReLU, Dropout(0.5) → FC 384 → 4  (Season)
```
Trained by `train_farl.py`. Saves to `models/farl_model.pth`.
Evaluated by `evaluate_baseline.py`.

### Model B — Hierarchical FaRL-64 (novel contribution)
```
FaRL-64 backbone (frozen, ViT-B/16)
    └── 768-d feature vector
         └── Shared FC: 768 → 384, ReLU, Dropout(0.5)
              ├── Stage 1: FC 384 → 4   (Season: Spring/Summer/Autumn/Winter)
              └── Stage 2: FC (384+4) → 12  (Sub-Type, conditioned on Season logits)
```
Trained by `train.py`. Saves best checkpoint to `results/best_hierarchical.pth`.
Evaluated by `evaluate_hierarchical.py`.

Stage 2 receives `concat(shared_features[384], stage1_softmax[4]) = 388-d`.
This is the key architectural difference from the original paper's flat head.

## Targets to beat (from original paper Tables 2 & 3)
| Task | Original best | Our target |
|---|---|---|
| Season (4-class) | FaRL-64: 0.554 accuracy | > 0.554 |
| Sub-Type (12-class) | FaRL-16: 0.318 accuracy | > 0.318 |

## Dataset structure
```
RGB-M/train/season/subtype/*.png    e.g. RGB-M/train/autumn/deep/10063.png
RGB-M/test/season/subtype/*.png
```
Seasons (4): autumn, spring, summer, winter
Sub-types (12): deep/soft/warm (autumn), bright/light/warm (spring),
                cool/light/soft (summer), bright/cool/deep (winter)

## Training — hierarchical model
```bash
python train.py --config configs/hierarchical.yaml
python train.py --config configs/hierarchical.yaml --resume results/checkpoint_epoch_25.pth
```
- Joint loss: `total = loss_season + lambda * loss_subtype`  (lambda=1.0)
- Saves `results/checkpoint_epoch_{N}.pth` every epoch (full optimizer state)
- Saves `results/best_hierarchical.pth` on season accuracy improvement
- Early stopping: patience = 10 epochs on season accuracy
- Optional `--drive_dir PATH`: syncs results/ to Google Drive every 5 epochs

## Training — baseline
```bash
python train_farl.py
```
Saves to `models/farl_model.pth`.

## Evaluation
```bash
python evaluate_baseline.py
python evaluate_hierarchical.py --checkpoint results/best_hierarchical.pth
```

## Original paper hyperparameters (used for both models)
- Backbone: FaRL-Base-Patch16, pretrained on LAION-Face, **weights frozen**
- Input: RGB-masked face crops, 3×224×224
- Train/test split: 80/20 (~4,008 train / ~912 test), no identity overlap
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-5)
- Scheduler: CosineAnnealingWarmRestarts (T_0=10, eta_min=1e-5)
- Epochs: 50, batch size: 64
- Augmentation: RandomResizedCrop(224), HorizontalFlip(p=0.5),
  ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2),
  RandomAdjustSharpness(factor=2, p=0.2)
- Metrics: accuracy, precision, recall, F1 (macro), top-2 (season) / top-3 (sub-type)

## Environment
- Python 3.10+
- PyTorch 2.3.0 + CUDA 12.1
- timm 0.9.0
- GPU: NVIDIA RTX 4070 (8 GB VRAM) / Google Colab T4

## Key conventions
- FaRL backbone always loaded with `model.eval()` and `torch.no_grad()`.
- `FaRLBackbone` in `hierarchical_head.py` auto-detects checkpoint format:
  handles CLIP key names (from `train_farl.py` output) and official FaRL keys.
- All metrics macro-averaged via sklearn.
- Confusion matrices saved as PNG to `results/`.

## Paper motivation (cite in writing)
The original paper conclusion states: *"we will explore those paradigms involving
hierarchical and ordinal learning, given that such nature matches the inner nature
of the Armocromia classes."* This work directly implements that suggestion.
