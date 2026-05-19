"""
train_dinov2_color.py — ToneFit ML Pipeline: Model B+ (DINOv2 + Color Fusion)
===============================================================================
Trains a DINOv2 ViT-B/14 classifier with color feature fusion to predict
personal color season (Spring / Summer / Autumn / Winter) from face images.

DINOv2 is a self-supervised Vision Transformer trained by Meta AI.
It produces powerful image embeddings without task-specific pretraining.

Color feature fusion: CIELab/HSV features from preprocess.py (L*, a*, b*, ITA,
H, S, V) are concatenated with deep features before the classifier head.
This directly targets the warm/cool confusion axis identified in Stacchio et al.
(2024) — a* encodes warm vs cool, ITA encodes skin undertone.

NOTE: DINOv2 uses a patch size of 14. Input images must be multiples of 14.
      We use 224x224 (224 / 14 = 16 patches per side). Do NOT use 256 here.

Requires:
    Run preprocess.py first to generate features.csv and models/scaler.pkl.

Usage (Colab or local):
    python train_dinov2_color.py

Outputs:
    models/dinov2_color_model.pth        — best model weights (backbone + head)
    results/dinov2_color_history.json    — per-epoch loss/accuracy history
    results/dinov2_color_training.png    — training curves (loss + accuracy)
    results/dinov2_color_test_report.txt — classification report on test set

Requirements:
    pip install torch torchvision scikit-learn pandas matplotlib
"""

import os
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for Colab/servers
import matplotlib.pyplot as plt

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Label mapping — alphabetical, must match ImageFolder order --------------
LABEL_MAP = {"autumn": 0, "spring": 1, "summer": 2, "winter": 3}
SEASONS   = ["autumn", "spring", "summer", "winter"]

# --- Paths (all relative to working directory — Colab-friendly) --------------
TRAIN_DIR     = Path("RGB-M/train")   # READ ONLY — ImageFolder reads sub-types recursively
TEST_DIR      = Path("RGB-M/test")    # READ ONLY
MODELS_DIR    = Path("models")
RESULTS_DIR   = Path("results")

MODEL_OUT     = MODELS_DIR / "dinov2_color_model.pth"
HISTORY_OUT   = RESULTS_DIR / "dinov2_color_history.json"
PLOT_OUT      = RESULTS_DIR / "dinov2_color_training.png"
REPORT_OUT    = RESULTS_DIR / "dinov2_color_test_report.txt"

# --- Hyperparameters ---------------------------------------------------------
EPOCHS           = 50
BATCH_SIZE       = 64
LR               = 1e-3        # AdamW learning rate for the classifier head
BACKBONE_LR      = 1e-5        # lower LR for unfrozen backbone blocks
WEIGHT_DECAY     = 1e-5
T_0              = 10          # CosineAnnealingWarmRestarts restart period
ETA_MIN          = 1e-5        # minimum LR at cosine bottom
DROPOUT          = 0.5
FEATURE_DIM      = 768         # DINOv2 ViT-B/14 output dimension
NUM_CLASSES      = 4
N_UNFREEZE_BLOCKS = 2          # number of final transformer blocks to unfreeze

# DINOv2 patch size = 14 → input must be a multiple of 14
# 224 = 14 × 16  ✓
INPUT_SIZE    = 224
RESIZE_SIZE   = 256            # used only for val/test Resize before CenterCrop

RANDOM_STATE  = 42
NUM_WORKERS   = 0              # 0 is safest for Windows and Google Colab

# Color feature fusion — from preprocess.py output
FEATURES_CSV  = "features.csv"
SCALER_PATH   = "models/scaler.pkl"
# 7 features selected for warm/cool discrimination (a* = warm/cool axis, ITA = undertone)
COLOR_COLS    = ["L_mean", "a_mean", "b_mean", "ITA", "H_mean", "S_mean", "V_mean"]
COLOR_DIM     = len(COLOR_COLS)
# All 12 columns as saved by preprocess.py (needed for scaler index lookup)
_ALL_FEAT_COLS = ["L_mean","a_mean","b_mean","L_std","a_std","b_std",
                  "ITA","H_mean","S_mean","V_mean","skin_ratio","blur_score"]

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# =============================================================================
# DEVICE
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
if device.type == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# DATA AUGMENTATION / TRANSFORMS
# =============================================================================

# Training transforms — aggressive augmentation to prevent overfitting
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2),
    transforms.RandomAdjustSharpness(2, p=0.2),
    transforms.ToTensor(),
    # ImageNet normalization — standard for DINOv2 pretrained weights
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Validation / test transforms — deterministic, no augmentation
val_transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# =============================================================================
# COLOR-AWARE DATASET
# =============================================================================

class ColorAwareDataset(Dataset):
    """
    Wraps an ImageFolder dataset and appends pre-extracted color features
    (from features.csv) to each sample. Returns (image, color_feats, label).

    Images without a features.csv entry (failed preprocessing quality filter)
    receive a zero vector — the model learns to ignore absent features gracefully.
    """

    def __init__(self, image_folder: ImageFolder, features_csv: str,
                 scaler_path: str, color_cols: list):
        self.dataset    = image_folder
        self.color_dim  = len(color_cols)
        self._zero      = np.zeros(self.color_dim, dtype=np.float32)
        self.color_lookup: dict = {}

        if os.path.exists(features_csv) and os.path.exists(scaler_path):
            df = pd.read_csv(features_csv)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            col_indices = [_ALL_FEAT_COLS.index(c) for c in color_cols]
            scaled = scaler.transform(df[_ALL_FEAT_COLS].values.astype(np.float32))

            for i, row in df.iterrows():
                key = row["filename"].replace("\\", "/")
                self.color_lookup[key] = scaled[i, col_indices].astype(np.float32)

            print(f"[INFO] Color features loaded: {len(self.color_lookup)} entries "
                  f"({self.color_dim} features per image)")
        else:
            print(f"[WARNING] features.csv or scaler.pkl not found — "
                  f"color features will be zero vectors. Run preprocess.py first.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        img_path = self.dataset.samples[idx][0]
        # features.csv keys are relative to RGB-M/ (e.g. "train/autumn/deep/img.jpg")
        rel_path = os.path.relpath(img_path, "RGB-M").replace("\\", "/")
        color_feats = self.color_lookup.get(rel_path, self._zero)
        return image, torch.tensor(color_feats), label


# =============================================================================
# MODEL DEFINITION
# =============================================================================

class DINOv2Classifier(nn.Module):
    """
    DINOv2 ViT-B/14 backbone + lightweight 2-layer classifier head.
    Color features (CIELab/HSV) are concatenated with deep features before the head.

    Architecture:
        DINOv2 backbone  →  [CLS] token (768-d)
        concat color_feats (7-d)
        → Linear(775, 387) → ReLU → Dropout(0.5) → Linear(387, 4)
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, color_dim: int = COLOR_DIM,
                 num_classes: int = NUM_CLASSES, dropout: float = DROPOUT):
        super().__init__()

        print("[INFO] Loading DINOv2 ViT-B/14 via timm ...")
        print("       (First run will download ~330 MB of weights)")
        import timm
        self.backbone = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,  # removes classification head, returns 768-d features
            img_size=224,   # interpolate positional embeddings from 518 to 224
        )

        # Freeze entire backbone first, then selectively unfreeze last N blocks
        for param in self.backbone.parameters():
            param.requires_grad = False

        for block in self.backbone.blocks[-N_UNFREEZE_BLOCKS:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        unfrozen = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"[INFO] Unfroze last {N_UNFREEZE_BLOCKS} backbone blocks + norm ({unfrozen:,} params).")

        combined_dim = feature_dim + color_dim  # 768 + 7 = 775
        self.head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, num_classes),
        )

    def forward(self, x, color_feats):
        features = self.backbone(x)                           # (B, 768)
        combined = torch.cat([features, color_feats], dim=1) # (B, 775)
        return self.head(combined)


# =============================================================================
# TRAINING & EVALUATION HELPERS
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer):
    """Run one full training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, color_feats, labels in loader:
        images      = images.to(device)
        color_feats = color_feats.to(device).float()
        labels      = labels.to(device)

        optimizer.zero_grad()
        logits = model(images, color_feats)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluate on val/test set. Returns (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    for images, color_feats, labels in loader:
        images      = images.to(device)
        color_feats = color_feats.to(device).float()
        labels      = labels.to(device)

        logits = model(images, color_feats)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_training_curves(history: dict, best_epoch: int, save_path: Path):
    """
    Plot loss and accuracy curves for train/val across all epochs.
    Marks the best epoch (lowest val loss) with a vertical dashed line.
    Saves to save_path.
    """
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DINOv2 + Color Features Training Curves — ToneFit Personal Color Classification",
                 fontsize=13, fontweight="bold")

    # --- Loss subplot --------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss",
             color="#2196F3", linewidth=2)
    ax1.plot(epochs_range, history["val_loss"],   label="Val Loss",
             color="#FF5722", linewidth=2)
    ax1.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=1.5,
                label=f"Best epoch ({best_epoch})")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Accuracy subplot ----------------------------------------------------
    ax2 = axes[1]
    ax2.plot(epochs_range, [a * 100 for a in history["train_acc"]],
             label="Train Acc", color="#2196F3", linewidth=2)
    ax2.plot(epochs_range, [a * 100 for a in history["val_acc"]],
             label="Val Acc",   color="#FF5722", linewidth=2)
    ax2.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=1.5,
                label=f"Best epoch ({best_epoch})")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training curves saved → {save_path}")


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    start_time = time.time()

    # --- Create output directories if they don't exist -----------------------
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- Verify dataset folders exist ----------------------------------------
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

    print("=" * 60)
    print("  ToneFit — DINOv2 Personal Color Classifier Training")
    print("=" * 60)

    # --- Datasets with ImageFolder (recursive — sub-types = same class) ------
    print("\n[INFO] Loading datasets ...")
    _train_folder = ImageFolder(root=str(TRAIN_DIR), transform=train_transform)
    _test_folder  = ImageFolder(root=str(TEST_DIR),  transform=val_transform)

    train_dataset = ColorAwareDataset(_train_folder, FEATURES_CSV, SCALER_PATH, COLOR_COLS)
    test_dataset  = ColorAwareDataset(_test_folder,  FEATURES_CSV, SCALER_PATH, COLOR_COLS)

    print(f"[INFO] Classes detected: {_train_folder.classes}")
    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Test  samples: {len(test_dataset)}")

    train_counts = Counter(label for _, label in _train_folder.samples)
    for i, s in enumerate(SEASONS):
        print(f"         {s:8s}: {train_counts[i]}")

    # --- Class weights -------------------------------------------------------
    train_labels = [label for _, label in _train_folder.samples]
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2, 3]),
        y=np.array(train_labels),
    )
    print(f"[INFO] Class weights (balanced):")
    for i, s in enumerate(SEASONS):
        print(f"         {s:8s} (label={i}): {weights[i]:.4f}")
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # --- DataLoaders ---------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # --- Model ---------------------------------------------------------------
    model = DINOv2Classifier().to(device)

    # Count trainable vs frozen parameters
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen      = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\n[INFO] Model parameters:")
    print(f"         Trainable (head):  {trainable:,}")
    print(f"         Frozen (backbone): {frozen:,}")

    # --- Loss, Optimizer, Scheduler ------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # class_weights is a FloatTensor on device

    # Differential LRs: low LR for unfrozen backbone blocks, higher for head
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params,     "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    # CosineAnnealingWarmRestarts: cosine decay with periodic restarts
    # T_0=10 means the first restart happens after 10 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, eta_min=ETA_MIN
    )

    # --- Training history storage --------------------------------------------
    history = {
        "train_loss": [],
        "val_loss":   [],
        "train_acc":  [],
        "val_acc":    [],
    }

    best_val_acc  = 0.0
    best_epoch    = 1

    # --- Table header --------------------------------------------------------
    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>10} {'LR':>10} {'Time':>8}")
    print("-" * 72)

    # --- Epoch loop ----------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc, _, _ = evaluate(model, val_loader, criterion)

        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_epoch    = epoch
            torch.save(
                {
                    "backbone": model.backbone.state_dict(),
                    "head":     model.head.state_dict(),
                },
                MODEL_OUT,
            )
            marker = " ← best"
        else:
            marker = ""

        elapsed  = time.time() - epoch_start
        curr_lr  = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>6} {train_loss:>12.4f} {train_acc * 100:>9.2f}% "
              f"{val_loss:>10.4f} {val_acc * 100:>9.2f}% "
              f"{curr_lr:>10.2e} {elapsed:>6.1f}s{marker}")

    total_time = time.time() - start_time
    print("-" * 72)
    print(f"[INFO] Training complete in {total_time / 60:.1f} min")
    print(f"[INFO] Best epoch: {best_epoch}  |  Best val acc: {best_val_acc:.4f}")
    print(f"[INFO] Best model saved → {MODEL_OUT}")

    # --- Save training history to JSON ---------------------------------------
    with open(HISTORY_OUT, "w") as f:
        json.dump({**history, "best_epoch": best_epoch}, f, indent=2)
    print(f"[INFO] History saved → {HISTORY_OUT}")

    # --- Plot training curves ------------------------------------------------
    plot_training_curves(history, best_epoch, PLOT_OUT)

    # --- Final evaluation on test set ----------------------------------------
    print("\n[INFO] Running final evaluation on test set (best model weights) ...")

    # Reload best weights
    checkpoint = torch.load(MODEL_OUT, map_location=device)
    model.backbone.load_state_dict(checkpoint["backbone"])
    model.head.load_state_dict(checkpoint["head"])
    model.eval()

    _, test_acc, y_pred, y_true = evaluate(model, val_loader, criterion)

    print(f"\n[RESULT] Test Accuracy: {test_acc * 100:.2f}%\n")

    report = classification_report(
        y_true, y_pred,
        target_names=SEASONS,
        digits=4,
    )
    print(report)

    # --- Confusion matrix (text) ---------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    cm_lines = ["\nConfusion Matrix (rows=true, cols=predicted):"]
    header = "           " + "  ".join(f"{s:>8}" for s in SEASONS)
    cm_lines.append(header)
    for i, row in enumerate(cm):
        row_str = f"{SEASONS[i]:10s}" + "  ".join(f"{v:>8}" for v in row)
        cm_lines.append(row_str)
    cm_text = "\n".join(cm_lines)
    print(cm_text)

    # --- Autumn recall note --------------------------------------------------
    autumn_idx    = LABEL_MAP["autumn"]
    autumn_recall = cm[autumn_idx, autumn_idx] / cm[autumn_idx].sum()
    print(f"\n[NOTE] Autumn recall: {autumn_recall * 100:.2f}%")
    print("       Autumn is historically the hardest class in personal color studies.")
    print("       Track this metric carefully when comparing models.")

    # --- Save report to file -------------------------------------------------
    report_content = (
        "ToneFit — DINOv2 Test Set Classification Report\n"
        "=" * 50 + "\n\n"
        f"Best epoch  : {best_epoch}\n"
        f"Test accuracy: {test_acc * 100:.2f}%\n\n"
        + report
        + "\n"
        + cm_text
        + f"\n\nAutumn recall: {autumn_recall * 100:.2f}%\n"
        f"\nTotal training time: {total_time / 60:.1f} min\n"
        f"Device used: {device}\n"
    )
    with open(REPORT_OUT, "w") as f:
        f.write(report_content)
    print(f"\n[INFO] Test report saved → {REPORT_OUT}")

    print("\n" + "=" * 60)
    print("  All DINOv2 outputs saved:")
    print(f"    Model weights  : {MODEL_OUT}")
    print(f"    Training history: {HISTORY_OUT}")
    print(f"    Training curves : {PLOT_OUT}")
    print(f"    Test report     : {REPORT_OUT}")
    print("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
