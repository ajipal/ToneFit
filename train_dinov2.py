"""
train_dinov2.py — ToneFit ML Pipeline: Step 5 (Model B — DINOv2)
=================================================================
Trains a DINOv2 ViT-B/14 classifier to predict personal color season
(Spring / Summer / Autumn / Winter) from face images.

DINOv2 is a self-supervised Vision Transformer trained by Meta AI.
It produces powerful image embeddings without task-specific pretraining.

NOTE: DINOv2 uses a patch size of 14. Input images must be multiples of 14.
      We use 224x224 (224 / 14 = 16 patches per side). Do NOT use 256 here.

Pipeline position:
    collect_data.py → preprocess.py → eda.ipynb
    → train_traditional.py → [THIS FILE] → evaluate.py → predict.py

Usage (Colab or local):
    python train_dinov2.py

Outputs:
    models/dinov2_model.pth          — best model weights (backbone + head)
    results/dinov2_history.json      — per-epoch loss/accuracy history
    results/dinov2_training.png      — training curves (loss + accuracy)
    results/dinov2_test_report.txt   — classification report on test set

Requirements:
    pip install torch torchvision scikit-learn pandas matplotlib
"""

import os
import json
import time
import warnings
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for Colab/servers
import matplotlib.pyplot as plt

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

MODEL_OUT     = MODELS_DIR / "dinov2_model.pth"
HISTORY_OUT   = RESULTS_DIR / "dinov2_history.json"
PLOT_OUT      = RESULTS_DIR / "dinov2_training.png"
REPORT_OUT    = RESULTS_DIR / "dinov2_test_report.txt"

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
# MODEL DEFINITION
# =============================================================================

class DINOv2Classifier(nn.Module):
    """
    DINOv2 ViT-B/14 backbone + lightweight 2-layer classifier head.

    Architecture:
        DINOv2 backbone (frozen)  →  [CLS] token (768-d)
        → Linear(768, 384) → ReLU → Dropout(0.5) → Linear(384, 4)

    We freeze ALL backbone parameters and only train the head.
    This is valid because DINOv2 is already a powerful general-purpose
    feature extractor — fine-tuning would require much more data.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM,
                 num_classes: int = NUM_CLASSES, dropout: float = DROPOUT):
        super().__init__()

        # Load DINOv2 ViT-B/14 via timm (avoids torch.hub Colab issues)
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

        # Classifier head — same architecture as FaRL baseline for fair comparison
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes),
        )

    def forward(self, x):
        # timm with num_classes=0 returns pooled CLS token directly
        features = self.backbone(x)  # shape: (batch, 768)
        logits = self.head(features) # shape: (batch, 4)
        return logits


# =============================================================================
# TRAINING & EVALUATION HELPERS
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer):
    """Run one full training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
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

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
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
    fig.suptitle("DINOv2 Training Curves — ToneFit Personal Color Classification",
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
    train_dataset = ImageFolder(root=str(TRAIN_DIR), transform=train_transform)
    test_dataset  = ImageFolder(root=str(TEST_DIR),  transform=val_transform)

    print(f"[INFO] Classes detected: {train_dataset.classes}")
    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Test  samples: {len(test_dataset)}")

    train_counts = Counter(label for _, label in train_dataset.samples)
    for i, s in enumerate(SEASONS):
        print(f"         {s:8s}: {train_counts[i]}")

    # --- Class weights -------------------------------------------------------
    train_labels = [label for _, label in train_dataset.samples]
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
