"""
train_clip.py — ToneFit ML Pipeline: CLIP ViT-B/16
====================================================
Trains a CLIP ViT-B/16 classifier to predict personal color season
(Spring / Summer / Autumn / Winter) from face images.

CLIP (Contrastive Language-Image Pretraining) — OpenAI.
The visual encoder is used as a frozen feature extractor.

CLIP ViT-B/16 specifics:
  - Feature dim : 512  (768-d CLS token projected to 512 via learned proj matrix)
  - Depth       : 12 transformer blocks
  - Normalization: CLIP-specific mean/std (different from ImageNet)
  - Loaded via  : open_clip (pip install open-clip-torch)

Usage:
    python train_clip.py

Outputs:
    models/clip_model.pth          — best model weights (backbone + head)
    results/clip_history.json      — per-epoch loss/accuracy history
    results/clip_training.png      — training curves (loss + accuracy)
    results/clip_test_report.txt   — classification report on test set
"""

import os
import json
import time
import warnings
import numpy as np
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

LABEL_MAP = {"autumn": 0, "spring": 1, "summer": 2, "winter": 3}
SEASONS   = ["autumn", "spring", "summer", "winter"]

TRAIN_DIR   = Path("RGB-M/train")
TEST_DIR    = Path("RGB-M/test")
MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")

MODEL_OUT   = MODELS_DIR  / "clip_model.pth"
HISTORY_OUT = RESULTS_DIR / "clip_history.json"
PLOT_OUT    = RESULTS_DIR / "clip_training.png"
REPORT_OUT  = RESULTS_DIR / "clip_test_report.txt"

EPOCHS            = 50
BATCH_SIZE        = 64
LR                = 1e-3      # head learning rate
BACKBONE_LR       = 1e-5      # unfrozen backbone blocks
WEIGHT_DECAY      = 1e-5
T_0               = 10
ETA_MIN           = 1e-5
DROPOUT           = 0.5
FEATURE_DIM       = 512       # CLIP ViT-B/16 projected output dim
NUM_CLASSES       = 4
N_UNFREEZE_BLOCKS = 2         # last 2 of 12 transformer blocks

INPUT_SIZE  = 224
RESIZE_SIZE = 256
NUM_WORKERS = 0

# CLIP-specific normalization — different from ImageNet
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# DEVICE
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
if device.type == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# TRANSFORMS  (CLIP normalization)
# =============================================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2),
    transforms.RandomAdjustSharpness(2, p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

# =============================================================================
# MODEL
# =============================================================================

class CLIPClassifier(nn.Module):
    """
    CLIP ViT-B/16 visual encoder + 2-layer classifier head.

    open_clip's visual encoder runs the full forward pass including the
    learned projection (768 → 512), so backbone(x) returns 512-d features.

    All backbone params frozen except the last N_UNFREEZE_BLOCKS transformer
    blocks, the final LayerNorm (ln_post), and the projection matrix (proj),
    which are fine-tuned at a low LR.
    """

    def __init__(self):
        super().__init__()

        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip is required. Install with: pip install open-clip-torch"
            )

        print("[INFO] Loading CLIP ViT-B/16 via open_clip ...")
        print("       (First run will download ~330 MB of weights)")
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )
        self.backbone = clip_model.visual

        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer blocks
        for block in self.backbone.transformer.resblocks[-N_UNFREEZE_BLOCKS:]:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze final LayerNorm and projection matrix
        for param in self.backbone.ln_post.parameters():
            param.requires_grad = True
        if self.backbone.proj is not None:
            self.backbone.proj.requires_grad = True

        total_bb  = sum(p.numel() for p in self.backbone.parameters())
        unfrozen  = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"[INFO] Backbone: CLIP ViT-B/16 — {total_bb:,} params total")
        print(f"[INFO] Unfroze last {N_UNFREEZE_BLOCKS} blocks + ln_post + proj ({unfrozen:,} params)")

        self.head = nn.Sequential(
            nn.Linear(FEATURE_DIM, FEATURE_DIM // 2),   # 512 → 256
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FEATURE_DIM // 2, NUM_CLASSES),   # 256 → 4
        )

    def forward(self, x):
        features = self.backbone(x)   # (B, 512)
        return self.head(features)    # (B, 4)


# =============================================================================
# TRAINING & EVALUATION HELPERS
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

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

def plot_training_curves(history, best_epoch, save_path):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CLIP ViT-B/16 Training Curves — ToneFit Personal Color Classification",
                 fontsize=13, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="#2196F3", linewidth=2)
    ax1.plot(epochs_range, history["val_loss"],   label="Val Loss",   color="#FF5722", linewidth=2)
    ax1.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=1.5,
                label=f"Best epoch ({best_epoch})")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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
# MAIN
# =============================================================================

def main():
    start_time = time.time()

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

    print("=" * 60)
    print("  ToneFit — CLIP ViT-B/16 Personal Color Classifier")
    print("=" * 60)

    # --- Datasets ---------------------------------------------------------------
    print("\n[INFO] Loading datasets ...")
    train_dataset = ImageFolder(root=str(TRAIN_DIR), transform=train_transform)
    test_dataset  = ImageFolder(root=str(TEST_DIR),  transform=val_transform)

    print(f"[INFO] Classes: {train_dataset.classes}")
    print(f"[INFO] Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    train_counts = Counter(label for _, label in train_dataset.samples)
    for i, s in enumerate(SEASONS):
        print(f"         {s:8s}: {train_counts[i]}")

    # --- Class weights ----------------------------------------------------------
    train_labels  = [label for _, label in train_dataset.samples]
    weights       = compute_class_weight("balanced", classes=np.array([0,1,2,3]),
                                         y=np.array(train_labels))
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"[INFO] Class weights: { {s: round(w,4) for s,w in zip(SEASONS, weights)} }")

    # --- DataLoaders ------------------------------------------------------------
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

    # --- Model ------------------------------------------------------------------
    model = CLIPClassifier().to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[INFO] Trainable: {trainable:,} | Frozen: {frozen:,}")

    # --- Loss, optimizer, scheduler ---------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params,     "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, eta_min=ETA_MIN
    )

    # --- Training loop ----------------------------------------------------------
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_epoch = 0.0, 1

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>10} {'LR':>10} {'Time':>8}")
    print("-" * 72)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc     = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _   = evaluate(model, val_loader, criterion)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save({"backbone": model.backbone.state_dict(),
                        "head":     model.head.state_dict()}, MODEL_OUT)
            marker = " ← best"

        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6} {train_loss:>12.4f} {train_acc*100:>9.2f}% "
              f"{val_loss:>10.4f} {val_acc*100:>9.2f}% "
              f"{curr_lr:>10.2e} {time.time()-t0:>6.1f}s{marker}")

    total_time = time.time() - start_time
    print("-" * 72)
    print(f"[INFO] Done in {total_time/60:.1f} min | Best epoch {best_epoch} | Val acc {best_val_acc:.4f}")
    print(f"[INFO] Best model saved → {MODEL_OUT}")

    # --- Save history -----------------------------------------------------------
    with open(HISTORY_OUT, "w") as f:
        json.dump({**history, "best_epoch": best_epoch}, f, indent=2)
    print(f"[INFO] History saved → {HISTORY_OUT}")

    plot_training_curves(history, best_epoch, PLOT_OUT)

    # --- Final test evaluation --------------------------------------------------
    print("\n[INFO] Final evaluation on test set (best weights) ...")
    checkpoint = torch.load(MODEL_OUT, map_location=device)
    model.backbone.load_state_dict(checkpoint["backbone"])
    model.head.load_state_dict(checkpoint["head"])

    _, test_acc, y_pred, y_true = evaluate(model, val_loader, criterion)
    print(f"\n[RESULT] Test Accuracy: {test_acc*100:.2f}%\n")

    report = classification_report(y_true, y_pred, target_names=SEASONS, digits=4)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    header = "           " + "  ".join(f"{s:>8}" for s in SEASONS)
    cm_lines = ["\nConfusion Matrix (rows=true, cols=predicted):", header]
    for i, row in enumerate(cm):
        cm_lines.append(f"{SEASONS[i]:10s}" + "  ".join(f"{v:>8}" for v in row))
    cm_text = "\n".join(cm_lines)
    print(cm_text)

    autumn_recall = cm[LABEL_MAP["autumn"], LABEL_MAP["autumn"]] / cm[LABEL_MAP["autumn"]].sum()
    print(f"\n[NOTE] Autumn recall: {autumn_recall*100:.2f}%")

    with open(REPORT_OUT, "w") as f:
        f.write("ToneFit — CLIP ViT-B/16 Test Set Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best epoch   : {best_epoch}\n")
        f.write(f"Test accuracy: {test_acc*100:.2f}%\n\n")
        f.write(report + "\n" + cm_text)
        f.write(f"\n\nAutumn recall: {autumn_recall*100:.2f}%\n")
        f.write(f"Total time   : {total_time/60:.1f} min\n")
        f.write(f"Device       : {device}\n")
    print(f"[INFO] Report saved → {REPORT_OUT}")

    print("\n" + "=" * 60)
    print("  CLIP ViT-B/16 outputs saved:")
    print(f"    {MODEL_OUT}")
    print(f"    {HISTORY_OUT}")
    print(f"    {PLOT_OUT}")
    print(f"    {REPORT_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
