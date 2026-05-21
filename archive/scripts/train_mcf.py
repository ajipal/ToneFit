"""
train_mcf.py — ToneFit ML Project
====================================
Model B: MCF (Mask Contrastive Face) Classifier
Backbone: ViT-B/16 loaded via timm, patched with MCF self-supervised weights.
         feature_dim=768 (CLS token output, no projection head).

MCF paper: "Learning Robust Facial Representations with Self-Supervised Contrastive
           Learning and Masked Image Modeling" — face-specialized, self-supervised ViT.

Pretrained weights (Google Drive):
  https://drive.google.com/file/d/1hZqMKUfc2J6zGMDLPB4m32fIzi0TSecU/view?usp=sharing
  Place at: models/mcf_weights.pth

Checkpoint structure:
  checkpoint['model'] — state dict with keys prefixed by 'online_encoder.'
  Only online_encoder keys are loaded; target encoder and projector keys are discarded.

Pipeline step: Step 4c — Deep Learning Training (MCF)

Usage:
    python train_mcf.py

Outputs:
    models/mcf_model.pth          — best model checkpoint (by val accuracy)
    results/mcf_history.json      — per-epoch loss/accuracy history
    results/mcf_training.png      — training curve plot
    results/mcf_test_report.txt   — classification report on test set
"""

import os
import json
import time
import warnings
import numpy as np
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

TRAIN_DIR   = "RGB-M/train"
TEST_DIR    = "RGB-M/test"
MCF_WEIGHTS = "models/mcf_weights.pth"
MODEL_OUT   = "models/mcf_model.pth"
HISTORY_OUT = "results/mcf_history.json"
PLOT_OUT    = "results/mcf_training.png"
REPORT_OUT  = "results/mcf_test_report.txt"

LABEL_MAP = {"autumn": 0, "spring": 1, "summer": 2, "winter": 3}
SEASONS   = ["autumn", "spring", "summer", "winter"]

EPOCHS       = 50
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-5
T_0          = 10
ETA_MIN      = 1e-5

FEATURE_DIM  = 768  # ViT-B/16 CLS token output

# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ---------------------------------------------------------------------------
# DATA TRANSFORMS
# MCF uses standard ImageNet normalization (not CLIP normalization).
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

def build_classifier_head(feature_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(feature_dim, feature_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(feature_dim // 2, 4),
    )


class MCFModel(nn.Module):
    """ViT-B/16 backbone patched with MCF weights + 4-class classifier head."""

    def __init__(self, vit_backbone, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.backbone = vit_backbone
        self.head = build_classifier_head(feature_dim)

    def forward(self, x):
        features = self.backbone(x)   # (B, 768) — timm num_classes=0 returns CLS token
        return self.head(features)


def load_model():
    """
    Load MCF-pretrained ViT-B/16 as a frozen feature extractor.

    The MCF checkpoint stores the full MIM+contrastive model under checkpoint['model'].
    Keys follow the pattern 'online_encoder.<vit_key>'.
    We extract and rename only those keys, then load into a plain timm ViT-B/16.

    Falls back to ImageNet-pretrained ViT-B/16 (timm) if weights file is missing.
    """
    import timm

    mcf_path = Path(MCF_WEIGHTS)

    vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)

    if mcf_path.exists():
        print("[INFO] MCF weights found. Loading MCF ViT-B/16 backbone...")
        checkpoint = torch.load(str(mcf_path), map_location="cpu", weights_only=False)

        # MCF checkpoint may be nested under 'model' key
        state_dict = checkpoint.get("model", checkpoint)

        # Extract online_encoder weights and strip the prefix
        online_enc = {
            k.replace("online_encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("online_encoder.")
        }

        if len(online_enc) == 0:
            # Some builds store without prefix — try loading directly
            print("[WARNING] No 'online_encoder.*' keys found. Attempting direct load...")
            online_enc = state_dict

        missing, unexpected = vit.load_state_dict(online_enc, strict=False)
        print(f"[INFO] MCF weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        backbone_name = "MCF (ViT-B/16, self-supervised face pretraining)"

    else:
        print(f"[WARNING] MCF weights not found at '{MCF_WEIGHTS}'.")
        print("[INFO] Falling back to ImageNet-pretrained ViT-B/16 via timm.")
        vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        backbone_name = "ViT-B/16 (ImageNet pretrained fallback)"

    # Freeze backbone entirely — only head is trained
    for param in vit.parameters():
        param.requires_grad = False

    model = MCFModel(vit, feature_dim=FEATURE_DIM)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Backbone: {backbone_name}")
    print(f"[INFO] Trainable params: {trainable:,} / {total:,}")

    return model, backbone_name


# ---------------------------------------------------------------------------
# TRAINING HELPERS
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


def get_all_predictions(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, best_epoch: int, save_path: str):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MCF Training Curves", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs_range, history["train_loss"], label="Train Loss", color="steelblue")
    ax.plot(epochs_range, history["val_loss"],   label="Val Loss",   color="coral")
    ax.axvline(x=best_epoch, color="green", linestyle="--", linewidth=1.5,
               label=f"Best epoch ({best_epoch})")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs_range, history["train_acc"], label="Train Acc", color="steelblue")
    ax.plot(epochs_range, history["val_acc"],   label="Val Acc",   color="coral")
    ax.axvline(x=best_epoch, color="green", linestyle="--", linewidth=1.5,
               label=f"Best epoch ({best_epoch})")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training plot saved to: {save_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()

    print("=" * 60)
    print("  ToneFit — MCF Personal Color Classifier (Model B)")
    print("=" * 60)

    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load datasets
    # ------------------------------------------------------------------
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

    train_dataset = ImageFolder(root=TRAIN_DIR, transform=train_transform)
    test_dataset  = ImageFolder(root=TEST_DIR,  transform=val_transform)

    print(f"[INFO] Classes detected: {train_dataset.classes}")
    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Test  samples: {len(test_dataset)}")

    train_counts = Counter(label for _, label in train_dataset.samples)
    test_counts  = Counter(label for _, label in test_dataset.samples)
    print("\n[INFO] Train class distribution:")
    for i, s in enumerate(SEASONS):
        print(f"  {s:<8}: {train_counts[i]}")
    print("[INFO] Test class distribution:")
    for i, s in enumerate(SEASONS):
        print(f"  {s:<8}: {test_counts[i]}")

    # ------------------------------------------------------------------
    # 2. Class weights
    # ------------------------------------------------------------------
    train_labels = [label for _, label in train_dataset.samples]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2, 3]),
        y=np.array(train_labels),
    )
    print(f"\n[INFO] Class weights: {dict(zip(SEASONS, class_weights.round(4)))}")
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # ------------------------------------------------------------------
    # 3. DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    print(f"\n[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Test  batches: {len(test_loader)}")

    # ------------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------------
    print("\n[INFO] Building model...")
    model, backbone_name = load_model()
    model = model.to(device)

    # ------------------------------------------------------------------
    # 5. Loss, optimizer, scheduler
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, eta_min=ETA_MIN
    )

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    print(f"\n[INFO] Starting training for {EPOCHS} epochs...")
    print(f"[INFO] Backbone: {backbone_name}")
    print("-" * 70)
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>8}  {'LR':>10}  {'Time':>6}")
    print("-" * 70)

    history = {
        "train_loss": [],
        "val_loss":   [],
        "train_acc":  [],
        "val_acc":    [],
    }

    best_val_acc = 0.0
    best_epoch   = 1

    for epoch in range(1, EPOCHS + 1):
        t_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc     = evaluate(model, test_loader, criterion, device)

        scheduler.step(epoch - 1)

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_acc"].append(round(val_acc, 6))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(
                {"backbone": model.backbone.state_dict(), "head": model.head.state_dict()},
                MODEL_OUT,
            )

        elapsed    = time.time() - t_start
        current_lr = optimizer.param_groups[0]["lr"]
        marker     = " <-- best" if epoch == best_epoch else ""

        print(
            f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.4f}  "
            f"{val_loss:>8.4f}  {val_acc:>8.4f}  {current_lr:>10.2e}  "
            f"{elapsed:>5.1f}s{marker}"
        )

    print("-" * 70)
    print(f"\n[INFO] Training complete. Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"[INFO] Best model saved to: {MODEL_OUT}")

    total_time_min = (time.time() - start_time) / 60
    print(f"[INFO] Total training time: {total_time_min:.1f} min")

    # ------------------------------------------------------------------
    # 7. Save training history
    # ------------------------------------------------------------------
    history["best_epoch"]        = best_epoch
    history["best_val_acc"]      = round(best_val_acc, 6)
    history["backbone"]          = backbone_name
    history["epochs"]            = EPOCHS
    history["batch_size"]        = BATCH_SIZE
    history["total_time_min"]    = round(total_time_min, 1)

    with open(HISTORY_OUT, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] History saved to: {HISTORY_OUT}")

    # ------------------------------------------------------------------
    # 8. Plot training curves
    # ------------------------------------------------------------------
    plot_training_curves(history, best_epoch, PLOT_OUT)

    # ------------------------------------------------------------------
    # 9. Final evaluation on test set using best model
    # ------------------------------------------------------------------
    print(f"\n[INFO] Loading best model for final test evaluation...")
    checkpoint = torch.load(MODEL_OUT, map_location=device)
    model.backbone.load_state_dict(checkpoint["backbone"])
    model.head.load_state_dict(checkpoint["head"])
    model.eval()

    all_labels, all_preds = get_all_predictions(model, test_loader, device)

    report_str = classification_report(
        all_labels, all_preds,
        target_names=SEASONS,
        digits=4,
    )

    print("\n" + "=" * 60)
    print("  Final Test Classification Report")
    print("=" * 60)
    print(f"  Backbone: {backbone_name}")
    print(f"  Best Epoch: {best_epoch}  |  Best Val Acc: {best_val_acc:.4f}")
    print("=" * 60)
    print(report_str)

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"{'':12s} " + "  ".join(f"{s:>8s}" for s in SEASONS))
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8d}" for v in row)
        print(f"{SEASONS[i]:12s} {row_str}")

    autumn_idx    = LABEL_MAP["autumn"]
    autumn_correct = cm[autumn_idx, autumn_idx]
    autumn_total   = cm[autumn_idx].sum()
    autumn_recall  = autumn_correct / autumn_total if autumn_total > 0 else 0.0
    print(f"\n[NOTE] Autumn recall: {autumn_recall:.4f} ({autumn_correct}/{autumn_total})")
    print("       (Autumn is historically the hardest class — track this carefully.)")

    with open(REPORT_OUT, "w") as f:
        f.write("ToneFit — MCF Classifier — Test Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Backbone     : {backbone_name}\n")
        f.write(f"Best Epoch   : {best_epoch}\n")
        f.write(f"Best Val Acc : {best_val_acc:.4f}\n")
        f.write(f"Training Time: {total_time_min:.1f} min\n")
        f.write(f"Device       : {device}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
        f.write("\nConfusion Matrix (rows=actual, cols=predicted):\n")
        f.write(f"{'':12s} " + "  ".join(f"{s:>8s}" for s in SEASONS) + "\n")
        for i, row in enumerate(cm):
            row_str = "  ".join(f"{v:>8d}" for v in row)
            f.write(f"{SEASONS[i]:12s} {row_str}\n")
        f.write(f"\nAutumn recall: {autumn_recall:.4f} ({autumn_correct}/{autumn_total})\n")

    print(f"\n[INFO] Test report saved to: {REPORT_OUT}")
    print("\n[DONE] MCF training pipeline finished successfully.")


if __name__ == "__main__":
    main()
