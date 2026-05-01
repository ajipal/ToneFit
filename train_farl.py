"""
train_farl.py — ToneFit ML Project
====================================
Model A: FaRL (Face Representation Learning) Classifier
Backbone: ViT-Base with FaRL pretrained weights (fallback: ResNeXt50)

Pipeline step: Step 4b — Deep Learning Training (FaRL)

Usage:
    python train_farl.py

Outputs:
    models/farl_model.pth          — best model checkpoint (by val accuracy)
    results/farl_history.json      — per-epoch loss/accuracy history
    results/farl_training.png      — training curve plot
    results/farl_test_report.txt   — classification report on test set
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for Colab and headless)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DATASET_DIR   = "dataset"          # root folder with spring/summer/autumn/winter subfolders
SPLIT_CSV     = "data_split.csv"   # columns: filename, season, split
FARL_WEIGHTS  = "models/farl_weights.pth"
MODEL_OUT     = "models/farl_model.pth"
HISTORY_OUT   = "results/farl_history.json"
PLOT_OUT      = "results/farl_training.png"
REPORT_OUT    = "results/farl_test_report.txt"

LABEL_MAP = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}
SEASONS   = ["spring", "summer", "autumn", "winter"]

EPOCHS      = 50
BATCH_SIZE  = 64
LR          = 1e-3
WEIGHT_DECAY = 1e-5
T_0         = 10          # CosineAnnealingWarmRestarts restart period
ETA_MIN     = 1e-5

FARL_FEATURE_DIM   = 768   # ViT-Base hidden size
RESNEXT_FEATURE_DIM = 2048  # ResNeXt50 output features

# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ---------------------------------------------------------------------------
# DATASET CLASS
# ---------------------------------------------------------------------------

class PersonalColorDataset(Dataset):
    """
    Loads face images listed in data_split.csv.

    Expected CSV columns:
        filename  — basename only (e.g. 'spring_IU_001.jpg')
        season    — one of spring / summer / autumn / winter
        split     — 'train' or 'test'

    Images are expected at:  dataset/{season}/{filename}
    """

    def __init__(self, df: pd.DataFrame, dataset_dir: str, transform=None):
        """
        Args:
            df           : filtered DataFrame (already subset to train or test)
            dataset_dir  : root dataset directory path (string)
            transform    : torchvision transforms to apply
        """
        self.df = df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        season   = row["season"].strip().lower()
        filename = row["filename"].strip()
        label    = LABEL_MAP[season]

        img_path = os.path.join(self.dataset_dir, season, filename)

        # Load image — convert to RGB to handle grayscale or RGBA inputs
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Check that data_split.csv filename column matches actual files."
            )

        if self.transform:
            img = self.transform(img)

        return img, label


# ---------------------------------------------------------------------------
# DATA TRANSFORMS
# ---------------------------------------------------------------------------

# ImageNet normalization (used by both ViT and ResNeXt)
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
# MODEL BUILDER
# ---------------------------------------------------------------------------

def build_classifier_head(feature_dim: int) -> nn.Sequential:
    """
    Shared classifier head for both FaRL and ResNeXt50 fallback.
    Maps (feature_dim) -> (4 classes).

    Architecture:
        Linear(feature_dim, feature_dim//2) -> ReLU -> Dropout(0.5) -> Linear(feature_dim//2, 4)
    """
    return nn.Sequential(
        nn.Linear(feature_dim, feature_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(feature_dim // 2, 4),
    )


class FaRLModel(nn.Module):
    """
    FaRL backbone (ViT-Base) + classifier head.
    The backbone is frozen; only the head is trained.
    """

    def __init__(self, vit_backbone, feature_dim: int = FARL_FEATURE_DIM):
        super().__init__()
        self.backbone = vit_backbone
        self.head = build_classifier_head(feature_dim)

    def forward(self, x):
        # timm ViT models return (batch, num_patches+1, hidden_dim) with
        # forward_features, then we take the CLS token at index 0.
        features = self.backbone.forward_features(x)   # (B, N+1, 768) or (B, 768)

        # Some timm versions already return the CLS token directly as (B, 768)
        if features.dim() == 3:
            features = features[:, 0, :]   # CLS token

        return self.head(features)


class ResNeXtFallbackModel(nn.Module):
    """
    ResNeXt50-32x4d backbone + classifier head.
    Used when FaRL weights are not found.
    The backbone is frozen; only the head is trained.
    """

    def __init__(self, resnext_backbone, feature_dim: int = RESNEXT_FEATURE_DIM):
        super().__init__()
        # Remove the original classification layer
        self.backbone = nn.Sequential(*list(resnext_backbone.children())[:-1])
        self.head = build_classifier_head(feature_dim)

    def forward(self, x):
        features = self.backbone(x)          # (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)   # (B, 2048)
        return self.head(features)


def load_model():
    """
    Attempt to load FaRL (ViT-Base + pretrained FaRL weights).
    Falls back to ResNeXt50 (ImageNet pretrained) if weights file is missing.

    Returns:
        model       : nn.Module ready to be placed on device
        backbone_name: str describing which backbone was loaded
    """
    farl_path = Path(FARL_WEIGHTS)

    if farl_path.exists():
        # ------------------------------------------------------------------ #
        # Load FaRL (ViT-Base)
        # ------------------------------------------------------------------ #
        print("[INFO] FaRL weights found. Loading ViT-Base backbone...")
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for FaRL. Install it with:  pip install timm"
            )

        # Create a ViT-Base model with timm (no pretrained weights yet —
        # we load the FaRL checkpoint manually below)
        vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)

        # Load the FaRL checkpoint
        checkpoint = torch.load(str(farl_path), map_location="cpu")

        # FaRL checkpoints may store weights under a 'model' or 'state_dict' key
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint

        # Strip any 'module.' prefix that appears when saved from DataParallel
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}

        missing, unexpected = vit.load_state_dict(cleaned, strict=False)
        print(f"[INFO] FaRL weights loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

        # Freeze backbone
        for param in vit.parameters():
            param.requires_grad = False

        model = FaRLModel(vit, feature_dim=FARL_FEATURE_DIM)
        backbone_name = "FaRL (ViT-Base)"

    else:
        # ------------------------------------------------------------------ #
        # Fallback: ResNeXt50-32x4d (ImageNet pretrained via torchvision)
        # ------------------------------------------------------------------ #
        print(f"[WARNING] FaRL weights not found at '{FARL_WEIGHTS}'.")
        print("[INFO] Falling back to ResNeXt50-32x4d (ImageNet pretrained).")
        from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

        resnext = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

        # Freeze backbone
        for param in resnext.parameters():
            param.requires_grad = False

        model = ResNeXtFallbackModel(resnext, feature_dim=RESNEXT_FEATURE_DIM)
        backbone_name = "ResNeXt50-32x4d (ImageNet fallback)"

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Backbone: {backbone_name}")
    print(f"[INFO] Trainable params: {trainable:,} / {total:,}")

    return model, backbone_name


# ---------------------------------------------------------------------------
# TRAINING HELPERS
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one full pass over the training set. Returns (avg_loss, accuracy)."""
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

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate on a DataLoader. Returns (avg_loss, accuracy)."""
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

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def get_all_predictions(model, loader, device):
    """Collect all predictions and ground-truth labels from a DataLoader."""
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, best_epoch: int, save_path: str):
    """
    Plot training and validation loss + accuracy curves.
    Marks the best epoch (highest val accuracy) with a vertical line.
    """
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("FaRL Training Curves", fontsize=14, fontweight="bold")

    # --- Loss subplot ---
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

    # --- Accuracy subplot ---
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
    print("=" * 60)
    print("  ToneFit — FaRL Personal Color Classifier (Model A)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Ensure output directories exist
    # ------------------------------------------------------------------
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Load data_split.csv
    # ------------------------------------------------------------------
    print(f"\n[INFO] Loading split CSV: {SPLIT_CSV}")
    if not os.path.exists(SPLIT_CSV):
        raise FileNotFoundError(
            f"'{SPLIT_CSV}' not found in the current directory.\n"
            "Run the preprocessing step first to generate this file."
        )

    df = pd.read_csv(SPLIT_CSV)
    # Normalise column names (strip whitespace, lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {"filename", "season", "split"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"data_split.csv must have columns: {required_cols}\n"
            f"Found: {list(df.columns)}"
        )

    # Normalise values
    df["season"] = df["season"].str.strip().str.lower()
    df["split"]  = df["split"].str.strip().str.lower()

    train_df = df[df["split"] == "train"].copy()
    test_df  = df[df["split"] == "test"].copy()

    print(f"[INFO] Train samples: {len(train_df)}")
    print(f"[INFO] Test  samples: {len(test_df)}")

    # Print class distribution
    print("\n[INFO] Train class distribution:")
    print(train_df["season"].value_counts().to_string())
    print("\n[INFO] Test class distribution:")
    print(test_df["season"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 3. Compute class weights (handle imbalance)
    # ------------------------------------------------------------------
    train_labels = train_df["season"].map(LABEL_MAP).values
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2, 3]),
        y=train_labels,
    )
    print(f"\n[INFO] Class weights: {dict(zip(SEASONS, class_weights.round(4)))}")
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # ------------------------------------------------------------------
    # 4. Build DataLoaders
    # ------------------------------------------------------------------
    train_dataset = PersonalColorDataset(train_df, DATASET_DIR, transform=train_transform)
    test_dataset  = PersonalColorDataset(test_df,  DATASET_DIR, transform=val_transform)

    # We use all test data as both val (during training) and final test
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
    # 5. Build model
    # ------------------------------------------------------------------
    print("\n[INFO] Building model...")
    model, backbone_name = load_model()
    model = model.to(device)

    # ------------------------------------------------------------------
    # 6. Loss, optimizer, scheduler
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # Only optimize parameters that are NOT frozen (the classifier head)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, eta_min=ETA_MIN
    )

    # ------------------------------------------------------------------
    # 7. Training loop
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

    best_val_acc  = 0.0
    best_epoch    = 1

    for epoch in range(1, EPOCHS + 1):
        t_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate (using test set — small pilot study, no separate val split)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        # Step scheduler
        scheduler.step(epoch - 1)   # CosineAnnealingWarmRestarts expects epoch index

        # Record history
        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_acc"].append(round(val_acc, 6))

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), MODEL_OUT)

        elapsed = time.time() - t_start
        current_lr = optimizer.param_groups[0]["lr"]

        marker = " <-- best" if epoch == best_epoch else ""
        print(
            f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.4f}  "
            f"{val_loss:>8.4f}  {val_acc:>8.4f}  {current_lr:>10.2e}  "
            f"{elapsed:>5.1f}s{marker}"
        )

    print("-" * 70)
    print(f"\n[INFO] Training complete. Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"[INFO] Best model saved to: {MODEL_OUT}")

    # ------------------------------------------------------------------
    # 8. Save training history
    # ------------------------------------------------------------------
    history["best_epoch"]    = best_epoch
    history["best_val_acc"]  = round(best_val_acc, 6)
    history["backbone"]      = backbone_name
    history["epochs"]        = EPOCHS
    history["batch_size"]    = BATCH_SIZE

    with open(HISTORY_OUT, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] History saved to: {HISTORY_OUT}")

    # ------------------------------------------------------------------
    # 9. Plot training curves
    # ------------------------------------------------------------------
    plot_training_curves(history, best_epoch, PLOT_OUT)

    # ------------------------------------------------------------------
    # 10. Final evaluation on test set using best model
    # ------------------------------------------------------------------
    print(f"\n[INFO] Loading best model for final test evaluation...")
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    model.eval()

    all_labels, all_preds = get_all_predictions(model, test_loader, device)

    # Classification report
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

    # Confusion matrix (printed as text)
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"{'':12s} " + "  ".join(f"{s:>8s}" for s in SEASONS))
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8d}" for v in row)
        print(f"{SEASONS[i]:12s} {row_str}")

    # Highlight Autumn recall (hardest class per literature)
    autumn_idx = LABEL_MAP["autumn"]
    autumn_correct = cm[autumn_idx, autumn_idx]
    autumn_total   = cm[autumn_idx].sum()
    autumn_recall  = autumn_correct / autumn_total if autumn_total > 0 else 0.0
    print(f"\n[NOTE] Autumn recall: {autumn_recall:.4f} ({autumn_correct}/{autumn_total})")
    print("       (Autumn is historically the hardest class — track this carefully.)")

    # Save report to file
    with open(REPORT_OUT, "w") as f:
        f.write("ToneFit — FaRL Classifier — Test Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Backbone   : {backbone_name}\n")
        f.write(f"Best Epoch : {best_epoch}\n")
        f.write(f"Best Val Acc: {best_val_acc:.4f}\n")
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
    print("\n[DONE] FaRL training pipeline finished successfully.")


if __name__ == "__main__":
    main()
