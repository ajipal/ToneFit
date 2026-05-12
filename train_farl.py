"""
train_farl.py — ToneFit ML Project
====================================
Model A: FaRL (Face Representation Learning) Classifier
Backbone: CLIP ViT-B/16 loaded via OpenAI CLIP, then patched with FaRL weights.
         model.visual is used as the frozen feature extractor (feature_dim=512).
         Fallback: ResNeXt50 (torchvision, ImageNet pretrained).

Feature caching: backbone runs once before training, outputs saved to
results/features_train_farl.pt and results/features_test_farl.pt.
Each epoch only trains the small classifier head — no ViT forward pass per epoch.
Note: augmentation is not applied when caching (val_transform used for extraction).

Requires:
    pip install git+https://github.com/openai/CLIP.git

Usage:
    python train_farl.py
    python train_farl.py --recache   # force re-extraction of features

Outputs:
    models/farl_model.pth              — best model checkpoint (by val accuracy)
    results/features_train_farl.pt     — cached backbone features (train)
    results/features_test_farl.pt      — cached backbone features (test)
    results/farl_history.json          — per-epoch loss/accuracy history
    results/farl_training.png          — training curve plot
    results/farl_test_report.txt       — classification report on test set
"""

import argparse
import os
import json
import time
import warnings
import numpy as np
from collections import Counter
from pathlib import Path

import torch
import torch.backends.cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from PIL import Image

torch.backends.cudnn.benchmark = True

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

TRAIN_DIR    = "RGB-M/train"
TEST_DIR     = "RGB-M/test"
FARL_WEIGHTS = "models/farl_weights.pth"
MODEL_OUT    = "models/farl_model.pth"
HISTORY_OUT  = "results/farl_history.json"
PLOT_OUT     = "results/farl_training.png"
REPORT_OUT   = "results/farl_test_report.txt"

CACHE_TRAIN = "results/features_train_farl.pt"
CACHE_TEST  = "results/features_test_farl.pt"

LABEL_MAP = {"autumn": 0, "spring": 1, "summer": 2, "winter": 3}
SEASONS   = ["autumn", "spring", "summer", "winter"]
SEASON_TO_IDX = {s: i for i, s in enumerate(SEASONS)}

EPOCHS       = 50
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-5
T_0          = 10
ETA_MIN      = 1e-5

FARL_FEATURE_DIM    = 512
RESNEXT_FEATURE_DIM = 2048

# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

class FaRLDataset(Dataset):
    """
    Reads from <root>/season/subtype/ hierarchy.
    Returns (image_tensor, season_idx) — sub-types treated as same class.
    """

    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        for season_dir in sorted(Path(root).iterdir()):
            if not season_dir.is_dir():
                continue
            season_name = season_dir.name.lower()
            if season_name not in SEASON_TO_IDX:
                continue
            season_idx = SEASON_TO_IDX[season_name]

            for subtype_dir in sorted(season_dir.iterdir()):
                if not subtype_dir.is_dir():
                    continue
                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    for img_path in sorted(subtype_dir.glob(ext)):
                        self.samples.append((str(img_path), season_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, season_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, season_idx


# ---------------------------------------------------------------------------
# DATA TRANSFORMS
# ---------------------------------------------------------------------------

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

resnext_val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# FEATURE CACHING
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_features(backbone, loader, device, flatten=False):
    all_feats, all_labels = [], []
    for images, labels in loader:
        images = images.to(device).float()
        feats  = backbone(images)
        if flatten:
            feats = feats.view(feats.size(0), -1)
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


def build_or_load_cache(model, backbone_name, train_loader_raw, test_loader_raw,
                        device, recache=False):
    """Cache frozen backbone features once. Returns train/test TensorDataset loaders."""
    need_cache = recache or not (
        os.path.exists(CACHE_TRAIN) and os.path.exists(CACHE_TEST)
    )
    is_resnext = "ResNeXt" in backbone_name

    if need_cache:
        print("[CACHE] Extracting backbone features (runs once)...")
        t0 = time.time()
        model.backbone.eval()
        tr_feats, tr_labels = _extract_features(
            model.backbone, train_loader_raw, device, flatten=is_resnext)
        te_feats, te_labels = _extract_features(
            model.backbone, test_loader_raw, device, flatten=is_resnext)
        torch.save({"features": tr_feats, "labels": tr_labels}, CACHE_TRAIN)
        torch.save({"features": te_feats, "labels": te_labels}, CACHE_TEST)
        elapsed = time.time() - t0
        print(f"[CACHE] Done in {elapsed:.1f}s. "
              f"Train: {len(tr_feats)} | Test: {len(te_feats)} | "
              f"Feature dim: {tr_feats.shape[1]}")
    else:
        tr = torch.load(CACHE_TRAIN, map_location="cpu")
        te = torch.load(CACHE_TEST,  map_location="cpu")
        tr_feats, tr_labels = tr["features"], tr["labels"]
        te_feats, te_labels = te["features"], te["labels"]
        print(f"[CACHE] Loaded cached features. "
              f"Train: {len(tr_feats)} | Test: {len(te_feats)} | "
              f"Feature dim: {tr_feats.shape[1]}")

    tr_ds = TensorDataset(tr_feats.to(device), tr_labels.to(device))
    te_ds = TensorDataset(te_feats.to(device), te_labels.to(device))
    return (
        DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False),
    )


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


class FaRLModel(nn.Module):
    """CLIP ViT-B/16 + FaRL weights → frozen backbone → 4-class season head."""

    def __init__(self, visual_backbone, feature_dim: int = FARL_FEATURE_DIM):
        super().__init__()
        self.backbone = visual_backbone
        self.head = build_classifier_head(feature_dim)

    def forward(self, x):
        return self.head(self.backbone(x))


class ResNeXtFallbackModel(nn.Module):
    """ResNeXt50-32x4d backbone + 4-class season head (fallback)."""

    def __init__(self, resnext_backbone, feature_dim: int = RESNEXT_FEATURE_DIM):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnext_backbone.children())[:-1])
        self.head = build_classifier_head(feature_dim)

    def forward(self, x):
        features = self.backbone(x).view(x.size(0), -1)
        return self.head(features)


def load_model():
    farl_path = Path(FARL_WEIGHTS)

    if farl_path.exists():
        print("[INFO] FaRL weights found. Loading CLIP ViT-B/16 backbone...")
        try:
            import clip
        except ImportError:
            raise ImportError(
                "CLIP is required for FaRL. Install it with:\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            )

        clip_model, _ = clip.load("ViT-B/16", device="cpu")
        farl_state = torch.load(str(farl_path), map_location="cpu")
        state_dict = farl_state.get("state_dict", farl_state)
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = clip_model.load_state_dict(cleaned, strict=False)
        print(f"[INFO] FaRL weights applied. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        vit = clip_model.visual.float()
        for param in vit.parameters():
            param.requires_grad = False

        model = FaRLModel(vit, feature_dim=FARL_FEATURE_DIM)
        backbone_name = "FaRL (CLIP ViT-B/16, feature_dim=512)"

    else:
        print(f"[WARNING] FaRL weights not found at '{FARL_WEIGHTS}'.")
        print("[INFO] Falling back to ResNeXt50-32x4d (ImageNet pretrained).")
        from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

        resnext = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        for param in resnext.parameters():
            param.requires_grad = False

        model = ResNeXtFallbackModel(resnext, feature_dim=RESNEXT_FEATURE_DIM)
        backbone_name = "ResNeXt50-32x4d (ImageNet fallback)"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Backbone: {backbone_name}")
    print(f"[INFO] Trainable params: {trainable:,} / {total:,}")
    return model, backbone_name


# ---------------------------------------------------------------------------
# TRAINING HELPERS (cached — no backbone forward per epoch)
# ---------------------------------------------------------------------------

def train_one_epoch_cached(model, loader, optimizer, criterion):
    model.head.train()
    total_loss = correct = total = 0

    for features, labels in loader:
        optimizer.zero_grad()
        logits = model.head(features)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs          = features.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += bs

    return total_loss / total, correct / total


def evaluate_cached(model, loader, criterion):
    model.head.eval()
    total_loss = correct = total = 0

    with torch.no_grad():
        for features, labels in loader:
            logits = model.head(features)
            loss   = criterion(logits, labels)

            bs          = features.size(0)
            total_loss += loss.item() * bs
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += bs

    return total_loss / total, correct / total


def get_all_predictions_cached(model, loader):
    model.head.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in loader:
            logits = model.head(features)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, best_epoch: int, save_path: str):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("FaRL Training Curves", fontsize=14, fontweight="bold")

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
    parser = argparse.ArgumentParser(description="Train FaRL Model A")
    parser.add_argument("--recache", action="store_true",
                        help="Force re-extraction of backbone features")
    args, _ = parser.parse_known_args()  # ignore Jupyter kernel args

    print("=" * 60)
    print("  ToneFit — FaRL Personal Color Classifier (Model A)")
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

    train_dataset = FaRLDataset(root=TRAIN_DIR, transform=val_transform)
    test_dataset  = FaRLDataset(root=TEST_DIR,  transform=val_transform)

    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Test  samples: {len(test_dataset)}")

    train_counts = Counter(szn for _, szn in train_dataset.samples)
    print("\n[INFO] Train class distribution:")
    for i, s in enumerate(SEASONS):
        print(f"  {s:<8}: {train_counts[i]}")

    # ------------------------------------------------------------------
    # 2. Compute class weights
    # ------------------------------------------------------------------
    train_labels  = [szn for _, szn in train_dataset.samples]
    class_weights = compute_class_weight("balanced", classes=np.arange(4),
                                         y=np.array(train_labels))
    print(f"\n[INFO] Class weights: {dict(zip(SEASONS, class_weights.round(4)))}")
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # ------------------------------------------------------------------
    # 3. Raw loaders for feature extraction
    # ------------------------------------------------------------------
    train_loader_raw = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    test_loader_raw = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------------
    print("\n[INFO] Building model...")
    model, backbone_name = load_model()
    model = model.to(device)

    # ------------------------------------------------------------------
    # 5. Cache backbone features (backbone forward runs only once)
    # ------------------------------------------------------------------
    train_loader, test_loader = build_or_load_cache(
        model, backbone_name, train_loader_raw, test_loader_raw,
        device, recache=args.recache,
    )

    # ------------------------------------------------------------------
    # 6. Loss, optimizer, scheduler (head parameters only)
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, eta_min=ETA_MIN
    )

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    print(f"\n[INFO] Starting training for {EPOCHS} epochs (head only)...")
    print(f"[INFO] Backbone: {backbone_name}")
    print("-" * 70)
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>8}  {'LR':>10}  {'Time':>6}")
    print("-" * 70)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch   = 1

    for epoch in range(1, EPOCHS + 1):
        t_start = time.time()

        train_loss, train_acc = train_one_epoch_cached(model, train_loader, optimizer, criterion)
        val_loss,   val_acc   = evaluate_cached(model, test_loader, criterion)

        scheduler.step(epoch - 1)

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_acc"].append(round(val_acc, 6))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({"backbone": model.backbone.state_dict(),
                        "head":     model.head.state_dict()}, MODEL_OUT)

        elapsed    = time.time() - t_start
        current_lr = optimizer.param_groups[0]["lr"]
        marker     = " <-- best" if epoch == best_epoch else ""
        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.4f}  "
              f"{val_loss:>8.4f}  {val_acc:>8.4f}  {current_lr:>10.2e}  "
              f"{elapsed:>5.1f}s{marker}")

        if epoch == 1:
            print(f"[INFO] Estimated total time: {elapsed * EPOCHS / 60:.1f} min")

    print("-" * 70)
    print(f"\n[INFO] Training complete. Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"[INFO] Best model saved to: {MODEL_OUT}")

    # ------------------------------------------------------------------
    # 8. Save history
    # ------------------------------------------------------------------
    history["best_epoch"]   = best_epoch
    history["best_val_acc"] = round(best_val_acc, 6)
    history["backbone"]     = backbone_name
    history["epochs"]       = EPOCHS
    history["batch_size"]   = BATCH_SIZE

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
    checkpoint = torch.load(MODEL_OUT, map_location=device)
    model.head.load_state_dict(checkpoint["head"])

    all_labels, all_preds = get_all_predictions_cached(model, test_loader)

    report_str = classification_report(all_labels, all_preds, target_names=SEASONS, digits=4)

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
        print(f"{SEASONS[i]:12s} " + "  ".join(f"{v:>8d}" for v in row))

    autumn_idx     = LABEL_MAP["autumn"]
    autumn_correct = cm[autumn_idx, autumn_idx]
    autumn_total   = cm[autumn_idx].sum()
    autumn_recall  = autumn_correct / autumn_total if autumn_total > 0 else 0.0
    print(f"\n[NOTE] Autumn recall: {autumn_recall:.4f} ({autumn_correct}/{autumn_total})")
    print("       (Autumn is historically the hardest class — track this carefully.)")

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
            f.write(f"{SEASONS[i]:12s} " + "  ".join(f"{v:>8d}" for v in row) + "\n")
        f.write(f"\nAutumn recall: {autumn_recall:.4f} ({autumn_correct}/{autumn_total})\n")

    print(f"\n[INFO] Test report saved to: {REPORT_OUT}")
    print("\n[DONE] FaRL training pipeline finished successfully.")


if __name__ == "__main__":
    main()
