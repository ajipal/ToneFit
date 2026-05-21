"""
train_farl_12class.py — ToneFit ML Project
============================================
Model B: FaRL-64 12-class Sub-Type Classifier
Backbone: CLIP ViT-B/16 loaded via OpenAI CLIP, patched with FaRL weights (frozen).
Head: flat FC 512 → 256 → 12 (one output per sub-type).

Derived 4-season accuracy: map each sub-type prediction back to its parent
season using subtype_to_season() — no separate season head needed.

Feature caching: backbone runs once, features saved to:
    results/features_train_12class.pt
    results/features_test_12class.pt
Each epoch trains only the small FC head (no ViT forward pass).

Requires:
    pip install git+https://github.com/openai/CLIP.git

Usage:
    python train_farl_12class.py
    python train_farl_12class.py --recache   # force re-extraction

Outputs:
    models/farl_12class_model.pth        — best checkpoint (by val 12-class acc)
    results/farl_12class_history.json    — per-epoch history
    results/farl_12class_training.png    — training curves
    results/farl_12class_report.txt      — final classification reports
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
MODEL_OUT    = "models/farl_12class_model.pth"
HISTORY_OUT  = "results/farl_12class_history.json"
PLOT_OUT     = "results/farl_12class_training.png"
REPORT_OUT   = "results/farl_12class_report.txt"

CACHE_TRAIN  = "results/features_train_12class.pt"
CACHE_TEST   = "results/features_test_12class.pt"

# 12 sub-type classes (sorted alphabetically — matches CLAUDE.md)
CLASS_NAMES_12 = sorted([
    "autumn_deep", "autumn_soft", "autumn_warm",
    "spring_bright", "spring_light", "spring_warm",
    "summer_cool", "summer_light", "summer_soft",
    "winter_bright", "winter_cool", "winter_deep",
])
CLASS_TO_IDX_12 = {c: i for i, c in enumerate(CLASS_NAMES_12)}

SEASONS      = ["autumn", "spring", "summer", "winter"]
SEASON_TO_IDX = {s: i for i, s in enumerate(SEASONS)}

FEATURE_DIM  = 512   # CLIP ViT-B/16 projected output
NUM_CLASSES  = 12

EPOCHS       = 50
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-5
T_0          = 10
ETA_MIN      = 1e-5

# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ---------------------------------------------------------------------------
# DATASET — SubTypeDataset (from CLAUDE.md spec)
# ---------------------------------------------------------------------------

class SubTypeDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
        # Walk two levels deep: season/subtype/
        for season in os.listdir(root):
            season_path = os.path.join(root, season)
            if not os.path.isdir(season_path):
                continue
            for subtype in os.listdir(season_path):
                subtype_path = os.path.join(season_path, subtype)
                if not os.path.isdir(subtype_path):
                    continue
                label = f"{season}_{subtype}"
                if label not in CLASS_TO_IDX_12:
                    continue
                for img_file in os.listdir(subtype_path):
                    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append(
                            (os.path.join(subtype_path, img_file),
                             CLASS_TO_IDX_12[label])
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# DATA TRANSFORMS (CLIP normalization)
# ---------------------------------------------------------------------------

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


# ---------------------------------------------------------------------------
# HELPER: derive season from sub-type label
# ---------------------------------------------------------------------------

def subtype_to_season(subtype_label: str) -> str:
    """'autumn_deep' → 'autumn'"""
    return subtype_label.split("_")[0]


def preds_to_season_preds(subtype_preds: np.ndarray) -> np.ndarray:
    """Convert array of sub-type indices to season indices."""
    return np.array([
        SEASON_TO_IDX[subtype_to_season(CLASS_NAMES_12[p])]
        for p in subtype_preds
    ])


def labels_to_season_labels(subtype_labels: np.ndarray) -> np.ndarray:
    """Convert array of sub-type ground-truth indices to season indices."""
    return np.array([
        SEASON_TO_IDX[subtype_to_season(CLASS_NAMES_12[l])]
        for l in subtype_labels
    ])


# ---------------------------------------------------------------------------
# BACKBONE LOADING
# ---------------------------------------------------------------------------

def load_farl_backbone():
    """
    Load CLIP ViT-B/16, patch with FaRL weights, freeze all params.
    Returns (visual_encoder, backbone_name_str).
    Falls back to ResNeXt50 if FaRL weights not found.
    """
    farl_path = Path(FARL_WEIGHTS)

    if farl_path.exists():
        print("[INFO] FaRL weights found. Loading CLIP ViT-B/16 backbone...")
        try:
            import clip
        except ImportError:
            raise ImportError(
                "CLIP is required. Install with:\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            )

        clip_model, _ = clip.load("ViT-B/16", device="cpu")
        farl_state = torch.load(str(farl_path), map_location="cpu")
        state_dict = farl_state.get("state_dict", farl_state)
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = clip_model.load_state_dict(cleaned, strict=False)
        print(f"[INFO] FaRL weights applied. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        backbone = clip_model.visual.float()
        for param in backbone.parameters():
            param.requires_grad = False

        return backbone, "FaRL (CLIP ViT-B/16, feature_dim=512)", FEATURE_DIM

    else:
        print(f"[WARNING] FaRL weights not found at '{FARL_WEIGHTS}'.")
        print("[INFO] Falling back to ResNeXt50-32x4d (ImageNet pretrained).")
        from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

        resnext = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        for param in resnext.parameters():
            param.requires_grad = False
        backbone = nn.Sequential(*list(resnext.children())[:-1])
        return backbone, "ResNeXt50-32x4d (ImageNet fallback)", 2048


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


def build_or_load_cache(backbone, backbone_name, train_loader, test_loader,
                        device, recache=False):
    need_cache = recache or not (
        os.path.exists(CACHE_TRAIN) and os.path.exists(CACHE_TEST)
    )
    is_resnext = "ResNeXt" in backbone_name

    if need_cache:
        print("[CACHE] Extracting backbone features (runs once)...")
        t0 = time.time()
        backbone.eval()
        tr_feats, tr_labels = _extract_features(backbone, train_loader, device, flatten=is_resnext)
        te_feats, te_labels = _extract_features(backbone, test_loader, device, flatten=is_resnext)
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

    feat_dim = tr_feats.shape[1]
    tr_ds = TensorDataset(tr_feats.to(device), tr_labels.to(device))
    te_ds = TensorDataset(te_feats.to(device), te_labels.to(device))
    return (
        DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False),
        feat_dim,
    )


# ---------------------------------------------------------------------------
# CLASSIFIER HEAD — 12 output classes
# ---------------------------------------------------------------------------

def build_head(feature_dim: int) -> nn.Sequential:
    """FC(feature_dim → 256) → ReLU → Dropout(0.5) → FC(256 → 12)"""
    return nn.Sequential(
        nn.Linear(feature_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES),
    )


# ---------------------------------------------------------------------------
# TOP-K ACCURACY
# ---------------------------------------------------------------------------

def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 3) -> float:
    """Fraction of samples where true label appears in top-k predictions."""
    topk = logits.topk(k, dim=1).indices          # [B, k]
    correct = topk.eq(labels.unsqueeze(1)).any(1)  # [B]
    return correct.float().mean().item()


# ---------------------------------------------------------------------------
# TRAINING HELPERS (cached — no backbone forward per epoch)
# ---------------------------------------------------------------------------

def train_one_epoch(head, loader, optimizer, criterion):
    head.train()
    total_loss = 0.0
    correct = total = 0
    top3_correct = 0

    for features, labels in loader:
        optimizer.zero_grad()
        logits = head(features)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = features.size(0)
        total_loss   += loss.item() * bs
        correct      += (logits.argmax(1) == labels).sum().item()
        top3_correct += sum(
            labels[i].item() in logits[i].topk(3).indices.tolist()
            for i in range(bs)
        )
        total += bs

    return total_loss / total, correct / total, top3_correct / total


@torch.no_grad()
def evaluate(head, loader, criterion):
    head.eval()
    total_loss = 0.0
    correct = total = 0
    top3_correct = 0

    for features, labels in loader:
        logits = head(features)
        loss   = criterion(logits, labels)

        bs = features.size(0)
        total_loss   += loss.item() * bs
        correct      += (logits.argmax(1) == labels).sum().item()
        top3_correct += sum(
            labels[i].item() in logits[i].topk(3).indices.tolist()
            for i in range(bs)
        )
        total += bs

    return total_loss / total, correct / total, top3_correct / total


@torch.no_grad()
def get_all_predictions(head, loader):
    head.eval()
    all_preds, all_labels = [], []
    all_logits = []

    for features, labels in loader:
        logits = head(features)
        all_logits.append(logits.cpu())
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (np.array(all_labels), np.array(all_preds),
            torch.cat(all_logits))


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, best_epoch: int, save_path: str):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("FaRL-64 12-Class Training Curves", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs_range, history["train_loss"], label="Train", color="steelblue")
    ax.plot(epochs_range, history["val_loss"],   label="Val",   color="coral")
    ax.axvline(x=best_epoch, color="green", linestyle="--", linewidth=1.5,
               label=f"Best ({best_epoch})")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs_range, history["train_acc"],  label="Train Top-1", color="steelblue")
    ax.plot(epochs_range, history["val_acc"],    label="Val Top-1",   color="coral")
    ax.plot(epochs_range, history["train_top3"], label="Train Top-3", color="steelblue",
            linestyle="--", alpha=0.7)
    ax.plot(epochs_range, history["val_top3"],   label="Val Top-3",   color="coral",
            linestyle="--", alpha=0.7)
    ax.axvline(x=best_epoch, color="green", linestyle="--", linewidth=1.5)
    ax.set_title("12-Class Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs_range, history["train_season_acc"], label="Train Season",
            color="steelblue")
    ax.plot(epochs_range, history["val_season_acc"],   label="Val Season",
            color="coral")
    ax.axvline(x=best_epoch, color="green", linestyle="--", linewidth=1.5)
    ax.set_title("Derived 4-Season Accuracy")
    ax.set_xlabel("Epoch")
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
    parser = argparse.ArgumentParser(description="Train FaRL-64 12-Class (Model B)")
    parser.add_argument("--recache", action="store_true",
                        help="Force re-extraction of backbone features")
    args, _ = parser.parse_known_args()  # ignore Jupyter kernel args

    print("=" * 60)
    print("  ToneFit — FaRL-64 12-Class Sub-Type Classifier (Model B)")
    print("=" * 60)

    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load datasets using SubTypeDataset (from CLAUDE.md spec)
    # ------------------------------------------------------------------
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

    train_dataset = SubTypeDataset(root=TRAIN_DIR, transform=val_transform)
    test_dataset  = SubTypeDataset(root=TEST_DIR,  transform=val_transform)

    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Test  samples: {len(test_dataset)}")

    train_counts = Counter(lbl for _, lbl in train_dataset.samples)
    print("\n[INFO] Train class distribution:")
    for i, name in enumerate(CLASS_NAMES_12):
        print(f"  {name:<22}: {train_counts[i]}")

    # ------------------------------------------------------------------
    # 2. Compute class weights for 12-class head
    # ------------------------------------------------------------------
    all_train_labels = [lbl for _, lbl in train_dataset.samples]
    class_weights = compute_class_weight(
        "balanced",
        classes=np.arange(NUM_CLASSES),
        y=np.array(all_train_labels),
    )
    print(f"\n[INFO] Class weights (12): {class_weights.round(3).tolist()}")
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
    # 4. Load backbone
    # ------------------------------------------------------------------
    print("\n[INFO] Loading backbone...")
    backbone, backbone_name, feat_dim = load_farl_backbone()
    backbone = backbone.to(device)

    # ------------------------------------------------------------------
    # 5. Cache backbone features
    # ------------------------------------------------------------------
    train_loader, test_loader, feat_dim = build_or_load_cache(
        backbone, backbone_name, train_loader_raw, test_loader_raw,
        device, recache=args.recache,
    )
    print(f"[INFO] Feature dim: {feat_dim}")

    # ------------------------------------------------------------------
    # 6. Build head, loss, optimizer, scheduler
    # ------------------------------------------------------------------
    head = build_head(feat_dim).to(device)

    trainable = sum(p.numel() for p in head.parameters())
    print(f"[INFO] Trainable head params: {trainable:,}")

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, eta_min=ETA_MIN
    )

    # ------------------------------------------------------------------
    # 7. Training loop — 50 epochs, no early stopping
    # ------------------------------------------------------------------
    print(f"\n[INFO] Starting training for {EPOCHS} epochs (head only)...")
    print(f"[INFO] Backbone: {backbone_name}")
    print("-" * 80)
    print(f"{'Ep':>4}  {'Loss':>8}  {'Tr Acc':>7}  {'Tr T3':>7}  "
          f"{'Val Acc':>8}  {'Val T3':>7}  {'Val Szn':>8}  {'LR':>10}  {'Time':>5}")
    print("-" * 80)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "train_top3": [], "val_top3": [],
        "train_season_acc": [], "val_season_acc": [],
    }

    best_val_acc = 0.0
    best_epoch   = 1

    for epoch in range(1, EPOCHS + 1):
        t_start = time.time()

        # Train
        tr_loss, tr_acc, tr_top3 = train_one_epoch(head, train_loader, optimizer, criterion)

        # Validate
        vl_loss, vl_acc, vl_top3 = evaluate(head, test_loader, criterion)

        # Derived season accuracy on val set
        val_true, val_pred, _ = get_all_predictions(head, test_loader)
        val_szn_true = labels_to_season_labels(val_true)
        val_szn_pred = preds_to_season_preds(val_pred)
        vl_szn_acc   = (val_szn_pred == val_szn_true).mean()

        # Same for train set (sampled — just for history)
        tr_true, tr_pred, _ = get_all_predictions(head, train_loader)
        tr_szn_true = labels_to_season_labels(tr_true)
        tr_szn_pred = preds_to_season_preds(tr_pred)
        tr_szn_acc  = (tr_szn_pred == tr_szn_true).mean()

        scheduler.step(epoch - 1)

        history["train_loss"].append(round(tr_loss, 6))
        history["val_loss"].append(round(vl_loss, 6))
        history["train_acc"].append(round(tr_acc, 6))
        history["val_acc"].append(round(vl_acc, 6))
        history["train_top3"].append(round(tr_top3, 6))
        history["val_top3"].append(round(vl_top3, 6))
        history["train_season_acc"].append(round(float(tr_szn_acc), 6))
        history["val_season_acc"].append(round(float(vl_szn_acc), 6))

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch   = epoch
            torch.save({
                "head":          head.state_dict(),
                "backbone_name": backbone_name,
                "feat_dim":      feat_dim,
                "epoch":         epoch,
                "val_acc":       best_val_acc,
                "val_top3":      vl_top3,
                "val_season_acc": float(vl_szn_acc),
            }, MODEL_OUT)

        elapsed    = time.time() - t_start
        current_lr = optimizer.param_groups[0]["lr"]
        marker     = " <--" if epoch == best_epoch else ""
        print(
            f"{epoch:>4}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {tr_top3:>7.4f}  "
            f"{vl_acc:>8.4f}  {vl_top3:>7.4f}  {vl_szn_acc:>8.4f}  "
            f"{current_lr:>10.2e}  {elapsed:>4.1f}s{marker}"
        )

        if epoch == 1:
            print(f"[INFO] Estimated total time: {elapsed * EPOCHS / 60:.1f} min")

    print("-" * 80)
    best_vl_top3    = history["val_top3"][best_epoch - 1]
    best_vl_szn_acc = history["val_season_acc"][best_epoch - 1]
    print(f"\n[INFO] Training complete.")
    print(f"[INFO] Best val 12-class acc : {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"[INFO] Top-3 acc at best     : {best_vl_top3:.4f}")
    print(f"[INFO] Derived season acc    : {best_vl_szn_acc:.4f}")
    print(f"[INFO] Best model saved to   : {MODEL_OUT}")

    # ------------------------------------------------------------------
    # 8. Save history
    # ------------------------------------------------------------------
    history["best_epoch"]          = best_epoch
    history["best_val_acc"]        = round(best_val_acc, 6)
    history["best_val_top3"]       = round(best_vl_top3, 6)
    history["best_val_season_acc"] = round(best_vl_szn_acc, 6)
    history["backbone"]            = backbone_name
    history["epochs"]              = EPOCHS
    history["batch_size"]          = BATCH_SIZE

    with open(HISTORY_OUT, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] History saved to: {HISTORY_OUT}")

    # ------------------------------------------------------------------
    # 9. Plot
    # ------------------------------------------------------------------
    plot_training_curves(history, best_epoch, PLOT_OUT)

    # ------------------------------------------------------------------
    # 10. Final classification reports on best checkpoint
    # ------------------------------------------------------------------
    print(f"\n[INFO] Loading best checkpoint for final evaluation...")
    ckpt = torch.load(MODEL_OUT, map_location=device)
    head.load_state_dict(ckpt["head"])

    all_true, all_pred, all_logits = get_all_predictions(head, test_loader)

    # Top-3 accuracy (paper standard)
    top3_acc = top_k_accuracy(all_logits, torch.from_numpy(all_true), k=3)

    # Derived season labels
    szn_true = labels_to_season_labels(all_true)
    szn_pred = preds_to_season_preds(all_pred)
    season_acc = (szn_pred == szn_true).mean()

    # 12-class report
    report_12 = classification_report(
        all_true, all_pred, target_names=CLASS_NAMES_12, digits=4)

    print("\n" + "=" * 60)
    print("  Final Test Report — Sub-Type (12-class)")
    print("=" * 60)
    print(f"  Backbone  : {backbone_name}")
    print(f"  Best Epoch: {best_epoch}  |  12-class Acc: {best_val_acc:.4f}")
    print(f"  Top-3 Acc : {top3_acc:.4f}  |  Derived Season Acc: {season_acc:.4f}")
    print("=" * 60)
    print(report_12)

    cm_12 = confusion_matrix(all_true, all_pred)
    print("Confusion Matrix — Sub-Type (rows=actual, cols=predicted):")
    print(f"{'':16s} " + " ".join(f"{s.split('_')[1]:>7s}" for s in CLASS_NAMES_12))
    for i, row in enumerate(cm_12):
        print(f"{CLASS_NAMES_12[i]:16s} " + " ".join(f"{v:>7d}" for v in row))

    # Autumn recall (hardest class — track separately per paper)
    autumn_classes = [i for i, n in enumerate(CLASS_NAMES_12) if n.startswith("autumn")]
    autumn_true_mask = np.isin(all_true, autumn_classes)
    autumn_correct   = (all_pred[autumn_true_mask] == all_true[autumn_true_mask]).sum()
    autumn_total     = autumn_true_mask.sum()
    autumn_recall    = autumn_correct / autumn_total if autumn_total > 0 else 0.0
    print(f"\n[NOTE] Autumn sub-types recall: {autumn_recall:.4f} ({autumn_correct}/{autumn_total})")
    print("       (Autumn is historically the hardest season — track this carefully.)")

    # Derived 4-season report
    season_report = classification_report(
        szn_true, szn_pred, target_names=SEASONS, digits=4)

    print("\n" + "=" * 60)
    print("  Derived 4-Season Report (from sub-type predictions)")
    print("=" * 60)
    print(f"  Season Acc: {season_acc:.4f}")
    print("=" * 60)
    print(season_report)

    szn_cm = confusion_matrix(szn_true, szn_pred)
    print("Confusion Matrix — Derived Season (rows=actual, cols=predicted):")
    print(f"{'':12s} " + "  ".join(f"{s:>8s}" for s in SEASONS))
    for i, row in enumerate(szn_cm):
        print(f"{SEASONS[i]:12s} " + "  ".join(f"{v:>8d}" for v in row))

    # ------------------------------------------------------------------
    # 11. Save report to file
    # ------------------------------------------------------------------
    with open(REPORT_OUT, "w") as f:
        f.write("ToneFit — FaRL-64 12-Class Sub-Type Classifier (Model B)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Backbone        : {backbone_name}\n")
        f.write(f"Best Epoch      : {best_epoch}\n")
        f.write(f"12-Class Acc    : {best_val_acc:.4f}\n")
        f.write(f"Top-3 Acc       : {top3_acc:.4f}\n")
        f.write(f"Derived Szn Acc : {season_acc:.4f}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Sub-Type Classification Report (12-class):\n")
        f.write(report_12)
        f.write(f"\nAutumn sub-types recall: {autumn_recall:.4f} ({autumn_correct}/{autumn_total})\n")
        f.write("\nDerived 4-Season Classification Report:\n")
        f.write(season_report)
        f.write("\nDerived Season Confusion Matrix:\n")
        f.write(f"{'':12s} " + "  ".join(f"{s:>8s}" for s in SEASONS) + "\n")
        for i, row in enumerate(szn_cm):
            f.write(f"{SEASONS[i]:12s} " + "  ".join(f"{v:>8d}" for v in row) + "\n")

    print(f"\n[INFO] Report saved to: {REPORT_OUT}")
    print("\n[DONE] FaRL-64 12-class training pipeline finished successfully.")


if __name__ == "__main__":
    main()
