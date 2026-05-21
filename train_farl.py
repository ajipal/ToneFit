"""
train_farl.py — ToneFit ML Project
====================================
Model A: FaRL (Face Representation Learning) Flat Baseline
Backbone: CLIP ViT-B/16 loaded via OpenAI CLIP, then patched with FaRL weights.
         model.visual is used as the frozen feature extractor (feature_dim=512).
         Fallback: ResNeXt50 (torchvision, ImageNet pretrained).

Architecture (flat two-head baseline — matches the original paper):
    backbone (512-d) → shared FC: 512→256, ReLU, Dropout(0.5)
        ├── season_head:  FC 256 → 4   (Season, 4-class)
        └── subtype_head: FC 256 → 12  (Sub-Type, 12-class)

Feature caching: backbone runs once before training, outputs saved to
results/features_train_farl.pt and results/features_test_farl.pt.
Each epoch only trains the small classifier heads — no ViT forward pass.

Requires:
    pip install git+https://github.com/openai/CLIP.git

Usage:
    python train_farl.py
    python train_farl.py --recache   # force re-extraction of features

Outputs:
    models/farl_model.pth              — best model checkpoint (by val season accuracy)
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

SEASONS = ["autumn", "spring", "summer", "winter"]
SUBTYPE_CLASSES = [
    "autumn_deep", "autumn_soft", "autumn_warm",
    "spring_bright", "spring_light", "spring_warm",
    "summer_cool", "summer_light", "summer_soft",
    "winter_bright", "winter_cool", "winter_deep",
]
SEASON_TO_IDX  = {s: i for i, s in enumerate(SEASONS)}
SUBTYPE_TO_IDX = {s: i for i, s in enumerate(SUBTYPE_CLASSES)}

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
    Returns (image_tensor, season_idx, subtype_idx).
    """

    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples: list[tuple[str, int, int]] = []

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
                key = f"{season_name}_{subtype_dir.name.lower()}"
                if key not in SUBTYPE_TO_IDX:
                    continue
                subtype_idx = SUBTYPE_TO_IDX[key]

                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    for img_path in sorted(subtype_dir.glob(ext)):
                        self.samples.append((str(img_path), season_idx, subtype_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, season_idx, subtype_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, season_idx, subtype_idx


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
    all_feats, all_seasons, all_subtypes = [], [], []
    for images, season_labels, subtype_labels in loader:
        images = images.to(device).float()
        feats  = backbone(images)
        if flatten:
            feats = feats.view(feats.size(0), -1)
        all_feats.append(feats.cpu())
        all_seasons.append(season_labels)
        all_subtypes.append(subtype_labels)
    return torch.cat(all_feats), torch.cat(all_seasons), torch.cat(all_subtypes)


def build_or_load_cache(model, backbone_name, train_loader, test_loader,
                        device, recache=False):
    """Cache frozen backbone features once. Returns train/test cached loaders."""
    def _cache_valid(path):
        if not os.path.exists(path):
            return False
        d = torch.load(path, map_location="cpu", weights_only=False)
        return "season_labels" in d and "subtype_labels" in d

    need_cache = recache or not (_cache_valid(CACHE_TRAIN) and _cache_valid(CACHE_TEST))
    is_resnext = "ResNeXt" in backbone_name

    if need_cache:
        print("[CACHE] Extracting backbone features (runs once)...")
        t0 = time.time()
        model.backbone.eval()
        tr_feats, tr_szn, tr_sub = _extract_features(
            model.backbone, train_loader, device, flatten=is_resnext)
        te_feats, te_szn, te_sub = _extract_features(
            model.backbone, test_loader, device, flatten=is_resnext)
        torch.save({"features": tr_feats, "season_labels": tr_szn,
                    "subtype_labels": tr_sub}, CACHE_TRAIN)
        torch.save({"features": te_feats, "season_labels": te_szn,
                    "subtype_labels": te_sub}, CACHE_TEST)
        elapsed = time.time() - t0
        print(f"[CACHE] Done in {elapsed:.1f}s. "
              f"Train: {len(tr_feats)} | Test: {len(te_feats)} | "
              f"Feature dim: {tr_feats.shape[1]}")
    else:
        tr = torch.load(CACHE_TRAIN, map_location="cpu", weights_only=False)
        te = torch.load(CACHE_TEST,  map_location="cpu", weights_only=False)
        tr_feats, tr_szn, tr_sub = tr["features"], tr["season_labels"], tr["subtype_labels"]
        te_feats, te_szn, te_sub = te["features"], te["season_labels"], te["subtype_labels"]
        print(f"[CACHE] Loaded cached features. "
              f"Train: {len(tr_feats)} | Test: {len(te_feats)} | "
              f"Feature dim: {tr_feats.shape[1]}")

    tr_ds = TensorDataset(tr_feats.to(device), tr_szn.to(device), tr_sub.to(device))
    te_ds = TensorDataset(te_feats.to(device), te_szn.to(device), te_sub.to(device))
    return (
        DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False),
    )


# ---------------------------------------------------------------------------
# MODEL — flat two-head baseline (matches original paper)
# ---------------------------------------------------------------------------

class FaRLModel(nn.Module):
    """
    CLIP ViT-B/16 + FaRL weights → frozen backbone → flat two-head classifier.
    Flat = season and sub-type heads independently connected to shared features.
    This matches the original paper's baseline architecture.
    """

    def __init__(self, visual_backbone, feature_dim: int = FARL_FEATURE_DIM):
        super().__init__()
        self.backbone = visual_backbone
        shared_dim = feature_dim // 2
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.season_head  = nn.Linear(shared_dim, 4)
        self.subtype_head = nn.Linear(shared_dim, 12)

    def forward(self, x):
        features = self.backbone(x)
        shared   = self.shared(features)
        return self.season_head(shared), self.subtype_head(shared)


class ResNeXtFallbackModel(nn.Module):
    """ResNeXt50-32x4d backbone + flat two-head classifier (fallback)."""

    def __init__(self, resnext_backbone, feature_dim: int = RESNEXT_FEATURE_DIM):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnext_backbone.children())[:-1])
        shared_dim = feature_dim // 2
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.season_head  = nn.Linear(shared_dim, 4)
        self.subtype_head = nn.Linear(shared_dim, 12)

    def forward(self, x):
        features = self.backbone(x).view(x.size(0), -1)
        shared   = self.shared(features)
        return self.season_head(shared), self.subtype_head(shared)


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
        farl_state = torch.load(str(farl_path), map_location="cpu", weights_only=False)
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

def train_one_epoch_cached(model, loader, optimizer, crit_szn, crit_sub):
    model.shared.train()
    model.season_head.train()
    model.subtype_head.train()

    total_loss = 0.0
    correct_szn = correct_sub = total = 0

    for features, season_labels, subtype_labels in loader:
        optimizer.zero_grad()
        shared         = model.shared(features)
        szn_logits     = model.season_head(shared)
        sub_logits     = model.subtype_head(shared)
        loss = crit_szn(szn_logits, season_labels) + crit_sub(sub_logits, subtype_labels)
        loss.backward()
        optimizer.step()

        bs = features.size(0)
        total_loss  += loss.item() * bs
        correct_szn += (szn_logits.argmax(1) == season_labels).sum().item()
        correct_sub += (sub_logits.argmax(1) == subtype_labels).sum().item()
        total       += bs

    return total_loss / total, correct_szn / total, correct_sub / total


def evaluate_cached(model, loader, crit_szn, crit_sub):
    model.shared.eval()
    model.season_head.eval()
    model.subtype_head.eval()

    total_loss = 0.0
    correct_szn = correct_sub = total = 0

    with torch.no_grad():
        for features, season_labels, subtype_labels in loader:
            shared     = model.shared(features)
            szn_logits = model.season_head(shared)
            sub_logits = model.subtype_head(shared)
            loss = crit_szn(szn_logits, season_labels) + crit_sub(sub_logits, subtype_labels)

            bs = features.size(0)
            total_loss  += loss.item() * bs
            correct_szn += (szn_logits.argmax(1) == season_labels).sum().item()
            correct_sub += (sub_logits.argmax(1) == subtype_labels).sum().item()
            total       += bs

    return total_loss / total, correct_szn / total, correct_sub / total


def get_all_predictions_cached(model, loader):
    model.shared.eval()
    model.season_head.eval()
    model.subtype_head.eval()

    szn_true, szn_pred = [], []
    sub_true, sub_pred = [], []

    with torch.no_grad():
        for features, season_labels, subtype_labels in loader:
            shared     = model.shared(features)
            szn_logits = model.season_head(shared)
            sub_logits = model.subtype_head(shared)

            szn_pred.extend(szn_logits.argmax(1).cpu().numpy())
            sub_pred.extend(sub_logits.argmax(1).cpu().numpy())
            szn_true.extend(season_labels.cpu().numpy())
            sub_true.extend(subtype_labels.cpu().numpy())

    return (np.array(szn_true), np.array(szn_pred),
            np.array(sub_true), np.array(sub_pred))


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, best_epoch: int, save_path: str):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("FaRL Baseline Training Curves", fontsize=14, fontweight="bold")

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
    ax.plot(epochs_range, history["train_season_acc"], label="Train Season", color="steelblue")
    ax.plot(epochs_range, history["val_season_acc"],   label="Val Season",   color="coral")
    ax.axvline(x=best_epoch, color="green", linestyle="--", linewidth=1.5,
               label=f"Best epoch ({best_epoch})")
    ax.set_title("Season Accuracy (4-class)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs_range, history["train_subtype_acc"], label="Train Sub-Type", color="steelblue")
    ax.plot(epochs_range, history["val_subtype_acc"],   label="Val Sub-Type",   color="coral")
    ax.axvline(x=best_epoch, color="green", linestyle="--", linewidth=1.5,
               label=f"Best epoch ({best_epoch})")
    ax.set_title("Sub-Type Accuracy (12-class)")
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
    parser = argparse.ArgumentParser(description="Train FaRL Model A (flat baseline)")
    parser.add_argument("--recache", action="store_true",
                        help="Force re-extraction of backbone features")
    args, _ = parser.parse_known_args()  # ignore Jupyter kernel args

    print("=" * 60)
    print("  ToneFit — FaRL Flat Baseline (Model A)")
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

    train_counts = Counter(sub for _, _, sub in train_dataset.samples)
    print("\n[INFO] Train sub-type distribution:")
    for i, name in enumerate(SUBTYPE_CLASSES):
        print(f"  {name:<22}: {train_counts[i]}")

    # ------------------------------------------------------------------
    # 2. Compute class weights
    # ------------------------------------------------------------------
    szn_labels_list = [szn for _, szn, _ in train_dataset.samples]
    sub_labels_list = [sub for _, _, sub in train_dataset.samples]

    szn_weights = compute_class_weight("balanced",
                                       classes=np.arange(4),
                                       y=np.array(szn_labels_list))
    sub_weights = compute_class_weight("balanced",
                                       classes=np.arange(12),
                                       y=np.array(sub_labels_list))

    print(f"\n[INFO] Season class weights:  {dict(zip(SEASONS, szn_weights.round(4)))}")
    szn_weight_tensor = torch.tensor(szn_weights, dtype=torch.float32).to(device)
    sub_weight_tensor = torch.tensor(sub_weights, dtype=torch.float32).to(device)

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
    # 5. Cache backbone features
    # ------------------------------------------------------------------
    train_loader, test_loader = build_or_load_cache(
        model, backbone_name, train_loader_raw, test_loader_raw,
        device, recache=args.recache,
    )

    # ------------------------------------------------------------------
    # 6. Loss, optimizer, scheduler (head parameters only)
    # ------------------------------------------------------------------
    crit_szn = nn.CrossEntropyLoss(weight=szn_weight_tensor)
    crit_sub = nn.CrossEntropyLoss(weight=sub_weight_tensor)

    head_params = list(model.shared.parameters()) + \
                  list(model.season_head.parameters()) + \
                  list(model.subtype_head.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, eta_min=ETA_MIN
    )

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    print(f"\n[INFO] Starting training for {EPOCHS} epochs (head only)...")
    print(f"[INFO] Backbone: {backbone_name}")
    print("-" * 80)
    print(f"{'Epoch':>6}  {'Loss':>8}  {'Tr Szn':>7}  {'Tr Sub':>7}  "
          f"{'Val Szn':>8}  {'Val Sub':>8}  {'LR':>10}  {'Time':>6}")
    print("-" * 80)

    history = {
        "train_loss": [], "val_loss": [],
        "train_season_acc": [], "val_season_acc": [],
        "train_subtype_acc": [], "val_subtype_acc": [],
    }

    best_val_acc = 0.0
    best_epoch   = 1

    for epoch in range(1, EPOCHS + 1):
        t_start = time.time()

        tr_loss, tr_szn, tr_sub = train_one_epoch_cached(
            model, train_loader, optimizer, crit_szn, crit_sub)
        vl_loss, vl_szn, vl_sub = evaluate_cached(
            model, test_loader, crit_szn, crit_sub)

        scheduler.step(epoch - 1)

        history["train_loss"].append(round(tr_loss, 6))
        history["val_loss"].append(round(vl_loss, 6))
        history["train_season_acc"].append(round(tr_szn, 6))
        history["val_season_acc"].append(round(vl_szn, 6))
        history["train_subtype_acc"].append(round(tr_sub, 6))
        history["val_subtype_acc"].append(round(vl_sub, 6))

        if vl_szn > best_val_acc:
            best_val_acc = vl_szn
            best_epoch   = epoch
            torch.save({
                "backbone":     model.backbone.state_dict(),
                "shared":       model.shared.state_dict(),
                "season_head":  model.season_head.state_dict(),
                "subtype_head": model.subtype_head.state_dict(),
            }, MODEL_OUT)

        elapsed    = time.time() - t_start
        current_lr = optimizer.param_groups[0]["lr"]
        marker     = " <-- best" if epoch == best_epoch else ""
        print(
            f"{epoch:>6}  {tr_loss:>8.4f}  {tr_szn:>7.4f}  {tr_sub:>7.4f}  "
            f"{vl_szn:>8.4f}  {vl_sub:>8.4f}  {current_lr:>10.2e}  "
            f"{elapsed:>5.1f}s{marker}"
        )

        if epoch == 1:
            print(f"[INFO] Estimated total time: {elapsed * EPOCHS / 60:.1f} min")

    print("-" * 80)
    print(f"\n[INFO] Training complete.")
    print(f"[INFO] Best val season accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    best_sub_at_best = history["val_subtype_acc"][best_epoch - 1]
    print(f"[INFO] Sub-type accuracy at best epoch: {best_sub_at_best:.4f}")
    print(f"[INFO] Best model saved to: {MODEL_OUT}")

    # ------------------------------------------------------------------
    # 8. Save history
    # ------------------------------------------------------------------
    history["best_epoch"]           = best_epoch
    history["best_val_acc"]         = round(best_val_acc, 6)       # season (kept for compat)
    history["best_val_subtype_acc"] = round(best_sub_at_best, 6)
    history["backbone"]             = backbone_name
    history["epochs"]               = EPOCHS
    history["batch_size"]           = BATCH_SIZE

    with open(HISTORY_OUT, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] History saved to: {HISTORY_OUT}")

    # ------------------------------------------------------------------
    # 9. Plot
    # ------------------------------------------------------------------
    plot_training_curves(history, best_epoch, PLOT_OUT)

    # ------------------------------------------------------------------
    # 10. Final classification reports
    # ------------------------------------------------------------------
    print(f"\n[INFO] Loading best model for final test evaluation...")
    ckpt = torch.load(MODEL_OUT, map_location=device, weights_only=False)
    model.shared.load_state_dict(ckpt["shared"])
    model.season_head.load_state_dict(ckpt["season_head"])
    model.subtype_head.load_state_dict(ckpt["subtype_head"])

    szn_true, szn_pred, sub_true, sub_pred = get_all_predictions_cached(model, test_loader)

    # Season report
    season_report = classification_report(
        szn_true, szn_pred, target_names=SEASONS, digits=4)

    print("\n" + "=" * 60)
    print("  Final Test Classification Report — Season (4-class)")
    print("=" * 60)
    print(f"  Backbone: {backbone_name}")
    print(f"  Best Epoch: {best_epoch}  |  Best Season Acc: {best_val_acc:.4f}")
    print("=" * 60)
    print(season_report)

    szn_cm = confusion_matrix(szn_true, szn_pred)
    print("Confusion Matrix — Season (rows=actual, cols=predicted):")
    print(f"{'':12s} " + "  ".join(f"{s:>8s}" for s in SEASONS))
    for i, row in enumerate(szn_cm):
        print(f"{SEASONS[i]:12s} " + "  ".join(f"{v:>8d}" for v in row))

    autumn_idx     = SEASONS.index("autumn")
    aut_correct    = szn_cm[autumn_idx, autumn_idx]
    aut_total      = szn_cm[autumn_idx].sum()
    print(f"\n[NOTE] Autumn recall: {aut_correct/aut_total:.4f} ({aut_correct}/{aut_total})")
    print("       (Autumn is historically the hardest class — track this carefully.)")

    # Sub-type report
    subtype_report = classification_report(
        sub_true, sub_pred, target_names=SUBTYPE_CLASSES, digits=4)

    print("\n" + "=" * 60)
    print("  Final Test Classification Report — Sub-Type (12-class)")
    print("=" * 60)
    print(f"  Best Epoch: {best_epoch}  |  Sub-Type Acc: {best_sub_at_best:.4f}")
    print("=" * 60)
    print(subtype_report)

    sub_cm = confusion_matrix(sub_true, sub_pred)
    print("Confusion Matrix — Sub-Type (rows=actual, cols=predicted):")
    print(f"{'':16s} " + " ".join(f"{s.split('_')[1]:>7s}" for s in SUBTYPE_CLASSES))
    for i, row in enumerate(sub_cm):
        print(f"{SUBTYPE_CLASSES[i]:16s} " + " ".join(f"{v:>7d}" for v in row))

    # Save report
    with open(REPORT_OUT, "w") as f:
        f.write("ToneFit — FaRL Flat Baseline (Model A) — Test Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Backbone   : {backbone_name}\n")
        f.write(f"Best Epoch : {best_epoch}\n")
        f.write(f"Best Season Acc: {best_val_acc:.4f}\n")
        f.write(f"SubType Acc    : {best_sub_at_best:.4f}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Season Classification Report:\n")
        f.write(season_report)
        f.write("\nSeason Confusion Matrix (rows=actual, cols=predicted):\n")
        f.write(f"{'':12s} " + "  ".join(f"{s:>8s}" for s in SEASONS) + "\n")
        for i, row in enumerate(szn_cm):
            f.write(f"{SEASONS[i]:12s} " + "  ".join(f"{v:>8d}" for v in row) + "\n")
        f.write(f"\n[NOTE] Autumn recall: {aut_correct/aut_total:.4f} ({aut_correct}/{aut_total})\n")
        f.write("\n\nSub-Type Classification Report:\n")
        f.write(subtype_report)
        f.write("\nSub-Type Confusion Matrix (rows=actual, cols=predicted):\n")
        f.write(f"{'':16s} " + " ".join(f"{s.split('_')[1]:>7s}" for s in SUBTYPE_CLASSES) + "\n")
        for i, row in enumerate(sub_cm):
            f.write(f"{SUBTYPE_CLASSES[i]:16s} " + " ".join(f"{v:>7d}" for v in row) + "\n")

    print(f"\n[INFO] Test report saved to: {REPORT_OUT}")
    print("\n[DONE] FaRL training pipeline finished successfully.")


if __name__ == "__main__":
    main()
