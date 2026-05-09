"""
train.py — Hierarchical Deep Armocromia Training (feature-cached)
-----------------------------------------------------------------
Trains DeepArmocromiaHierarchical (frozen FaRL-64 backbone + two-stage head)
on the RGB-M dataset using AdamW + CosineAnnealingWarmRestarts.

Feature caching: backbone runs once before training starts, outputs are saved
to results/features_{train,val}.pt. Subsequent runs load the cache directly,
making each epoch ~10× faster (head-only forward/backward, no ViT pass).

Usage:
    python train.py --config configs/hierarchical.yaml
    python train.py --config configs/hierarchical.yaml --resume results/checkpoint_epoch_25.pth
    python train.py --config configs/hierarchical.yaml --recache   # force rebuild cache

Outputs:
    results/features_train.pt         — cached backbone features (train set)
    results/features_val.pt           — cached backbone features (val/test set)
    results/best_hierarchical.pth     — best checkpoint (by val season accuracy)
    results/checkpoint_epoch_{N}.pth  — per-epoch checkpoint with full optimizer state
    results/hierarchical_history.json — full loss/accuracy history
"""

import argparse
import glob
import json
import os
import shutil
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from PIL import Image
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from hierarchical_head import DeepArmocromiaHierarchical, HierarchicalLoss

torch.backends.cudnn.benchmark = True

# ── Label definitions ─────────────────────────────────────────────────────────

SEASON_CLASSES = ["autumn", "spring", "summer", "winter"]
SUBTYPE_CLASSES = [
    "autumn_deep", "autumn_soft", "autumn_warm",
    "spring_bright", "spring_light", "spring_warm",
    "summer_cool", "summer_light", "summer_soft",
    "winter_bright", "winter_cool", "winter_deep",
]
SEASON_TO_IDX  = {s: i for i, s in enumerate(SEASON_CLASSES)}
SUBTYPE_TO_IDX = {s: i for i, s in enumerate(SUBTYPE_CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]



# ── Dataset ───────────────────────────────────────────────────────────────────

class ArmocromiaDataset(Dataset):
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


# ── Transforms ────────────────────────────────────────────────────────────────

def build_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Feature caching ───────────────────────────────────────────────────────────

def _extract_features(backbone, dataset, device, batch_size):
    """
    Run backbone over all images in dataset (no augmentation).
    Returns (features [N,512], season_labels [N], subtype_labels [N]).
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    all_feats, all_seasons, all_subtypes = [], [], []

    backbone.eval()
    with torch.no_grad():
        for imgs, season_lbl, subtype_lbl in loader:
            feats = backbone(imgs.to(device))
            all_feats.append(feats.cpu())
            all_seasons.append(season_lbl)
            all_subtypes.append(subtype_lbl)

    return (
        torch.cat(all_feats,    dim=0),
        torch.cat(all_seasons,  dim=0),
        torch.cat(all_subtypes, dim=0),
    )


def build_or_load_cache(model, train_ds, val_ds, out_dir, device, batch_size,
                        recache=False):
    """
    Cache backbone features for train and val sets.
    Creates cache files if missing (or recache=True), otherwise loads them.

    Returns:
        train_cache, val_cache — each a dict with keys:
            'features' [N,512], 'season_labels' [N], 'subtype_labels' [N]
    """
    cache_train = os.path.join(out_dir, "features_train.pt")
    cache_val   = os.path.join(out_dir, "features_val.pt")

    need_cache = recache or not (
        os.path.exists(cache_train) and os.path.exists(cache_val)
    )

    if need_cache:
        print("[CACHE] Extracting backbone features (runs once)...")
        t0 = time.time()

        val_tf = build_val_transform()
        train_cache_ds = _DatasetFromSamples(train_ds.samples, val_tf)
        val_cache_ds   = _DatasetFromSamples(val_ds.samples,   val_tf)

        tr_feats, tr_szn, tr_sub = _extract_features(
            model.backbone, train_cache_ds, device, batch_size
        )
        vl_feats, vl_szn, vl_sub = _extract_features(
            model.backbone, val_cache_ds, device, batch_size
        )

        train_payload = {
            "features":       tr_feats,
            "season_labels":  tr_szn,
            "subtype_labels": tr_sub,
        }
        val_payload = {
            "features":       vl_feats,
            "season_labels":  vl_szn,
            "subtype_labels": vl_sub,
        }
        torch.save(train_payload, cache_train)
        torch.save(val_payload,   cache_val)

        elapsed = time.time() - t0
        print(f"[CACHE] Features cached in {elapsed:.1f}s. "
              f"Train: {len(tr_feats)} samples, Val: {len(vl_feats)} samples")
    else:
        train_payload = torch.load(cache_train, map_location="cpu")
        val_payload   = torch.load(cache_val,   map_location="cpu")
        print(f"[CACHE] Loaded cached features. "
              f"Train: {len(train_payload['features'])} samples, "
              f"Val: {len(val_payload['features'])} samples")

    return train_payload, val_payload


class _DatasetFromSamples(Dataset):
    """Minimal dataset that reuses pre-scanned sample list with a given transform."""

    def __init__(self, samples: list[tuple[str, int, int]], transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, season_idx, subtype_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, season_idx, subtype_idx


def _make_tensor_loader(cache: dict, batch_size: int, shuffle: bool,
                        device: torch.device) -> DataLoader:
    """Wrap cached feature tensors in a TensorDataset DataLoader."""
    ds = TensorDataset(
        cache["features"].to(device),
        cache["season_labels"].to(device),
        cache["subtype_labels"].to(device),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ── Train / eval on cached features ──────────────────────────────────────────

def train_one_epoch(head, loader, optimizer, criterion):
    head.train()
    total_loss = season_loss_sum = subtype_loss_sum = 0.0
    correct_season = correct_subtype = total = 0

    for features, season_labels, subtype_labels in loader:
        optimizer.zero_grad()
        season_logits, subtype_logits = head(features)
        loss, ls, lsub = criterion(season_logits, subtype_logits,
                                   season_labels, subtype_labels)
        loss.backward()
        optimizer.step()

        bs = features.size(0)
        total_loss       += loss.item() * bs
        season_loss_sum  += ls.item() * bs
        subtype_loss_sum += lsub.item() * bs
        correct_season   += (season_logits.argmax(1) == season_labels).sum().item()
        correct_subtype  += (subtype_logits.argmax(1) == subtype_labels).sum().item()
        total += bs

    return {
        "total_loss":   total_loss   / total,
        "season_loss":  season_loss_sum  / total,
        "subtype_loss": subtype_loss_sum / total,
        "season_acc":   correct_season   / total,
        "subtype_acc":  correct_subtype  / total,
    }


@torch.no_grad()
def evaluate(head, loader, criterion):
    head.eval()
    total_loss = season_loss_sum = subtype_loss_sum = 0.0
    correct_season = correct_subtype = total = 0

    for features, season_labels, subtype_labels in loader:
        season_logits, subtype_logits = head(features)
        loss, ls, lsub = criterion(season_logits, subtype_logits,
                                   season_labels, subtype_labels)

        bs = features.size(0)
        total_loss       += loss.item() * bs
        season_loss_sum  += ls.item() * bs
        subtype_loss_sum += lsub.item() * bs
        correct_season   += (season_logits.argmax(1) == season_labels).sum().item()
        correct_subtype  += (subtype_logits.argmax(1) == subtype_labels).sum().item()
        total += bs

    return {
        "total_loss":   total_loss   / total,
        "season_loss":  season_loss_sum  / total,
        "subtype_loss": subtype_loss_sum / total,
        "season_acc":   correct_season   / total,
        "subtype_acc":  correct_subtype  / total,
    }


@torch.no_grad()
def get_all_predictions(head, loader):
    """Collect season and subtype predictions from a cached-feature loader."""
    head.eval()
    season_true, season_pred = [], []
    sub_true,    sub_pred    = [], []

    for features, season_labels, subtype_labels in loader:
        szn_logits, sub_logits = head(features, hard_season=False)
        season_pred.extend(szn_logits.argmax(1).cpu().numpy())
        sub_pred.extend(sub_logits.argmax(1).cpu().numpy())
        season_true.extend(season_labels.cpu().numpy())
        sub_true.extend(subtype_labels.cpu().numpy())

    return (np.array(season_true), np.array(season_pred),
            np.array(sub_true),    np.array(sub_pred))


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_epoch_checkpoint(path, epoch, model, optimizer, scheduler,
                          history, best_season_acc, best_epoch, cfg):
    torch.save({
        "epoch":           epoch,
        "head_state":      model.head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history":         history,
        "best_season_acc": best_season_acc,
        "best_epoch":      best_epoch,
        "cfg":             cfg,
    }, path)


def load_resume_checkpoint(path, model, optimizer, scheduler, device):
    print(f"[RESUME] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model.head.load_state_dict(ckpt["head_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch     = ckpt["epoch"] + 1
    history         = ckpt.get("history", [])
    best_season_acc = ckpt.get("best_season_acc", 0.0)
    best_epoch      = ckpt.get("best_epoch", ckpt["epoch"])
    print(f"[RESUME] Resuming from epoch {start_epoch} "
          f"| best season acc so far: {best_season_acc:.4f} (epoch {best_epoch})")
    return start_epoch, history, best_season_acc, best_epoch


def sync_to_drive(results_dir: str, drive_dir: str):
    os.makedirs(drive_dir, exist_ok=True)
    copied = 0
    for pattern in ("*.pth", "*.json", "*.png"):
        for f in glob.glob(os.path.join(results_dir, pattern)):
            shutil.copy(f, drive_dir)
            copied += 1
    print(f"[DRIVE] Synced {copied} files → {drive_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/hierarchical.yaml")
    parser.add_argument("--resume",    default=None,
                        help="Path to a checkpoint_epoch_N.pth to resume from")
    parser.add_argument("--drive_dir", default=None,
                        help="Google Drive path — syncs results/ here every 5 epochs")
    parser.add_argument("--recache",   action="store_true",
                        help="Force re-extraction of backbone features even if cache exists")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    out_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ── Scan datasets ─────────────────────────────────────────────────────
    train_ds = ArmocromiaDataset(cfg["data"]["train_dir"])
    val_ds   = ArmocromiaDataset(cfg["data"]["test_dir"])
    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_counts = Counter(sub for _, _, sub in train_ds.samples)
    for i, name in enumerate(SUBTYPE_CLASSES):
        print(f"  {name:<22}: {train_counts[i]}")

    # ── Build model (backbone needed for caching) ─────────────────────────
    ckpt_path = cfg["model"].get("checkpoint_path")
    model = DeepArmocromiaHierarchical(
        checkpoint_path=ckpt_path if ckpt_path else None,
        feature_dim=cfg["model"]["feature_dim"],
        shared_dim=cfg["model"]["shared_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable params: {trainable:,} / {total_p:,}")

    # ── Cache backbone features ───────────────────────────────────────────
    bs = cfg["training"]["batch_size"]
    train_cache, val_cache = build_or_load_cache(
        model, train_ds, val_ds, out_dir, device, bs,
        recache=args.recache,
    )

    feature_dim = train_cache["features"].shape[1]
    print(f"[INFO] Feature dim: {feature_dim}")

    train_loader = _make_tensor_loader(train_cache, bs, shuffle=True,  device=device)
    val_loader   = _make_tensor_loader(val_cache,   bs, shuffle=False, device=device)

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    criterion = HierarchicalLoss(lambda_subtype=cfg["training"]["lambda_subtype"])
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["scheduler"]["T_0"],
        eta_min=cfg["scheduler"]["eta_min"],
    )

    # ── Resume or start fresh ─────────────────────────────────────────────
    epochs = cfg["training"]["epochs"]
    best_season_acc = 0.0
    best_epoch      = 1
    history         = []
    start_epoch     = 1

    if args.resume:
        start_epoch, history, best_season_acc, best_epoch = \
            load_resume_checkpoint(args.resume, model, optimizer, scheduler, device)
    out_best = os.path.join(out_dir, cfg["output"]["checkpoint_name"])

    hdr = (f"{'Ep':>4}  {'TrSzn':>6} {'TrSub':>6} {'TrTot':>7}  "
           f"{'VlSzn':>6} {'VlSub':>6} {'VlTot':>7}  {'LR':>8}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    train_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(model.head, train_loader, optimizer, criterion)
        vl = evaluate(model.head, val_loader, criterion)
        scheduler.step(epoch - 1)

        lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            **{f"train_{k}": round(v, 6) for k, v in tr.items()},
            **{f"val_{k}":   round(v, 6) for k, v in vl.items()},
        }
        history.append(record)

        # ── Best-model checkpoint ──────────────────────────────────────────
        improved = vl["season_acc"] > best_season_acc
        if improved:
            best_season_acc = vl["season_acc"]
            best_epoch      = epoch
            torch.save({
                "epoch":       epoch,
                "head_state":  model.head.state_dict(),
                "season_acc":  best_season_acc,
                "subtype_acc": vl["subtype_acc"],
                "cfg":         cfg,
            }, out_best)
        # ── Per-epoch checkpoint ───────────────────────────────────────────
        epoch_ckpt = os.path.join(out_dir, f"checkpoint_epoch_{epoch}.pth")
        save_epoch_checkpoint(epoch_ckpt, epoch, model, optimizer, scheduler,
                              history, best_season_acc, best_epoch, cfg)

        # ── Console output ─────────────────────────────────────────────────
        elapsed = time.time() - t0
        marker  = " ← best" if improved else ""
        print(
            f"{epoch:>4}  {tr['season_acc']:>6.4f} {tr['subtype_acc']:>6.4f} {tr['total_loss']:>7.4f}"
            f"  {vl['season_acc']:>6.4f} {vl['subtype_acc']:>6.4f} {vl['total_loss']:>7.4f}"
            f"  {lr:.2e}{marker}  ({elapsed:.1f}s)"
        )
        print(f"       Epoch {epoch}/{epochs} | Season Acc: {vl['season_acc']:.4f} "
              f"| Subtype Acc: {vl['subtype_acc']:.4f} | Best so far: {best_season_acc:.4f}")

        if epoch == start_epoch:
            remaining = epochs - start_epoch
            est_total_secs = elapsed * remaining
            est_mins = int(est_total_secs // 60)
            est_secs = int(est_total_secs % 60)
            print(f"       Epoch 1 done in {elapsed:.1f}s. "
                  f"Estimated remaining: {est_mins}m {est_secs:02d}s "
                  f"({remaining} epochs left)")

        # ── Drive sync every 5 epochs ──────────────────────────────────────
        if args.drive_dir and epoch % 5 == 0:
            sync_to_drive(out_dir, args.drive_dir)

    if args.drive_dir:
        sync_to_drive(out_dir, args.drive_dir)

    total_secs = time.time() - train_start
    total_mins, secs = divmod(int(total_secs), 60)
    total_hrs,  mins = divmod(total_mins, 60)
    elapsed_str = (f"{total_hrs}h {mins:02d}m {secs:02d}s" if total_hrs
                   else f"{mins}m {secs:02d}s")

    print(f"\n{'='*55}")
    print(f"  Training finished in {elapsed_str}")
    print(f"  Best season acc : {best_season_acc:.4f}  (epoch {best_epoch})")
    best_sub = next((r["val_subtype_acc"] for r in history if r["epoch"] == best_epoch), 0)
    print(f"  Subtype acc     : {best_sub:.4f}  (at same epoch)")
    print(f"  Best checkpoint : {out_best}")
    print(f"{'='*55}")

    hist_path = os.path.join(out_dir, "hierarchical_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] History: {hist_path}")

    # ── Final classification reports (mirrors train_farl.py output) ───────
    print(f"\n[INFO] Loading best checkpoint for final evaluation...")
    best_ckpt = torch.load(out_best, map_location=device)
    model.head.load_state_dict(best_ckpt["head_state"])

    szn_true, szn_pred, sub_true, sub_pred = get_all_predictions(model.head, val_loader)

    # Season report
    print("\n" + "=" * 60)
    print("  Final Test Classification Report — Season (4-class)")
    print("=" * 60)
    print(f"  Best Epoch: {best_epoch}  |  Best Season Acc: {best_season_acc:.4f}")
    print("=" * 60)
    print(classification_report(szn_true, szn_pred, target_names=SEASON_CLASSES, digits=4))

    szn_cm = confusion_matrix(szn_true, szn_pred)
    print("Confusion Matrix — Season (rows=actual, cols=predicted):")
    print(f"{'':12s} " + "  ".join(f"{s:>8s}" for s in SEASON_CLASSES))
    for i, row in enumerate(szn_cm):
        print(f"{SEASON_CLASSES[i]:12s} " + "  ".join(f"{v:>8d}" for v in row))

    autumn_idx = SEASON_CLASSES.index("autumn")
    aut_correct = szn_cm[autumn_idx, autumn_idx]
    aut_total   = szn_cm[autumn_idx].sum()
    print(f"\n[NOTE] Autumn recall: {aut_correct/aut_total:.4f} ({aut_correct}/{aut_total})")
    print("       (Autumn is historically the hardest class — track this carefully.)")

    # Sub-type report
    print("\n" + "=" * 60)
    print("  Final Test Classification Report — Sub-Type (12-class)")
    print("=" * 60)
    print(f"  Best Epoch: {best_epoch}  |  Best SubType Acc: {best_sub:.4f}")
    print("=" * 60)
    print(classification_report(sub_true, sub_pred, target_names=SUBTYPE_CLASSES, digits=4))

    sub_cm = confusion_matrix(sub_true, sub_pred)
    print("Confusion Matrix — Sub-Type (rows=actual, cols=predicted):")
    print(f"{'':16s} " + " ".join(f"{s.split('_')[1]:>7s}" for s in SUBTYPE_CLASSES))
    for i, row in enumerate(sub_cm):
        label = SUBTYPE_CLASSES[i]
        print(f"{label:16s} " + " ".join(f"{v:>7d}" for v in row))

    print("\n[DONE] Hierarchical training pipeline finished successfully.")


if __name__ == "__main__":
    main()
