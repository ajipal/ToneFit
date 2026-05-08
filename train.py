"""
train.py — Hierarchical Deep Armocromia Training
-------------------------------------------------
Trains DeepArmocromiaHierarchical (frozen FaRL-64 backbone + two-stage head)
on the RGB-M dataset using AdamW + CosineAnnealingWarmRestarts.

Usage:
    python train.py --config configs/hierarchical.yaml
    python train.py --config configs/hierarchical.yaml --resume results/checkpoint_epoch_25.pth

Outputs:
    results/best_hierarchical.pth        — best checkpoint (by val season accuracy)
    results/checkpoint_epoch_{N}.pth     — checkpoint saved every epoch
    results/hierarchical_history.json    — full loss/accuracy history
"""

import argparse
import glob
import json
import os
import shutil
import time
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import yaml

from hierarchical_head import DeepArmocromiaHierarchical, HierarchicalLoss

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

EARLY_STOP_PATIENCE = 10   # stop if season acc stagnates this many epochs


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

def build_transforms(training: bool):
    if training:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Train / eval helpers ──────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = season_loss_sum = subtype_loss_sum = 0.0
    correct_season = correct_subtype = total = 0

    for imgs, season_labels, subtype_labels in loader:
        imgs           = imgs.to(device)
        season_labels  = season_labels.to(device)
        subtype_labels = subtype_labels.to(device)

        optimizer.zero_grad()
        season_logits, subtype_logits = model(imgs)
        loss, ls, lsub = criterion(season_logits, subtype_logits, season_labels, subtype_labels)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss       += loss.item() * bs
        season_loss_sum  += ls.item() * bs
        subtype_loss_sum += lsub.item() * bs
        correct_season   += (season_logits.argmax(1) == season_labels).sum().item()
        correct_subtype  += (subtype_logits.argmax(1) == subtype_labels).sum().item()
        total += bs

    n = total
    return {
        "total_loss":   total_loss / n,
        "season_loss":  season_loss_sum / n,
        "subtype_loss": subtype_loss_sum / n,
        "season_acc":   correct_season / n,
        "subtype_acc":  correct_subtype / n,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = season_loss_sum = subtype_loss_sum = 0.0
    correct_season = correct_subtype = total = 0

    for imgs, season_labels, subtype_labels in loader:
        imgs           = imgs.to(device)
        season_labels  = season_labels.to(device)
        subtype_labels = subtype_labels.to(device)

        season_logits, subtype_logits = model(imgs)
        loss, ls, lsub = criterion(season_logits, subtype_logits, season_labels, subtype_labels)

        bs = imgs.size(0)
        total_loss       += loss.item() * bs
        season_loss_sum  += ls.item() * bs
        subtype_loss_sum += lsub.item() * bs
        correct_season   += (season_logits.argmax(1) == season_labels).sum().item()
        correct_subtype  += (subtype_logits.argmax(1) == subtype_labels).sum().item()
        total += bs

    n = total
    return {
        "total_loss":   total_loss / n,
        "season_loss":  season_loss_sum / n,
        "subtype_loss": subtype_loss_sum / n,
        "season_acc":   correct_season / n,
        "subtype_acc":  correct_subtype / n,
    }


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
    start_epoch      = ckpt["epoch"] + 1
    history          = ckpt.get("history", [])
    best_season_acc  = ckpt.get("best_season_acc", 0.0)
    best_epoch       = ckpt.get("best_epoch", ckpt["epoch"])
    print(f"[RESUME] Resuming from epoch {start_epoch} "
          f"| best season acc so far: {best_season_acc:.4f} (epoch {best_epoch})")
    return start_epoch, history, best_season_acc, best_epoch


def sync_to_drive(results_dir: str, drive_dir: str):
    """Copy all .pth and .json files from results_dir to drive_dir."""
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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    out_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Datasets & loaders
    train_ds = ArmocromiaDataset(cfg["data"]["train_dir"], transform=build_transforms(True))
    test_ds  = ArmocromiaDataset(cfg["data"]["test_dir"],  transform=build_transforms(False))
    print(f"[INFO] Train: {len(train_ds)} | Test: {len(test_ds)}")

    train_counts = Counter(sub for _, _, sub in train_ds.samples)
    for i, name in enumerate(SUBTYPE_CLASSES):
        print(f"  {name:<22}: {train_counts[i]}")

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    # Model
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

    # Resume or start fresh
    epochs = cfg["training"]["epochs"]
    best_season_acc  = 0.0
    best_epoch       = 1
    epochs_no_improve = 0
    history          = []
    start_epoch      = 1

    if args.resume:
        start_epoch, history, best_season_acc, best_epoch = \
            load_resume_checkpoint(args.resume, model, optimizer, scheduler, device)
        epochs_no_improve = best_epoch  # will be recalculated from history if needed

    out_best  = os.path.join(out_dir, cfg["output"]["checkpoint_name"])

    # Column header
    hdr = (f"{'Ep':>4}  {'TrSzn':>6} {'TrSub':>6} {'TrTot':>7}  "
           f"{'VlSzn':>6} {'VlSub':>6} {'VlTot':>7}  {'LR':>8}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl = evaluate(model, test_loader, criterion, device)
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
            epochs_no_improve = 0
            torch.save({
                "epoch":       epoch,
                "head_state":  model.head.state_dict(),
                "season_acc":  best_season_acc,
                "subtype_acc": vl["subtype_acc"],
                "cfg":         cfg,
            }, out_best)
        else:
            epochs_no_improve += 1

        # ── Per-epoch checkpoint (full state for resuming) ─────────────────
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

        # ── Drive sync every 5 epochs ──────────────────────────────────────
        if args.drive_dir and epoch % 5 == 0:
            sync_to_drive(out_dir, args.drive_dir)

        # ── Early stopping ─────────────────────────────────────────────────
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n[EARLY STOP] Season accuracy did not improve for "
                  f"{EARLY_STOP_PATIENCE} consecutive epochs.")
            print(f"[EARLY STOP] Best: {best_season_acc:.4f} at epoch {best_epoch}. Stopping.")
            break

    # Final Drive sync
    if args.drive_dir:
        sync_to_drive(out_dir, args.drive_dir)

    print(f"\n[INFO] Best season acc: {best_season_acc:.4f} at epoch {best_epoch}")
    print(f"[INFO] Best checkpoint: {out_best}")

    hist_path = os.path.join(out_dir, "hierarchical_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] History: {hist_path}")


if __name__ == "__main__":
    main()
