"""
evaluate_hierarchical.py — Hierarchical Model Evaluation
---------------------------------------------------------
Loads a trained DeepArmocromiaHierarchical checkpoint and evaluates it on the
test set, reporting metrics for both Season (4-class) and Sub-Type (12-class).

Usage:
    python evaluate_hierarchical.py --checkpoint results/best_hierarchical.pth

Outputs (to results/):
    confusion_season_hierarchical.png
    confusion_subtype_hierarchical.png
    hierarchical_metrics.json
"""

import argparse
import json
import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from hierarchical_head import DeepArmocromiaHierarchical

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

RESULTS_DIR = "results"

# Paper targets
BASELINE_SEASON_ACC  = 0.554   # FaRL-64
BASELINE_SUBTYPE_ACC = 0.318   # FaRL-16


# ── Dataset ───────────────────────────────────────────────────────────────────

class ArmocromiaDataset(Dataset):
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


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_season_true, all_season_pred, all_season_prob = [], [], []
    all_subtype_true, all_subtype_pred, all_subtype_prob = [], [], []

    for imgs, season_labels, subtype_labels in loader:
        imgs = imgs.to(device)
        season_logits, subtype_logits = model(imgs, hard_season=False)

        season_prob  = torch.softmax(season_logits, dim=1).cpu().numpy()
        subtype_prob = torch.softmax(subtype_logits, dim=1).cpu().numpy()

        all_season_true.extend(season_labels.numpy())
        all_season_pred.extend(season_prob.argmax(1))
        all_season_prob.extend(season_prob)

        all_subtype_true.extend(subtype_labels.numpy())
        all_subtype_pred.extend(subtype_prob.argmax(1))
        all_subtype_prob.extend(subtype_prob)

    return (
        np.array(all_season_true),  np.array(all_season_pred),  np.array(all_season_prob),
        np.array(all_subtype_true), np.array(all_subtype_pred), np.array(all_subtype_prob),
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def top_k_accuracy(y_true, y_prob, k):
    correct = 0
    for true, prob in zip(y_true, y_prob):
        top_k = np.argsort(prob)[-k:]
        if true in top_k:
            correct += 1
    return correct / len(y_true)


def compute_metrics(y_true, y_pred, y_prob, class_names, k_topk):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    topk = top_k_accuracy(y_true, y_prob, k_topk)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, f"top{k_topk}_acc": topk}


def print_per_class_metrics(y_true, y_pred, class_names):
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() == 0:
            continue
        p = precision_score(y_true == i, y_pred == i, zero_division=0)
        r = recall_score(y_true == i, y_pred == i, zero_division=0)
        f = f1_score(y_true == i, y_pred == i, zero_division=0)
        n = mask.sum()
        print(f"  {name:<22}  prec={p:.4f}  rec={r:.4f}  f1={f:.4f}  (n={n})")


# ── Confusion matrix ──────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    labels = [c.replace("_", "\n") for c in class_names]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax)
    acc = accuracy_score(y_true, y_pred)
    ax.set_title(f"{title}\nAccuracy: {acc:.4f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/best_hierarchical.pth")
    parser.add_argument("--test_dir",   default="RGB-M/test")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt.get("cfg", {})

    model_cfg = cfg.get("model", {})
    model = DeepArmocromiaHierarchical(
        checkpoint_path=None,
        feature_dim=model_cfg.get("feature_dim", 768),
        shared_dim=model_cfg.get("shared_dim", 384),
        dropout=model_cfg.get("dropout", 0.5),
    ).to(device)
    model.head.load_state_dict(ckpt["head_state"])
    model.eval()
    print(f"  Restored from epoch {ckpt.get('epoch', '?')} "
          f"(saved season_acc={ckpt.get('season_acc', 0):.4f})")

    # Dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ds     = ArmocromiaDataset(args.test_dir, transform=transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[INFO] Test set: {len(ds)} images")

    # Inference
    print("[INFO] Running inference...")
    (season_true, season_pred, season_prob,
     subtype_true, subtype_pred, subtype_prob) = run_inference(model, loader, device)

    # Season metrics
    print("\n" + "=" * 65)
    print("  SEASON (4-class)")
    print("=" * 65)
    sm = compute_metrics(season_true, season_pred, season_prob, SEASON_CLASSES, k_topk=2)
    print(f"  Accuracy  : {sm['accuracy']:.4f}   (target > {BASELINE_SEASON_ACC})")
    print(f"  Precision : {sm['precision']:.4f} (macro)")
    print(f"  Recall    : {sm['recall']:.4f} (macro)")
    print(f"  F1        : {sm['f1']:.4f} (macro)")
    print(f"  Top-2 Acc : {sm['top2_acc']:.4f}")
    print("\nPer-class breakdown:")
    print_per_class_metrics(season_true, season_pred, SEASON_CLASSES)

    # Sub-type metrics
    print("\n" + "=" * 65)
    print("  SUB-TYPE (12-class)")
    print("=" * 65)
    stm = compute_metrics(subtype_true, subtype_pred, subtype_prob, SUBTYPE_CLASSES, k_topk=3)
    print(f"  Accuracy  : {stm['accuracy']:.4f}   (target > {BASELINE_SUBTYPE_ACC})")
    print(f"  Precision : {stm['precision']:.4f} (macro)")
    print(f"  Recall    : {stm['recall']:.4f} (macro)")
    print(f"  F1        : {stm['f1']:.4f} (macro)")
    print(f"  Top-3 Acc : {stm['top3_acc']:.4f}")
    print("\nPer-class breakdown:")
    print_per_class_metrics(subtype_true, subtype_pred, SUBTYPE_CLASSES)

    # Comparison table
    print("\n" + "=" * 65)
    print("  COMPARISON VS PAPER BASELINES")
    print("=" * 65)
    rows = [
        ("FaRL-64 (paper)",  BASELINE_SEASON_ACC,  "—"),
        ("FaRL-16 (paper)",  "—",                  BASELINE_SUBTYPE_ACC),
        ("Ours (hierarchical)", f"{sm['accuracy']:.4f}", f"{stm['accuracy']:.4f}"),
    ]
    print(f"  {'Model':<25} {'Season Acc':>12} {'SubType Acc':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    for name, sa, sta in rows:
        beat_s   = " ✓" if isinstance(sa,  str) and sa  != "—" and float(sa)  > BASELINE_SEASON_ACC  else ""
        beat_sub = " ✓" if isinstance(sta, str) and sta != "—" and float(sta) > BASELINE_SUBTYPE_ACC else ""
        print(f"  {name:<25} {str(sa):>12}{beat_s} {str(sta):>12}{beat_sub}")

    # Confusion matrices
    print("\n[INFO] Saving confusion matrices...")
    save_confusion_matrix(season_true, season_pred, SEASON_CLASSES,
                          "Hierarchical Model — Season", "confusion_season_hierarchical.png")
    save_confusion_matrix(subtype_true, subtype_pred, SUBTYPE_CLASSES,
                          "Hierarchical Model — Sub-Type", "confusion_subtype_hierarchical.png")

    # Save JSON summary
    summary = {
        "season":  {k: round(float(v), 6) for k, v in sm.items()},
        "subtype": {k: round(float(v), 6) for k, v in stm.items()},
        "baselines": {
            "FaRL-64_season_acc":   BASELINE_SEASON_ACC,
            "FaRL-16_subtype_acc":  BASELINE_SUBTYPE_ACC,
        },
    }
    json_path = os.path.join(RESULTS_DIR, "hierarchical_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
