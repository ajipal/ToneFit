"""
evaluate_baseline.py — FaRL-64 Baseline Evaluation
----------------------------------------------------
Evaluates the FaRL-64 flat-head baseline on the held-out test set
(Season classification, 4 classes).

Usage:
    python evaluate_baseline.py

Requires:
    - models/farl_model.pth    (from train_farl.py)
    - RGB-M/test/              (READ ONLY — ImageFolder, seasons as top-level dirs)
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from collections import Counter
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────

TEST_DIR    = "RGB-M/test"
FARL_PATH   = "models/farl_model.pth"
RESULTS_DIR = "results"

SEASONS   = ["autumn", "spring", "summer", "winter"]
BATCH_SIZE = 64

# Paper baselines from Stacchio et al. (2024) Table 2
PAPER_BASELINES = {
    "FaRL-16 (paper)":    {"accuracy": 0.525, "f1": 0.516, "top2": 0.815},
    "FaRL-64 (paper)":    {"accuracy": 0.554, "f1": 0.548, "top2": 0.808},
    "ResNeXt50 (paper)":  {"accuracy": 0.513, "f1": 0.502, "top2": 0.789},
}

# ── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── DATASET ───────────────────────────────────────────────────────────────────

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── MODEL LOADER ──────────────────────────────────────────────────────────────

def build_classifier_head(feature_dim):
    return nn.Sequential(
        nn.Linear(feature_dim, feature_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(feature_dim // 2, 4),
    )


def load_farl_baseline(path):
    """Load FaRL-64 baseline (backbone + 4-class season head)."""
    try:
        import timm
        backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        feature_dim = 768
    except Exception:
        from torchvision import models as tvm
        _b = tvm.resnext50_32x4d(pretrained=False)
        feature_dim = _b.fc.in_features
        backbone = nn.Sequential(*list(_b.children())[:-1], nn.Flatten())

    head = build_classifier_head(feature_dim)

    checkpoint = torch.load(path, map_location=DEVICE)
    if "backbone" in checkpoint and "head" in checkpoint:
        backbone.load_state_dict(checkpoint["backbone"], strict=False)
        head.load_state_dict(checkpoint["head"])
    else:
        backbone.load_state_dict(checkpoint, strict=False)

    class _Model(nn.Module):
        def __init__(self, bb, h):
            super().__init__()
            self.backbone = bb
            self.head = h
        def forward(self, x):
            with torch.no_grad():
                feats = self.backbone(x)
            return self.head(feats)

    model = _Model(backbone, head).to(DEVICE)
    model.eval()
    return model

# ── INFERENCE ─────────────────────────────────────────────────────────────────

def run_inference(model, loader):
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs  = imgs.to(DEVICE)
            probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(np.argmax(probs, axis=1))
            all_probs.extend(probs)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    top2 = float(top_k_accuracy_score(y_true, y_prob, k=2))
    rep  = classification_report(y_true, y_pred, target_names=SEASONS, output_dict=True)

    log.info(f"\n  Classification Report — {name}:")
    log.info("  " + classification_report(y_true, y_pred, target_names=SEASONS))
    log.info(f"  Top-2 Accuracy: {top2:.4f}")
    log.info(f"  Autumn recall : {rep['autumn']['recall']:.4f}  (hardest class)")

    return {
        "Model":             name,
        "Accuracy":          round(acc, 4),
        "Top-2 Accuracy":    round(top2, 4),
        "Precision (macro)": round(rep["macro avg"]["precision"], 4),
        "Recall (macro)":    round(rep["macro avg"]["recall"], 4),
        "F1 (macro)":        round(rep["macro avg"]["f1-score"], 4),
        "F1 (weighted)":     round(rep["weighted avg"]["f1-score"], 4),
        "Recall (Spring)":   round(rep["spring"]["recall"], 4),
        "Recall (Summer)":   round(rep["summer"]["recall"], 4),
        "Recall (Autumn)":   round(rep["autumn"]["recall"], 4),
        "Recall (Winter)":   round(rep["winter"]["recall"], 4),
    }

# ── PLOTS ─────────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, name, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[s.capitalize() for s in SEASONS],
                yticklabels=[s.capitalize() for s in SEASONS],
                linewidths=0.5, ax=ax)
    ax.set_title(f"{name} — Test Set\nAccuracy: {accuracy_score(y_true, y_pred):.4f}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {path}")


def save_comparison_chart(metrics, baselines):
    """Bar chart: FaRL-64 baseline vs paper baselines."""
    names      = list(baselines.keys()) + [metrics["Model"]]
    accuracies = [v["accuracy"] for v in baselines.values()] + [metrics["Accuracy"]]
    f1s        = [v["f1"]       for v in baselines.values()] + [metrics["F1 (weighted)"]]
    colors     = ["#AAAAAA"] * len(baselines) + ["#4C72B0"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, vals, title in [
        (axes[0], accuracies, "Season Accuracy"),
        (axes[1], f1s,        "F1 (Weighted)"),
    ]:
        bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 0.75)
        ax.axhline(0.554, color="red", linestyle="--", linewidth=0.8,
                   alpha=0.6, label="FaRL-64 paper (0.554)")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

    fig.suptitle("FaRL-64 Baseline vs Deep Armocromia Paper (Stacchio et al., 2024)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "baseline_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {path}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 65)
    log.info("  ToneFit — FaRL-64 Baseline Evaluation (Season, 4-class)")
    log.info("  Deep Armocromia dataset — Stacchio et al., ECCV 2024")
    log.info("=" * 65)
    log.info(f"  Device: {DEVICE}")

    if not os.path.isdir(TEST_DIR):
        log.error(f"  {TEST_DIR} not found.")
        return
    if not os.path.exists(FARL_PATH):
        log.error(f"  {FARL_PATH} not found. Run train_farl.py first.")
        return

    log.info("\n[1] Loading test set...")
    test_ds = ImageFolder(root=TEST_DIR, transform=TEST_TRANSFORM)
    log.info(f"  {len(test_ds)} images")
    counts = Counter(label for _, label in test_ds.samples)
    for i, s in enumerate(SEASONS):
        log.info(f"    {s:<8}: {counts[i]}")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    log.info("\n[2] Loading FaRL-64 baseline...")
    model = load_farl_baseline(FARL_PATH)

    log.info("\n[3] Running inference...")
    y_true, y_pred, y_prob = run_inference(model, test_loader)
    metrics = compute_metrics("FaRL-64 Baseline", y_true, y_pred, y_prob)

    log.info("\n[4] Saving confusion matrix...")
    save_confusion_matrix(y_true, y_pred, "FaRL-64 Baseline", "confusion_baseline.png")

    log.info("\n[5] Saving comparison chart vs paper...")
    save_comparison_chart(metrics, PAPER_BASELINES)

    log.info("\n[6] Saving results CSV...")
    rows = [metrics] + [
        {"Model": n, "Accuracy": v["accuracy"], "F1 (weighted)": v["f1"],
         "Top-2 Accuracy": v["top2"]}
        for n, v in PAPER_BASELINES.items()
    ]
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    df.to_csv(csv_path, index=False)
    log.info(f"  Saved: {csv_path}")

    log.info("\n" + "=" * 65)
    log.info("  FINAL — FaRL-64 Baseline (Season, 4-class)")
    log.info("=" * 65)
    log.info(f"  Accuracy    : {metrics['Accuracy']:.4f}  (FaRL-64 paper: 0.554)")
    log.info(f"  Top-2 Acc   : {metrics['Top-2 Accuracy']:.4f}")
    log.info(f"  F1 (macro)  : {metrics['F1 (macro)']:.4f}")
    log.info(f"  Autumn recall: {metrics['Recall (Autumn)']:.4f}  (hardest class)")
    beat = metrics["Accuracy"] > 0.554
    log.info(f"  Beats FaRL-64 paper? {'YES' if beat else 'Not yet'}")
    log.info("=" * 65)


if __name__ == "__main__":
    run()
