"""
ToneFit ML — Model Evaluation (Step 6)
========================================
Evaluates FaRL and DINOv2 on the held-out test set and compares them
against each other and against the Deep Armocromia paper baselines.

Usage:
    python evaluate.py

Requires:
    - models/farl_model.pth    (from train_farl.py)
    - models/dinov2_model.pth  (from train_dinov2.py)
    - RGB-M/test/              (READ ONLY — loaded via ImageFolder)
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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score,
)

# ── CONFIG ─────────────────────────────────────────────────────────────────────

TEST_DIR      = "RGB-M/test"   # READ ONLY — ImageFolder reads sub-types recursively
MODELS_DIR    = "models"
RESULTS_DIR   = "results"

FARL_PATH     = os.path.join(MODELS_DIR, "farl_model.pth")
DINOV2_PATH   = os.path.join(MODELS_DIR, "dinov2_model.pth")

SEASONS       = ["autumn", "spring", "summer", "winter"]  # alphabetical — ImageFolder order
LABEL_MAP     = {"autumn": 0, "spring": 1, "summer": 2, "winter": 3}
BATCH_SIZE    = 64
IMG_SIZE      = 224

# Paper baselines from Stacchio et al. (2024) Table 2
PAPER_BASELINES = {
    "FaRL16 (paper)":    {"accuracy": 0.525, "f1": 0.516, "top2": 0.815},
    "FaRL64 (paper)":    {"accuracy": 0.554, "f1": 0.548, "top2": 0.808},
    "ResNeXt50 (paper)": {"accuracy": 0.513, "f1": 0.502, "top2": 0.789},
}

# ── LOGGING ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── DEVICE ─────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── DATASET ────────────────────────────────────────────────────────────────────

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── MODEL LOADERS ──────────────────────────────────────────────────────────────

def build_classifier_head(feature_dim):
    return nn.Sequential(
        nn.Linear(feature_dim, feature_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(feature_dim // 2, 4),
    )


def load_farl(path):
    """Load FaRL or ResNeXt50 fallback model."""
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
    # Support both full model checkpoint and split backbone/head
    if "backbone" in checkpoint and "head" in checkpoint:
        backbone.load_state_dict(checkpoint["backbone"], strict=False)
        head.load_state_dict(checkpoint["head"])
    else:
        # Flat state dict — try loading directly
        try:
            backbone.load_state_dict(checkpoint, strict=False)
        except Exception:
            pass

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


def load_dinov2(path):
    """Load DINOv2 model."""
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                              verbose=False)
    feature_dim = 768
    head = build_classifier_head(feature_dim)

    checkpoint = torch.load(path, map_location=DEVICE)
    if "backbone" in checkpoint:
        backbone.load_state_dict(checkpoint["backbone"], strict=False)
    if "head" in checkpoint:
        head.load_state_dict(checkpoint["head"])

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

# ── INFERENCE ──────────────────────────────────────────────────────────────────

def run_inference(model, loader):
    """Run model on all batches, return (y_true, y_pred, y_prob)."""
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )

# ── METRICS ────────────────────────────────────────────────────────────────────

def compute_metrics(name, y_true, y_pred, y_prob):
    acc   = accuracy_score(y_true, y_pred)
    top2  = float(top_k_accuracy_score(y_true, y_prob, k=2))
    rep   = classification_report(y_true, y_pred,
                                   target_names=SEASONS, output_dict=True)

    log.info(f"\n  Classification Report — {name}:")
    log.info("  " + classification_report(y_true, y_pred, target_names=SEASONS))

    log.info(f"  Top-2 Accuracy: {top2:.4f}")
    log.info(f"  Autumn recall : {rep['autumn']['recall']:.4f}  ← watch this")

    return {
        "Model":              name,
        "Accuracy":           round(acc, 4),
        "Top-2 Accuracy":     round(top2, 4),
        "Precision (macro)":  round(rep["macro avg"]["precision"], 4),
        "Recall (macro)":     round(rep["macro avg"]["recall"], 4),
        "F1 (macro)":         round(rep["macro avg"]["f1-score"], 4),
        "F1 (weighted)":      round(rep["weighted avg"]["f1-score"], 4),
        "Recall (Spring)":    round(rep["spring"]["recall"], 4),
        "Recall (Summer)":    round(rep["summer"]["recall"], 4),
        "Recall (Autumn)":    round(rep["autumn"]["recall"], 4),
        "Recall (Winter)":    round(rep["winter"]["recall"], 4),
    }

# ── PLOTS ──────────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, name, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[s.capitalize() for s in SEASONS],
        yticklabels=[s.capitalize() for s in SEASONS],
        linewidths=0.5, ax=ax,
    )
    acc = accuracy_score(y_true, y_pred)
    ax.set_title(f"{name} — Test Set\nAccuracy: {acc:.4f}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {path}")


def save_combined_confusion_matrices(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = confusion_matrix(r["y_true"], r["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[s.capitalize() for s in SEASONS],
            yticklabels=[s.capitalize() for s in SEASONS],
            linewidths=0.5, ax=ax,
        )
        ax.set_title(f"{r['name']}\nAcc: {r['metrics']['Accuracy']:.4f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
    fig.suptitle("Confusion Matrices — FaRL vs DINOv2 (Test Set)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {path}")


def save_comparison_chart(df_our, baselines):
    """Bar chart: our models vs paper baselines on Accuracy and F1."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    all_names  = list(baselines.keys()) + df_our["Model"].tolist()
    accuracies = ([v["accuracy"] for v in baselines.values()] +
                  df_our["Accuracy"].tolist())
    f1s        = ([v["f1"] for v in baselines.values()] +
                  df_our["F1 (weighted)"].tolist())

    # Color: gray for paper baselines, colored for ours
    n_base  = len(baselines)
    colors  = ["#AAAAAA"] * n_base + ["#4C72B0", "#DD8452"][:len(df_our)]

    for ax, vals, title, ylabel in [
        (axes[0], accuracies, "Accuracy",  "Accuracy"),
        (axes[1], f1s,        "F1 (Weighted)", "F1 Score"),
    ]:
        bars = ax.bar(all_names, vals, color=colors,
                      edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, 0.75)
        ax.axhline(0.554, color="red", linestyle="--", linewidth=0.8,
                   alpha=0.6, label="FaRL64 paper best (0.554)")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)

    fig.suptitle("ToneFit Results vs Deep Armocromia Paper (Stacchio et al., 2024)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {path}")

# ── MAIN ───────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 65)
    log.info("  ToneFit ML — Model Evaluation")
    log.info("  FaRL vs DINOv2 | Deep Armocromia dataset")
    log.info("=" * 65)
    log.info(f"  Device: {DEVICE}")

    if not os.path.isdir(TEST_DIR):
        log.error(f"  {TEST_DIR} not found. Check that RGB-M/test/ exists.")
        return

    log.info("\n[Step 1] Loading test set...")
    from torchvision.datasets import ImageFolder
    test_ds = ImageFolder(root=TEST_DIR, transform=TEST_TRANSFORM)
    log.info(f"  Test set: {len(test_ds)} images")
    from collections import Counter
    counts = Counter(label for _, label in test_ds.samples)
    for i, s in enumerate(SEASONS):
        log.info(f"    {s:<8}: {counts[i]}")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    model_results = []

    # ── FaRL ──────────────────────────────────────────────────────────────────
    log.info("\n" + "─" * 65)
    log.info("  MODEL A: FaRL (or ResNeXt50 fallback)")
    log.info("─" * 65)
    if not os.path.exists(FARL_PATH):
        log.warning(f"  {FARL_PATH} not found — skipping FaRL.")
    else:
        log.info(f"  Loading from {FARL_PATH} ...")
        farl = load_farl(FARL_PATH)
        y_true_f, y_pred_f, y_prob_f = run_inference(farl, test_loader)
        metrics_f = compute_metrics("FaRL (ours)", y_true_f, y_pred_f, y_prob_f)
        save_confusion_matrix(y_true_f, y_pred_f, "FaRL (ours)", "confusion_farl.png")
        model_results.append({
            "name": "FaRL (ours)", "y_true": y_true_f,
            "y_pred": y_pred_f, "metrics": metrics_f,
        })

    # ── DINOv2 ────────────────────────────────────────────────────────────────
    log.info("\n" + "─" * 65)
    log.info("  MODEL B: DINOv2 (ViT-B/14)")
    log.info("─" * 65)
    if not os.path.exists(DINOV2_PATH):
        log.warning(f"  {DINOV2_PATH} not found — skipping DINOv2.")
    else:
        log.info(f"  Loading from {DINOV2_PATH} ...")
        dinov2 = load_dinov2(DINOV2_PATH)
        y_true_d, y_pred_d, y_prob_d = run_inference(dinov2, test_loader)
        metrics_d = compute_metrics("DINOv2 (ours)", y_true_d, y_pred_d, y_prob_d)
        save_confusion_matrix(y_true_d, y_pred_d, "DINOv2 (ours)", "confusion_dinov2.png")
        model_results.append({
            "name": "DINOv2 (ours)", "y_true": y_true_d,
            "y_pred": y_pred_d, "metrics": metrics_d,
        })

    if not model_results:
        log.error("  No models evaluated. Train both models first.")
        return

    # ── Comparison table ──────────────────────────────────────────────────────
    log.info("\n[Step 2] Building comparison table...")
    df_our = pd.DataFrame([r["metrics"] for r in model_results])

    # Append paper baselines as extra rows (accuracy + F1 only, rest NaN)
    baseline_rows = []
    for name, vals in PAPER_BASELINES.items():
        row = {col: None for col in df_our.columns}
        row["Model"] = name
        row["Accuracy"] = vals["accuracy"]
        row["F1 (weighted)"] = vals["f1"]
        baseline_rows.append(row)
    df_full = pd.concat([df_our, pd.DataFrame(baseline_rows)], ignore_index=True)

    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    df_full.to_csv(csv_path, index=False)
    log.info(f"  Saved: {csv_path}")

    # ── Combined confusion matrices ───────────────────────────────────────────
    if len(model_results) >= 1:
        log.info("\n[Step 3] Saving confusion matrices...")
        save_combined_confusion_matrices(model_results)

    # ── Comparison chart vs paper ─────────────────────────────────────────────
    log.info("\n[Step 4] Saving comparison chart...")
    save_comparison_chart(df_our, PAPER_BASELINES)

    # ── Final printed table ───────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  FINAL RESULTS — Test Set")
    log.info("=" * 65)

    display_cols = ["Model", "Accuracy", "F1 (weighted)", "Top-2 Accuracy",
                    "Recall (Autumn)"]
    available = [c for c in display_cols if c in df_our.columns]
    log.info("\n" + df_our[available].to_string(index=False))

    log.info("\n  Paper baselines (Stacchio et al., 2024):")
    for name, vals in PAPER_BASELINES.items():
        log.info(f"    {name:<22}: acc={vals['accuracy']:.3f}  f1={vals['f1']:.3f}")

    log.info("\n  Autumn recall (expected hardest class):")
    for r in model_results:
        log.info(f"    {r['name']:<22}: {r['metrics']['Recall (Autumn)']:.4f}")

    best = max(model_results, key=lambda r: r["metrics"]["Accuracy"])
    log.info(f"\n  Best model: {best['name']}  "
             f"(acc={best['metrics']['Accuracy']:.4f})")

    log.info("\n" + "=" * 65)
    log.info("  Saved files:")
    log.info(f"    {RESULTS_DIR}/comparison_table.csv")
    log.info(f"    {RESULTS_DIR}/confusion_farl.png")
    log.info(f"    {RESULTS_DIR}/confusion_dinov2.png")
    log.info(f"    {RESULTS_DIR}/confusion_matrices.png")
    log.info(f"    {RESULTS_DIR}/model_comparison.png")
    log.info("=" * 65)


if __name__ == "__main__":
    run()
