"""
ToneFit ML — Model Evaluation (Step 6)
========================================
Evaluates all three trained models on the held-out test set and compares them.

Models evaluated:
  A1. SVM            — models/svm_model.pkl
  A2. Random Forest  — models/rf_model.pkl
  B.  MobileNetV2    — models/mobilenetv2_model.h5

Metrics computed per model:
  - Accuracy
  - Precision, Recall, F1-score (per class + macro average)
  - Top-2 accuracy (is the correct season in the top 2 predictions?)
  - Confusion matrix (saved as heatmap)

What this script does:
  1. Loads X_test / y_test for SVM and RF evaluation
  2. Loads test images from dataset/ for MobileNetV2 evaluation
  3. Generates classification reports and confusion matrix plots
  4. Computes top-2 accuracy for all models
  5. Saves a side-by-side comparison table to results/comparison_table.csv
  6. Prints the final comparison

Requirements:
  pip install scikit-learn tensorflow numpy pandas matplotlib seaborn

Usage:
  python evaluate.py

Run preprocess.py, train_traditional.py, and train_deeplearning.py first.
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATASET_DIR = "dataset"
MODELS_DIR  = "models"
RESULTS_DIR = "results"
SEASONS     = ["spring", "summer", "autumn", "winter"]
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
RANDOM_STATE = 42

SVM_PATH  = os.path.join(MODELS_DIR, "svm_model.pkl")
RF_PATH   = os.path.join(MODELS_DIR, "rf_model.pkl")
DL_PATH   = os.path.join(MODELS_DIR, "mobilenetv2_model.h5")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── SETUP ────────────────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_traditional_test_data():
    """Load the preprocessed test arrays for SVM and RF."""
    for f in ["X_test.npy", "y_test.npy"]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"'{f}' not found. Run preprocess.py first.")
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
    log.info(f"  X_test: {X_test.shape}  —  y_test: {y_test.shape}")
    return X_test, y_test


def load_dl_test_generator():
    """
    Build a test image generator for MobileNetV2.
    Uses the same 20% validation split as training — no augmentation.
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.20,
    )
    gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=SEASONS,
        subset="validation",
        seed=RANDOM_STATE,
        shuffle=False,   # must be False so y_true order matches predictions
    )
    return gen


def top2_accuracy(y_true, y_prob):
    """
    Top-2 accuracy: the correct class is among the 2 highest-probability predictions.
    Useful when the model is unsure between two adjacent seasons (e.g. Spring vs Autumn).
    """
    return float(top_k_accuracy_score(y_true, y_prob, k=2))


def save_confusion_matrix(y_true, y_pred, model_name, filename):
    """Save a labeled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[s.capitalize() for s in SEASONS],
        yticklabels=[s.capitalize() for s in SEASONS],
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}\n(Test Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {path}")


def print_report(model_name, y_true, y_pred):
    """Print a formatted classification report."""
    log.info(f"\n  Classification Report — {model_name}:")
    report = classification_report(y_true, y_pred, target_names=SEASONS)
    for line in report.split("\n"):
        log.info("    " + line)


def evaluate_model(name, y_true, y_pred, y_prob, cm_filename):
    """
    Run all metrics for one model.
    Returns a dict of scalar metrics for the comparison table.
    """
    acc   = accuracy_score(y_true, y_pred)
    top2  = top2_accuracy(y_true, y_prob)
    report_dict = classification_report(
        y_true, y_pred,
        target_names=SEASONS,
        output_dict=True,
    )

    print_report(name, y_true, y_pred)
    save_confusion_matrix(y_true, y_pred, name, cm_filename)

    # Per-class recall (important for Autumn which is historically hard)
    per_class_recall = {s: round(report_dict[s]["recall"], 4) for s in SEASONS}
    log.info(f"\n  Per-class Recall (Autumn is historically the hardest):")
    for season, recall in per_class_recall.items():
        flag = "  ← watch this" if season == "autumn" else ""
        log.info(f"    {season:<8}: {recall:.4f}{flag}")

    return {
        "Model":        name,
        "Accuracy":     round(acc, 4),
        "Top-2 Acc":    round(top2, 4),
        "Precision (macro)": round(report_dict["macro avg"]["precision"], 4),
        "Recall (macro)":    round(report_dict["macro avg"]["recall"], 4),
        "F1 (macro)":        round(report_dict["macro avg"]["f1-score"], 4),
        "Recall (Spring)":   per_class_recall["spring"],
        "Recall (Summer)":   per_class_recall["summer"],
        "Recall (Autumn)":   per_class_recall["autumn"],
        "Recall (Winter)":   per_class_recall["winter"],
    }


# ─── COMBINED CONFUSION MATRIX PLOT ───────────────────────────────────────────

def save_combined_confusion_matrices(results_data, y_trues, y_preds):
    """
    Save all three confusion matrices side by side in one figure.
    Useful for the paper — easy visual comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    season_labels = [s.capitalize() for s in SEASONS]

    for ax, (name, y_true, y_pred) in zip(axes, zip(
        [r["Model"] for r in results_data], y_trues, y_preds
    )):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=season_labels,
            yticklabels=season_labels,
            linewidths=0.5,
            ax=ax,
        )
        acc = accuracy_score(y_true, y_pred)
        ax.set_title(f"{name}\nAccuracy: {acc:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

    fig.suptitle("Confusion Matrices — All Models (Test Set)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrices_combined.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {path}")


# ─── COMPARISON BAR CHART ─────────────────────────────────────────────────────

def save_comparison_chart(df):
    """Bar chart comparing key metrics across all three models."""
    metrics  = ["Accuracy", "F1 (macro)", "Top-2 Acc"]
    models   = df["Model"].tolist()
    x        = np.arange(len(metrics))
    width    = 0.25
    colors   = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [float(df.loc[df["Model"] == model, m].values[0]) for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model, color=color,
                      edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Test Set Performance",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "model_comparison_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info("  ToneFit ML — Model Evaluation")
    log.info("=" * 60)

    results_data = []
    all_y_true   = []
    all_y_pred   = []

    # ─────────────────────────────────────────────────────────────────────
    # Load traditional test data (shared by SVM and RF)
    # ─────────────────────────────────────────────────────────────────────
    log.info("\n[Step 1] Loading test data for traditional models...")
    X_test, y_test = load_traditional_test_data()

    # ─────────────────────────────────────────────────────────────────────
    # MODEL A1: SVM
    # ─────────────────────────────────────────────────────────────────────
    log.info("\n" + "─" * 60)
    log.info("  Evaluating: SVM")
    log.info("─" * 60)

    if not os.path.exists(SVM_PATH):
        log.warning(f"  {SVM_PATH} not found — skipping SVM.")
    else:
        with open(SVM_PATH, "rb") as f:
            svm = pickle.load(f)

        y_pred_svm  = svm.predict(X_test)
        y_prob_svm  = svm.predict_proba(X_test)   # shape: (n, 4)

        result = evaluate_model(
            name="SVM",
            y_true=y_test,
            y_pred=y_pred_svm,
            y_prob=y_prob_svm,
            cm_filename="confusion_matrix_svm.png",
        )
        results_data.append(result)
        all_y_true.append(y_test)
        all_y_pred.append(y_pred_svm)

    # ─────────────────────────────────────────────────────────────────────
    # MODEL A2: Random Forest
    # ─────────────────────────────────────────────────────────────────────
    log.info("\n" + "─" * 60)
    log.info("  Evaluating: Random Forest")
    log.info("─" * 60)

    if not os.path.exists(RF_PATH):
        log.warning(f"  {RF_PATH} not found — skipping Random Forest.")
    else:
        with open(RF_PATH, "rb") as f:
            rf = pickle.load(f)

        y_pred_rf  = rf.predict(X_test)
        y_prob_rf  = rf.predict_proba(X_test)

        result = evaluate_model(
            name="Random Forest",
            y_true=y_test,
            y_pred=y_pred_rf,
            y_prob=y_prob_rf,
            cm_filename="confusion_matrix_rf.png",
        )
        results_data.append(result)
        all_y_true.append(y_test)
        all_y_pred.append(y_pred_rf)

    # ─────────────────────────────────────────────────────────────────────
    # MODEL B: MobileNetV2
    # ─────────────────────────────────────────────────────────────────────
    log.info("\n" + "─" * 60)
    log.info("  Evaluating: MobileNetV2")
    log.info("─" * 60)

    if not os.path.exists(DL_PATH):
        log.warning(f"  {DL_PATH} not found — skipping MobileNetV2.")
    else:
        dl_model = tf.keras.models.load_model(DL_PATH)

        log.info("  Loading test image generator...")
        test_gen = load_dl_test_generator()

        log.info(f"  Running predictions on {test_gen.samples} images...")
        y_prob_dl = dl_model.predict(test_gen, verbose=1)   # shape: (n, 4)
        y_pred_dl = np.argmax(y_prob_dl, axis=1)
        y_true_dl = test_gen.classes                         # ground truth integers

        result = evaluate_model(
            name="MobileNetV2",
            y_true=y_true_dl,
            y_pred=y_pred_dl,
            y_prob=y_prob_dl,
            cm_filename="confusion_matrix_mobilenetv2.png",
        )
        results_data.append(result)
        all_y_true.append(y_true_dl)
        all_y_pred.append(y_pred_dl)

    # ─────────────────────────────────────────────────────────────────────
    # Comparison table and charts
    # ─────────────────────────────────────────────────────────────────────
    if len(results_data) == 0:
        log.error("\nNo models were evaluated. Train all models first.")
        return

    log.info("\n[Step 2] Building comparison table...")
    df = pd.DataFrame(results_data)

    # Save to CSV
    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    log.info(f"  Saved: {csv_path}")

    # Combined confusion matrix plot (if all 3 ran)
    if len(all_y_true) == 3:
        log.info("\n[Step 3] Saving combined confusion matrix plot...")
        save_combined_confusion_matrices(results_data, all_y_true, all_y_pred)

    # Bar chart comparison
    log.info("\n[Step 4] Saving comparison bar chart...")
    save_comparison_chart(df)

    # ─────────────────────────────────────────────────────────────────────
    # Final printed table
    # ─────────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  FINAL COMPARISON — Test Set Results")
    log.info("=" * 60)

    # Print key columns
    display_cols = ["Model", "Accuracy", "F1 (macro)", "Top-2 Acc",
                    "Recall (Autumn)"]
    log.info("\n  " + df[display_cols].to_string(index=False))

    # Highlight best model
    best_idx   = df["F1 (macro)"].idxmax()
    best_model = df.loc[best_idx, "Model"]
    best_f1    = df.loc[best_idx, "F1 (macro)"]
    log.info(f"\n  Best model by F1 (macro): {best_model}  ({best_f1:.4f})")

    # Autumn recall note (per paper requirement)
    log.info("\n  Autumn recall per model (historically hardest class):")
    for _, row in df.iterrows():
        log.info(f"    {row['Model']:<20}: {row['Recall (Autumn)']:.4f}")

    log.info("\n" + "=" * 60)
    log.info("  DONE")
    log.info("=" * 60)
    log.info(f"\n  Saved files:")
    log.info(f"    {RESULTS_DIR}/comparison_table.csv")
    log.info(f"    {RESULTS_DIR}/confusion_matrix_svm.png")
    log.info(f"    {RESULTS_DIR}/confusion_matrix_rf.png")
    log.info(f"    {RESULTS_DIR}/confusion_matrix_mobilenetv2.png")
    log.info(f"    {RESULTS_DIR}/confusion_matrices_combined.png")
    log.info(f"    {RESULTS_DIR}/model_comparison_chart.png")
    log.info(f"\n  Next step → python predict.py <image_path>")
    log.info("=" * 60)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
