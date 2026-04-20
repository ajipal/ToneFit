"""
ToneFit ML — Deep Learning Training (Step 5)
==============================================
Trains a MobileNetV2 image classifier on the face crop dataset.

Architecture:
  MobileNetV2 (pretrained on ImageNet, frozen base)
  → GlobalAveragePooling2D
  → Dense(128, ReLU)
  → Dropout(0.3)
  → Dense(4, Softmax)   ← 4 personal color seasons

Training strategy:
  Phase 1 — Feature extraction (base frozen, 30 epochs max)
    Train only the new top layers while MobileNetV2 weights are locked.
    This lets the model learn to use ImageNet features for our task
    without destroying them with random gradients from the new head.

  Phase 2 — Fine-tuning (top 30 layers unfrozen, 20 epochs max)
    Unfreeze the top portion of the base and continue training
    at a much lower learning rate so the pretrained weights
    are adjusted gently, not overwritten.

What this script does:
  1. Loads images from dataset/ using ImageDataGenerator (with augmentation)
  2. Builds MobileNetV2 transfer learning model
  3. Trains Phase 1 with early stopping
  4. Unfreezes top layers, trains Phase 2 (fine-tuning)
  5. Plots and saves training/validation loss and accuracy curves
  6. Saves: models/mobilenetv2_model.h5

Requirements:
  pip install tensorflow scikit-learn numpy matplotlib

Usage:
  python train_deeplearning.py

Notes:
  - Google Colab with GPU is strongly recommended (much faster than CPU)
  - To enable GPU on Colab: Runtime → Change runtime type → T4 GPU
  - Expects images in dataset/spring/, dataset/summer/, etc.
  - Run collect_data.py first to build the dataset
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATASET_DIR  = "dataset"
MODELS_DIR   = "models"
RESULTS_DIR  = "results"
MODEL_PATH   = os.path.join(MODELS_DIR, "mobilenetv2_model.h5")

SEASONS      = ["spring", "summer", "autumn", "winter"]
NUM_CLASSES  = 4
IMG_SIZE     = (224, 224)   # MobileNetV2 default input size
BATCH_SIZE   = 32
RANDOM_STATE = 42

# Phase 1 — frozen base
P1_EPOCHS    = 30
P1_LR        = 1e-4
P1_PATIENCE  = 5            # early stopping patience

# Phase 2 — fine-tuning (unfreeze top N layers of base)
P2_UNFREEZE  = 30           # unfreeze the last 30 layers of MobileNetV2
P2_EPOCHS    = 20
P2_LR        = 1e-5         # much smaller LR to avoid destroying pretrained weights
P2_PATIENCE  = 5

VALIDATION_SPLIT = 0.20

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── SETUP ────────────────────────────────────────────────────────────────────

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Reproducibility
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ─── STEP 1: DATA LOADING & AUGMENTATION ──────────────────────────────────────

def build_generators():
    """
    Create training and validation data generators.

    Training augmentation:
      - Horizontal flip (faces can be mirrored)
      - Small rotation (±15°) — slight head tilt variation
      - Small zoom (±10%) — different photo crops
      - Small brightness shift (±10%) — lighting variation
    Validation: no augmentation, only rescaling.

    MobileNetV2 expects pixel values in [-1, 1], which
    preprocess_input() handles for us.
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=VALIDATION_SPLIT,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.10,
        brightness_range=[0.9, 1.1],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=VALIDATION_SPLIT,
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=SEASONS,        # fixes label order: spring=0, summer=1, autumn=2, winter=3
        subset="training",
        seed=RANDOM_STATE,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=SEASONS,
        subset="validation",
        seed=RANDOM_STATE,
        shuffle=False,
    )

    return train_gen, val_gen


# ─── STEP 2: BUILD MODEL ──────────────────────────────────────────────────────

def build_model():
    """
    Build the transfer learning model.

    Base: MobileNetV2 pretrained on ImageNet.
      - include_top=False removes the original 1000-class classifier
      - weights="imagenet" loads pretrained weights
      - We freeze the base so Phase 1 only trains our new top layers

    Top layers we add:
      GlobalAveragePooling2D  — flattens the spatial feature map
      Dense(128, relu)        — learns season-specific combinations
      Dropout(0.3)            — reduces overfitting
      Dense(4, softmax)       — outputs probability for each season
    """
    # Load MobileNetV2 base (no top classifier)
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False   # freeze all base layers for Phase 1
    log.info(f"  MobileNetV2 base: {len(base_model.layers)} layers (all frozen)")

    # Build full model
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)   # training=False keeps BN layers in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    # Class weights to handle imbalance (Autumn/Winter skew in Filipino celebrities)
    # Keras computes these for us later from y labels
    model.compile(
        optimizer=optimizers.Adam(learning_rate=P1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    log.info(f"  Total parameters    : {model.count_params():,}")
    log.info(f"  Trainable parameters: {sum(tf.size(w).numpy() for w in model.trainable_weights):,}")

    return model, base_model


# ─── STEP 3: CLASS WEIGHTS ────────────────────────────────────────────────────

def compute_class_weights(train_gen):
    """
    Compute per-class weights inversely proportional to class frequency.
    Passed to model.fit() to handle class imbalance.
    """
    from sklearn.utils.class_weight import compute_class_weight

    labels = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    log.info("\n  Class weights (to handle imbalance):")
    for i, season in enumerate(SEASONS):
        log.info(f"    {season:<8}: {class_weight_dict[i]:.4f}")
    return class_weight_dict


# ─── STEP 4: TRAINING CALLBACKS ───────────────────────────────────────────────

def make_callbacks(monitor="val_accuracy", prefix="p1"):
    """
    EarlyStopping   — stops training if val_accuracy stops improving
    ModelCheckpoint — saves the best weights during training
    """
    checkpoint_path = os.path.join(MODELS_DIR, f"best_{prefix}.h5")
    return [
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=P1_PATIENCE if prefix == "p1" else P2_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ─── STEP 5: PLOT CURVES ──────────────────────────────────────────────────────

def plot_history(history_p1, history_p2=None):
    """
    Plot training and validation loss/accuracy across both phases.
    A vertical dashed line marks where Phase 2 (fine-tuning) begins.
    """
    # Combine histories if both phases ran
    if history_p2 is not None:
        acc     = history_p1.history["accuracy"]     + history_p2.history["accuracy"]
        val_acc = history_p1.history["val_accuracy"] + history_p2.history["val_accuracy"]
        loss    = history_p1.history["loss"]         + history_p2.history["loss"]
        val_loss= history_p1.history["val_loss"]     + history_p2.history["val_loss"]
        phase2_start = len(history_p1.history["accuracy"])
    else:
        acc     = history_p1.history["accuracy"]
        val_acc = history_p1.history["val_accuracy"]
        loss    = history_p1.history["loss"]
        val_loss= history_p1.history["val_loss"]
        phase2_start = None

    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy ──────────────────────────────────────────────────────────
    ax1.plot(epochs, acc,     label="Train Accuracy", color="#4C72B0", linewidth=2)
    ax1.plot(epochs, val_acc, label="Val Accuracy",   color="#DD8452", linewidth=2)
    if phase2_start:
        ax1.axvline(phase2_start + 0.5, color="gray", linestyle="--", linewidth=1.2,
                    label="Fine-tuning starts")
    ax1.set_title("Model Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Loss ──────────────────────────────────────────────────────────────
    ax2.plot(epochs, loss,     label="Train Loss", color="#4C72B0", linewidth=2)
    ax2.plot(epochs, val_loss, label="Val Loss",   color="#DD8452", linewidth=2)
    if phase2_start:
        ax2.axvline(phase2_start + 0.5, color="gray", linestyle="--", linewidth=1.2,
                    label="Fine-tuning starts")
    ax2.set_title("Model Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Categorical Cross-Entropy", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("MobileNetV2 Training Curves", fontsize=15, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "mobilenetv2_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info("  ToneFit ML — Deep Learning Training (MobileNetV2)")
    log.info("=" * 60)
    log.info(f"\n  TensorFlow version : {tf.__version__}")
    log.info(f"  GPU available      : {len(tf.config.list_physical_devices('GPU')) > 0}")

    # ── Load data ──────────────────────────────────────────────────────────
    log.info("\n[Step 1] Loading dataset from dataset/ ...")
    train_gen, val_gen = build_generators()

    log.info(f"\n  Training images  : {train_gen.samples}")
    log.info(f"  Validation images: {val_gen.samples}")
    log.info(f"  Image size       : {IMG_SIZE}")
    log.info(f"  Batch size       : {BATCH_SIZE}")
    log.info(f"  Class mapping    : {train_gen.class_indices}")

    # ── Build model ────────────────────────────────────────────────────────
    log.info("\n[Step 2] Building MobileNetV2 model...")
    model, base_model = build_model()

    # ── Class weights ──────────────────────────────────────────────────────
    class_weights = compute_class_weights(train_gen)

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 1 — Feature Extraction (base frozen)
    # ─────────────────────────────────────────────────────────────────────
    log.info("\n" + "─" * 60)
    log.info(f"  PHASE 1 — Feature Extraction (base frozen)")
    log.info(f"  Max epochs: {P1_EPOCHS}   LR: {P1_LR}   Early stopping patience: {P1_PATIENCE}")
    log.info("─" * 60)

    history_p1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=P1_EPOCHS,
        callbacks=make_callbacks(prefix="p1"),
        class_weight=class_weights,
        verbose=1,
    )

    p1_best_val_acc = max(history_p1.history["val_accuracy"])
    log.info(f"\n  Phase 1 best val accuracy: {p1_best_val_acc:.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 2 — Fine-tuning (unfreeze top layers)
    # ─────────────────────────────────────────────────────────────────────
    log.info("\n" + "─" * 60)
    log.info(f"  PHASE 2 — Fine-tuning (unfreeze top {P2_UNFREEZE} base layers)")
    log.info(f"  Max epochs: {P2_EPOCHS}   LR: {P2_LR}   Early stopping patience: {P2_PATIENCE}")
    log.info("─" * 60)

    # Unfreeze base model, then re-freeze everything except the top N layers
    base_model.trainable = True
    total_base_layers = len(base_model.layers)
    freeze_until = total_base_layers - P2_UNFREEZE

    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    trainable_now = sum(1 for l in base_model.layers if l.trainable)
    log.info(f"  Base layers now trainable: {trainable_now} / {total_base_layers}")

    # Recompile with lower learning rate — must recompile after changing trainability
    model.compile(
        optimizer=optimizers.Adam(learning_rate=P2_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_p2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=P2_EPOCHS,
        callbacks=make_callbacks(prefix="p2"),
        class_weight=class_weights,
        verbose=1,
    )

    p2_best_val_acc = max(history_p2.history["val_accuracy"])
    log.info(f"\n  Phase 2 best val accuracy: {p2_best_val_acc:.4f}")

    # ── Plot training curves ───────────────────────────────────────────────
    log.info("\n[Step 3] Saving training curves...")
    plot_history(history_p1, history_p2)

    # ── Save final model ───────────────────────────────────────────────────
    log.info("\n[Step 4] Saving model...")
    model.save(MODEL_PATH)
    log.info(f"  Model saved: {MODEL_PATH}")

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  DONE — Deep Learning Training Complete")
    log.info("=" * 60)
    log.info(f"  Phase 1 best val accuracy : {p1_best_val_acc:.4f}")
    log.info(f"  Phase 2 best val accuracy : {p2_best_val_acc:.4f}")
    log.info(f"\n  Saved files:")
    log.info(f"    {MODEL_PATH}")
    log.info(f"    {RESULTS_DIR}/mobilenetv2_training_curves.png")
    log.info(f"\n  Next step → python evaluate.py")
    log.info("=" * 60)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
