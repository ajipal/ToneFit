"""
inference.py — Deep Armocromia Hierarchical Inference
------------------------------------------------------
Runs the full preprocessing pipeline on a raw photo and predicts
Season + Sub-Type using the trained hierarchical FaRL-64 model.

Preprocessing mirrors the RGB-M dataset creation pipeline:
  1. Face detection (Facer)
  2. Face segmentation — extract hair, skin, nose, brows, eyes, mouth masks
  3. Merge segments → single binary mask
  4. Apply RGB mask (zero out non-face pixels)
  5. Resize to 224×224
  6. Val transforms: Resize(256) → CenterCrop(224) → ToTensor → Normalize

Usage:
    python inference.py --image path/to/photo.jpg
    python inference.py --image photo.jpg --checkpoint results/best_hierarchical.pth
    python inference.py --image photo.jpg --no_facer   # skip masking, use raw crop
"""

import argparse
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from hierarchical_head import DeepArmocromiaHierarchical

# ── Label definitions (must match train.py) ───────────────────────────────────

SEASON_CLASSES = ["autumn", "spring", "summer", "winter"]
SUBTYPE_CLASSES = [
    "autumn_deep", "autumn_soft", "autumn_warm",
    "spring_bright", "spring_light", "spring_warm",
    "summer_cool", "summer_light", "summer_soft",
    "winter_bright", "winter_cool", "winter_deep",
]

# Facer segment names to include in the face mask (covers all models)
# CelebAMask-HQ label names used by farl/celebm models
INCLUDE_SEGMENTS = {
    "skin", "face", "nose", "l_brow", "r_brow", "l_eye", "r_eye",
    "eye_g", "mouth", "u_lip", "l_lip", "inner_mouth", "hair",
    # LaPa model names
    "left_brow", "right_brow", "left_eye", "right_eye",
    "upper_lip", "lower_lip",
}

# ── Transforms (val — no augmentation, matches train.py) ─────────────────────

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Facer helpers ─────────────────────────────────────────────────────────────

def _load_facer():
    try:
        import facer
        detector = facer.face_detector("retinaface/mobilenet", device=DEVICE)
        parser   = facer.face_parser("farl/celebm/448", device=DEVICE)
        return facer, detector, parser
    except ImportError:
        return None, None, None
    except Exception:
        # Try alternative parser if celebm not available
        try:
            import facer
            detector = facer.face_detector("retinaface/mobilenet", device=DEVICE)
            parser   = facer.face_parser("farl/lapa/448", device=DEVICE)
            return facer, detector, parser
        except Exception:
            return None, None, None


def _detect_and_crop(facer_mod, detector, img_pil):
    """
    Detect faces, return the crop PIL image for the largest face.
    Returns (cropped_pil, bbox) or (None, None) if no face found.
    """
    import torch

    img_np = np.array(img_pil.convert("RGB"))
    # facer expects a uint8 HWC BGR or RGB tensor — use torch tensor
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        faces = detector(img_t)

    if faces is None or len(faces.get("rects", [])) == 0:
        return None, None

    rects = faces["rects"][0]  # [N, 4] in (x1, y1, x2, y2) format
    if rects.numel() == 0:
        return None, None

    # Pick largest bounding box by area
    areas = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
    best  = areas.argmax().item()
    x1, y1, x2, y2 = rects[best].int().tolist()

    H, W = img_np.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = img_pil.crop((x1, y1, x2, y2)).convert("RGB")
    return crop, (x1, y1, x2, y2)


def _segment_and_mask(facer_mod, parser, img_pil):
    """
    Run Facer segmentation on a face crop.
    Returns masked PIL image (non-face pixels set to 0) or the original
    crop if segmentation produces an empty mask.
    """
    import torch

    img_np = np.array(img_pil.convert("RGB"))
    H, W   = img_np.shape[:2]
    img_t  = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        # Parser expects a dict with image tensor — build minimal input
        faces_input = {"image": img_t, "rects": torch.tensor([[0, 0, W, H]], dtype=torch.float32, device=DEVICE).unsqueeze(0)}
        try:
            parsed = parser(faces_input)
        except Exception:
            # Some Facer versions take raw tensor
            try:
                parsed = parser(img_t)
            except Exception:
                return img_pil

    # Extract segmentation map
    seg = None
    for key in ("seg", "seg_logits", "parsing", "label"):
        if key in parsed:
            seg = parsed[key]
            break

    if seg is None:
        return img_pil

    # seg shape: [1, C, H, W] logits or [1, H, W] label map
    if seg.dim() == 4:
        seg = seg.squeeze(0).argmax(dim=0)   # [H, W]
    elif seg.dim() == 3:
        seg = seg.squeeze(0)                  # [H, W]

    # Build binary mask from included segment classes
    label_names = None
    if hasattr(parser, "labels"):
        label_names = parser.labels
    elif hasattr(parser, "label_names"):
        label_names = parser.label_names

    if label_names is not None:
        mask = torch.zeros_like(seg, dtype=torch.bool)
        for idx, name in enumerate(label_names):
            if name.lower() in INCLUDE_SEGMENTS:
                mask |= (seg == idx)
    else:
        # No label names available — include everything except background (0)
        mask = seg > 0

    mask_np = mask.cpu().numpy().astype(np.uint8)   # H×W, 0 or 1

    if mask_np.sum() < 100:
        # Mask too small — return original crop
        return img_pil

    # Apply mask: zero out non-face pixels
    masked = img_np.copy()
    masked[mask_np == 0] = 0

    return Image.fromarray(masked)


def preprocess_image(image_path: str, use_facer: bool = True):
    """
    Full preprocessing pipeline for a raw input photo.

    Steps:
      1. Load image
      2. Detect face (Facer) — largest bbox if multiple faces
      3. Segment face → binary mask (hair + skin + eyes + nose + mouth)
      4. Apply mask (zero out background)
      5. Resize masked crop to 224×224

    Returns:
      tensor (1, 3, 224, 224) ready for model inference,
      or raises RuntimeError if no face is found.
    """
    img_pil = Image.open(image_path).convert("RGB")

    if use_facer:
        facer_mod, detector, parser = _load_facer()
    else:
        facer_mod = None

    if facer_mod is not None and detector is not None:
        crop, bbox = _detect_and_crop(facer_mod, detector, img_pil)
        if crop is None:
            raise RuntimeError(
                "No face detected in image. Please use a clear front-facing photo."
            )
        masked = _segment_and_mask(facer_mod, parser, crop)
    else:
        # Facer unavailable — use center crop as fallback
        W, H   = img_pil.size
        margin = 0.1
        x1, y1 = int(W * margin), int(H * margin)
        x2, y2 = int(W * (1 - margin)), int(H * (1 - margin))
        masked = img_pil.crop((x1, y1, x2, y2)).convert("RGB")

    # Resize to 224×224 before val transforms
    masked = masked.resize((224, 224), Image.LANCZOS)

    # Apply val transforms (Resize(256) → CenterCrop(224) handled internally)
    tensor = VAL_TRANSFORM(masked).unsqueeze(0)  # [1, 3, 224, 224]
    return tensor


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str):
    """Load DeepArmocromiaHierarchical from best_hierarchical.pth."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "  Run: python train.py --config configs/hierarchical.yaml"
        )

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    # Determine backbone checkpoint path (same directory as model checkpoint,
    # or fall back to configs/hierarchical.yaml setting)
    try:
        import yaml
        with open("configs/hierarchical.yaml") as f:
            cfg = yaml.safe_load(f)
        backbone_ckpt = cfg["model"].get("checkpoint_path")
    except Exception:
        backbone_ckpt = None

    model = DeepArmocromiaHierarchical(
        checkpoint_path=backbone_ckpt,
        feature_dim=512,
        shared_dim=256,
        dropout=0.5,
    )

    if "head_state" in ckpt:
        model.head.load_state_dict(ckpt["head_state"])
    else:
        # Attempt to load whole state_dict
        model.load_state_dict(ckpt, strict=False)

    model.to(DEVICE)
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, tensor):
    """
    Run inference and return (season_probs, subtype_probs) as numpy arrays.
    Uses soft conditioning (hard_season=False) — matches training behaviour.
    """
    tensor = tensor.to(DEVICE)
    season_logits, subtype_logits = model(tensor, hard_season=False)
    season_probs  = F.softmax(season_logits,  dim=1).squeeze(0).cpu().numpy()
    subtype_probs = F.softmax(subtype_logits, dim=1).squeeze(0).cpu().numpy()
    return season_probs, subtype_probs


# ── Output formatting ─────────────────────────────────────────────────────────

def _display_name(label: str) -> str:
    """'autumn_deep' → 'Autumn-Deep'"""
    return label.replace("_", "-").title()


def print_results(season_probs, subtype_probs):
    season_idx  = int(np.argmax(season_probs))
    subtype_idx = int(np.argmax(subtype_probs))

    season_name  = _display_name(SEASON_CLASSES[season_idx])
    subtype_name = _display_name(SUBTYPE_CLASSES[subtype_idx])

    print(f"\nSeason:   {season_name} ({season_probs[season_idx]*100:.1f}%)")
    print(f"Sub-Type: {subtype_name} ({subtype_probs[subtype_idx]*100:.1f}%)")

    print("\nAll season confidences:")
    order = np.argsort(season_probs)[::-1]
    for i in order:
        print(f"  {_display_name(SEASON_CLASSES[i]):<8}: {season_probs[i]*100:5.1f}%")

    print("\nAll sub-type confidences:")
    order = np.argsort(subtype_probs)[::-1]
    for i in order:
        print(f"  {_display_name(SUBTYPE_CLASSES[i]):<16}: {subtype_probs[i]*100:5.1f}%")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict personal color season from a photo."
    )
    parser.add_argument("--image",      required=True, help="Path to input photo")
    parser.add_argument(
        "--checkpoint",
        default="results/best_hierarchical.pth",
        help="Path to trained model checkpoint (default: results/best_hierarchical.pth)",
    )
    parser.add_argument(
        "--no_facer",
        action="store_true",
        help="Skip Facer face detection/segmentation (use center crop fallback)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Device : {DEVICE}")
    print(f"[INFO] Image  : {args.image}")

    try:
        print("[INFO] Preprocessing...")
        tensor = preprocess_image(args.image, use_facer=not args.no_facer)
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Loading model...")
    try:
        model = load_model(args.checkpoint)
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    season_probs, subtype_probs = predict(model, tensor)
    print_results(season_probs, subtype_probs)


if __name__ == "__main__":
    main()
