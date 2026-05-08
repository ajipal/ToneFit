"""
hierarchical_head.py
--------------------
Two-stage hierarchical classifier head for Deep Armocromia.

Architecture:
    FaRL-64 backbone: timm ViT-B/16 CLS token → 768-d (frozen)
        └── Shared FC: 768 → 384, ReLU, Dropout(0.5)
             ├── Stage 1: FC 384 → 4   (Season classification)
             └── Stage 2: FC 388 → 12  (Sub-Type, conditioned on Stage 1 softmax output)

The key novelty over the original paper:
    Stage 2 input = concat(shared_features[384], stage1_softmax[4]) = 388-d
    This lets the sub-type head explicitly condition on the predicted season.

Usage:
    model = HierarchicalArmocromiaHead(feature_dim=768)
    season_logits, subtype_logits = model(features)

    # Joint loss
    loss = criterion_season(season_logits, season_labels) \
         + lambda_weight * criterion_subtype(subtype_logits, subtype_labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Label definitions (must match dataset) ────────────────────────────────────

SEASON_CLASSES = ["Autumn", "Summer", "Winter", "Spring"]          # 4 classes
SUBTYPE_CLASSES = [                                                 # 12 classes
    "Autumn_Deep", "Autumn_Soft", "Autumn_Warm",
    "Summer_Cool", "Summer_Light", "Summer_Soft",
    "Winter_Bright", "Winter_Cool", "Winter_Deep",
    "Spring_Bright", "Spring_Light", "Spring_Warm",
]

NUM_SEASONS = len(SEASON_CLASSES)    # 4
NUM_SUBTYPES = len(SUBTYPE_CLASSES)  # 12


# ── Main model ─────────────────────────────────────────────────────────────────

class HierarchicalArmocromiaHead(nn.Module):
    """
    Two-stage hierarchical head on top of a frozen FaRL-64 backbone.

    Args:
        feature_dim (int): Output dimension of the FaRL backbone (default: 768,
                           matching timm ViT-B/16 CLS token output).
        shared_dim (int): Hidden dimension of the shared FC layer (default: 384,
                          i.e. feature_dim // 2, matching the original paper).
        dropout (float): Dropout probability (default: 0.5, matching original paper).
    """

    def __init__(
        self,
        feature_dim: int = 768,
        shared_dim: int = 384,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Shared feature compression (mirrors original paper's first FC layer)
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # Stage 1 — Season classification (4 classes)
        self.stage1 = nn.Linear(shared_dim, NUM_SEASONS)

        # Stage 2 — Sub-type classification (12 classes)
        # Input: shared features (384) + season softmax (4) = 388
        self.stage2 = nn.Linear(shared_dim + NUM_SEASONS, NUM_SUBTYPES)

    def forward(
        self,
        x: torch.Tensor,
        hard_season: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (Tensor): FaRL feature vectors, shape [B, feature_dim].
            hard_season (bool): If True, Stage 2 uses the argmax one-hot of
                                Stage 1 instead of soft probabilities.
                                Use False during training, optionally True
                                at inference for cleaner conditioning.

        Returns:
            season_logits  (Tensor): Shape [B, 4]
            subtype_logits (Tensor): Shape [B, 12]
        """
        # Shared compression
        shared = self.shared(x)                         # [B, 384]

        # Stage 1 — season
        season_logits = self.stage1(shared)             # [B, 4]

        # Conditioning signal for Stage 2
        if hard_season:
            season_idx = season_logits.argmax(dim=1)    # [B]
            season_cond = F.one_hot(                    # [B, 4]
                season_idx, num_classes=NUM_SEASONS
            ).float()
        else:
            season_cond = F.softmax(season_logits, dim=1)   # [B, 4]

        # Concatenate shared features + season conditioning
        stage2_input = torch.cat([shared, season_cond], dim=1)  # [B, 388]

        # Stage 2 — sub-type
        subtype_logits = self.stage2(stage2_input)      # [B, 12]

        return season_logits, subtype_logits


# ── Joint loss helper ──────────────────────────────────────────────────────────

class HierarchicalLoss(nn.Module):
    """
    Weighted sum of season CE loss + sub-type CE loss.

    Args:
        lambda_subtype (float): Weight for the sub-type loss (default: 1.0).
                                Increase if sub-type performance lags.
    """

    def __init__(self, lambda_subtype: float = 1.0):
        super().__init__()
        self.lambda_subtype = lambda_subtype
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        season_logits: torch.Tensor,
        subtype_logits: torch.Tensor,
        season_labels: torch.Tensor,
        subtype_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss, season_loss, subtype_loss
        """
        loss_season = self.ce(season_logits, season_labels)
        loss_subtype = self.ce(subtype_logits, subtype_labels)
        total = loss_season + self.lambda_subtype * loss_subtype
        return total, loss_season, loss_subtype


# ── FaRL backbone wrapper ──────────────────────────────────────────────────────

def _remap_clip_to_timm(state: dict) -> dict:
    """
    Remap CLIP ViT-B/16 parameter names to timm ViT-B/16 names.

    Called automatically when the checkpoint contains CLIP-style keys
    (e.g. conv1.weight, transformer.resblocks.X, class_embedding).
    The CLIP projection layer (768→512) is dropped so the backbone always
    returns 768-d CLS token features, matching the hierarchical head's
    expected feature_dim.
    """
    out = {}
    for k, v in state.items():
        # Drop the final linear projection (CLIP-specific; timm doesn't have it)
        if k == "proj":
            continue
        # Pre-transformer norm not present in timm's standard ViT
        if k in ("ln_pre.weight", "ln_pre.bias"):
            continue

        if k == "class_embedding":          # [768] → [1, 1, 768]
            out["cls_token"] = v.unsqueeze(0).unsqueeze(0)
        elif k == "positional_embedding":   # [197, 768] → [1, 197, 768]
            out["pos_embed"] = v.unsqueeze(0)
        elif k == "conv1.weight":
            out["patch_embed.proj.weight"] = v
        elif k in ("ln_post.weight", "ln_post.bias"):
            out[k.replace("ln_post.", "norm.")] = v
        elif k.startswith("transformer.resblocks."):
            nk = k.replace("transformer.resblocks.", "blocks.")
            nk = nk.replace(".attn.in_proj_weight", ".attn.qkv.weight")
            nk = nk.replace(".attn.in_proj_bias",   ".attn.qkv.bias")
            nk = nk.replace(".attn.out_proj.",       ".attn.proj.")
            nk = nk.replace(".ln_1.", ".norm1.")
            nk = nk.replace(".ln_2.", ".norm2.")
            nk = nk.replace(".mlp.c_fc.",   ".mlp.fc1.")
            nk = nk.replace(".mlp.c_proj.", ".mlp.fc2.")
            out[nk] = v
        else:
            out[k] = v
    return out


class FaRLBackbone(nn.Module):
    """
    Frozen FaRL-64 feature extractor.

    Accepts three checkpoint formats:
      1. Official FaRL/timm checkpoint  — weights under 'model' key, timm key names
      2. train_farl.py output           — weights under 'backbone' key, CLIP key names
      3. Raw CLIP visual encoder        — CLIP key names at the top level

    Returns the [CLS] token embedding (768-d) before CLIP's projection layer.
    NOTE: DeepArmocromiaHierarchical does NOT use FaRLBackbone — it uses the
    same CLIP encode_image() path as train_farl.py (512-d output). FaRLBackbone
    is kept here for standalone use only.
    """

    def __init__(self, checkpoint_path: str | None = None):
        super().__init__()
        import timm

        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
        )

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")

            # Unwrap nested checkpoints
            if isinstance(state, dict):
                if "backbone" in state:
                    state = state["backbone"]       # train_farl.py format
                elif "model" in state:
                    state = state["model"]          # official FaRL format

            # Strip 'visual.' prefix (official FaRL stores weights under it)
            if any(k.startswith("visual.") for k in state):
                state = {k.replace("visual.", ""): v
                         for k, v in state.items()
                         if k.startswith("visual.") or "visual." not in k}

            # Remap CLIP key names → timm key names when needed
            if any(k in state for k in ("conv1.weight", "class_embedding", "transformer.resblocks.0.attn.in_proj_weight")):
                state = _remap_clip_to_timm(state)

            missing, unexpected = self.model.load_state_dict(state, strict=False)
            loaded = len(state) - len(unexpected)
            print(f"[FaRLBackbone] Loaded {loaded}/{len(state)} keys "
                  f"| missing={len(missing)} unexpected={len(unexpected)}")

        # Freeze all backbone parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Preprocessed images, shape [B, 3, 224, 224].

        Returns:
            Tensor: CLS token features, shape [B, 768].
        """
        return self.model(x)


# ── Full pipeline (backbone + head) ───────────────────────────────────────────

class DeepArmocromiaHierarchical(nn.Module):
    """
    End-to-end model: frozen FaRL-64 backbone + hierarchical classification head.

    This is the model you instantiate for training and evaluation.
    Only the head parameters are trainable.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        feature_dim: int = 768,
        shared_dim: int = 384,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.backbone = FaRLBackbone(checkpoint_path)
        self.head = HierarchicalArmocromiaHead(feature_dim, shared_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        hard_season: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.head(features, hard_season=hard_season)

    def trainable_parameters(self):
        """Returns only the head parameters (backbone is frozen)."""
        return self.head.parameters()


# ── Quick smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running smoke test...")

    head = HierarchicalArmocromiaHead(feature_dim=768)
    dummy_features = torch.randn(8, 768)    # batch of 8

    season_logits, subtype_logits = head(dummy_features)

    assert season_logits.shape == (8, 4),  f"Expected (8,4), got {season_logits.shape}"
    assert subtype_logits.shape == (8, 12), f"Expected (8,12), got {subtype_logits.shape}"

    # Test loss
    criterion = HierarchicalLoss(lambda_subtype=1.0)
    season_labels = torch.randint(0, 4, (8,))
    subtype_labels = torch.randint(0, 12, (8,))
    total, ls, lsub = criterion(season_logits, subtype_logits, season_labels, subtype_labels)

    print(f"  season_logits:  {season_logits.shape}")
    print(f"  subtype_logits: {subtype_logits.shape}")
    print(f"  total_loss={total.item():.4f}  season_loss={ls.item():.4f}  subtype_loss={lsub.item():.4f}")

    # Count trainable params
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"  trainable head params: {n_params:,}")
    print("Smoke test passed.")
