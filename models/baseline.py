"""
models/baseline.py
------------------
Flat 2-FC head replicating the original Deep Armocromia paper (Stacchio et al., 2024).

Architecture (per task):
    FaRL-64 features (768-d, frozen)
        └── FC 768 → 384, ReLU, Dropout(0.5) → FC 384 → N

Two independent heads — one for Season (4-class), one for Sub-Type (12-class).
No cross-conditioning; season and sub-type are predicted independently.

Usage:
    python models/baseline.py     # runs smoke test
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from hierarchical_head import FaRLBackbone

NUM_SEASONS  = 4
NUM_SUBTYPES = 12


class FlatHead(nn.Module):
    """Standard 2-FC classification head from the original paper."""

    def __init__(self, feature_dim: int = 768, num_classes: int = 4, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineArmocromia(nn.Module):
    """
    Original Deep Armocromia flat baseline.

    Both heads share the same frozen FaRL-64 features but are fully
    independent — no conditioning of sub-type on season.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        feature_dim: int = 768,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.backbone    = FaRLBackbone(checkpoint_path)
        self.season_head = FlatHead(feature_dim, NUM_SEASONS, dropout)
        self.subtype_head = FlatHead(feature_dim, NUM_SUBTYPES, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.season_head(features), self.subtype_head(features)

    def trainable_parameters(self):
        return list(self.season_head.parameters()) + list(self.subtype_head.parameters())


if __name__ == "__main__":
    print("Running baseline smoke test...")
    model = BaselineArmocromia()
    dummy = torch.randn(4, 3, 224, 224)
    season_logits, subtype_logits = model(dummy)
    assert season_logits.shape  == (4, NUM_SEASONS),  f"Bad shape: {season_logits.shape}"
    assert subtype_logits.shape == (4, NUM_SUBTYPES), f"Bad shape: {subtype_logits.shape}"
    n = sum(p.numel() for p in model.season_head.parameters()) \
      + sum(p.numel() for p in model.subtype_head.parameters())
    print(f"  season_logits:  {season_logits.shape}")
    print(f"  subtype_logits: {subtype_logits.shape}")
    print(f"  trainable head params: {n:,}")
    print("Smoke test passed.")
