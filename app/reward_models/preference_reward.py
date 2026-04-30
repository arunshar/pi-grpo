"""Preference reward (cross-encoder over (prompt, response) pairs).

Used as the `R_pref` term in `PhysicsReward` when configured. In
production this is a fine-tuned DeBERTa-v3-large; the scaffold uses a
deterministic stub for tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PreferenceReward(nn.Module):
    """Stub preference head. Replace with a real cross-encoder."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.proj(embeddings).squeeze(-1)
