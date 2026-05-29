"""GRPO advantage normalization."""

from __future__ import annotations

import torch


def test_group_advantage_is_zero_mean_unit_var() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True).clamp_min(1e-6)
    adv = (rewards - mean) / std
    assert torch.allclose(adv.mean(dim=1), torch.zeros(1), atol=1e-6)
    assert torch.allclose(adv.std(dim=1, unbiased=True), torch.ones(1), atol=1e-3)
