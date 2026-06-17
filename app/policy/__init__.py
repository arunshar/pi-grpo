"""Policy networks and training drivers for the Pi-GRPO trainers.

`CausalPolicy` is the token policy the PPO/DPO/GRPO trainers optimize;
`MotionCodebook` decodes its tokens into S-KBM control sequences; the `driver`
module runs real end-to-end training on the physics-aware reward.
"""

from __future__ import annotations

from app.policy.decode import CodebookConfig, MotionCodebook
from app.policy.driver import SMOKE, TrainConfig, TrainResult, train
from app.policy.model import CausalPolicy, PolicyConfig, ValueHead, build_policy_pair

__all__ = [
    "SMOKE",
    "CausalPolicy",
    "CodebookConfig",
    "MotionCodebook",
    "PolicyConfig",
    "TrainConfig",
    "TrainResult",
    "ValueHead",
    "build_policy_pair",
    "train",
]
