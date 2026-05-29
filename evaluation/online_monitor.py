"""Online monitor. Diffs the prod policy against the previous checkpoint
on a 1 percent shadow stream and flags reward / KL drift.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)


def report_drift(prev_reward: float, curr_reward: float, kl: float) -> None:
    log.info("drift", prev_reward=prev_reward, curr_reward=curr_reward, kl=kl)
