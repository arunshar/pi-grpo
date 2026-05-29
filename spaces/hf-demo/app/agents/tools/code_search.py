"""Code search tool stub."""

from __future__ import annotations

from typing import Any

_PLAYBOOK = {
    "kl_explosion": "If KL > 2*target for 3 windows, halve LR and bump kl_coef by 1.5x.",
    "reward_hacking": "If reward_mean rises while physics_violation_rate also rises, raise w_hard.",
}


async def run(args: dict[str, Any]) -> dict[str, Any]:
    q = (args.get("query") or "").lower()
    return {"hits": [{"id": k, "text": v} for k, v in _PLAYBOOK.items() if q in v.lower()]}
