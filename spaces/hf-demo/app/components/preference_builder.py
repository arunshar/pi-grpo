"""Build (prompt, chosen, rejected) triples for DPO.

Sources:
- HITL feedback rows from the sibling `geotrace-agent` project.
- Synthetic pairs sampled from a base policy and scored by the
  PhysicsReward (chosen = higher reward, rejected = lower reward).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    margin: float
    source: str

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self))


def from_hitl_jsonl(path: str) -> list[PreferencePair]:
    """Load HITL items exported from `geotrace-agent` and convert."""

    out: list[PreferencePair] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("label") not in {"correct", "incorrect"}:
            continue
        chosen = item["regions_chosen"]
        rejected = item["regions_rejected"]
        if not chosen or not rejected:
            continue
        out.append(PreferencePair(
            prompt=item["question"],
            chosen=json.dumps(chosen),
            rejected=json.dumps(rejected),
            margin=float(item.get("margin", 0.0)),
            source="hitl",
        ))
    return out


def synthesize_from_reward(
    base_outputs: Iterable[tuple[str, list[str], list[float]]],
    *,
    margin_min: float = 0.5,
) -> list[PreferencePair]:
    """For each prompt with K base outputs and K rewards, build pairs:

    chosen = argmax reward
    rejected = the lowest-reward output whose margin to chosen is >= margin_min

    Returning at most K-1 pairs per prompt.
    """

    pairs: list[PreferencePair] = []
    for prompt, outputs, rewards in base_outputs:
        if not outputs or len(outputs) != len(rewards):
            continue
        idxs = sorted(range(len(rewards)), key=lambda i: -rewards[i])
        top = idxs[0]
        for j in idxs[1:]:
            margin = rewards[top] - rewards[j]
            if margin < margin_min:
                continue
            pairs.append(PreferencePair(
                prompt=prompt,
                chosen=outputs[top],
                rejected=outputs[j],
                margin=margin,
                source="synthetic",
            ))
    return pairs


def write_jsonl(path: str, pairs: list[PreferencePair]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w") as fh:
        for pair in pairs:
            fh.write(pair.to_jsonl() + "\n")
            n += 1
    log.info("preferences_written", path=path, n=n)
    return n
