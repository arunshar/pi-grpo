"""OPT-IN Ray sharding for the GRPO rollout (sampling) phase.

This is an *additive* helper. It does not modify or replace the pure-torch
`train_grpo` in `app.policy.driver`; it offers an alternative way to run the
SAMPLE phase (step (1) of the GRPO loop: `policy.generate` -> (B, K, T_r)
tokens) by splitting the B prompts across Ray actors and reassembling the
rollouts in the original prompt order.

HONEST PERFORMANCE NOTE / TODO
------------------------------
For the *tiny* motion-primitive policy used in the tests and the headline study
(d_model<=64, a couple of layers, a CPU forward over a vocab of ~25 tokens), the
rollout itself is cheap. Ray actor scheduling, the policy state-dict transfer to
each actor, and the (B, K, T_r) tensor serialization back to the driver will very
likely COST MORE than the rollout work they parallelize. Measured wins from this
path are expected only when:

  * the policy is large (a real LLM / vLLM-backed generator) so per-prompt
    `generate` dominates actor overhead, and/or
  * generation is the bottleneck rather than reward evaluation.

In the current project the documented bottleneck is the *reward* matrix
(`driver._reward_matrix`, a serial B*K Python loop over `PhysicsReward`), NOT the
rollout. So:

  TODO(perf): benchmark this against the serial `policy.generate` on the real
  target policy before wiring it into `train_grpo`. For the tiny policy, keep the
  serial fallback (`generate_rollouts(..., backend="serial")`, the default).
  TODO(scale): when moving to a vLLM/LLM generator, replace the per-actor
  `policy.generate` call with the vLLM engine handle and shard prompts, not the
  weights.
  TODO(determinism): per-shard RNG. `torch.Generator` objects are not
  serializable across the actor boundary, so this helper takes an integer
  `seed` and derives a per-shard generator INSIDE each actor. That changes the
  sampling stream relative to a single-process generator; the equivalence we
  assert in tests is the *reassembly* (split/merge) logic, exercised with a
  deterministic stub generator, not bitwise sampling parity with the serial
  driver. Bitwise parity across the actor boundary is left as future work.

The split/merge logic here is pure and tensor-shape-correct, and is what the
unit tests pin down without needing a live Ray cluster.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

# A generator function with the SAME call surface as `CausalPolicy.generate`,
# returning (rollout_ids, logp_old) both shaped (B_shard, K, T_r).
GenerateFn = Callable[..., tuple[torch.Tensor, torch.Tensor]]


# --------------------------------------------------------------------------- split


def split_prompt_indices(b: int, n_shards: int) -> list[list[int]]:
    """Partition prompt row indices 0..b-1 into <= n_shards contiguous shards.

    Contiguous, near-even, and order-preserving: the concatenation of the shards
    is exactly range(b). Empty shards are dropped, so the number of returned
    shards is min(n_shards, b) for b > 0 (and 0 for b == 0).
    """
    if b < 0:
        raise ValueError(f"b must be >= 0, got {b}")
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if b == 0:
        return []
    n = min(n_shards, b)
    base, extra = divmod(b, n)
    shards: list[list[int]] = []
    start = 0
    for s in range(n):
        size = base + (1 if s < extra else 0)
        shards.append(list(range(start, start + size)))
        start += size
    return shards


def split_prompts(prompts: torch.Tensor, n_shards: int) -> list[torch.Tensor]:
    """(B, T_p) -> list of (B_shard, T_p) prompt shards, in original row order."""
    if prompts.dim() != 2:
        raise ValueError(f"prompts must be 2-D (B, T_p), got shape {tuple(prompts.shape)}")
    b = prompts.shape[0]
    return [prompts[idx] for idx in split_prompt_indices(b, n_shards)]


# --------------------------------------------------------------------------- merge


def merge_rollouts(
    shard_indices: Sequence[Sequence[int]],
    shard_rollouts: Sequence[torch.Tensor],
    shard_logps: Sequence[torch.Tensor],
    b: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter per-shard (B_shard, K, T_r) outputs back into (B, K, T_r).

    `shard_indices[i]` are the ORIGINAL prompt rows that produced
    `shard_rollouts[i]` / `shard_logps[i]`. The result places each shard's rows
    at their original positions, so prompt order is preserved regardless of the
    order in which shards completed. Every prompt row must be covered exactly
    once across all shards.
    """
    if not (len(shard_indices) == len(shard_rollouts) == len(shard_logps)):
        raise ValueError("shard_indices / shard_rollouts / shard_logps length mismatch")
    if len(shard_rollouts) == 0:
        raise ValueError("cannot merge zero shards")

    ref = shard_rollouts[0]
    if ref.dim() != 3:
        raise ValueError(f"each shard rollout must be 3-D (B_shard, K, T_r), got {tuple(ref.shape)}")
    _, k, t_r = ref.shape

    out_roll = torch.empty(b, k, t_r, dtype=ref.dtype)
    out_logp = torch.empty(b, k, t_r, dtype=shard_logps[0].dtype)

    seen = torch.zeros(b, dtype=torch.bool)
    for idx, roll, logp in zip(shard_indices, shard_rollouts, shard_logps, strict=True):
        if roll.shape[1:] != (k, t_r) or logp.shape[1:] != (k, t_r):
            raise ValueError("inconsistent (K, T_r) across shards")
        if len(idx) != roll.shape[0] or len(idx) != logp.shape[0]:
            raise ValueError("shard index count does not match shard batch size")
        for local, original in enumerate(idx):
            if original < 0 or original >= b:
                raise ValueError(f"prompt index {original} out of range [0, {b})")
            if seen[original]:
                raise ValueError(f"prompt index {original} produced by more than one shard")
            seen[original] = True
            out_roll[original] = roll[local]
            out_logp[original] = logp[local]

    if not bool(seen.all()):
        missing = [i for i in range(b) if not bool(seen[i])]
        raise ValueError(f"prompt rows not covered by any shard: {missing}")
    return out_roll, out_logp


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class RayRolloutConfig:
    """Knobs for the opt-in parallel rollout."""

    n_shards: int = 2            # number of prompt shards / actors
    k: int = 6                   # rollouts per prompt
    max_new_tokens: int = 12     # T_r control tokens per rollout
    temperature: float = 1.0
    seed: int = 1234             # base seed; shard s derives seed + s
    num_cpus_per_actor: float = 1.0


# --------------------------------------------------------------------------- serial


def generate_rollouts_serial(
    generate_fn: GenerateFn,
    prompts: torch.Tensor,
    cfg: RayRolloutConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Serial fallback: shard, call `generate_fn` per shard, merge.

    This exercises EXACTLY the same split/merge path as the Ray backend but
    without any actors, so it is the reference the parallel path must match
    on the reassembly. `generate_fn` has the `CausalPolicy.generate` surface.
    """
    b = prompts.shape[0]
    index_shards = split_prompt_indices(b, cfg.n_shards)
    roll_shards: list[torch.Tensor] = []
    logp_shards: list[torch.Tensor] = []
    for s, idx in enumerate(index_shards):
        gen = torch.Generator().manual_seed(cfg.seed + s)
        roll, logp = generate_fn(
            prompts[idx],
            k=cfg.k,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            generator=gen,
        )
        roll_shards.append(roll)
        logp_shards.append(logp)
    return merge_rollouts(index_shards, roll_shards, logp_shards, b)


# --------------------------------------------------------------------------- ray


def _build_actor_cls(num_cpus_per_actor: float):  # pragma: no cover - needs ray
    """Lazily define the Ray actor so importing this module never requires ray.

    Returns a `@ray.remote`-decorated class whose `.generate` holds a private
    `CausalPolicy` rebuilt from a state-dict + `PolicyConfig` and runs the
    rollout for one prompt shard.
    """
    import ray

    from app.policy.model import CausalPolicy, PolicyConfig

    @ray.remote(num_cpus=num_cpus_per_actor)
    class _RolloutActor:
        def __init__(self, policy_cfg: PolicyConfig, state_dict: dict) -> None:
            self._policy = CausalPolicy(policy_cfg)
            self._policy.load_state_dict(state_dict)
            self._policy.eval()

        def generate(
            self,
            prompts_shard: torch.Tensor,
            *,
            k: int,
            max_new_tokens: int,
            temperature: float,
            seed: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            gen = torch.Generator().manual_seed(seed)
            return self._policy.generate(
                prompts_shard,
                k=k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                generator=gen,
            )

    return _RolloutActor


def generate_rollouts_ray(
    policy,
    prompts: torch.Tensor,
    cfg: RayRolloutConfig,
) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - needs a live cluster
    """Parallel rollout across Ray actors. Requires `ray` and an initialized cluster.

    The caller is responsible for `ray.init(...)` / `ray.shutdown()`. Targets
    Ray 2.x; do NOT rely on `local_mode` (removed in Ray >= 2.40). For unit
    testing without a cluster, drive `generate_rollouts(..., backend="serial")`
    or monkeypatch the dispatch, see `tests/test_ray_rollout.py`.

    The policy weights are broadcast to each actor via a CPU state-dict. For a
    large/LLM policy this transfer is the part to optimize (put the weights in
    the object store once with `ray.put`), see the module TODOs.
    """
    import ray

    from app.policy.model import PolicyConfig  # noqa: F401  (type clarity)

    b = prompts.shape[0]
    index_shards = split_prompt_indices(b, cfg.n_shards)

    actor_cls = _build_actor_cls(cfg.num_cpus_per_actor)
    cpu_state = {kk: v.detach().cpu() for kk, v in policy.state_dict().items()}
    state_ref = ray.put(cpu_state)
    cfg_ref = ray.put(policy.cfg)

    actors = [actor_cls.remote(cfg_ref, state_ref) for _ in index_shards]
    futures = [
        actor.generate.remote(
            prompts[idx],
            k=cfg.k,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            seed=cfg.seed + s,
        )
        for s, (actor, idx) in enumerate(zip(actors, index_shards, strict=True))
    ]
    results = ray.get(futures)
    roll_shards = [r for r, _ in results]
    logp_shards = [lp for _, lp in results]
    return merge_rollouts(index_shards, roll_shards, logp_shards, b)


# --------------------------------------------------------------------------- entry


def generate_rollouts(
    policy,
    prompts: torch.Tensor,
    cfg: RayRolloutConfig = RayRolloutConfig(),
    *,
    backend: str = "serial",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample K rollouts per prompt, optionally sharded across Ray actors.

    backend="serial" (default): shard the prompts and run `policy.generate`
    per shard in-process, then merge. Safe everywhere, no ray needed. This is
    the recommended path for the tiny motion-primitive policy.

    backend="ray": dispatch each shard to a Ray actor. Requires `ray` and an
    initialized cluster (the caller owns init/shutdown). See the module
    docstring: only expected to pay off for a large/LLM policy.

    Returns (rollout_ids, logp_old), both (B, K, T_r), in original prompt order.
    """
    if backend == "serial":
        return generate_rollouts_serial(policy.generate, prompts, cfg)
    if backend == "ray":
        return generate_rollouts_ray(policy, prompts, cfg)
    raise ValueError(f"unknown backend {backend!r}; choose 'serial' or 'ray'")


__all__ = [
    "GenerateFn",
    "RayRolloutConfig",
    "generate_rollouts",
    "generate_rollouts_ray",
    "generate_rollouts_serial",
    "merge_rollouts",
    "split_prompt_indices",
    "split_prompts",
]
