"""Unit + mock tests for the opt-in Ray rollout sharding helper.

All tests are CPU-only, tiny, and fast. They do NOT require a live Ray cluster:
the split/merge logic is pure tensor bookkeeping, and the dispatch test
monkeypatches `generate_rollouts_ray` with a local in-process implementation
that reuses the real `merge_rollouts`, so the actor `.remote` boundary is
mocked away. No `torch` model and no `ray` import are needed.

What is pinned down here:
  * split + merge reassembles (B, K, T_r) in the ORIGINAL prompt order, even
    when shards complete out of order;
  * the dispatch covers every prompt exactly once (no drop, no duplicate);
  * the serial fallback equals the mocked-parallel merge on the same stub.
"""

from __future__ import annotations

import pytest
import torch

from app.policy import ray_rollout as rr

# --------------------------------------------------------------------------- stub


def _make_stub_generate(t_r: int):
    """A deterministic stand-in for `CausalPolicy.generate`.

    The rollout tokens for a prompt encode that prompt's first-token VALUE, so
    after a merge we can assert each output row went back to its original prompt
    position. Shape contract matches the real policy: (B_shard, K, T_r).
    """

    def stub_generate(prompts_shard, *, k, max_new_tokens, temperature, generator=None):
        assert max_new_tokens == t_r
        bsh = prompts_shard.shape[0]
        roll = torch.empty(bsh, k, t_r, dtype=torch.long)
        logp = torch.empty(bsh, k, t_r, dtype=torch.float32)
        for i in range(bsh):
            tag = int(prompts_shard[i, 0].item())  # identity tag for this prompt
            roll[i] = tag
            logp[i] = float(tag)
        return roll, logp

    return stub_generate


def _tagged_prompts(b: int, t_p: int = 2) -> torch.Tensor:
    """Prompts whose first column is the unique row id, so rows are traceable."""
    p = torch.zeros(b, t_p, dtype=torch.long)
    p[:, 0] = torch.arange(b)
    return p


# --------------------------------------------------------------------------- split


@pytest.mark.parametrize(
    "b,n,expected_sizes",
    [
        (7, 3, [3, 2, 2]),   # uneven: remainder goes to the front shards
        (6, 2, [3, 3]),      # even
        (4, 8, [1, 1, 1, 1]),  # more shards than prompts -> one per prompt
        (1, 4, [1]),
    ],
)
def test_split_prompt_indices_partition(b, n, expected_sizes) -> None:
    shards = rr.split_prompt_indices(b, n)
    assert [len(s) for s in shards] == expected_sizes
    # contiguous, order-preserving, exact cover of range(b)
    flat = [i for s in shards for i in s]
    assert flat == list(range(b))


def test_split_prompt_indices_edges() -> None:
    assert rr.split_prompt_indices(0, 3) == []
    with pytest.raises(ValueError):
        rr.split_prompt_indices(3, 0)
    with pytest.raises(ValueError):
        rr.split_prompt_indices(-1, 2)


def test_split_prompts_preserves_rows() -> None:
    prompts = _tagged_prompts(5)
    shards = rr.split_prompts(prompts, 2)
    # reconcatenating the shards must reproduce the input exactly
    assert torch.equal(torch.cat(shards, dim=0), prompts)
    with pytest.raises(ValueError):
        rr.split_prompts(torch.zeros(5), 2)  # 1-D is rejected


# --------------------------------------------------------------------------- merge (test 1)


def test_merge_reassembles_in_prompt_order_even_out_of_order() -> None:
    """Pure split/merge logic reassembles (B, K, T_r) in the ORIGINAL order.

    Build per-shard outputs with a deterministic stub, then feed the shards to
    `merge_rollouts` in a SHUFFLED order to prove ordering comes from the index
    bookkeeping, not from shard arrival order.
    """
    b, k, t_r = 7, 4, 3
    prompts = _tagged_prompts(b)
    stub = _make_stub_generate(t_r)

    index_shards = rr.split_prompt_indices(b, 3)
    roll_shards, logp_shards = [], []
    for idx in index_shards:
        roll, logp = stub(prompts[idx], k=k, max_new_tokens=t_r, temperature=1.0)
        roll_shards.append(roll)
        logp_shards.append(logp)

    # complete the shards out of order: reverse everything in lockstep
    order = list(reversed(range(len(index_shards))))
    merged_roll, merged_logp = rr.merge_rollouts(
        [index_shards[i] for i in order],
        [roll_shards[i] for i in order],
        [logp_shards[i] for i in order],
        b,
    )
    assert merged_roll.shape == (b, k, t_r)
    assert merged_logp.shape == (b, k, t_r)
    # row r must carry tag r everywhere (stub encodes the prompt id into tokens)
    for r in range(b):
        assert torch.all(merged_roll[r] == r)
        assert torch.all(merged_logp[r] == float(r))


def test_merge_rejects_duplicate_and_missing() -> None:
    b, k, t_r = 4, 2, 2
    roll = torch.zeros(2, k, t_r, dtype=torch.long)
    logp = torch.zeros(2, k, t_r)
    # duplicate coverage of row 0
    with pytest.raises(ValueError):
        rr.merge_rollouts([[0, 1], [0, 1]], [roll, roll], [logp, logp], b)
    # missing rows 2,3
    with pytest.raises(ValueError):
        rr.merge_rollouts([[0, 1]], [roll], [logp], b)


# --------------------------------------------------------------------------- dispatch (test 2)


def test_ray_dispatch_covers_every_prompt_exactly_once(monkeypatch) -> None:
    """MOCK the actor `.remote` dispatch: assert all B prompts covered once.

    `generate_rollouts_ray` normally builds Ray actors and calls `.remote`. We
    monkeypatch it with a local stand-in that runs the stub per shard and reuses
    the REAL `merge_rollouts`, so the actor boundary is mocked but the coverage
    contract is exercised. A coverage counter proves each prompt is generated
    exactly once.
    """
    b, k, t_r = 6, 3, 4
    prompts = _tagged_prompts(b)
    stub = _make_stub_generate(t_r)
    cfg = rr.RayRolloutConfig(n_shards=4, k=k, max_new_tokens=t_r, temperature=1.0)

    gen_counts = torch.zeros(b, dtype=torch.long)

    def fake_ray(policy, prompts_in, cfg_in):
        index_shards = rr.split_prompt_indices(prompts_in.shape[0], cfg_in.n_shards)
        roll_shards, logp_shards = [], []
        for idx in index_shards:
            for orig in idx:
                gen_counts[orig] += 1  # this prompt was dispatched to an actor
            roll, logp = stub(prompts_in[idx], k=cfg_in.k, max_new_tokens=cfg_in.max_new_tokens, temperature=cfg_in.temperature)
            roll_shards.append(roll)
            logp_shards.append(logp)
        return rr.merge_rollouts(index_shards, roll_shards, logp_shards, prompts_in.shape[0])

    monkeypatch.setattr(rr, "generate_rollouts_ray", fake_ray)

    roll, _logp = rr.generate_rollouts(policy=None, prompts=prompts, cfg=cfg, backend="ray")
    assert roll.shape == (b, k, t_r)
    # every prompt dispatched exactly once: no drop, no duplicate
    assert torch.all(gen_counts == 1)
    # and reassembled in order
    for r in range(b):
        assert torch.all(roll[r] == r)


# --------------------------------------------------------------------------- fallback parity (test 3)


def test_serial_fallback_equals_parallel_merge_on_same_stub(monkeypatch) -> None:
    """Serial fallback == mocked-parallel merge on the same deterministic stub.

    Both paths shard the same prompts and run the same stub; the only
    difference is serial-in-process vs the (mocked) ray dispatch. Because the
    stub is deterministic in the prompt id (not in any RNG stream), the merged
    (B, K, T_r) outputs must be bit-identical, proving the reassembly is
    equivalent across backends.
    """
    b, k, t_r = 5, 3, 4
    prompts = _tagged_prompts(b)
    stub = _make_stub_generate(t_r)
    cfg = rr.RayRolloutConfig(n_shards=3, k=k, max_new_tokens=t_r, temperature=1.0)

    # serial path: drive the public entry with the stub as the policy.generate
    class _StubPolicy:
        generate = staticmethod(stub)

    serial_roll, serial_logp = rr.generate_rollouts(
        policy=_StubPolicy(), prompts=prompts, cfg=cfg, backend="serial"
    )

    # mocked-parallel path: same stub, same merge, dispatched "across actors"
    def fake_ray(policy, prompts_in, cfg_in):
        index_shards = rr.split_prompt_indices(prompts_in.shape[0], cfg_in.n_shards)
        rolls, logps = [], []
        for idx in index_shards:
            roll, logp = stub(prompts_in[idx], k=cfg_in.k, max_new_tokens=cfg_in.max_new_tokens, temperature=cfg_in.temperature)
            rolls.append(roll)
            logps.append(logp)
        return rr.merge_rollouts(index_shards, rolls, logps, prompts_in.shape[0])

    monkeypatch.setattr(rr, "generate_rollouts_ray", fake_ray)
    par_roll, par_logp = rr.generate_rollouts(policy=None, prompts=prompts, cfg=cfg, backend="ray")

    assert torch.equal(serial_roll, par_roll)
    assert torch.equal(serial_logp, par_logp)


def test_generate_rollouts_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        rr.generate_rollouts(policy=None, prompts=_tagged_prompts(2), backend="nope")
