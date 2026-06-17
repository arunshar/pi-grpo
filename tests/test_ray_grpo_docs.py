"""Content-sanity tests for the Ray-GRPO runbook (docs/RAY_GRPO.md).

These are CPU-only, dependency-free string checks. They do NOT import Ray,
torch, or any application module; they only read the markdown doc from the repo
root and assert that it actually documents the agreed Ray-GRPO contract:

  * the two public entrypoints other agents build against
    (``train_grpo_ray`` and ``RayRewardPool``), and
  * the three design sections the task requires
    (convergence-parity, async/staleness, reward-actor pool).

The point is to fail loudly if the doc drifts away from the entrypoint names or
silently drops a required section, so the docs cannot rot independently of the
code that other parallel agents are writing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# tests/ lives at <repo>/tests/, so the repo root is one level up.
REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "RAY_GRPO.md"


@pytest.fixture(scope="module")
def doc_text() -> str:
    assert DOC_PATH.exists(), f"missing Ray-GRPO doc at {DOC_PATH}"
    text = DOC_PATH.read_text(encoding="utf-8")
    assert text.strip(), "Ray-GRPO doc is empty"
    return text


def test_doc_exists_and_nonempty(doc_text: str) -> None:
    # A real floor on substance, not just existence: the runbook is multi-section.
    assert len(doc_text) > 1500, "Ray-GRPO doc is suspiciously short for a runbook"
    assert doc_text.lower().count("\n#") >= 3, "expected several markdown sections"


@pytest.mark.parametrize("entrypoint", ["train_grpo_ray", "RayRewardPool"])
def test_references_key_entrypoints(doc_text: str, entrypoint: str) -> None:
    assert entrypoint in doc_text, f"doc must reference the {entrypoint!r} entrypoint"


def test_reuses_serial_reference_names(doc_text: str) -> None:
    # The Ray path must point back at the serial reward path it parallelizes,
    # so a reader knows what the parity gate compares against.
    for symbol in ("_reward_matrix", "_trajectory_reward", "PhysicsReward"):
        assert symbol in doc_text, f"doc must mention the serial reference {symbol!r}"


@pytest.mark.parametrize(
    "section_terms",
    [
        ("convergence-parity", ("convergence-parity", "convergence parity")),
        ("staleness", ("staleness", "async")),
        ("reward-actor", ("reward-actor", "reward actor", "RayRewardPool")),
    ],
    ids=["convergence-parity", "staleness", "reward-actor"],
)
def test_covers_required_sections(doc_text: str, section_terms: tuple[str, tuple[str, ...]]) -> None:
    name, accepted = section_terms
    low = doc_text.lower()
    assert any(term.lower() in low for term in accepted), f"doc must cover the {name!r} section"


def test_three_backend_framing_present(doc_text: str) -> None:
    # The PC-RF three-backend story: laptop / MSI / Anyscale.
    low = doc_text.lower()
    for backend in ("laptop", "msi", "anyscale"):
        assert backend in low, f"doc must describe the {backend!r} backend"


def test_honest_about_tiny_model(doc_text: str) -> None:
    # The task demands honesty that tiny-model rollout/learner curves may be flat
    # (overhead-bound) while the reward-eval curve is the clean win.
    low = doc_text.lower()
    assert "overhead-bound" in low, "doc must state the tiny-model curves are overhead-bound"
    assert "flat" in low, "doc must admit the tiny-model curves may be flat"


def test_no_local_mode_recommended(doc_text: str) -> None:
    # local_mode was removed in Ray >= 2.40; the doc must not recommend it as a
    # live-cluster substitute. If mentioned at all, it must be in a negative
    # ("removed" / "do not use") context, never as guidance.
    low = doc_text.lower()
    if "local_mode" in low:
        assert ("removed" in low) or ("not be used" in low) or ("do not use" in low), (
            "local_mode may only appear flagged as removed / not-to-be-used"
        )


def test_no_em_or_en_dashes(doc_text: str) -> None:
    # House style: no em/en dashes anywhere in generated content.
    assert "—" not in doc_text, "doc contains an em dash"
    assert "–" not in doc_text, "doc contains an en dash"
