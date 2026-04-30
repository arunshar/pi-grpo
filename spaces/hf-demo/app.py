"""Pi-GRPO Hugging Face Space.

Free-tier (CPU-only) Streamlit demo of the physics-aware reward and a
reasoner tab backed by the free Hugging Face Inference API. Imports
the project's components directly (no vLLM, no GPU dependency).
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --- Make the project source importable --------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent  # repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.components.kinematic_bicycle import SkbmConfig, evaluate  # noqa: E402
from app.components.physics_reward import (  # noqa: E402
    EmpiricalEnvelope,
    PhysicsReward,
    RewardWeights,
)
from app.components.pidpm_scorer import PiDpmScorer  # noqa: E402

st.set_page_config(page_title="Pi-GRPO", page_icon="🔬", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trajectory_from_speed(v: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Build (x, y, theta, v) from a 1-D speed series, straight east-bound."""

    n = v.shape[0]
    states = np.zeros((n, 4), dtype=np.float64)
    for i in range(1, n):
        states[i, 0] = states[i - 1, 0] + v[i - 1] * dt
    states[:, 3] = v
    return states


SAMPLES = {
    "Clean (5 m/s straight)": _trajectory_from_speed(np.full(20, 5.0)),
    "Speeding (rises to 30 m/s)": _trajectory_from_speed(np.linspace(5.0, 30.0, 20)),
    "Hard accel (jerk spike)": _trajectory_from_speed(
        np.concatenate([np.full(8, 5.0), np.linspace(5.0, 12.0, 4), np.full(8, 12.0)])
    ),
    "Sub-max with curvature": (lambda: (
        np.column_stack([
            np.cumsum(np.cos(np.linspace(0, np.pi / 4, 20)) * 5.0),
            np.cumsum(np.sin(np.linspace(0, np.pi / 4, 20)) * 5.0),
            np.linspace(0, np.pi / 4, 20),
            np.full(20, 5.0),
        ])
    ))(),
}


@st.cache_resource(show_spinner=False)
def _reward() -> PhysicsReward:
    cfg = SkbmConfig()  # default vessel envelope: 25 kt = 12.86 m/s
    return PhysicsReward(cfg, weights=RewardWeights(), envelope=EmpiricalEnvelope())


@st.cache_resource(show_spinner=False)
def _pidpm() -> PiDpmScorer:
    ckpt = os.environ.get("PG_PIDPM_CHECKPOINT")
    return PiDpmScorer(ckpt, device="cpu")


def _read_csv(file: io.BytesIO) -> np.ndarray:
    df = pd.read_csv(file)
    if {"lat", "lon", "t", "v"}.issubset(df.columns):
        # build (x, y, theta, v) with x, y from a local equirectangular projection
        lat0 = df["lat"].iloc[0]
        x = (df["lon"] - df["lon"].iloc[0]) * np.cos(np.radians(lat0)) * 6_371_000.0 * (np.pi / 180.0)
        y = (df["lat"] - df["lat"].iloc[0]) * 6_371_000.0 * (np.pi / 180.0)
        theta = np.arctan2(np.diff(y, prepend=y.iloc[0]), np.diff(x, prepend=x.iloc[0]))
        return np.column_stack([x, y, theta, df["v"].to_numpy()])
    if {"x", "y", "theta", "v"}.issubset(df.columns):
        return df[["x", "y", "theta", "v"]].to_numpy()
    raise ValueError("CSV must have columns lat,lon,t,v or x,y,theta,v")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔬 Pi-GRPO")
    st.caption(
        "Physics-informed reinforcement learning. PPO + DPO + GRPO under "
        "a hybrid reward whose hard term is unbounded above so a single "
        "S-KBM violation dominates the gradient."
    )
    sample = st.selectbox("Sample trajectory", list(SAMPLES.keys()))
    uploaded = st.file_uploader("...or upload CSV (lat,lon,t,v) or (x,y,theta,v)", type=["csv"])
    st.markdown("**Reward weights**")
    w_hard = st.slider("w_hard", 0.0, 10.0, 5.0, step=0.1)
    w_soft = st.slider("w_soft", 0.0, 5.0, 1.0, step=0.1)
    w_data = st.slider("w_data", 0.0, 5.0, 1.0, step=0.1)
    w_pref = st.slider("w_pref", 0.0, 5.0, 1.0, step=0.1)


# build trajectory
if uploaded is not None:
    try:
        traj = _read_csv(uploaded)
    except Exception as exc:
        st.error(f"Bad CSV: {exc}")
        st.stop()
else:
    traj = SAMPLES[sample]


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_reward, tab_reasoner = st.tabs(["Reward & violations", "Reasoner"])

with tab_reward:
    reward = _reward()
    reward.weights = RewardWeights(hard=w_hard, soft=w_soft, data=w_data, pref=w_pref)
    pidpm = _pidpm()
    log_p = pidpm.log_prob(traj)
    rb = reward.score(traj, pi_dpm_log_prob=log_p, pref_logit=0.0)
    v = evaluate(traj, cfg=reward.cfg)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("R total", f"{rb.total:.3f}")
    c2.metric("R_hard", f"{rb.hard:.3f}", delta=("violation" if rb.hard < 0 else "ok"))
    c3.metric("R_soft", f"{rb.soft:.3f}")
    c4.metric("R_data", f"{rb.data:.3f}")
    c5.metric("R_pref", f"{rb.pref:.3f}")

    st.subheader("S-KBM violations")
    st.write({
        "speed > v_max (frac)":  f"{v.speed_max_pct:.2%}",
        "|a| > a_max (frac)":    f"{v.accel_max_pct:.2%}",
        "|delta| > d_max (frac)": f"{v.steer_max_pct:.2%}",
        "curvature p95 (rad)":   f"{v.curvature_p95:.4f}",
        "jerk p95 (m/s^3)":      f"{v.jerk_p95:.4f}",
    })

    st.subheader("Reward decomposition")
    st.bar_chart(pd.DataFrame({
        "term":  ["R_hard", "R_soft", "R_data", "R_pref"],
        "value": [rb.hard, rb.soft, rb.data, rb.pref],
    }).set_index("term"))

    st.subheader("Trajectory")
    df = pd.DataFrame(traj, columns=["x", "y", "theta", "v"])
    df["step"] = np.arange(len(df))
    st.line_chart(df.set_index("step")[["v"]])

    # red markers at speeding segments
    speeding = df[df["v"] > reward.cfg.v_max_mps]
    if not speeding.empty:
        st.warning(f"{len(speeding)} segment(s) exceed v_max = {reward.cfg.v_max_mps:.2f} m/s")
    else:
        st.success("No hard violations in this trajectory.")


with tab_reasoner:
    st.write(
        "Calls the free Hugging Face Inference API on a small instruct model. "
        "Set `HF_TOKEN` as a Space secret to avoid rate limits."
    )
    model_id = st.selectbox("Model", [
        "Qwen/Qwen2-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/gemma-2-9b-it",
    ])

    prompt = st.text_area("Prompt", (
        "You are a physical-plausibility reasoner. The trajectory is\n"
        f"{traj.tolist()[:8]}\n... (truncated). Decide PASS / SOFT_VIOLATION / "
        "HARD_VIOLATION and give one sentence of reasoning."
    ), height=200)

    if st.button("Ask the reasoner", type="primary"):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(model=model_id, token=token)
            out = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            text = out.choices[0].message.content if hasattr(out, "choices") else str(out)
            st.success(text)
        except Exception as exc:
            # Deterministic fallback: explain the reward decomposition
            verdict = (
                "HARD_VIOLATION" if rb.hard < 0
                else "SOFT_VIOLATION" if rb.soft < -0.01
                else "PASS"
            )
            st.warning(f"HF Inference unavailable ({exc}); using deterministic fallback.")
            st.info(f"VERDICT: {verdict}.\n"
                    f"R_hard = {rb.hard:.3f}, R_soft = {rb.soft:.3f}, R_data = {rb.data:.3f}.")
