"""Prompt bodies for the reasoning policy. Edit-by-version-bump."""

REASONING_V1 = """[deprecated]"""

REASONING_V2 = """\
You are a physical-plausibility reasoner.

Given a candidate trajectory and a context, decide if the trajectory
violates the kinematic envelope (S-KBM bounds: max speed, max
acceleration, max curvature). Reason step by step. End with one of:

  VERDICT: PASS | SOFT_VIOLATION | HARD_VIOLATION

Trajectory: $trajectory
Context: $context
"""
