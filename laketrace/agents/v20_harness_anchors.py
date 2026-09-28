"""Score-blind non-LLM anchors used to validate the harness, not model ability."""

from __future__ import annotations

from typing import Any

from laketrace.agents.v20_baselines import run_baseline


HARNESS_ANCHORS = (
    "always_anomaly",
    "always_cannot_determine",
    "random_walk_forced_submit",
    "systematic_scan",
    "classic_gradient_tracking",
)


def run_harness_anchor(env: Any, anchor_id: str, seed: int = 0) -> None:
    """Execute one public-information anchor through typed environment actions."""

    if anchor_id not in HARNESS_ANCHORS:
        raise ValueError(f"unknown harness anchor: {anchor_id}")
    if anchor_id == "always_anomaly":
        env.submit({"anomaly": "anomaly", "decision": "cannot_determine"})
        return
    if anchor_id == "always_cannot_determine":
        env.submit({"anomaly": "no_anomaly", "decision": "cannot_determine"})
        return
    mapped = {
        "random_walk_forced_submit": "random",
        "systematic_scan": "lawnmower",
        "classic_gradient_tracking": "greedy_gradient",
    }[anchor_id]
    run_baseline(env, mapped, seed)


__all__ = ["HARNESS_ANCHORS", "run_harness_anchor"]
