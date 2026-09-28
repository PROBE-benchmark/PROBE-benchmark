"""Bounded, observation-only inverse calculation under explicit transport assumptions.

No DataStore, evaluator, file access, executable code, or model calls. The Gaussian
pulse is the analytical solution on R^3, not a lake boundary-condition model.
"""
from __future__ import annotations

import math
import numpy as np


MODEL = "uniform_flow_gaussian_pulse"


def validate_physics(value):
    if not isinstance(value, dict) or set(value) != {"model", "age_s", "diffusivity_m2_s", "velocity_m_s"}:
        raise ValueError("physics requires model, age_s, diffusivity_m2_s and velocity_m_s")
    if value["model"] != MODEL:
        raise ValueError("only uniform_flow_gaussian_pulse is implemented; do not fit other source types as validated physics")
    age = value["age_s"]
    vectors = [value["diffusivity_m2_s"], value["velocity_m_s"]]
    if type(age) not in (int, float) or not math.isfinite(age) or not 0 < age <= 1e7:
        raise ValueError("age_s must be positive and at most 1e7")
    if any(not isinstance(v, list) or len(v) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) for x in v) for v in vectors):
        raise ValueError("diffusivity and velocity require three finite numbers in x,y,z order")
    if any(not 1e-8 <= d <= 1e5 for d in vectors[0]) or any(abs(u) > 100 for u in vectors[1]):
        raise ValueError("transport parameters exceed declared numerical bounds")
    return value


def pulse_kernel(observations, sources, physics):
    """Concentration per unit released mass, m^-3; positive z points down."""
    validate_physics(physics)
    t = physics["age_s"]
    d = np.asarray(physics["diffusivity_m2_s"])
    delta = np.asarray(observations)[:, None, :] - np.asarray(sources)[None, :, :] - t * np.asarray(physics["velocity_m_s"])
    return np.exp(-np.sum(delta * delta / (4 * t * d), axis=2)) / ((4 * math.pi * t) ** 1.5 * np.sqrt(np.prod(d)))


def fit_source(samples, navigation, physics, *, max_candidates=4096, max_observations=4096, result_schema=1):
    """Rank all declared grid candidates using training data only.

    Fits nonnegative mass and a constant background. Every fifth spatially ordered
    measured node is held out, never used for fitting or candidate ranking. Repeated
    calls reuse this split: held-out error is diagnostic, not an independent test.
    """
    validate_physics(physics)
    if not 12 <= len(samples) <= max_observations:
        raise ValueError(f"fit_source needs 12-{max_observations} distinct measured nodes; collect a spatial survey first")
    if not 1 <= len(navigation) <= max_candidates:
        raise ValueError(f"candidate grid must contain 1-{max_candidates} nodes; supply explicit xyz bounds to narrow geometry")
    if any(not math.isfinite(float(v)) for row in samples for v in row):
        raise ValueError("fit input contains non-finite measurements")
    # rows: node_id,x,y,z,target_value; order independent of observation arrival.
    data = np.asarray(sorted(samples, key=lambda r: (r[1], r[2], r[3], r[0])), dtype=float)
    nav = np.asarray(sorted(navigation, key=lambda r: r[0]), dtype=float)
    train = np.arange(len(data)) % 5 != 0
    held = ~train
    y = data[:, 4]
    # Bounded chunks avoid N_observed * N_candidates * 3 temporary allocation.
    masses, backgrounds, errors = [], [], []
    for start in range(0, len(nav), 128):
        k = pulse_kernel(data[train, 1:4], nav[start:start+128, 1:4], physics)
        centered = k - k.mean(axis=0)
        denom = np.sum(centered * centered, axis=0)
        mass = np.maximum(0, np.divide(centered.T @ (y[train] - y[train].mean()), denom,
                                     out=np.zeros_like(denom), where=denom > 1e-30))
        background = y[train].mean() - mass * k.mean(axis=0)
        errors.extend(np.mean((k * mass + background - y[train, None]) ** 2, axis=0).tolist())
        masses.extend(mass.tolist())
        backgrounds.extend(background.tolist())
    order = sorted(range(len(nav)), key=lambda i: (errors[i], int(nav[i, 0])))
    ranked = []
    for i in order[:5]:
        prediction = pulse_kernel(data[:, 1:4], nav[i:i+1, 1:4], physics)[:, 0] * masses[i] + backgrounds[i]
        ranked.append({"node_id": int(nav[i, 0]), "position": dict(zip("xyz", nav[i, 1:4].tolist())),
                       "mass": masses[i], "background": backgrounds[i],
                       "train_rmse": math.sqrt(errors[i]),
                       "heldout_rmse": float(np.sqrt(np.mean((prediction[held] - y[held]) ** 2)))})
    best = order[0]
    # An error-based diagnostic set, deliberately not called a confidence interval.
    floor = max(float(np.var(y[train])) * .01, 1e-12)
    near = [i for i in order if errors[i] <= errors[best] + floor]
    spread = np.ptp(nav[near, 1:4], axis=0).tolist()
    measured = set(int(i) for i in data[:, 0])
    unmeasured = nav[[int(n[0]) not in measured for n in nav]]
    suggestions = []
    if len(unmeasured) and len(ranked) > 1:
        k = pulse_kernel(unmeasured[:, 1:4], nav[order[:5], 1:4], physics)
        predicted = k * np.asarray(masses)[order[:5]] + np.asarray(backgrounds)[order[:5]]
        disagreement = np.ptp(predicted, axis=1)
        for i in sorted(range(len(unmeasured)), key=lambda i: (-disagreement[i], int(unmeasured[i, 0])))[:3]:
            suggestions.append({"position": dict(zip("xyz", unmeasured[i, 1:4].tolist())),
                                "predicted_disagreement": float(disagreement[i])})
    result = {"model": MODEL, "assumptions": physics,
            "scope": "Numerical hypothesis from measured values only. Infinite homogeneous medium, instantaneous point release, known age/constant velocity/diagonal diffusivity, constant fitted background. Not a validated lake model, posterior probability or mechanism classifier.",
            "candidate_count": len(nav), "measured_nodes": len(data), "train_nodes": int(train.sum()), "heldout_nodes": int(held.sum()),
            "ranking_uses": "training residual only; held-out nodes are spatially interleaved and reused across calls",
            "top_candidates": ranked, "near_best_count": len(near), "near_best_span_m": dict(zip("xyz", spread)),
            "near_best_rule": "training MSE <= minimum + max(0.01*training variance,1e-12); not a confidence region",
            "constant_background_heldout_rmse": float(np.sqrt(np.mean((y[held] - y[train].mean()) ** 2))),
            "suggested_measurements": suggestions,
            "prediction_note": "Suggested coordinates use model disagreement, never unobserved concentrations. Check residuals, alternative assumptions and spatial coverage before declaring."}
    if result_schema >= 2:
        for candidate in ranked:
            node_id = candidate.pop("node_id")
            candidate["source_grid_node_id"] = node_id
            candidate["actually_measured"] = node_id in measured
        peak = data[int(np.argmax(y))]
        result["measured_target_maximum"] = {"node_id": int(peak[0]), "position": dict(zip("xyz", peak[1:4].tolist())), "value": float(peak[4]),
            "interpretation": "Maximum among actual measurements, not a source position or a certified whole-domain plume threshold."}
        result["identifier_note"] = "source_grid_node_id identifies a hypothetical source location and may be unmeasured. Only actually measured node_ids may be cited as plume_node_id or evidence_node_ids."
    return result
