from __future__ import annotations

import math
from typing import Any

from .geometry import injection_support_error

LEVELS = ("L1", "L2", "L3", "L4", "L5")


def region_score(prediction, positions):
    """Node-set overlap; empty assertion differs from an unknown prediction."""
    if prediction is None:
        return {"answered": False, "iou": None, "precision": None, "recall": None}
    predicted = {tuple(float(p[a]) for a in "xyz") for p in prediction}
    expected = {tuple(float(p[a]) for a in "xyz") for p in positions}
    overlap = len(predicted & expected)
    return {"answered": True, "predicted_nodes": len(predicted), "true_nodes": len(expected),
            "iou": overlap / len(predicted | expected) if predicted | expected else 1.,
            "precision": overlap / len(predicted) if predicted else float(not expected),
            "recall": overlap / len(expected) if expected else float(not predicted)}


def _position(answer: dict[str, Any], key: str) -> tuple[float, float, float] | None:
    value = answer.get(key)
    if not isinstance(value, dict):
        return None
    try:
        xyz = float(value["x"]), float(value["y"]), float(value["z"])
    except (KeyError, TypeError, ValueError):
        return None
    return xyz if all(math.isfinite(item) for item in xyz) else None


def score_submission(answer: dict[str, Any], truth: dict[str, Any], tolerances: dict[str, float]) -> dict[str, Any]:
    detectable = bool(truth["detectable_source"])
    expected_anomaly = "anomaly" if detectable else "no_anomaly"
    l1 = answer.get("anomaly") == expected_anomaly

    l2_applicable = detectable and bool(truth.get("L2_plume_positions"))
    l2_h = l2_v = l2_distance = l2_plume_distance = None
    l2 = None
    peak = _position(answer, "peak_position")
    if l2_applicable:
        if peak is None:
            l2 = False
        else:
            # Distance to the complete-field maximum is retained as a
            # diagnostic, but it no longer determines L2 correctness.
            distances = [
                (math.hypot(peak[0] - item["x"], peak[1] - item["y"]), abs(peak[2] - item["z"]))
                for item in truth["L2_peak_positions"]
            ]
            l2_h, l2_v = min(distances, key=lambda item: (max(item[0] / tolerances["horizontal_m"], item[1] / tolerances["vertical_m"]), item))
            l2_distance = math.hypot(l2_h, l2_v)
            l2_plume_distance = min(
                math.dist(peak, (float(item["x"]), float(item["y"]), float(item["z"])))
                for item in truth["L2_plume_positions"]
            )
            # The environment is a discrete node graph and arrival readings
            # expose exact node coordinates.  The epsilon only absorbs JSON
            # floating-point round trips; it is not a spatial tolerance.
            l2 = l2_plume_distance <= 1e-6

    l3_applicable = detectable
    l3 = answer.get("mechanism") == truth.get("L3_mechanism") if l3_applicable else None

    physical_support = truth.get("source_geometry_contract") in {"injection_support_v1", "boundary_faces_v1"}
    if physical_support and (not truth.get("injection_geometry") or not truth.get("injection_provenance")):
        raise ValueError("physical support contract requires injection geometry and generator provenance")
    l4_applicable = detectable and bool(truth.get("injection_geometry") if physical_support else truth.get("source_nodes"))
    l4_h = l4_v = l4_distance = None
    l4 = None
    source = _position(answer, "source_position")
    if l4_applicable:
        if source is None:
            l4 = False
        elif physical_support:
            l4_h, l4_v = injection_support_error(source, truth["injection_geometry"], tolerances)
            l4_distance = math.hypot(l4_h, l4_v)
            l4 = l4_h <= tolerances["horizontal_m"] and l4_v <= tolerances["vertical_m"]
        else:
            distances = [
                (
                    math.dist((source[0], source[1]), (float(node["x"]), float(node["y"]))),
                    abs(source[2] - float(node["z"])),
                )
                for node in truth["source_nodes"]
            ]
            l4_h, l4_v = min(distances, key=lambda item: (max(item[0]/tolerances["horizontal_m"],item[1]/tolerances["vertical_m"]) if truth.get("case_contract")=="source_event_v2" else math.hypot(item[0],item[1]), item))
            l4_distance = math.hypot(l4_h, l4_v)
            l4 = l4_h <= tolerances["horizontal_m"] and l4_v <= tolerances["vertical_m"]

    decision = answer.get("decision")
    if decision not in {"declare", "cannot_determine"}:
        expected_decision = "declare" if detectable and l4 is True else "cannot_determine"
        l5_category = "unsubmitted"
    elif not detectable:
        expected_decision = "cannot_determine"
        l5_category = "correct_true_negative" if decision == expected_decision else "false_declaration_true_negative"
    elif l4 is True:
        expected_decision = "declare"
        l5_category = "correct_declaration" if decision == expected_decision else "overconservative"
    else:
        expected_decision = "cannot_determine"
        l5_category = "correct_uncertainty" if decision == expected_decision else "false_declaration"
    l5 = decision in {"declare", "cannot_determine"} and decision == expected_decision
    candidate_answer = answer.get("source_candidate_id")
    candidate_match = None if not l4_applicable or candidate_answer is None else candidate_answer == truth.get("L4_candidate_id")

    result = {
        "L1_pass": l1,
        "L2_applicable": l2_applicable, "L2_pass": l2,
        "L2_horizontal_error_m": l2_h, "L2_vertical_error_m": l2_v, "L2_peak_distance_m": l2_distance,
        "L2_plume_distance_m": l2_plume_distance,
        "L2_reported_position_in_plume": l2,
        "L2_scoring_target": "registered_F1_all_anomalous_footprint_components",
        "L3_applicable": l3_applicable, "L3_pass": l3,
        "L4_applicable": l4_applicable, "L4_pass": l4,
        "L4_horizontal_error_m": l4_h, "L4_vertical_error_m": l4_v, "L4_source_distance_m": l4_distance,
        "L4_scoring_target": "physical_injection_support" if physical_support else "nearest_registered_source_node",
        "L4_candidate_match": candidate_match,
        "L5_pass": l5, "L5_expected_decision": expected_decision, "L5_category": l5_category,
    }
    if truth.get("case_contract") == "source_event_v1":
        # These cases register event presence independently of a nonempty
        # thresholded footprint. An empty region is a legitimate L2 answer.
        if not truth["L2_plume_positions"]:
            result.update(L2_applicable=True, L2_pass=answer.get("plume_region_positions") == [])
        result["L2_scoring_target"] = "registered_source_excess_threshold_node_set"
        result["L2_region"] = region_score(answer.get("plume_region_positions"), truth["L2_plume_positions"])
        result["L4_region"] = region_score(answer.get("source_region_positions"), truth["source_nodes"])
        result["L4_scoring_target"] = "actual_weighted_injection_cells"
        if 'temporal_excess' in truth:
            frames=truth['temporal_excess']
            sample=answer.get('plume_measurement')
            hit=False
            if sample is not None:
                t=sample['time_s']; node=int(sample['node_id'])
                if frames[0]['time_s']<=t<=frames[-1]['time_s'] and 0<=node<len(frames[0]['values']):
                    upper=next((i for i,f in enumerate(frames) if f['time_s']>=t),len(frames)-1)
                    lower=max(0,upper-1)
                    fraction=(t-frames[lower]['time_s'])/max(frames[upper]['time_s']-frames[lower]['time_s'],1e-12)
                    excess=frames[lower]['values'][node]+fraction*(frames[upper]['values'][node]-frames[lower]['values'][node])
                    hit=excess>=truth['plume_excess_threshold']
            elif not truth['L2_plume_positions']:
                hit=answer.get('plume_region_positions')==[]
            result.update(L2_applicable=True,L2_pass=hit,L2_scoring_target='source_excess_at_actual_acquisition_time')
    if truth.get("case_contract") == "source_event_v2":
        from .source_event import apply_v2
        result = apply_v2(result, answer, truth)
    return result
