"""Unified, fail-closed data layer for the v20 paper-figure pipeline.

This module deliberately sits beside, rather than inside, the byte-frozen
legacy Nature renderer.  It normalizes two explicitly different sources into
one set of publication tables:

* a visibly labelled synthetic layout rehearsal; and
* replay-verified v20 formal evidence.

Synthetic rows can never be promoted by the formal adapter.  The adapter also
recomputes every score from the sealed answer and evaluator truth before a row
is admitted.  Plotting code should consume only :class:`PaperDataBundle`, not
raw logs or evaluator truth directly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from laketrace.data.store import DataStore
from laketrace.scoring.scorer import score_submission
from laketrace.util import jsonl, read_csv, sha256_file, stable_hash, write_csv


PAPER_SCHEMA_VERSION = "laketrace-v20-paper-data-v1"
PROTOCOL_ID = "laketrace-v20-3d-single-edge-v1"
SYNTHETIC_KIND = "synthetic_layout_rehearsal"
FORMAL_KIND = "formal_replay_verified"
REGISTERED_CONTROL_KIND = "registered_control_evidence"
SYNTHETIC_WATERMARK = "SIMULATED DATA — LAYOUT REHEARSAL"
DEFAULT_CONTROL_ROOT = (
    Path(__file__).resolve().parents[2]
    / "results" / "validation" / "v20_public_baselines_final_v4"
)


OUTCOME_FIELDS = (
    "schema_version", "dataset_kind", "synthetic", "protocol_id", "scenario_id",
    "run_id", "episode_id", "actor", "model_display_name", "model_snapshot",
    "instance_id", "seed", "attempt_index", "science_eligible", "replay_verified",
    "waterbody", "domain_id", "difficulty_tier", "event_cluster_id", "background_id",
    "detectable_source", "source_geometry_type", "dstar",
    "termination_reason", "submit_only_reason", "termination_class", "submitted",
    "turns", "move_calls", "analyze_calls", "submit_calls", "invalid_calls",
    "infeasible_calls", "cancelled_unobserved_calls", "visited_nodes", "sample_count",
    "n_total_nodes", "coverage_fraction", "endurance_used_fraction",
    "range_used_fraction", "repeat_diagnostic_count", "action_success_n",
    "action_attempt_n", "action_parse_success_rate", "move_analyze_ratio",
    "repetition_fraction", "entered_anomaly_footprint",
    "input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens",
    "cost_public_price", "latency_s", "provider_attempts", "identity_failures",
    "operational_metric_origin",
    "entered_l4_tolerance", "trajectory_source_closest_horizontal_m",
    "trajectory_source_closest_vertical_m", "trajectory_source_closest_distance_m",
    "footprint_entry_step", "source_approach_step", "source_approach_after_footprint",
    "source_approach_before_footprint", "bypass_source_without_prior_footprint",
    "bypass_correct_without_footprint", "bypass_correct_without_post_footprint_source",
    "checkpoint_A_footprint_entry", "checkpoint_B_source_approach_after_A",
    "checkpoint_C_correct_L4_submission",
    "answer_anomaly", "answer_peak_x", "answer_peak_y", "answer_peak_z",
    "answer_mechanism", "answer_source_x", "answer_source_y", "answer_source_z",
    "answer_source_candidate_id", "answer_decision", "terminal_confidence",
    "rho", "rho_status", "rho_source_distance_tau", "rho_peak_distance_tau",
    "L1_pass", "L2_applicable", "L2_pass", "L2_horizontal_error_m",
    "L2_vertical_error_m", "L2_peak_distance_m", "L2_plume_distance_m",
    "L2_reported_position_in_plume", "L2_scoring_target", "L3_applicable", "L3_pass",
    "L4_applicable", "L4_pass", "L4_horizontal_error_m", "L4_vertical_error_m",
    "L4_source_distance_m", "L4_scoring_target", "L4_candidate_match", "L5_pass",
    "L5_expected_decision", "L5_category", "first_behavioral_stage",
    "first_failure_stage", "storyboard_selected", "storyboard_pair_id",
    "storyboard_actor_slot", "analyze_fraction", "revisit_fraction",
    "breadth_model_task_mean", "endurance_model_task_mean",
    "analysis_model_task_mean", "revisit_model_task_mean",
    "breadth_tertile", "endurance_tertile", "analysis_tertile", "revisit_tertile",
    "submission_model_task_mean", "repetition_model_task_mean",
    "proximity_model_task_mean", "submission_tertile", "repetition_tertile",
    "proximity_tertile",
    "model_config_hash", "presentation_mode", "source_episode_sha256",
)

STEP_FIELDS = (
    "schema_version", "dataset_kind", "synthetic", "episode_id", "actor",
    "instance_id", "seed", "step_index", "provider_round_index", "action", "ok",
    "status", "origin_node", "arrival_node", "node_id", "x", "y", "z",
    "direction", "cost_s", "path_length_m", "endurance_used_fraction",
    "range_used_fraction", "visited_nodes", "sample_sequence", "visit_index",
    "measurement_mode", "new_samples_since_last_analyze", "no_new_evidence",
    "cancelled_unobserved_calls", "intent", "evidence_node_ids_csv", "confidence",
    "decision_note", "storyboard_selected", "storyboard_pair_id",
    "storyboard_actor_slot",
)

GEOMETRY_FIELDS = (
    "schema_version", "dataset_kind", "synthetic", "record_origin", "instance_id",
    "waterbody", "domain_id", "difficulty_tier", "event_cluster_id", "background_id",
    "detectable_source", "source_geometry_type", "geometry_role", "node_id", "x", "y",
    "z", "value", "primary_tracer", "dstar", "source_tolerance_horizontal_m",
    "source_tolerance_vertical_m", "plume_component_count", "plume_node_count",
    "d_star", "d_star_stratum", "representative", "panel",
)

DECISION_FIELDS = (
    "schema_version", "dataset_kind", "synthetic", "episode_id", "actor",
    "instance_id", "seed", "decision_record_index", "provider_round_index",
    "basis_observation_hash", "chosen_action_name", "status", "intent",
    "evidence_node_ids_csv", "cited_measured_node_count", "confidence", "decision_note",
    "fed_back_to_agent", "truth_visible",
)

SCORE_FIELDS = (
    "L1_pass", "L2_applicable", "L2_pass", "L2_horizontal_error_m",
    "L2_vertical_error_m", "L2_peak_distance_m", "L2_plume_distance_m",
    "L2_reported_position_in_plume", "L2_scoring_target", "L3_applicable", "L3_pass",
    "L4_applicable", "L4_pass", "L4_horizontal_error_m", "L4_vertical_error_m",
    "L4_source_distance_m", "L4_scoring_target", "L4_candidate_match", "L5_pass",
    "L5_expected_decision", "L5_category",
)

CONTROL_FIELDS = (
    "schema_version", "dataset_kind", "synthetic", "protocol_id", "control_kind",
    "model_matrix_member", "actor", "instance_id", "seed", "waterbody", "domain_id",
    "difficulty_tier", "event_cluster_id", "background_id", "detectable_source",
    "source_geometry_type", "dstar", "evidence_origin", "evidence_sha256",
    "evidence_manifest", "evidence_manifest_sha256", *SCORE_FIELDS,
)


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (1, "1", "true", "True", "TRUE"):
        return True
    if value in (0, "0", "false", "False", "FALSE"):
        return False
    return None


def _float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _position(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        result = float(value["x"]), float(value["y"]), float(value["z"])
    except (KeyError, TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _xyz_key(value: Mapping[str, Any] | Sequence[float]) -> tuple[float, float, float]:
    if isinstance(value, Mapping):
        values = value["x"], value["y"], value["z"]
    else:
        values = value
    return tuple(round(float(item), 9) for item in values)  # type: ignore[return-value]


def _source_error(
    position: tuple[float, float, float], source_nodes: Sequence[Mapping[str, Any]],
) -> tuple[float, float, float] | None:
    if not source_nodes:
        return None
    pairs = []
    for node in source_nodes:
        horizontal = math.dist(position[:2], (float(node["x"]), float(node["y"])))
        vertical = abs(position[2] - float(node["z"]))
        pairs.append((horizontal, vertical, math.hypot(horizontal, vertical)))
    return min(pairs, key=lambda item: (item[2], item[0], item[1]))


def task_dstar(store: DataStore, truth: Mapping[str, Any]) -> float | None:
    """Return the registered set-invariant normalized peak/source separation."""

    peaks = truth.get("L2_peak_positions") or []
    sources = truth.get("source_nodes") or []
    if not peaks or not sources:
        return None
    tolerance = store.protocol["tolerances"][str(truth["waterbody"])]
    values = []
    for peak in peaks:
        for source in sources:
            horizontal = math.dist(
                (float(peak["x"]), float(peak["y"])),
                (float(source["x"]), float(source["y"])),
            )
            vertical = abs(float(peak["z"]) - float(source["z"]))
            values.append(math.sqrt(
                (horizontal / float(tolerance["horizontal_m"])) ** 2
                + (vertical / float(tolerance["vertical_m"])) ** 2
            ))
    return min(values)


def _dstar_stratum(value: float | None) -> str | None:
    if value is None:
        return None
    if value < 1:
        return "D* < 1"
    if value < 2:
        return "1 <= D* < 2"
    if value < 4:
        return "2 <= D* < 4"
    return "D* >= 4"


def build_task_geometry(
    store: DataStore, *, dataset_kind: str, instance_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Materialize evaluator-only geometry with an explicit truth-origin label."""

    if not store.evaluator:
        raise PermissionError("task geometry requires evaluator truth")
    wanted = set(instance_ids) if instance_ids is not None else {
        str(row["instance_id"]) for row in store.tasks()
    }
    synthetic = dataset_kind == SYNTHETIC_KIND
    tasks = [task for task in store.tasks() if str(task["instance_id"]) in wanted]
    representative_candidates = []
    for task in tasks:
        truth = store.truth(str(task["instance_id"]))
        if (
            truth.get("detectable_source")
            and truth.get("source_nodes")
            and truth.get("L2_peak_positions")
        ):
            representative_candidates.append(str(task["instance_id"]))
    representative = min(
        representative_candidates,
        key=lambda instance_id: stable_hash({
            "selection": "v20-outcome-blind-representative-v1",
            "protocol_id": PROTOCOL_ID,
            "instance_id": instance_id,
        }),
        default=None,
    )
    rows: list[dict[str, Any]] = []
    for task in tasks:
        instance_id = str(task["instance_id"])
        if instance_id not in wanted:
            continue
        truth = store.truth(instance_id)
        tolerance = store.protocol["tolerances"][truth["waterbody"]]
        dstar = task_dstar(store, truth)
        common = {
            "schema_version": PAPER_SCHEMA_VERSION,
            "dataset_kind": dataset_kind,
            "synthetic": synthetic,
            "record_origin": "registered_evaluator_truth",
            "instance_id": instance_id,
            "waterbody": truth["waterbody"],
            "domain_id": truth["domain_id"],
            "difficulty_tier": truth.get("difficulty_tier"),
            "event_cluster_id": truth.get("event_cluster_id"),
            "background_id": truth.get("background_id"),
            "detectable_source": bool(truth.get("detectable_source")),
            "source_geometry_type": truth.get("source_geometry_type"),
            "primary_tracer": truth.get("L2_primary_tracer"),
            "dstar": dstar,
            "d_star": dstar,
            "d_star_stratum": _dstar_stratum(dstar),
            "representative": instance_id == representative,
            "panel": "Fig1a" if instance_id == representative else None,
            "source_tolerance_horizontal_m": tolerance["horizontal_m"],
            "source_tolerance_vertical_m": tolerance["vertical_m"],
            "plume_component_count": truth.get("L2_plume_component_count", 0),
            "plume_node_count": truth.get("L2_plume_node_count", 0),
        }
        rows.append({
            **common, "geometry_role": "task_summary", "node_id": None,
            "x": None, "y": None, "z": None, "value": None,
        })
        for node in truth.get("source_nodes") or []:
            rows.append({
                **common, "geometry_role": "source_node", "node_id": node.get("node_id"),
                "x": node["x"], "y": node["y"], "z": node["z"], "value": None,
            })
        for index, node in enumerate(truth.get("L2_peak_positions") or []):
            rows.append({
                **common, "geometry_role": "peak_node", "node_id": index,
                "x": node["x"], "y": node["y"], "z": node["z"],
                "value": node.get("value"),
            })
        for node_id, node in zip(
            truth.get("L2_plume_node_ids") or [], truth.get("L2_plume_positions") or [],
        ):
            rows.append({
                **common, "geometry_role": "anomaly_footprint_node", "node_id": node_id,
                "x": node["x"], "y": node["y"], "z": node["z"], "value": None,
            })
    return rows


def _trajectory_positions(episode: Mapping[str, Any]) -> list[tuple[float, float, float]]:
    positions = []
    for row in episode.get("sample_ledger") or []:
        point = _position(row)
        if point is not None:
            positions.append(point)
    if positions:
        return positions
    for event in episode.get("events") or []:
        readings = event.get("readings") or (
            [event["reading"]] if isinstance(event.get("reading"), Mapping) else []
        )
        for row in readings:
            point = _position(row)
            if point is not None:
                positions.append(point)
    return positions


def trajectory_diagnostics(
    store: DataStore, episode: Mapping[str, Any], truth: Mapping[str, Any],
) -> dict[str, Any]:
    positions = _trajectory_positions(episode)
    plume = {_xyz_key(row) for row in truth.get("L2_plume_positions") or []}
    plume_flags = [_xyz_key(position) in plume for position in positions]
    footprint_index = next((index for index, value in enumerate(plume_flags) if value), None)
    entered_plume = footprint_index is not None
    source_nodes = truth.get("source_nodes") or []
    errors = [
        value for point in positions
        if (value := _source_error(point, source_nodes)) is not None
    ]
    closest = min(errors, key=lambda item: (item[2], item[0], item[1])) if errors else None
    tolerance = store.protocol["tolerances"][truth["waterbody"]]
    source_flags = [
        horizontal <= float(tolerance["horizontal_m"])
        and vertical <= float(tolerance["vertical_m"])
        for horizontal, vertical, _ in errors
    ]
    source_index = next((index for index, value in enumerate(source_flags) if value), None)
    entered_l4 = source_index is not None
    after_index = next(
        (
            index for index, value in enumerate(source_flags)
            if value and footprint_index is not None and index >= footprint_index
        ),
        None,
    )
    before_footprint = source_index is not None and (
        footprint_index is None or source_index < footprint_index
    )
    return {
        "entered_anomaly_footprint": entered_plume,
        "entered_l4_tolerance": entered_l4,
        "footprint_entry_step": footprint_index,
        "source_approach_step": source_index,
        "source_approach_after_footprint": after_index is not None,
        "source_approach_before_footprint": before_footprint,
        "trajectory_source_closest_horizontal_m": closest[0] if closest else None,
        "trajectory_source_closest_vertical_m": closest[1] if closest else None,
        "trajectory_source_closest_distance_m": closest[2] if closest else None,
    }


def _termination_class(reason: Any, submit_only_reason: Any) -> str:
    if reason == "submitted":
        if submit_only_reason == "physical_budget_exhausted":
            return "submitted_after_physical_limit"
        if submit_only_reason in {"provider_round_guard", "engineering_provider_round_cap"}:
            return "submitted_after_round_guard"
        return "submitted_direct"
    if reason == "infra_failure":
        return "infrastructure_failure"
    if submit_only_reason == "physical_budget_exhausted":
        return "physical_limit_no_submission"
    if reason == "provider_round_guard_exhausted":
        return "round_guard_no_submission"
    return "schema_or_other_no_submission"


def _behavioral_stage(diagnostics: Mapping[str, Any], score: Mapping[str, Any]) -> str:
    if not score.get("L4_applicable"):
        return "not_applicable_control"
    # Success is classified first, while bypasses remain explicit audit flags.
    if score.get("L4_pass") is True:
        return "success"
    if not diagnostics["entered_anomaly_footprint"]:
        return "sampling_coverage"
    if not diagnostics["source_approach_after_footprint"]:
        return "source_approach"
    return "conclusion_submission"


def _rho_diagnostic(
    position: tuple[float, float, float] | None,
    truth: Mapping[str, Any],
    tolerance: Mapping[str, Any],
) -> dict[str, Any]:
    """Registered relative set-distance rho; no source--peak axis projection."""

    if position is None:
        return {
            "rho": None, "rho_status": "no_valid_submitted_coordinate",
            "rho_source_distance_tau": None, "rho_peak_distance_tau": None,
        }
    sources = truth.get("source_nodes") or []
    peaks = truth.get("L2_peak_positions") or []
    if not sources:
        status = "no_registered_source_node"
    elif not peaks:
        status = "no_registered_peak"
    elif float(tolerance.get("horizontal_m", 0)) <= 0 or float(
        tolerance.get("vertical_m", 0)
    ) <= 0:
        status = "nonpositive_tolerance"
    else:
        status = "ok"
    if status != "ok":
        return {
            "rho": None, "rho_status": status,
            "rho_source_distance_tau": None, "rho_peak_distance_tau": None,
        }

    def distance_to_set(nodes: Sequence[Mapping[str, Any]]) -> float:
        values = []
        for node in nodes:
            horizontal = math.dist(
                position[:2], (float(node["x"]), float(node["y"])),
            )
            vertical = abs(position[2] - float(node["z"]))
            values.append(math.sqrt(
                (horizontal / float(tolerance["horizontal_m"])) ** 2
                + (vertical / float(tolerance["vertical_m"])) ** 2
            ))
        return min(values)

    source_distance = distance_to_set(sources)
    peak_distance = distance_to_set(peaks)
    denominator = source_distance + peak_distance
    if denominator <= 0:
        return {
            "rho": None, "rho_status": "zero_denominator_overlapping_sets",
            "rho_source_distance_tau": source_distance,
            "rho_peak_distance_tau": peak_distance,
        }
    return {
        "rho": source_distance / denominator,
        "rho_status": "ok",
        "rho_source_distance_tau": source_distance,
        "rho_peak_distance_tau": peak_distance,
    }


def _terminal_confidence(episode: Mapping[str, Any]) -> float | None:
    for row in reversed(list(episode.get("decision_record_calls") or [])):
        record = row.get("record") or row.get("raw_record") or {}
        value = _float(record.get("confidence"))
        if value is not None:
            return value
    return None


def _operational_metrics(
    episode: Mapping[str, Any], *, dataset_kind: str,
) -> dict[str, Any]:
    """Normalize optional cost/latency fields without inventing formal evidence."""

    usage = episode.get("token_usage")
    if not isinstance(usage, Mapping):
        usage = {}
    provider_calls = episode.get("provider_calls")
    attempts = episode.get("provider_attempts")
    if attempts is None and isinstance(provider_calls, Sequence) and not isinstance(
        provider_calls, (str, bytes)
    ):
        attempts = len(provider_calls)
    origin = episode.get("operational_metric_origin")
    if origin is None and dataset_kind == SYNTHETIC_KIND:
        origin = "synthetic_layout_only"
    elif origin is None and any(
        episode.get(field) is not None
        for field in ("input_tokens", "output_tokens", "latency_s", "cost_public_price")
    ):
        origin = "recorded_episode_field"
    return {
        "input_tokens": episode.get("input_tokens", usage.get("input_tokens")),
        "output_tokens": episode.get("output_tokens", usage.get("output_tokens")),
        "reasoning_tokens": episode.get(
            "reasoning_tokens", usage.get("reasoning_tokens")
        ),
        "cached_tokens": episode.get("cached_tokens", usage.get("cached_tokens")),
        "cost_public_price": episode.get(
            "cost_public_price", episode.get("public_price_cost")
        ),
        "latency_s": episode.get("latency_s", episode.get("wall_duration_s")),
        "provider_attempts": attempts,
        "identity_failures": episode.get("identity_failures"),
        "operational_metric_origin": origin,
    }


def outcome_from_episode(
    store: DataStore,
    episode: Mapping[str, Any],
    score: Mapping[str, Any],
    *,
    dataset_kind: str,
    replay_verified: bool | None,
    scenario_id: str = "",
    model_display_name: str | None = None,
    model_snapshot: str | None = None,
    source_episode_sha256: str | None = None,
) -> dict[str, Any]:
    if not store.evaluator:
        raise PermissionError("paper outcomes require evaluator truth")
    instance_id = str(episode["instance_id"])
    truth = store.truth(instance_id)
    answer = episode.get("answer") if isinstance(episode.get("answer"), Mapping) else {}
    diagnostics = trajectory_diagnostics(store, episode, truth)
    events = list(episode.get("events") or [])
    actions = [str(row.get("action") or "") for row in events]
    attempted = [row for row in events if row.get("action") != "infrastructure"]
    move_calls = actions.count("move")
    analyze_calls = actions.count("analyze")
    success_n = sum(row.get("ok") is True for row in attempted)
    sample_count = int(episode.get("sample_count") or len(episode.get("sample_ledger") or []))
    visited_nodes = int(episode.get("visited_nodes") or len({
        row.get("node_index") for row in episode.get("sample_ledger") or []
        if row.get("node_index") is not None
    }))
    n_total_nodes = int(episode.get("n_total_nodes") or 0)
    repeat_count = sum(bool(row.get("same_result_no_progress")) for row in events)
    if not repeat_count:
        repeat_count = max(0, sample_count - visited_nodes)
    peak = _position(answer.get("peak_position"))
    source = _position(answer.get("source_position"))
    tolerance = store.protocol["tolerances"][truth["waterbody"]]
    rho = _rho_diagnostic(source, truth, tolerance)
    l4_correct = score.get("L4_applicable") is True and score.get("L4_pass") is True
    bypass_source = bool(diagnostics["source_approach_before_footprint"])
    bypass_correct_without_footprint = bool(
        l4_correct and not diagnostics["entered_anomaly_footprint"]
    )
    bypass_correct_without_post_source = bool(
        l4_correct and not diagnostics["source_approach_after_footprint"]
    )
    behavioral_stage = _behavioral_stage(diagnostics, score)
    row: dict[str, Any] = {
        "schema_version": PAPER_SCHEMA_VERSION,
        "dataset_kind": dataset_kind,
        "synthetic": dataset_kind == SYNTHETIC_KIND,
        "protocol_id": episode.get("protocol_id", PROTOCOL_ID),
        "scenario_id": scenario_id,
        "run_id": episode.get("run_id"),
        "episode_id": episode.get("episode_id"),
        "actor": episode.get("actor"),
        "model_display_name": model_display_name or episode.get("model_display_name") or episode.get("actor"),
        "model_snapshot": model_snapshot or episode.get("configured_model"),
        "instance_id": instance_id,
        "seed": episode.get("seed"),
        "attempt_index": episode.get("attempt_index", 0),
        "science_eligible": bool(episode.get("science_eligible", False)),
        "replay_verified": replay_verified,
        "waterbody": truth["waterbody"],
        "domain_id": truth["domain_id"],
        "difficulty_tier": truth.get("difficulty_tier"),
        "event_cluster_id": truth.get("event_cluster_id"),
        "background_id": truth.get("background_id"),
        "detectable_source": bool(truth.get("detectable_source")),
        "source_geometry_type": truth.get("source_geometry_type"),
        "dstar": task_dstar(store, truth),
        "termination_reason": episode.get("termination_reason"),
        "submit_only_reason": episode.get("submit_only_reason"),
        "termination_class": _termination_class(
            episode.get("termination_reason"), episode.get("submit_only_reason"),
        ),
        "submitted": bool(answer),
        "turns": episode.get("turns", len(events)),
        "move_calls": move_calls,
        "analyze_calls": analyze_calls,
        "submit_calls": actions.count("submit"),
        "invalid_calls": actions.count("invalid"),
        "infeasible_calls": actions.count("infeasible"),
        "cancelled_unobserved_calls": episode.get("multi_call_cancelled_count", 0),
        "visited_nodes": visited_nodes,
        "sample_count": sample_count,
        "n_total_nodes": n_total_nodes,
        "coverage_fraction": visited_nodes / n_total_nodes if n_total_nodes else None,
        "endurance_used_fraction": episode.get("endurance_used_fraction"),
        "range_used_fraction": episode.get("range_used_fraction"),
        "repeat_diagnostic_count": repeat_count,
        "move_analyze_ratio": move_calls / analyze_calls if analyze_calls > 0 else None,
        "repetition_fraction": repeat_count / len(attempted) if attempted else None,
        "analyze_fraction": analyze_calls / len(attempted) if attempted else None,
        "revisit_fraction": repeat_count / sample_count if sample_count else None,
        "action_success_n": success_n,
        "action_attempt_n": len(attempted),
        "action_parse_success_rate": success_n / len(attempted) if attempted else None,
        **_operational_metrics(episode, dataset_kind=dataset_kind),
        **diagnostics,
        "bypass_source_without_prior_footprint": bypass_source,
        "bypass_correct_without_footprint": bypass_correct_without_footprint,
        "bypass_correct_without_post_footprint_source": bypass_correct_without_post_source,
        "checkpoint_A_footprint_entry": (
            diagnostics["entered_anomaly_footprint"]
            if score.get("L4_applicable") is True else None
        ),
        "checkpoint_B_source_approach_after_A": (
            diagnostics["source_approach_after_footprint"]
            if score.get("L4_applicable") is True else None
        ),
        "checkpoint_C_correct_L4_submission": (
            l4_correct if score.get("L4_applicable") is True else None
        ),
        "answer_anomaly": answer.get("anomaly"),
        "answer_peak_x": peak[0] if peak else None,
        "answer_peak_y": peak[1] if peak else None,
        "answer_peak_z": peak[2] if peak else None,
        "answer_mechanism": answer.get("mechanism"),
        "answer_source_x": source[0] if source else None,
        "answer_source_y": source[1] if source else None,
        "answer_source_z": source[2] if source else None,
        "answer_source_candidate_id": answer.get("source_candidate_id"),
        "answer_decision": answer.get("decision"),
        "terminal_confidence": _terminal_confidence(episode),
        **rho,
        **{field: score.get(field) for field in SCORE_FIELDS},
        "first_behavioral_stage": behavioral_stage,
        "first_failure_stage": (
            behavioral_stage if score.get("L4_applicable") is True else None
        ),
        "storyboard_selected": False,
        "model_config_hash": episode.get("model_config_hash"),
        "presentation_mode": episode.get("presentation_mode"),
        "source_episode_sha256": source_episode_sha256,
    }
    return {field: row.get(field) for field in OUTCOME_FIELDS}


BEHAVIOR_TERTILE_METRICS: Mapping[str, tuple[str, str, str]] = {
    "endurance": (
        "endurance_used_fraction", "endurance_model_task_mean", "endurance_tertile",
    ),
    "submission": ("submitted", "submission_model_task_mean", "submission_tertile"),
    "repetition": (
        "repetition_fraction", "repetition_model_task_mean", "repetition_tertile",
    ),
    "proximity": (
        "trajectory_source_closest_distance_m", "proximity_model_task_mean",
        "proximity_tertile",
    ),
}

# Retain the previously published data columns for downstream compatibility.
# They are explicitly not the Figure Outline v18 Fig. 4 metric set.
LEGACY_BEHAVIOR_TERTILE_METRICS: Mapping[str, tuple[str, str, str]] = {
    "breadth": ("coverage_fraction", "breadth_model_task_mean", "breadth_tertile"),
    "analysis": ("analyze_fraction", "analysis_model_task_mean", "analysis_tertile"),
    "revisit": ("revisit_fraction", "revisit_model_task_mean", "revisit_tertile"),
}


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one finite value")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def attach_model_behavior_tertiles(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Publish direction-neutral tertiles after seed-first, task-equal aggregation."""

    summary: dict[str, Any] = {
        "method": (
            "for each actor×task, arithmetic mean over finite seed values; then "
            "equal-task actor mean; empirical 1/3 and 2/3 thresholds"
        ),
        "published_metric_order": list(BEHAVIOR_TERTILE_METRICS),
        "metrics": {},
        "compatibility_metrics": {},
    }

    def attach_metric(raw_field: str, summary_field: str, tertile_field: str) -> dict[str, Any]:
        cell_values: dict[tuple[str, str], list[float]] = {}
        for row in rows:
            value = _float(row.get(raw_field))
            if value is None:
                continue
            key = str(row.get("actor")), str(row.get("instance_id"))
            cell_values.setdefault(key, []).append(value)
        task_means = {
            key: sum(values) / len(values) for key, values in cell_values.items()
        }
        actor_tasks: dict[str, list[float]] = {}
        for (actor, _), value in task_means.items():
            actor_tasks.setdefault(actor, []).append(value)
        actor_means = {
            actor: sum(values) / len(values) for actor, values in actor_tasks.items()
        }
        finite = list(actor_means.values())
        if finite and not math.isclose(min(finite), max(finite), abs_tol=1e-15):
            lower, upper = _quantile(finite, 1 / 3), _quantile(finite, 2 / 3)
            labels = {
                actor: "low" if value < lower else "high" if value > upper else "middle"
                for actor, value in actor_means.items()
            }
        else:
            lower = upper = finite[0] if finite else None
            labels = {actor: "middle" for actor in actor_means}
        for row in rows:
            actor = str(row.get("actor"))
            row[summary_field] = actor_means.get(actor)
            row[tertile_field] = labels.get(actor)
        return {
            "source_field": raw_field,
            "lower_cut": lower,
            "upper_cut": upper,
            "actor_task_means": actor_means,
            "task_counts": {actor: len(values) for actor, values in actor_tasks.items()},
            "labels": labels,
        }

    for metric, fields in BEHAVIOR_TERTILE_METRICS.items():
        summary["metrics"][metric] = attach_metric(*fields)
    for metric, fields in LEGACY_BEHAVIOR_TERTILE_METRICS.items():
        summary["compatibility_metrics"][metric] = attach_metric(*fields)
    return summary


def step_rows_from_episode(
    episode: Mapping[str, Any], *, dataset_kind: str,
) -> list[dict[str, Any]]:
    decisions = {
        int(row.get("provider_round_index", row.get("decision_record_index", -1))): row
        for row in episode.get("decision_record_calls") or []
    }
    output = []
    for index, event in enumerate(episode.get("events") or []):
        round_index = int(event.get("provider_round_index", index))
        decision = decisions.get(round_index, {})
        record = decision.get("record") or decision.get("raw_record") or {}
        reading = event.get("reading") if isinstance(event.get("reading"), Mapping) else None
        budget = event.get("budget_after") or {}
        endurance = budget.get("endurance") or {}
        range_budget = budget.get("range") or {}
        row = {
            "schema_version": PAPER_SCHEMA_VERSION,
            "dataset_kind": dataset_kind,
            "synthetic": dataset_kind == SYNTHETIC_KIND,
            "episode_id": episode.get("episode_id"),
            "actor": episode.get("actor"),
            "instance_id": episode.get("instance_id"),
            "seed": episode.get("seed"),
            "step_index": index,
            "provider_round_index": round_index,
            "action": event.get("action"),
            "ok": event.get("ok"),
            "status": "executed" if event.get("ok") is True else event.get("code", "failed"),
            "origin_node": event.get("origin_node"),
            "arrival_node": event.get("arrival_node"),
            "node_id": reading.get("node_index") if reading else None,
            "x": reading.get("x") if reading else None,
            "y": reading.get("y") if reading else None,
            "z": reading.get("z") if reading else None,
            "direction": event.get("direction"),
            "cost_s": event.get("cost_s", event.get("physical_cost_s")),
            "path_length_m": event.get("path_length_m"),
            "endurance_used_fraction": (
                float(endurance.get("used", 0)) / float(endurance["limit"])
                if endurance.get("limit") else None
            ),
            "range_used_fraction": (
                float(range_budget.get("used", 0)) / float(range_budget["limit"])
                if range_budget.get("limit") else None
            ),
            "visited_nodes": event.get("visited_nodes"),
            "sample_sequence": reading.get("sample_sequence") if reading else None,
            "visit_index": reading.get("visit_index") if reading else None,
            "measurement_mode": reading.get("mode") if reading else None,
            "new_samples_since_last_analyze": event.get("new_samples_since_last_analyze"),
            "no_new_evidence": event.get("no_new_evidence"),
            "cancelled_unobserved_calls": event.get("cancelled_tool_call_count", 0),
            "intent": record.get("intent"),
            "evidence_node_ids_csv": record.get("evidence_node_ids_csv"),
            "confidence": record.get("confidence"),
            "decision_note": record.get("decision_note"),
        }
        output.append({field: row.get(field) for field in STEP_FIELDS})
    return output


def decision_rows_from_episode(
    episode: Mapping[str, Any], *, dataset_kind: str,
) -> list[dict[str, Any]]:
    output = []
    for index, raw in enumerate(episode.get("decision_record_calls") or []):
        record = raw.get("record") or raw.get("raw_record") or {}
        evidence = str(record.get("evidence_node_ids_csv") or "").strip()
        cited = [value for value in evidence.replace(";", ",").split(",") if value.strip()]
        row = {
            "schema_version": PAPER_SCHEMA_VERSION,
            "dataset_kind": dataset_kind,
            "synthetic": dataset_kind == SYNTHETIC_KIND,
            "episode_id": episode.get("episode_id"),
            "actor": episode.get("actor"),
            "instance_id": episode.get("instance_id"),
            "seed": episode.get("seed"),
            "decision_record_index": raw.get("decision_record_index", index),
            "provider_round_index": raw.get("provider_round_index", index),
            "basis_observation_hash": raw.get("basis_observation_hash"),
            "chosen_action_name": raw.get("chosen_action_name"),
            "status": raw.get("status"),
            "intent": record.get("intent"),
            "evidence_node_ids_csv": record.get("evidence_node_ids_csv"),
            "cited_measured_node_count": len(cited),
            "confidence": record.get("confidence"),
            "decision_note": record.get("decision_note"),
            "fed_back_to_agent": raw.get("fed_back_to_agent"),
            "truth_visible": raw.get("truth_visible"),
        }
        output.append({field: row.get(field) for field in DECISION_FIELDS})
    return output


def attach_storyboard_selection(
    outcomes: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    *,
    selection_seed: Any,
    require_contrast: bool,
) -> dict[str, Any]:
    """Select one same-task/same-seed two-actor storyboard by stable hash."""

    for row in outcomes:
        row["storyboard_selected"] = False
        row["storyboard_pair_id"] = None
        row["storyboard_actor_slot"] = None
    for row in steps:
        row["storyboard_selected"] = False
        row["storyboard_pair_id"] = None
        row["storyboard_actor_slot"] = None

    steps_by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in steps:
        episode_id = str(row.get("episode_id"))
        if _float(row.get("x")) is not None and _float(row.get("y")) is not None:
            steps_by_episode.setdefault(episode_id, []).append(row)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in outcomes:
        episode_id = str(row.get("episode_id"))
        if row.get("L4_applicable") is not True or len(steps_by_episode.get(episode_id, [])) < 2:
            continue
        groups.setdefault((str(row.get("instance_id")), int(row.get("seed"))), []).append(row)

    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    fallback: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for group in groups.values():
        for left_index, left in enumerate(group):
            for right in group[left_index + 1:]:
                if left.get("actor") == right.get("actor"):
                    continue
                pair = left, right
                fallback.append(pair)
                if left.get("first_behavioral_stage") != right.get("first_behavioral_stage"):
                    candidates.append(pair)
    if not candidates:
        if require_contrast:
            raise ValueError("no complete contrasting two-actor storyboard pair exists")
        candidates = fallback
    if not candidates:
        return {"selected": False, "reason": "no_two_actor_complete_pair"}

    selected = min(
        candidates,
        key=lambda pair: stable_hash({
            "selection": "v20-storyboard-stable-hash-v1",
            "seed": selection_seed,
            "cell": [pair[0].get("instance_id"), pair[0].get("seed")],
            "episodes": sorted([pair[0].get("episode_id"), pair[1].get("episode_id")]),
        }),
    )
    selected = tuple(sorted(
        selected,
        key=lambda row: stable_hash({
            "selection": "v20-storyboard-actor-slot-v1",
            "seed": selection_seed,
            "episode_id": row.get("episode_id"),
        }),
    ))
    pair_hash = stable_hash({
        "selection": "v20-storyboard-pair-id-v1",
        "seed": selection_seed,
        "episodes": [row.get("episode_id") for row in selected],
    })
    prefix = "SYN-STORYBOARD" if all(row.get("synthetic") is True for row in selected) else "STORYBOARD"
    pair_id = f"{prefix}-{pair_hash[:16]}"
    slots: dict[str, int] = {}
    for slot, row in enumerate(selected, 1):
        episode_id = str(row.get("episode_id"))
        slots[episode_id] = slot
        row["storyboard_selected"] = True
        row["storyboard_pair_id"] = pair_id
        row["storyboard_actor_slot"] = slot
    for row in steps:
        episode_id = str(row.get("episode_id"))
        if episode_id in slots:
            row["storyboard_selected"] = True
            row["storyboard_pair_id"] = pair_id
            row["storyboard_actor_slot"] = slots[episode_id]
    return {
        "selected": True,
        "selection_rule": (
            "same registered task and seed; two distinct actors; complete measured paths; "
            "contrasting observable first stage when required; stable-hash tie-break"
        ),
        "pair_id": pair_id,
        "instance_id": selected[0].get("instance_id"),
        "seed": selected[0].get("seed"),
        "episodes": [row.get("episode_id") for row in selected],
        "actors": [row.get("actor") for row in selected],
        "stages": [row.get("first_behavioral_stage") for row in selected],
    }


def _typed_score_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    boolean_fields = {
        "L1_pass", "L2_applicable", "L2_pass", "L2_reported_position_in_plume",
        "L3_applicable", "L3_pass", "L4_applicable", "L4_pass",
        "L4_candidate_match", "L5_pass",
    }
    numeric_fields = {
        "L2_horizontal_error_m", "L2_vertical_error_m", "L2_peak_distance_m",
        "L2_plume_distance_m", "L4_horizontal_error_m", "L4_vertical_error_m",
        "L4_source_distance_m",
    }
    output: dict[str, Any] = {}
    for field in SCORE_FIELDS:
        value = raw.get(field)
        if field in boolean_fields:
            output[field] = _bool(value)
        elif field in numeric_fields:
            output[field] = _float(value)
        else:
            output[field] = None if value in (None, "") else value
    return output


def load_registered_controls(
    store: DataStore,
    control_root: str | Path = DEFAULT_CONTROL_ROOT,
) -> list[dict[str, Any]]:
    """Mechanically import the registered non-model controls used by Fig. 1c/4a.

    Controls remain a separate score-only table: they have no model episode or
    provider provenance and are never counted in the six-model rehearsal matrix.
    The source bytes must match the passing v20 artifact manifest.
    """

    if not store.evaluator:
        raise PermissionError("registered controls require evaluator task metadata")
    root = Path(control_root)
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("passed") is not True:
        raise ValueError("registered control artifact manifest is not passing")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("registered controls use a different protocol")
    registered_hashes = {
        str(row.get("file")): str(row.get("sha256"))
        for row in manifest.get("artifacts") or []
        if isinstance(row, Mapping)
    }
    specifications = (
        ("design_upper_scores.csv", "reachable_upper", "reachable_upper", {0}),
        ("design_random_scores_3seed.csv", "design_random_3seed", "random", {0, 1, 2}),
    )
    task_by_id = {str(row["instance_id"]): row for row in store.tasks()}
    task_ids = set(task_by_id)
    manifest_sha = sha256_file(manifest_path)
    output: list[dict[str, Any]] = []
    for filename, control_kind, expected_actor, expected_seeds in specifications:
        path = root / filename
        observed_sha = sha256_file(path)
        if registered_hashes.get(filename) != observed_sha:
            raise ValueError(f"registered control hash mismatch: {filename}")
        raw_rows = read_csv(path)
        keys: set[tuple[str, int]] = set()
        for raw in raw_rows:
            if _is_synthetic_mapping(raw):
                raise ValueError(f"registered control unexpectedly marked synthetic: {filename}")
            actor = str(raw.get("actor") or "")
            instance_id = str(raw.get("instance_id") or "")
            seed = int(raw.get("seed", -1))
            if actor != expected_actor:
                raise ValueError(f"unexpected actor in {filename}: {actor!r}")
            if instance_id not in task_by_id or seed not in expected_seeds:
                raise ValueError(f"unexpected registered control cell: {(instance_id, seed)}")
            key = instance_id, seed
            if key in keys:
                raise ValueError(f"duplicate registered control cell: {(filename, *key)}")
            keys.add(key)
            task = task_by_id[instance_id]
            truth = store.truth(instance_id)
            row = {
                "schema_version": PAPER_SCHEMA_VERSION,
                "dataset_kind": REGISTERED_CONTROL_KIND,
                "synthetic": False,
                "protocol_id": PROTOCOL_ID,
                "control_kind": control_kind,
                "model_matrix_member": False,
                "actor": actor,
                "instance_id": instance_id,
                "seed": seed,
                "waterbody": truth["waterbody"],
                "domain_id": truth["domain_id"],
                "difficulty_tier": truth.get("difficulty_tier"),
                "event_cluster_id": truth.get("event_cluster_id"),
                "background_id": truth.get("background_id"),
                "detectable_source": bool(truth.get("detectable_source")),
                "source_geometry_type": truth.get("source_geometry_type"),
                "dstar": task_dstar(store, truth),
                "evidence_origin": f"v20_public_baselines_final_v4/{filename}",
                "evidence_sha256": observed_sha,
                "evidence_manifest": "v20_public_baselines_final_v4/artifact_manifest.json",
                "evidence_manifest_sha256": manifest_sha,
                **_typed_score_row(raw),
            }
            output.append({field: row.get(field) for field in CONTROL_FIELDS})
        expected_keys = {(instance_id, seed) for instance_id in task_ids for seed in expected_seeds}
        if keys != expected_keys:
            raise ValueError(
                f"registered control matrix incomplete for {filename}: "
                f"observed {len(keys)}, expected {len(expected_keys)}"
            )
    output.sort(key=lambda row: (str(row["control_kind"]), str(row["instance_id"]), int(row["seed"])))
    return output


@dataclass
class PaperDataBundle:
    metadata: dict[str, Any]
    episode_outcomes: list[dict[str, Any]]
    episode_steps: list[dict[str, Any]]
    task_geometry: list[dict[str, Any]]
    decision_records: list[dict[str, Any]]
    control_rows: list[dict[str, Any]]

    def validate(self) -> None:
        kind = self.metadata.get("dataset_kind")
        if kind not in {SYNTHETIC_KIND, FORMAL_KIND}:
            raise ValueError(f"unsupported paper dataset kind: {kind!r}")
        synthetic = kind == SYNTHETIC_KIND
        if bool(self.metadata.get("synthetic")) != synthetic:
            raise ValueError("bundle synthetic flag disagrees with dataset_kind")
        keys = set()
        for row in self.episode_outcomes:
            if row.get("dataset_kind") != kind or bool(row.get("synthetic")) != synthetic:
                raise ValueError("outcome row provenance disagrees with bundle")
            key = row.get("actor"), row.get("instance_id"), int(row.get("seed"))
            if key in keys:
                raise ValueError(f"duplicate paper outcome cell: {key}")
            keys.add(key)
            if synthetic and row.get("replay_verified") is not None:
                raise ValueError("synthetic outcomes must not claim replay verification")
            if synthetic and row.get("science_eligible") is not False:
                raise ValueError("synthetic outcomes must be science-ineligible")
            if synthetic and row.get("operational_metric_origin") != "synthetic_layout_only":
                raise ValueError("synthetic operational fields require a layout-only origin")
            if not synthetic and row.get("replay_verified") is not True:
                raise ValueError("formal outcomes require exact replay verification")
            if not synthetic and row.get("science_eligible") is not True:
                raise ValueError("formal outcomes require science eligibility")
        if int(self.metadata.get("episodes", -1)) != len(self.episode_outcomes):
            raise ValueError("metadata episode count does not match outcomes")
        control_keys = set()
        for row in self.control_rows:
            if row.get("dataset_kind") != REGISTERED_CONTROL_KIND:
                raise ValueError("control row has the wrong evidence kind")
            if _bool(row.get("synthetic")) is not False:
                raise ValueError("registered controls must be explicitly non-synthetic")
            if _bool(row.get("model_matrix_member")) is not False:
                raise ValueError("registered controls cannot enter the model matrix")
            key = row.get("actor"), row.get("instance_id"), int(row.get("seed"))
            if key in control_keys:
                raise ValueError(f"duplicate registered control cell: {key}")
            control_keys.add(key)
        if int(self.metadata.get("control_rows", -1)) != len(self.control_rows):
            raise ValueError("metadata control count does not match control_rows")


def write_bundle(bundle: PaperDataBundle, output_dir: str | Path) -> dict[str, Any]:
    """Write a deterministic five-table bundle and its explicit provenance."""

    bundle.validate()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "episode_outcomes.csv": (OUTCOME_FIELDS, bundle.episode_outcomes),
        "episode_steps.csv": (STEP_FIELDS, bundle.episode_steps),
        "task_geometry.csv": (GEOMETRY_FIELDS, bundle.task_geometry),
        "decision_records.csv": (DECISION_FIELDS, bundle.decision_records),
        "control_rows.csv": (CONTROL_FIELDS, bundle.control_rows),
    }
    for name, (fields, rows) in paths.items():
        write_csv(target / name, list(fields), rows)
    metadata = dict(bundle.metadata)
    metadata["schema_version"] = PAPER_SCHEMA_VERSION
    metadata["files"] = {
        name: {"rows": len(rows), "sha256": sha256_file(target / name)}
        for name, (_, rows) in paths.items()
    }
    metadata["bundle_sha256"] = stable_hash({
        "metadata": {key: value for key, value in metadata.items() if key != "files"},
        "files": metadata["files"],
    })
    (target / "bundle_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if metadata.get("dataset_kind") == SYNTHETIC_KIND:
        (target / "SYNTHETIC_DO_NOT_CITE.md").write_text(
            "# SIMULATED DATA — LAYOUT REHEARSAL\n\n"
            "These rows are deterministic, capability-informed synthetic scenarios. "
            "They are not measured model performance, provider logs, or replay evidence.\n",
            encoding="utf-8",
        )
    return metadata


def _is_synthetic_mapping(value: Mapping[str, Any]) -> bool:
    if _bool(value.get("synthetic")) is True:
        return True
    if _bool(value.get("rehearsal")) is True:
        return True
    if value.get("watermark") == SYNTHETIC_WATERMARK:
        return True
    kind = str(value.get("dataset_kind") or value.get("data_contract_version") or "").lower()
    return any(marker in kind for marker in ("synthetic", "rehearsal", "simulated", "mock", "demo", "placeholder"))


def _episode_key(value: Mapping[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(value.get("run_id")), str(value.get("actor")), str(value.get("instance_id")),
        int(value.get("seed", 0)), int(value.get("attempt_index", 0)),
    )


def _score_value_matches(raw: Any, expected: Any) -> bool:
    if expected is None:
        return raw in (None, "", "None", "nan", "NaN")
    if isinstance(expected, bool):
        return _bool(raw) is expected
    if isinstance(expected, (int, float)):
        value = _float(raw)
        return value is not None and math.isclose(value, float(expected), rel_tol=1e-12, abs_tol=1e-12)
    return str(raw) == str(expected)


def _load_validations(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    output = []
    for raw in paths:
        path = Path(raw)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or _is_synthetic_mapping(value):
            raise ValueError(f"formal adapter rejected synthetic/invalid validation: {path}")
        if value.get("passed") is not True:
            raise ValueError(f"formal validation did not pass: {path}")
        if (value.get("attempt_journal") or {}).get("passed") is not True:
            raise ValueError(f"formal attempt journal did not pass: {path}")
        if value.get("replay_failures"):
            raise ValueError(f"formal validation contains replay failures: {path}")
        output.append(value)
    return output


def compile_formal_bundle(
    *,
    store: DataStore,
    score_paths: Sequence[str | Path],
    episode_paths: Sequence[str | Path],
    replay_paths: Sequence[str | Path],
    validation_paths: Sequence[str | Path],
    expected_tasks: int = 84,
    expected_seeds: int = 3,
    expected_models: int | None = None,
    control_root: str | Path = DEFAULT_CONTROL_ROOT,
) -> PaperDataBundle:
    """Normalize only complete, replay-verified v20 evidence.

    This is intentionally strict.  It is a post-score publication adapter, not
    a recovery path: synthetic markers, incomplete matrices, missing replay
    rows, failed journals, and score mismatches all raise before any bundle is
    returned.
    """

    if not store.evaluator:
        raise PermissionError("formal paper adapter requires evaluator truth")
    if not score_paths or not episode_paths or not replay_paths or not validation_paths:
        raise ValueError("formal adapter requires scores, episodes, replay, and validation")
    for path in [*score_paths, *episode_paths, *replay_paths, *validation_paths]:
        if any(part.lower() == "synthetic" for part in Path(path).parts):
            raise ValueError(f"formal adapter rejected synthetic path: {path}")
    validations = _load_validations(validation_paths)

    score_rows = [row for path in score_paths for row in read_csv(path)]
    episode_rows: list[dict[str, Any]] = []
    episode_source_hash: dict[str, str] = {}
    for path in episode_paths:
        source_hash = sha256_file(path)
        for raw in jsonl(path):
            row = dict(raw)
            episode_id = str(row.get("episode_id"))
            if episode_id in episode_source_hash:
                raise ValueError(f"duplicate episode id across formal inputs: {episode_id}")
            episode_source_hash[episode_id] = source_hash
            episode_rows.append(row)
    replay_rows = [row for path in replay_paths for row in read_csv(path)]
    if not score_rows or not episode_rows or not replay_rows:
        raise ValueError("formal adapter received an empty evidence table")
    if any(_is_synthetic_mapping(row) for row in [*score_rows, *episode_rows, *replay_rows]):
        raise ValueError("formal adapter rejected synthetic evidence")

    episodes: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    for episode in episode_rows:
        if episode.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("formal adapter encountered a mixed/non-v20 protocol")
        if episode.get("valid_for_scoring") is not True:
            continue
        if episode.get("science_eligible") is not True:
            continue
        key = _episode_key(episode)
        if key in episodes:
            raise ValueError(f"duplicate formal scoreable episode: {key}")
        episodes[key] = episode
    replay_by_episode: dict[str, dict[str, Any]] = {}
    required_replay_checks = (
        "replay_verified", "event_match", "stored_event_hash_match",
        "stored_sample_ledger_hash_match", "sample_ledger_content_match",
        "sample_ledger_match", "answer_submit_event_match",
        "termination_turn_budget_summary_match", "provider_audit_hash_match",
        "decision_audit_hash_match", "provider_guard_match",
        "model_config_and_provider_input_match", "raw_provider_call_execution_match",
        "provider_decision_links_match",
    )
    for row in replay_rows:
        episode_id = str(row.get("episode_id"))
        if episode_id in replay_by_episode:
            raise ValueError(f"duplicate replay verdict: {episode_id}")
        missing_or_failed = [
            field for field in required_replay_checks if _bool(row.get(field)) is not True
        ]
        if missing_or_failed:
            raise ValueError(
                f"formal replay row is not exact for {episode_id}: {missing_or_failed}"
            )
        replay_by_episode[episode_id] = row

    expected_task_ids = {str(row["instance_id"]) for row in store.tasks()}
    if len(expected_task_ids) != expected_tasks:
        raise ValueError(
            f"registered task count is {len(expected_task_ids)}, expected {expected_tasks}"
        )
    normalized: list[dict[str, Any]] = []
    all_steps: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    seen_score_keys = set()
    for raw_score in score_rows:
        key = _episode_key(raw_score)
        if key in seen_score_keys:
            raise ValueError(f"duplicate formal score row: {key}")
        seen_score_keys.add(key)
        episode = episodes.get(key)
        if episode is None:
            raise ValueError(f"formal score row has no sealed episode: {key}")
        episode_id = str(episode.get("episode_id"))
        replay = replay_by_episode.get(episode_id)
        if replay is None:
            raise ValueError(f"formal episode lacks a passing replay verdict: {episode_id}")
        for field in ("run_id", "actor", "instance_id", "seed", "attempt_index"):
            if str(replay.get(field)) != str(episode.get(field)):
                raise ValueError(f"formal replay identity mismatch for {episode_id}: {field}")
        if replay.get("event_chain_sha256") != episode.get("event_chain_sha256"):
            raise ValueError(f"formal replay is not bound to the episode chain: {episode_id}")
        truth = store.truth(str(episode["instance_id"]))
        recomputed = score_submission(
            episode.get("answer") or {}, truth,
            store.protocol["tolerances"][truth["waterbody"]],
        )
        mismatches = [
            field for field, expected in recomputed.items()
            if field not in raw_score or not _score_value_matches(raw_score.get(field), expected)
        ]
        if mismatches:
            raise ValueError(f"formal score mismatch for {episode_id}: {mismatches}")
        normalized.append(outcome_from_episode(
            store, episode, recomputed, dataset_kind=FORMAL_KIND,
            replay_verified=True,
            source_episode_sha256=episode_source_hash.get(episode_id),
        ))
        all_steps.extend(step_rows_from_episode(episode, dataset_kind=FORMAL_KIND))
        all_decisions.extend(decision_rows_from_episode(episode, dataset_kind=FORMAL_KIND))

    actors = sorted({str(row["actor"]) for row in normalized})
    validation_actors = {
        str(value) for row in validations
        for value in (row.get("actor"), row.get("expected_model")) if value
    }
    if not set(actors) <= validation_actors:
        raise ValueError(
            f"formal actors lack actor-bound validation: {sorted(set(actors) - validation_actors)}"
        )
    if expected_models is not None and len(actors) != expected_models:
        raise ValueError(f"formal model count is {len(actors)}, expected {expected_models}")
    expected_seeds_set = set(range(expected_seeds))
    for actor in actors:
        subset = [row for row in normalized if row["actor"] == actor]
        cells = {(str(row["instance_id"]), int(row["seed"])) for row in subset}
        task_ids = {instance_id for instance_id, _ in cells}
        seeds = {seed for _, seed in cells}
        if task_ids != expected_task_ids or seeds != expected_seeds_set:
            raise ValueError(
                f"formal matrix incomplete for {actor}: tasks={len(task_ids)}, seeds={sorted(seeds)}"
            )
        if len(cells) != expected_tasks * expected_seeds:
            raise ValueError(f"formal matrix has duplicate/missing cells for {actor}")
    if len(normalized) != len(actors) * expected_tasks * expected_seeds:
        raise ValueError("formal score table contains unexpected cells")

    behavior_tertiles = attach_model_behavior_tertiles(normalized)
    storyboard = attach_storyboard_selection(
        normalized,
        all_steps,
        selection_seed="formal-v20-preregistered-storyboard-v1",
        require_contrast=bool(expected_models is not None and expected_models >= 2),
    )
    normalized.sort(key=lambda row: (str(row["actor"]), str(row["instance_id"]), int(row["seed"])))
    all_steps.sort(key=lambda row: (str(row["actor"]), str(row["instance_id"]), int(row["seed"]), int(row["step_index"])))
    all_decisions.sort(key=lambda row: (str(row["actor"]), str(row["instance_id"]), int(row["seed"]), int(row["decision_record_index"])))
    geometry = build_task_geometry(
        store, dataset_kind=FORMAL_KIND, instance_ids=expected_task_ids,
    )
    representative_task = next(
        (str(row["instance_id"]) for row in geometry if row.get("representative") is True),
        None,
    )
    metadata = {
        "schema_version": PAPER_SCHEMA_VERSION,
        "dataset_kind": FORMAL_KIND,
        "synthetic": False,
        "protocol_id": PROTOCOL_ID,
        "episodes": len(normalized),
        "models": len(actors),
        "tasks": expected_tasks,
        "seeds": expected_seeds,
        "control_rows": 0,
        "replay_verified": len(normalized),
        "validation_files": [
            {"path": str(Path(path)), "sha256": sha256_file(path)} for path in validation_paths
        ],
        "validation_expected_cells": sum(int(row.get("expected_cells", 0)) for row in validations),
        "behavior_tertiles": behavior_tertiles,
        "storyboard": storyboard,
        "representative_task": {
            "instance_id": representative_task,
            "selection_rule": (
                "outcome-blind minimum stable hash among registered detectable tasks "
                "with nonempty source and peak geometry"
            ),
        },
    }
    controls = load_registered_controls(store, control_root)
    metadata["control_rows"] = len(controls)
    metadata["registered_controls"] = sorted({str(row["control_kind"]) for row in controls})
    bundle = PaperDataBundle(
        metadata=metadata,
        episode_outcomes=normalized,
        episode_steps=all_steps,
        task_geometry=geometry,
        decision_records=all_decisions,
        control_rows=controls,
    )
    bundle.validate()
    return bundle


__all__ = [
    "BEHAVIOR_TERTILE_METRICS", "CONTROL_FIELDS", "DECISION_FIELDS",
    "DEFAULT_CONTROL_ROOT", "FORMAL_KIND",
    "GEOMETRY_FIELDS", "OUTCOME_FIELDS", "REGISTERED_CONTROL_KIND",
    "PAPER_SCHEMA_VERSION", "PROTOCOL_ID", "PaperDataBundle", "SCORE_FIELDS",
    "STEP_FIELDS", "SYNTHETIC_KIND", "SYNTHETIC_WATERMARK",
    "attach_model_behavior_tertiles", "attach_storyboard_selection", "build_task_geometry",
    "compile_formal_bundle", "decision_rows_from_episode", "load_registered_controls",
    "outcome_from_episode",
    "step_rows_from_episode", "task_dstar", "trajectory_diagnostics", "write_bundle",
]
