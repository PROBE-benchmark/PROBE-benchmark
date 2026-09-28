from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from laketrace.contracts.v16_1_runner import DATA_CONTRACT_VERSION, PROMPT_ID
from laketrace.data.store import DataStore
from laketrace.env.core import Environment
from laketrace.scoring.scorer import score_submission
from laketrace.util import jsonl, load_config, sha256_file, stable_hash, write_csv


MODEL_FIELDS = [
    "model_id", "provider", "model_snapshot", "api_date_utc", "prompt_id",
    "representation", "primary_condition", "temperature", "top_p",
    "max_output_tokens", "seed", "vision_supported", "n_registered_repeats",
    "notes", "actor_id", "model_config_hash",
]
TASK_FIELDS = [
    "task_id", "waterbody_id", "connected_domain_id", "event_cluster_id",
    "background_id", "split", "anomaly_present", "source_class",
    "environment_condition", "source_x_m", "source_y_m", "source_z_m",
    "peak_x_m", "peak_y_m", "peak_z_m", "peak_to_source_h_m",
    "peak_to_source_v_m", "source_tol_h_m", "source_tol_v_m", "peak_tol_h_m",
    "peak_tol_v_m", "peak_source_disp_norm", "domain_node_count", "grid_dx_m",
    "grid_dy_m", "grid_dz_m", "stratified", "recirculation", "multi_peak",
    "complex_hydro", "ground_truth_closable", "solvability_pass",
    "exclusion_reason", "task_version_hash", "source_truth_kind",
    "source_node_count", "diagnostic_pair_rule", "primary_tracer",
    "difficulty_tier", "source_plume_class", "phase",
]
EPISODE_FIELDS = [
    "episode_id", "model_id", "model_snapshot", "prompt_id", "representation",
    "task_id", "run_id", "run_seed", "is_primary_condition", "started",
    "analysis_eligible", "infrastructure_failure", "pred_anomaly", "pred_l2_x_m",
    "pred_l2_y_m", "pred_l2_z_m", "pred_source_class", "pred_l4_x_m",
    "pred_l4_y_m", "pred_l4_z_m", "pred_closable", "confidence_l1",
    "confidence_l2", "confidence_l3", "confidence_l4", "confidence_l5",
    "l1_pass", "l2_pass", "l3_pass", "l4_pass", "l5_pass", "l5_category",
    "l2_error_h_m", "l2_error_v_m", "source_error_h_m", "source_error_v_m",
    "valid_submission", "terminal_state", "endurance_used_s", "endurance_budget_s",
    "distance_used_m", "distance_budget_m", "turns_used", "turns_budget", "n_moves",
    "n_analyse", "n_invalid_actions", "n_action_steps", "n_repeated_action_steps",
    "nodes_visited_unique", "min_dist_to_source_h_m", "min_dist_to_source_v_m",
    "min_dist_to_peak_h_m", "context_tokens_max", "context_tokens_mean",
    "record_nodes_final", "api_cost_usd", "wall_time_s", "raw_output_path",
    "log_hash", "input_tokens", "output_tokens", "termination_reason_raw",
    "valid_for_scoring", "model_identity_match", "event_cluster_id", "background_id",
    "difficulty_tier", "source_plume_class", "min_dist_to_peak_v_m",
    "chain_of_thought_requested", "chain_of_thought_stored",
]
STEP_FIELDS = [
    "episode_id", "step_index", "sim_time_s", "node_id", "x_m", "y_m", "z_m",
    "action_type", "action_payload_json", "action_valid", "invalid_reason",
    "observation_json", "observation_hash", "state_hash",
    "budget_endurance_remaining_s", "budget_distance_remaining_m",
    "budget_turns_remaining", "visited_nodes_count", "state_record_count",
    "context_tokens_to_model", "hypothesis_source_x_m", "hypothesis_source_y_m",
    "hypothesis_source_z_m", "hypothesis_source_class", "hypothesis_confidence",
    "hypothesis_basis_observation_hash",
    "is_repeated_action", "termination_after_step", "provider_round_index",
    "tool_call_index", "messages_hash", "tool_schema_hash",
    "evidence_content_hash", "visited_mask_hash", "measurement_history_count",
]
ROUND_FIELDS = [
    "episode_id", "provider_round_index", "observation_json", "observation_hash",
    "messages_hash", "tool_schema_hash", "state_record_count", "input_tokens",
    "output_tokens", "reported_model", "assistant_visible_content_sha256",
    "assistant_visible_content_char_count",
    "provider_error", "tool_call_count",
    "evidence_content_hash", "visited_mask_hash", "measurement_history_count",
]


def _write_parquet(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError(
            "v16.1 Parquet export requires pandas and pyarrow; install "
            "requirements-v16.1.txt in an isolated environment"
        ) from exc
    try:
        frame = pd.DataFrame(rows, columns=columns)
        frame.to_parquet(path, index=False, engine="pyarrow")
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError(
            "v16.1 Parquet export requires pyarrow; install requirements-v16.1.txt"
        ) from exc


def _positive_spacing(values: Iterable[float]) -> float | None:
    unique = sorted(set(map(float, values)))
    differences = [right - left for left, right in zip(unique, unique[1:]) if right - left > 1e-9]
    return min(differences) if differences else None


def _mechanism(value: Any) -> Any:
    return "external_nonpoint" if value == "external_diffuse" else value


def _representation(value: str | None) -> str:
    return {
        "records": "flat_list",
        "sparse_map": "text_map",
    }.get(str(value or "records"), str(value or "records"))


def _read_logs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    output = []
    for raw_path in paths:
        path = raw_path.resolve()
        files = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
        for item in files:
            for line_number, row in enumerate(jsonl(item), 1):
                value = dict(row)
                value["_raw_output_path"] = str(item.resolve())
                value["_raw_line_number"] = line_number
                output.append(value)
    return output


def _load_csv_by_id(path: Path, key: str = "instance_id") -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def _historical_flags(source_root: Path | None) -> dict[str, dict[str, Any]]:
    flags: dict[str, dict[str, Any]] = defaultdict(dict)
    if source_root is None:
        return flags
    validation = source_root / "validation"
    thermoclines = _load_csv_by_id(validation / "role_thermoclines.csv")
    reversal_rows = []
    path = validation / "role_reversal_series.csv"
    if path.is_file():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reversal_rows = list(csv.DictReader(handle))
    topology = _load_csv_by_id(validation / "role_topology_checks.csv")
    extrema = _load_csv_by_id(validation / "local_extrema_profile_v14.csv")
    for instance_id in thermoclines:
        flags[instance_id]["stratified"] = True
    for row in reversal_rows:
        flags[row["instance_id"]]["recirculation"] = True
    for instance_id, row in topology.items():
        if row.get("different_topology_zones") == "True":
            flags[instance_id]["complex_hydro"] = True
    for instance_id, row in extrema.items():
        try:
            flags[instance_id]["multi_peak"] = int(row["maxima_count"]) > 1
        except (KeyError, TypeError, ValueError):
            pass
    return flags


def _diagnostic_pair(
    truth: dict[str, Any], tolerance: dict[str, float],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, float | None, float | None, float | None]:
    pairs = []
    for peak in truth.get("L2_peak_positions") or []:
        for source in truth.get("source_nodes") or []:
            horizontal = math.dist(
                (float(peak["x"]), float(peak["y"])),
                (float(source["x"]), float(source["y"])),
            )
            vertical = abs(float(peak["z"]) - float(source["z"]))
            dstar = math.hypot(
                horizontal / float(tolerance["horizontal_m"]),
                vertical / float(tolerance["vertical_m"]),
            )
            pairs.append((dstar, horizontal, vertical, peak, source))
    if not pairs:
        return None, None, None, None, None
    dstar, horizontal, vertical, peak, source = min(
        pairs,
        key=lambda item: (item[0], item[1], item[2], stable_hash([item[3], item[4]])),
    )
    return peak, source, horizontal, vertical, dstar


def build_task_artifacts(
    store: DataStore, source_root: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    flags = _historical_flags(source_root)
    registry: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    field_grid: list[dict[str, Any]] = []
    for task in store.tasks():
        instance_id = task["instance_id"]
        truth = store.truth(instance_id)
        frame = store.frame(instance_id, truth["phase"])
        tolerance = store.protocol["tolerances"][truth["waterbody"]]
        peak, source, horizontal, vertical, dstar = _diagnostic_pair(truth, tolerance)
        x_spacing = _positive_spacing(node.x for node in frame.nodes)
        y_spacing = _positive_spacing(node.y for node in frame.nodes)
        z_spacing = _positive_spacing(node.z for node in frame.nodes)
        local_flags = flags.get(instance_id, {})
        hydro = truth.get("hydrodynamic_context")
        task_version_hash = stable_hash({
            "task": task,
            "truth": truth,
            "protocol": store.protocol,
        })
        registry.append({
            "task_id": instance_id,
            "waterbody_id": truth["waterbody"],
            "connected_domain_id": truth["domain_id"],
            "event_cluster_id": truth["event_cluster_id"],
            "background_id": truth["background_id"],
            "split": "test",
            "anomaly_present": int(bool(truth["detectable_source"])),
            "source_class": _mechanism(truth.get("L3_mechanism")) or "none",
            "environment_condition": hydro,
            "source_x_m": source.get("x") if source else None,
            "source_y_m": source.get("y") if source else None,
            "source_z_m": source.get("z") if source else None,
            "peak_x_m": peak.get("x") if peak else None,
            "peak_y_m": peak.get("y") if peak else None,
            "peak_z_m": peak.get("z") if peak else None,
            "peak_to_source_h_m": horizontal,
            "peak_to_source_v_m": vertical,
            "source_tol_h_m": tolerance["horizontal_m"],
            "source_tol_v_m": tolerance["vertical_m"],
            "peak_tol_h_m": tolerance["horizontal_m"],
            "peak_tol_v_m": tolerance["vertical_m"],
            "peak_source_disp_norm": dstar,
            "domain_node_count": len(frame.nodes),
            "grid_dx_m": x_spacing,
            "grid_dy_m": y_spacing,
            "grid_dz_m": z_spacing,
            "stratified": int(bool(local_flags.get("stratified", False))),
            "recirculation": int(bool(local_flags.get("recirculation", hydro == "reversing"))),
            "multi_peak": int(bool(local_flags.get("multi_peak", len(truth.get("L2_peak_positions") or []) > 1))),
            "complex_hydro": int(bool(local_flags.get("complex_hydro", hydro not in {None, "ordinary"}))),
            # v4 has an answer-dependent L5 and therefore no task-level
            # closability truth.  It is deliberately left unresolved instead
            # of silently changing the scoring construct during data upgrade.
            "ground_truth_closable": None,
            "solvability_pass": 1,
            "exclusion_reason": None,
            "task_version_hash": task_version_hash,
            "source_truth_kind": "source_nodes",
            "source_node_count": len(truth.get("source_nodes") or []),
            "diagnostic_pair_rule": "minimum_normalized_peak_node_to_source_node; never_used_for_L4_scoring",
            "primary_tracer": truth.get("L2_primary_tracer"),
            "difficulty_tier": truth.get("difficulty_tier"),
            "source_plume_class": truth.get("source_plume_class"),
            "phase": truth["phase"],
        })
        plume_ids = set(map(int, truth.get("L2_plume_node_ids") or []))
        peak_ids = set()
        for position in truth.get("L2_peak_positions") or []:
            node = frame.by_xyz.get((float(position["x"]), float(position["y"]), float(position["z"])))
            if node is not None:
                peak_ids.add(node.index)
        source_ids = set(map(int, truth.get("source_node_ids") or []))
        source_neighbourhood = set()
        peak_neighbourhood = set()
        for node in frame.nodes:
            if any(
                math.dist((node.x, node.y), (float(item["x"]), float(item["y"]))) <= float(tolerance["horizontal_m"])
                and abs(node.z - float(item["z"])) <= float(tolerance["vertical_m"])
                for item in truth.get("source_nodes") or []
            ):
                source_neighbourhood.add(node.index)
            if any(
                math.dist((node.x, node.y), (float(item["x"]), float(item["y"]))) <= float(tolerance["horizontal_m"])
                and abs(node.z - float(item["z"])) <= float(tolerance["vertical_m"])
                for item in truth.get("L2_peak_positions") or []
            ):
                peak_neighbourhood.add(node.index)
        region_sets = {
            "contaminated_region": plume_ids,
            "evidence_region": plume_ids,
            "peak_node": peak_ids,
            "peak_neighbourhood": peak_neighbourhood,
            "source_node": source_ids,
            "source_neighbourhood": source_neighbourhood,
        }
        for region_type, node_ids in region_sets.items():
            for node_id in sorted(node_ids):
                regions.append({
                    "task_id": instance_id,
                    "region_type": region_type,
                    "node_id": node_id,
                    "region_rule_version": (
                        "v4_L2_registered_F1_footprint" if region_type in {"contaminated_region", "evidence_region"}
                        else "v4_nearest_source_nodes_and_registered_tolerances"
                    ),
                })
        tracer = truth.get("L2_primary_tracer")
        for node in frame.nodes:
            row = {
                "task_id": instance_id,
                "node_id": node.index,
                "x_m": node.x,
                "y_m": node.y,
                "z_m": node.z,
                "concentration": node.channels.get(tracer) if tracer else None,
                "flow_u_m_s": node.flow["u_m_s"],
                "flow_v_m_s": node.flow["v_m_s"],
                "flow_w_m_s": node.flow["w_m_s"],
                "valid_cell": True,
                "primary_tracer": tracer,
                "phase": truth["phase"],
            }
            row.update({f"channel_{key}": value for key, value in node.channels.items()})
            field_grid.append(row)
    return registry, regions, field_grid


def build_task_audit(store: DataStore, source_root: Path | None) -> list[dict[str, Any]]:
    repair = {}
    distinctness: dict[str, list[dict[str, str]]] = defaultdict(list)
    if source_root is not None:
        repair = _load_csv_by_id(source_root / "validation" / "candidate_repair_log.csv")
        path = source_root / "validation" / "candidate_distinctness.csv"
        if path.is_file():
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    distinctness[row["instance_id"]].append(row)
    rows = []
    for task in store.tasks():
        truth = store.truth(task["instance_id"])
        base = {
            "candidate_event_id": truth["event_cluster_id"],
            "task_id": task["instance_id"],
            "source_class": _mechanism(truth.get("L3_mechanism")) or "none",
            "environment_condition": truth.get("hydrodynamic_context"),
            "event_cluster_id": truth["event_cluster_id"],
            "background_id": truth["background_id"],
            "task_version_hash": stable_hash({"task": task, "truth": truth}),
            "audit_scope": "post_registration_release_pipeline",
        }
        stages = [
            ("field_package_registration", 1, True, None, "data/laketrace_v2/tasks/instances.jsonl"),
            (
                "candidate_set_repair", 2,
                not repair or repair.get(task["instance_id"], {}).get("final_k", "8") == "8",
                None,
                "validation/candidate_repair_log.csv" if repair else "not_available",
            ),
            (
                "candidate_pair_distinctness", 3,
                not distinctness or all(row.get("equivalent") != "True" for row in distinctness.get(task["instance_id"], [])),
                None,
                "validation/candidate_distinctness.csv" if distinctness else "not_available",
            ),
            ("v4_solvability_gate", 4, True, None, "results/reference/design_validation.json"),
        ]
        for stage, order, retained, reason, artifact in stages:
            rows.append(base | {
                "stage": stage,
                "stage_order": order,
                "retained": int(retained),
                "exclusion_reason": reason if not retained else None,
                "source_artifact": artifact,
            })
    return rows


def build_interface_equivalence(store: DataStore) -> list[dict[str, Any]]:
    from laketrace.contracts.v16_1_runner import InstrumentedEnvironment

    by_domain = {}
    for task in store.tasks():
        by_domain.setdefault(task["domain_id"], task["instance_id"])
    output = []
    for index, (domain, instance_id) in enumerate(sorted(by_domain.items()), 1):
        legacy = Environment(store, instance_id, 20260828)
        instrumented = InstrumentedEnvironment(store, instance_id, 20260828)
        observations_a = [legacy.observation()]
        observations_b = [instrumented.observation()]
        budgets_a = [legacy._budget_snapshot()]
        budgets_b = [instrumented._budget_snapshot()]
        directions = [name for name, edge in legacy.graph.neighbors(legacy.current).items() if edge is not None]
        actions: list[tuple[str, dict[str, Any]]] = []
        if directions:
            actions.append(("move", {"direction": directions[0], "steps": 1}))
        actions.extend([
            ("analyze", {}),
            ("submit", {"answer": {"anomaly": "no_anomaly", "decision": "cannot_determine"}}),
        ])
        mismatch = None
        for step, (name, arguments) in enumerate(actions, 1):
            if name == "move":
                legacy.move(**arguments)
                instrumented.move(**arguments)
            elif name == "analyze":
                legacy.analyze()
                instrumented.analyze()
            else:
                legacy.submit(arguments["answer"])
                instrumented.submit(arguments["answer"])
            observations_a.append(legacy.observation())
            observations_b.append(instrumented.observation())
            budgets_a.append(legacy._budget_snapshot())
            budgets_b.append(instrumented._budget_snapshot())
            if mismatch is None and (
                stable_hash(observations_a[-1]) != stable_hash(observations_b[-1])
                or stable_hash(budgets_a[-1]) != stable_hash(budgets_b[-1])
                or legacy.termination_reason != instrumented.termination_reason
            ):
                mismatch = step
        observation_equal = stable_hash(observations_a) == stable_hash(observations_b)
        budget_equal = stable_hash(budgets_a) == stable_hash(budgets_b)
        termination_equal = legacy.termination_reason == instrumented.termination_reason
        output.append({
            "test_id": f"equivalence-{index:02d}",
            "task_id": instance_id,
            "action_sequence_id": f"{domain}-move-analyse-submit",
            "runner_a": "frozen_v4_environment",
            "runner_b": "v16.1_instrumented_environment",
            "n_steps": len(actions),
            "observation_hash_a": stable_hash(observations_a),
            "observation_hash_b": stable_hash(observations_b),
            "budget_hash_a": stable_hash(budgets_a),
            "budget_hash_b": stable_hash(budgets_b),
            "termination_a": legacy.termination_reason,
            "termination_b": instrumented.termination_reason,
            "observation_equal": int(observation_equal),
            "budget_equal": int(budget_equal),
            "termination_equal": int(termination_equal),
            "mismatch_step": mismatch,
            "passed": int(observation_equal and budget_equal and termination_equal),
        })
    return output


def _trajectory(episode: dict[str, Any], store: DataStore) -> list[tuple[float, float, float]]:
    task = store.task(episode["instance_id"])
    access = store.access_points["access_points"][task["domain_id"]]
    return [
        (float(access["x"]), float(access["y"]), float(access["z"])),
        *[
            (float(reading["x"]), float(reading["y"]), float(reading["z"]))
            for event in episode.get("events") or []
            for reading in event.get("readings") or []
        ],
    ]


def _minimum_errors(
    trajectory: list[tuple[float, float, float]], targets: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    pairs = [
        (
            math.dist((position[0], position[1]), (float(target["x"]), float(target["y"]))),
            abs(position[2] - float(target["z"])),
        )
        for position in trajectory
        for target in targets
    ]
    return min(pairs, key=lambda value: (math.hypot(*value), value)) if pairs else (None, None)


def _terminal(value: str | None) -> str:
    return {
        "turn_limit": "step_limit",
        "engineering_round_cap": "step_limit",
        "consecutive_invalid": "invalid_action",
        "infra_failure": "infrastructure",
        "endurance_exhausted": "step_limit",
        "range_exhausted": "step_limit",
    }.get(str(value), str(value or "invalid_action"))


def build_episode_artifacts(
    store: DataStore,
    episodes: list[dict[str, Any]],
    actor_registry: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    results = []
    steps = []
    rounds = []
    for position, episode in enumerate(episodes):
        actor = str(episode.get("actor") or "unknown")
        model = actor_registry.get(actor, {})
        episode_id = str(episode.get("episode_id") or f"{episode.get('run_id')}::{actor}::{episode.get('instance_id')}::{episode.get('seed')}::row{position}")
        truth = store.truth(episode["instance_id"])
        tolerance = store.protocol["tolerances"][truth["waterbody"]]
        valid = episode.get("valid_for_scoring") is not False and episode.get("termination_reason") != "infra_failure"
        answer = episode.get("answer") or {}
        score = score_submission(answer, truth, tolerance) if valid else {}
        peak = answer.get("peak_position") or {}
        source = answer.get("source_position") or {}
        confidence = answer.get("confidence") or {}
        trajectory = _trajectory(episode, store)
        min_source_h, min_source_v = _minimum_errors(trajectory, truth.get("source_nodes") or [])
        min_peak_h, min_peak_v = _minimum_errors(trajectory, truth.get("L2_peak_positions") or [])
        events = episode.get("events") or []
        calls = episode.get("provider_calls") or []
        context = [int((call.get("usage") or {}).get("input_tokens", 0)) for call in calls]
        raw_path = episode.get("_raw_output_path")
        row_hash = stable_hash({key: value for key, value in episode.items() if not key.startswith("_")})
        results.append({
            "episode_id": episode_id,
            "model_id": model.get("model_id", actor),
            "model_snapshot": model.get("model_snapshot") or next(iter(episode.get("reported_models") or []), actor),
            "prompt_id": episode.get("prompt_id") or model.get("prompt_id") or PROMPT_ID,
            "representation": _representation(episode.get("presentation_mode")),
            "task_id": episode["instance_id"],
            "run_id": episode.get("run_id"),
            "run_seed": episode.get("seed"),
            "is_primary_condition": model.get("primary_condition", int(_representation(episode.get("presentation_mode")) == "flat_list")),
            "started": episode.get("started_at_utc"),
            "analysis_eligible": int(valid),
            "infrastructure_failure": int(episode.get("termination_reason") == "infra_failure"),
            "pred_anomaly": answer.get("anomaly"),
            "pred_l2_x_m": peak.get("x"),
            "pred_l2_y_m": peak.get("y"),
            "pred_l2_z_m": peak.get("z"),
            "pred_source_class": _mechanism(answer.get("mechanism")),
            "pred_l4_x_m": source.get("x"),
            "pred_l4_y_m": source.get("y"),
            "pred_l4_z_m": source.get("z"),
            "pred_closable": int(answer.get("decision") == "declare") if answer.get("decision") else None,
            **{f"confidence_l{level}": confidence.get(f"L{level}") for level in range(1, 6)},
            **{f"l{level}_pass": score.get(f"L{level}_pass") for level in range(1, 6)},
            "l5_category": score.get("L5_category"),
            "l2_error_h_m": score.get("L2_horizontal_error_m"),
            "l2_error_v_m": score.get("L2_vertical_error_m"),
            "source_error_h_m": score.get("L4_horizontal_error_m"),
            "source_error_v_m": score.get("L4_vertical_error_m"),
            "valid_submission": int(bool(answer) and episode.get("termination_reason") == "submitted"),
            "terminal_state": _terminal(episode.get("termination_reason")),
            "endurance_used_s": episode.get("endurance_used_s"),
            "endurance_budget_s": task_budget(store, episode["instance_id"], "endurance_s"),
            "distance_used_m": episode.get("range_used_m"),
            "distance_budget_m": task_budget(store, episode["instance_id"], "range_m"),
            "turns_used": episode.get("turns"),
            "turns_budget": store.protocol["max_tool_turns"],
            "n_moves": sum(event.get("action") == "move" for event in events),
            "n_analyse": sum(event.get("action") == "analyze" for event in events),
            "n_invalid_actions": sum(event.get("action") in {"invalid", "infeasible"} for event in events),
            "n_action_steps": len(events),
            "n_repeated_action_steps": sum(bool(event.get("is_repeated_action")) for event in events),
            "nodes_visited_unique": episode.get("visited_nodes"),
            "min_dist_to_source_h_m": min_source_h,
            "min_dist_to_source_v_m": min_source_v,
            "min_dist_to_peak_h_m": min_peak_h,
            "context_tokens_max": max(context, default=0),
            "context_tokens_mean": statistics.mean(context) if context else 0,
            "record_nodes_final": episode.get("visited_nodes"),
            "api_cost_usd": episode.get("api_cost_usd"),
            "wall_time_s": episode.get("wall_time_s"),
            "raw_output_path": raw_path,
            "log_hash": row_hash,
            "input_tokens": (episode.get("token_usage") or {}).get("input_tokens", 0),
            "output_tokens": (episode.get("token_usage") or {}).get("output_tokens", 0),
            "termination_reason_raw": episode.get("termination_reason"),
            "valid_for_scoring": int(valid),
            "model_identity_match": episode.get("model_identity_match"),
            "event_cluster_id": truth.get("event_cluster_id"),
            "background_id": truth.get("background_id"),
            "difficulty_tier": truth.get("difficulty_tier"),
            "source_plume_class": truth.get("source_plume_class"),
            "min_dist_to_peak_v_m": min_peak_v,
            "chain_of_thought_requested": int(bool(episode.get("chain_of_thought_requested", False))),
            "chain_of_thought_stored": int(bool(episode.get("chain_of_thought_stored", False))),
        })
        for event_index, event in enumerate(events):
            steps.append({field: event.get(field) for field in STEP_FIELDS} | {
                "episode_id": episode_id,
                "step_index": event.get("step_index", event_index),
                "action_type": event.get("action_type") or ("analyse" if event.get("action") == "analyze" else event.get("action")),
                "action_payload_json": event.get("action_payload_json") or json.dumps(event.get("arguments") or {}, sort_keys=True),
                "action_valid": event.get("action_valid", bool(event.get("ok"))),
                "invalid_reason": event.get("invalid_reason") or event.get("code"),
                "is_repeated_action": bool(event.get("is_repeated_action", False)),
                "termination_after_step": event.get("termination_after_step") or (episode.get("termination_reason") if event_index == len(events) - 1 else None),
                "hypothesis_source_class": _mechanism(event.get("hypothesis_source_class")),
            })
        for round_index, call in enumerate(calls):
            usage = call.get("usage") or {}
            rounds.append({
                "episode_id": episode_id,
                "provider_round_index": call.get("provider_round_index", round_index),
                "observation_json": call.get("observation_json"),
                "observation_hash": call.get("observation_hash"),
                "messages_hash": call.get("messages_hash"),
                "tool_schema_hash": call.get("tool_schema_hash"),
                "state_record_count": call.get("state_record_count"),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "reported_model": call.get("reported_model"),
                "assistant_visible_content_sha256": call.get("content_sha256"),
                "assistant_visible_content_char_count": call.get("content_char_count", 0),
                "provider_error": call.get("provider_error"),
                "tool_call_count": len(call.get("tool_calls") or []),
                "evidence_content_hash": call.get("evidence_content_hash"),
                "visited_mask_hash": call.get("visited_mask_hash"),
                "measurement_history_count": call.get("measurement_history_count"),
            })
    return results, steps, rounds


def task_budget(store: DataStore, instance_id: str, key: str) -> float:
    budget = store.task(instance_id)["budget"]
    if key == "range_m" and key not in budget:
        return float(budget["endurance_s"]) * float(budget["cruise_ms"])
    return float(budget[key])


def build_model_registry(
    configs: list[dict[str, Any]], episodes: list[dict[str, Any]], registered_repeats: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    by_actor = {str(config["id"]): config for config in configs}
    actors = sorted({str(episode.get("actor") or "unknown") for episode in episodes} | set(by_actor))
    rows = []
    mapping = {}
    for actor in actors:
        config = by_actor.get(actor, {})
        matching = [episode for episode in episodes if str(episode.get("actor") or "unknown") == actor]
        representation = _representation(config.get("presentation_mode") or next((episode.get("presentation_mode") for episode in matching), "records"))
        model_id = str(config.get("registry_model_id") or actor)
        snapshot = str(config.get("model") or next((next(iter(episode.get("reported_models") or []), None) for episode in matching if episode.get("reported_models")), actor))
        row = {
            "model_id": model_id,
            "provider": config.get("provider") or config.get("adapter") or "unregistered",
            "model_snapshot": snapshot,
            "api_date_utc": config.get("api_date_utc"),
            "prompt_id": config.get("prompt_id") or PROMPT_ID,
            "representation": representation,
            "primary_condition": int(config.get("primary_condition", representation == "flat_list")),
            "temperature": config.get("temperature", 1.0),
            "top_p": config.get("top_p"),
            "max_output_tokens": config.get("max_output_tokens"),
            "seed": config.get("seed"),
            "vision_supported": int(bool(config.get("vision_supported", False))),
            "n_registered_repeats": int(config.get("n_registered_repeats", registered_repeats)),
            "notes": config.get("notes") or config.get("purpose"),
            "actor_id": actor,
            "model_config_hash": stable_hash(config) if config else None,
        }
        rows.append(row)
        mapping[actor] = row
    return rows, mapping


def default_thresholds() -> dict[str, Any]:
    return {
        "analysis_version": DATA_CONTRACT_VERSION,
        "primary_filter": {"is_primary_condition": 1},
        "bootstrap": {
            "n_resamples": 10000,
            "seed": 20260828,
            "cluster_1": "event_cluster_id",
            "cluster_2": "background_id",
        },
        "source_tolerance_rule": "horizontal <= source_tol_h_m AND vertical <= source_tol_v_m",
        "source_truth_rule": "minimum distance to canonical source_nodes set; scalar diagnostic pair never scores L4",
        "peak_pair_rule": "minimum normalized peak-node to source-node pair; diagnostic only",
        "distinguishability_rule": "Dstar >= 1",
        "loop_rule": {"window_steps": 6, "min_repeated_identical_actions": 3},
        "confidence_bins": {"method": "equal_frequency", "preferred_bins": 5, "minimum_per_bin": 10},
        "representation_readout": {
            "protocol": "frozen_evidence_terminal_readout_v1",
            "source_trajectory": "budgeted_serpentine",
            "representations": ["flat_list", "text_map"],
            "registered_repeats": 1,
            "pair_key": ["model_id", "task_id", "run_seed"],
            "paired_test": "exact_McNemar",
            "image_map_status": "optional_supplementary_vision_protocol_not_part_of_Fig5c",
        },
        "case_selection_rule": "max distinct failure modes; tie by abs(Dstar - taskset_median); then task_id",
        "diagnostic_thresholds": {
            "budget_use_pct": [33.3, 66.7],
            "submit_rate_pct": [33.3, 66.7],
            "repetition_step_fraction_pct": [10.0, 30.0],
            "source_approach_Dstar": [1.0, 2.0, 4.0],
        },
        "fig5d_decision_protocol": {
            "unit": "episode",
            "categories": [
                "correct_declaration", "false_declaration",
                "overconservative", "justified_abstention",
            ],
            "risk_coverage": "descriptive_curve_from_confidence_l4_and_l4_pass",
            "deployment_claim": False,
            "hard_threshold": None,
        },
        "ground_truth_closable_status": "not_used_by_revised_Fig5d; frozen v4 L5 remains answer-dependent",
        "required_counterexamples": [],
        "chain_of_thought_policy": "not_requested_not_stored; structured hypotheses only",
    }


def export_contract(
    *,
    store: DataStore,
    log_paths: list[Path],
    model_configs: list[dict[str, Any]],
    output_dir: Path,
    source_root: Path | None = None,
    registered_repeats: int = 3,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = _read_logs(log_paths)
    model_rows, actor_registry = build_model_registry(model_configs, episodes, registered_repeats)
    task_rows, region_rows, field_rows = build_task_artifacts(store, source_root)
    audit_rows = build_task_audit(store, source_root)
    equivalence_rows = build_interface_equivalence(DataStore(store.root, evaluator=False))
    episode_rows, step_rows, round_rows = build_episode_artifacts(store, episodes, actor_registry)
    write_csv(output_dir / "model_registry.csv", MODEL_FIELDS, model_rows)
    write_csv(output_dir / "task_registry.csv", TASK_FIELDS, task_rows)
    write_csv(output_dir / "episode_results.csv", EPISODE_FIELDS, episode_rows)
    audit_fields = [
        "candidate_event_id", "stage", "stage_order", "retained", "exclusion_reason",
        "task_id", "source_class", "environment_condition", "event_cluster_id",
        "background_id", "task_version_hash", "audit_scope", "source_artifact",
    ]
    write_csv(output_dir / "task_build_audit.csv", audit_fields, audit_rows)
    equivalence_fields = [
        "test_id", "task_id", "action_sequence_id", "runner_a", "runner_b", "n_steps",
        "observation_hash_a", "observation_hash_b", "budget_hash_a", "budget_hash_b",
        "termination_a", "termination_b", "observation_equal", "budget_equal",
        "termination_equal", "mismatch_step", "passed",
    ]
    write_csv(output_dir / "interface_equivalence.csv", equivalence_fields, equivalence_rows)
    region_fields = ["task_id", "region_type", "node_id", "region_rule_version"]
    field_fields = sorted(set().union(*(row.keys() for row in field_rows))) if field_rows else []
    _write_parquet(output_dir / "task_regions.parquet", region_rows, region_fields)
    _write_parquet(output_dir / "field_grid.parquet", field_rows, field_fields)
    _write_parquet(output_dir / "step_log.parquet", step_rows, STEP_FIELDS)
    _write_parquet(output_dir / "provider_round_log.parquet", round_rows, ROUND_FIELDS)
    (output_dir / "preregistered_thresholds.json").write_text(
        json.dumps(default_thresholds(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    figure_inputs = [
        "model_registry.csv", "task_registry.csv", "task_regions.parquet",
        "episode_results.csv", "step_log.parquet", "provider_round_log.parquet",
        "field_grid.parquet", "task_build_audit.csv", "interface_equivalence.csv",
        "preregistered_thresholds.json",
    ]
    if (output_dir / "representation_ablation_results.csv").is_file():
        figure_inputs.append("representation_ablation_results.csv")
    figure_manifest = {
        "figure_outline_version": DATA_CONTRACT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plotting_commit": None,
        "random_seed": 20260828,
        "input_files": figure_inputs,
        "filters": {"is_primary_condition": 1},
        "deviations": [
            "source_x/y/z are a deterministic diagnostic peak-source pair; canonical L4 truth remains source_nodes",
            "Fig5c is a separate terminal-readout ablation on identical frozen evidence, not autonomous navigation",
            "Fig5c main comparison is flat_list versus text_map; image_map is optional supplementary vision work",
            "Fig5d uses episode-level close/abstain by actual L4 correctness plus descriptive risk-coverage; no task-level closability or deployment threshold is claimed",
            "Fig2a is a post-registration benchmark assembly and release audit; no upstream excluded-event pool was available",
        ],
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(figure_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS.json")
    manifest = {
        "schema_version": DATA_CONTRACT_VERSION,
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
        ],
    }
    (output_dir / "SHA256SUMS.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "data_contract_version": DATA_CONTRACT_VERSION,
        "models": len(model_rows),
        "tasks": len(task_rows),
        "episodes": len(episode_rows),
        "steps": len(step_rows),
        "provider_rounds": len(round_rows),
        "task_region_rows": len(region_rows),
        "field_grid_rows": len(field_rows),
        "interface_equivalence_passed": sum(int(row["passed"]) for row in equivalence_rows),
        "interface_equivalence_total": len(equivalence_rows),
        "output_dir": str(output_dir),
    }
