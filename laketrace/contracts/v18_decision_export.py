from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from laketrace.contracts.v18_decision_repairs import merge_episode_decision_records, repair_usage
from laketrace.data.store import DataStore
from laketrace.scoring.scorer import score_submission
from laketrace.util import jsonl, sha256_file, stable_hash, write_csv


DECISION_FIELDS = [
    "episode_id", "model_id", "task_id", "run_seed", "decision_record_index",
    "main_provider_round_index", "tool_call_index", "linked_step_index",
    "event_index_before_action", "event_index_before_execution", "basis_observation_hash", "chosen_action_name",
    "chosen_action_arguments", "chosen_action_hash", "status", "error",
    "reported_model", "model_identity_match", "fed_back_to_agent",
    "truth_visible", "input_tokens", "output_tokens", "objective",
    "evidence_refs", "evidence_refs_parse_valid", "evidence_refs_valid",
    "source_x", "source_y", "source_z",
    "source_class", "action_rationale", "expected_evidence", "confidence",
    "main_uncertainty", "source_error_h_m", "source_error_v_m",
    "hypothesis_correct", "plume_entered_before_record", "action_valid",
    "action_result_code", "readout_attempt_count", "readout_prompt_hash", "readout_schema_hash",
    "record_origin", "original_status", "repair_attempt_index",
]

EPISODE_FIELDS = [
    "episode_id", "model_id", "task_id", "run_seed", "detectable_positive",
    "replay_verified", "termination_reason", "valid_submission",
    "decision_record_attempts", "decision_record_valid", "decision_record_coverage",
    "decision_record_repaired", "repair_attempt_rows",
    "decision_record_identity_match", "evidence_reference_valid_rate",
    "entered_plume", "first_plume_entry_event_index",
    "trajectory_entered_l4_tolerance", "min_source_horizontal_m",
    "min_source_vertical_m", "hypothesis_correct_at_any_step",
    "hypothesis_correct_after_plume", "first_correct_hypothesis_step",
    "final_hypothesis_correct", "l4_pass", "stage1_entered_plume",
    "stage2_correct_hypothesis_given_entry",
    "stage3_correct_submission_given_hypothesis", "first_error_stage",
    "main_input_tokens", "decision_record_input_tokens", "total_input_tokens",
    "repair_input_tokens",
    "input_token_multiplier", "main_output_tokens", "decision_record_output_tokens",
    "repair_output_tokens", "total_output_tokens", "behavior_chain_sha256", "event_chain_sha256",
]


def _log_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        files = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
        for item in files:
            yield from jsonl(item)


def _read_replay(path: Path | None) -> dict[str, bool]:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["episode_id"]: str(row["replay_verified"]).lower() == "true"
            for row in csv.DictReader(handle)
        }


def _source_error(position: dict[str, Any] | None, source_nodes: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    if not position or not source_nodes:
        return None, None
    pairs = [
        (
            math.dist(
                (float(position["x"]), float(position["y"])),
                (float(node["x"]), float(node["y"])),
            ),
            abs(float(position["z"]) - float(node["z"])),
        )
        for node in source_nodes
    ]
    return min(pairs, key=lambda pair: (math.hypot(*pair), pair))


def _visited_prefixes(store: DataStore, episode: dict[str, Any]) -> list[set[int]]:
    task = store.task(episode["instance_id"])
    frame = store.frame(episode["instance_id"], task["active_phase"])
    access = store.access_points["access_points"][task["domain_id"]]
    start = frame.by_xyz[(float(access["x"]), float(access["y"]), float(access["z"]))].index
    visited = {int(start)}
    prefixes = [set(visited)]
    for event in episode.get("events") or []:
        for reading in event.get("readings") or []:
            visited.add(int(reading["node_index"]))
        prefixes.append(set(visited))
    return prefixes


def _trajectory_positions(store: DataStore, episode: dict[str, Any]) -> list[dict[str, float]]:
    task = store.task(episode["instance_id"])
    access = store.access_points["access_points"][task["domain_id"]]
    output = [{"x": float(access["x"]), "y": float(access["y"]), "z": float(access["z"])}]
    output.extend(
        {"x": float(row["x"]), "y": float(row["y"]), "z": float(row["z"])}
        for event in episode.get("events") or []
        for row in event.get("readings") or []
    )
    return output


def export_v18_decisions(
    *,
    store: DataStore,
    log_paths: list[Path],
    replay_path: Path | None,
    output_dir: Path,
    repair_paths: list[Path] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    replay = _read_replay(replay_path)
    decision_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    all_episodes = list(_log_rows(log_paths))
    episodes = [
        row for row in all_episodes
        if row.get("valid_for_scoring") is not False and row.get("termination_reason") != "infra_failure"
    ]
    all_repairs = list(_log_rows(repair_paths or []))
    for episode_position, episode in enumerate(episodes):
        task_id = episode["instance_id"]
        truth = store.truth(task_id)
        tolerance = store.protocol["tolerances"][truth["waterbody"]]
        source_nodes = truth.get("source_nodes") or []
        plume_ids = set(map(int, truth.get("L2_plume_node_ids") or []))
        prefixes = _visited_prefixes(store, episode)
        all_visited = prefixes[-1]
        entered_plume = bool(all_visited & plume_ids) if plume_ids else False
        first_plume_entry = next(
            (index for index, nodes in enumerate(prefixes) if nodes & plume_ids),
            None,
        )
        trajectory = _trajectory_positions(store, episode)
        trajectory_errors = [_source_error(position, source_nodes) for position in trajectory]
        trajectory_pairs = [pair for pair in trajectory_errors if pair[0] is not None]
        if trajectory_pairs:
            min_h, min_v = min(trajectory_pairs, key=lambda pair: (math.hypot(*pair), pair))
            entered_l4 = any(
                h <= float(tolerance["horizontal_m"]) and v <= float(tolerance["vertical_m"])
                for h, v in trajectory_pairs
            )
        else:
            min_h = min_v = None
            entered_l4 = False
        episode_id = str(episode.get("episode_id") or f"row-{episode_position}")
        episode_repairs = [row for row in all_repairs if str(row.get("episode_id")) == episode_id]
        raw_records = merge_episode_decision_records(episode, episode_repairs)
        valid_records: list[dict[str, Any]] = []
        reference_valid_values: list[bool] = []
        for raw in raw_records:
            record = raw.get("record") or {}
            event_index = int(raw.get("event_index_before_action") or 0)
            visited_before = prefixes[min(event_index, len(prefixes) - 1)]
            evidence_text = str(record.get("evidence_refs") or "").strip()
            evidence_parse_valid = True
            evidence_nodes: list[int] = []
            if evidence_text:
                parts = [part.strip() for part in evidence_text.split(",")]
                try:
                    evidence_nodes = [int(part) for part in parts]
                except ValueError:
                    evidence_parse_valid = False
            record_eligible = raw.get("status") == "ok" and raw.get("model_identity_match") is True
            evidence_valid = (
                evidence_parse_valid and all(node in visited_before for node in evidence_nodes)
            ) if record_eligible else None
            if evidence_valid is not None:
                reference_valid_values.append(evidence_valid)
            position = None
            if all(record.get(f"source_{axis}") is not None for axis in "xyz"):
                position = {axis: float(record[f"source_{axis}"]) for axis in "xyz"}
            source_h, source_v = _source_error(position, source_nodes)
            hypothesis_correct = (
                source_h is not None
                and source_h <= float(tolerance["horizontal_m"])
                and source_v <= float(tolerance["vertical_m"])
            )
            plume_before = bool(visited_before & plume_ids) if plume_ids else False
            row = {
                "episode_id": episode_id,
                "model_id": episode["actor"],
                "task_id": task_id,
                "run_seed": episode["seed"],
                "decision_record_index": raw.get("decision_record_index"),
                "main_provider_round_index": raw.get("main_provider_round_index"),
                "tool_call_index": raw.get("tool_call_index"),
                "linked_step_index": raw.get("linked_step_index"),
                "event_index_before_action": event_index,
                "event_index_before_execution": raw.get("event_index_before_execution"),
                "basis_observation_hash": raw.get("basis_observation_hash"),
                "chosen_action_name": raw.get("chosen_action_name"),
                "chosen_action_arguments": raw.get("chosen_action_arguments"),
                "chosen_action_hash": raw.get("chosen_action_hash"),
                "status": raw.get("status"),
                "error": raw.get("error"),
                "reported_model": raw.get("reported_model"),
                "model_identity_match": raw.get("model_identity_match"),
                "fed_back_to_agent": raw.get("fed_back_to_agent"),
                "truth_visible": raw.get("truth_visible"),
                "input_tokens": (raw.get("usage") or {}).get("input_tokens", 0),
                "output_tokens": (raw.get("usage") or {}).get("output_tokens", 0),
                **{field: record.get(field) for field in (
                    "objective", "evidence_refs", "source_x", "source_y", "source_z",
                    "source_class", "action_rationale", "expected_evidence", "confidence",
                    "main_uncertainty",
                )},
                "evidence_refs_parse_valid": evidence_parse_valid if record_eligible else None,
                "evidence_refs_valid": evidence_valid,
                "source_error_h_m": source_h,
                "source_error_v_m": source_v,
                "hypothesis_correct": hypothesis_correct if position else None,
                "plume_entered_before_record": plume_before,
                "action_valid": raw.get("action_valid"),
                "action_result_code": raw.get("action_result_code"),
                "readout_attempt_count": raw.get("readout_attempt_count"),
                "readout_prompt_hash": raw.get("readout_prompt_hash"),
                "readout_schema_hash": raw.get("readout_schema_hash"),
                "record_origin": raw.get("record_origin", "inline"),
                "original_status": raw.get("original_status", raw.get("status")),
                "repair_attempt_index": raw.get("repair_attempt_index"),
            }
            decision_rows.append(row)
            if record_eligible and position is not None:
                valid_records.append(row)
        correct_any = any(row["hypothesis_correct"] is True for row in valid_records)
        correct_after = [
            row for row in valid_records
            if row["plume_entered_before_record"] and row["hypothesis_correct"] is True
        ]
        hypotheses_after = [row for row in valid_records if row["plume_entered_before_record"]]
        first_correct = min((int(row["linked_step_index"]) for row in correct_after if row["linked_step_index"] is not None), default=None)
        final_correct = hypotheses_after[-1]["hypothesis_correct"] if hypotheses_after else None
        score = score_submission(
            episode.get("answer") or {}, truth, tolerance
        )
        l4_pass = score.get("L4_pass") is True
        expected_attempts = sum(
            len(call.get("tool_calls") or [])
            for call in episode.get("provider_calls") or []
        )
        valid_count = sum(
            raw.get("status") == "ok" and raw.get("model_identity_match") is True
            for raw in raw_records
        )
        coverage = valid_count / expected_attempts if expected_attempts else 0.0
        complete_for_inference = expected_attempts > 0 and valid_count == expected_attempts
        if not entered_plume:
            first_error = "sampling_position"
        elif not correct_after and not complete_for_inference:
            first_error = "record_missing"
        elif not correct_after:
            first_error = "observation_interpretation"
        elif l4_pass:
            first_error = "success"
        elif final_correct is False:
            first_error = "hypothesis_update"
        elif final_correct is True:
            first_error = "conclusion_submission"
        else:
            first_error = "record_missing"
        main_usage = episode.get("main_token_usage") or {}
        readout_usage = episode.get("decision_record_token_usage") or {}
        total_usage = episode.get("token_usage") or {}
        repaired_usage = repair_usage(episode_repairs)
        main_input = int(main_usage.get("input_tokens", 0))
        readout_input = int(readout_usage.get("input_tokens", 0))
        repair_input = int(repaired_usage.get("input_tokens", 0))
        repair_output = int(repaired_usage.get("output_tokens", 0))
        episode_rows.append({
            "episode_id": episode_id,
            "model_id": episode["actor"],
            "task_id": task_id,
            "run_seed": episode["seed"],
            "detectable_positive": bool(truth.get("detectable_source")),
            "replay_verified": replay.get(episode.get("episode_id")),
            "termination_reason": episode.get("termination_reason"),
            "valid_submission": episode.get("answer") is not None,
            "decision_record_attempts": expected_attempts,
            "decision_record_valid": valid_count,
            "decision_record_coverage": coverage,
            "decision_record_repaired": sum(raw.get("record_origin") == "repair" for raw in raw_records),
            "repair_attempt_rows": len(episode_repairs),
            "decision_record_identity_match": bool(raw_records) and all(
                raw.get("status") == "ok" and raw.get("model_identity_match") is True
                for raw in raw_records
            ),
            "evidence_reference_valid_rate": (
                sum(reference_valid_values) / len(reference_valid_values)
                if reference_valid_values else None
            ),
            "entered_plume": entered_plume,
            "first_plume_entry_event_index": first_plume_entry,
            "trajectory_entered_l4_tolerance": entered_l4,
            "min_source_horizontal_m": min_h,
            "min_source_vertical_m": min_v,
            "hypothesis_correct_at_any_step": correct_any,
            "hypothesis_correct_after_plume": bool(correct_after),
            "first_correct_hypothesis_step": first_correct,
            "final_hypothesis_correct": final_correct,
            "l4_pass": score.get("L4_pass"),
            "stage1_entered_plume": entered_plume,
            "stage2_correct_hypothesis_given_entry": bool(correct_after) if entered_plume else None,
            "stage3_correct_submission_given_hypothesis": l4_pass if correct_after else None,
            "first_error_stage": first_error,
            "main_input_tokens": main_input,
            "decision_record_input_tokens": readout_input,
            "repair_input_tokens": repair_input,
            "total_input_tokens": int(total_usage.get("input_tokens", main_input + readout_input)) + repair_input,
            "input_token_multiplier": ((main_input + readout_input + repair_input) / main_input) if main_input else None,
            "main_output_tokens": int(main_usage.get("output_tokens", 0)),
            "decision_record_output_tokens": int(readout_usage.get("output_tokens", 0)),
            "repair_output_tokens": repair_output,
            "total_output_tokens": int(total_usage.get("output_tokens", 0)) + repair_output,
            "behavior_chain_sha256": episode.get("behavior_chain_sha256"),
            "event_chain_sha256": episode.get("event_chain_sha256"),
        })
    decision_path = output_dir / "decision_records.csv"
    episode_path = output_dir / "fig6_episode_diagnostics.csv"
    write_csv(decision_path, DECISION_FIELDS, decision_rows)
    write_csv(episode_path, EPISODE_FIELDS, episode_rows)
    summary = {
        "data_contract_version": "v18.decision-record.4",
        "attempt_rows": len(all_episodes),
        "excluded_infrastructure_attempts": len(all_episodes) - len(episodes),
        "episodes": len(episode_rows),
        "decision_record_attempts": sum(int(row["decision_record_attempts"]) for row in episode_rows),
        "decision_record_valid": sum(int(row["decision_record_valid"]) for row in episode_rows),
        "record_coverage": (
            sum(int(row["decision_record_valid"]) for row in episode_rows)
            / sum(int(row["decision_record_attempts"]) for row in episode_rows)
            if sum(int(row["decision_record_attempts"]) for row in episode_rows) else 0.0
        ),
        "replay_verified": sum(row["replay_verified"] is True for row in episode_rows),
        "submitted": sum(bool(row["valid_submission"]) for row in episode_rows),
        "repair_attempt_rows": len(all_repairs),
        "repaired_records": sum(int(row["decision_record_repaired"]) for row in episode_rows),
        "fig6_stage_rows": len(episode_rows),
        "outputs": {
            "decision_records.csv": sha256_file(decision_path),
            "fig6_episode_diagnostics.csv": sha256_file(episode_path),
        },
        "source_logs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in log_paths if path.is_file()
        ],
        "repair_logs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in (repair_paths or []) if path.is_file()
        ],
        "contract_hash": stable_hash({"decision_fields": DECISION_FIELDS, "episode_fields": EPISODE_FIELDS}),
    }
    summary_path = output_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
