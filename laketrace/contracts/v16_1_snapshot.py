from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from laketrace.agents.model import _identity_matches, _present_observation, _provider, validate_model_config
from laketrace.data.store import DataStore
from laketrace.env.core import Environment
from laketrace.providers.openai_compatible import ProviderResponse
from laketrace.scoring.scorer import score_submission
from laketrace.util import append_jsonl, jsonl, stable_hash, write_csv


SNAPSHOT_PROTOCOL_VERSION = "v16.1-frozen-evidence-readout-1"
SNAPSHOT_REPRESENTATIONS = ("flat_list", "text_map")
PRESENTATION_MODE = {"flat_list": "records", "text_map": "sparse_map"}

READOUT_PROMPT = """You are given one frozen LakeTrace measurement snapshot.
You cannot move or obtain more measurements. Use only the supplied evidence and
submit the best L2 contaminated-region coordinate and L4 source coordinate.
Use three nulls when a coordinate cannot be determined. Confidence values are
numbers from 0 to 1. Return exactly one readout_submit tool call and no prose."""

READOUT_TOOL = [{
    "type": "function",
    "function": {
        "name": "readout_submit",
        "description": "Submit L2 and L4 coordinates and confidence from the frozen evidence snapshot.",
        "parameters": {
            "type": "object",
            "properties": {
                "peak_x": {"type": ["number", "null"]},
                "peak_y": {"type": ["number", "null"]},
                "peak_z": {"type": ["number", "null"]},
                "source_x": {"type": ["number", "null"]},
                "source_y": {"type": ["number", "null"]},
                "source_z": {"type": ["number", "null"]},
                "confidence_l2": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "confidence_l4": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": [
                "peak_x", "peak_y", "peak_z", "source_x", "source_y", "source_z",
                "confidence_l2", "confidence_l4",
            ],
            "additionalProperties": False,
        },
    },
}]

READOUT_FIELDS = [
    "readout_id", "model_id", "model_snapshot", "task_id", "run_seed",
    "representation", "snapshot_protocol_version", "snapshot_id",
    "canonical_evidence_hash", "visited_mask_hash", "measurement_history_count",
    "rendered_observation_hash", "valid_submission", "infrastructure_failure",
    "peak_x_m", "peak_y_m", "peak_z_m", "source_x_m", "source_y_m", "source_z_m",
    "confidence_l2", "confidence_l4", "l2_applicable", "l2_pass",
    "l2_peak_distance_m", "l2_plume_distance_m", "l4_applicable", "l4_pass",
    "l4_source_distance_m", "input_tokens", "output_tokens", "reported_model",
    "model_identity_match", "content_sha256", "content_char_count", "provider_error",
    "chain_of_thought_requested", "chain_of_thought_stored", "raw_output_path", "log_hash",
]


def _keys_recursive(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys_recursive(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys_recursive(item)


def _rounded_floats(value: Any) -> Any:
    """Normalize harmless cross-Python summation differences for replay audit."""

    if isinstance(value, dict):
        return {key: _rounded_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded_floats(item) for item in value]
    if isinstance(value, float):
        return round(value, 9)
    return value


def _position(arguments: dict[str, Any], prefix: str) -> dict[str, float] | None:
    values = [arguments.get(f"{prefix}_{axis}") for axis in "xyz"]
    if all(value is None for value in values):
        return None
    if any(value is None or isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"{prefix} must contain three finite numbers or three nulls")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{prefix} must contain three finite numbers or three nulls")
    return dict(zip("xyz", result))


def _confidence(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number from 0 to 1")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a number from 0 to 1")
    return result


def _parse_readout(call: dict[str, Any]) -> dict[str, Any]:
    if call.get("name") != "readout_submit":
        raise ValueError("the only accepted tool is readout_submit")
    arguments = json.loads(call.get("arguments") or "{}")
    if not isinstance(arguments, dict):
        raise ValueError("readout_submit arguments must be an object")
    required = {
        "peak_x", "peak_y", "peak_z", "source_x", "source_y", "source_z",
        "confidence_l2", "confidence_l4",
    }
    if set(arguments) != required:
        raise ValueError("readout_submit fields do not match the registered flat schema")
    return {
        "peak_position": _position(arguments, "peak"),
        "source_position": _position(arguments, "source"),
        "confidence_l2": _confidence(arguments["confidence_l2"], "confidence_l2"),
        "confidence_l4": _confidence(arguments["confidence_l4"], "confidence_l4"),
    }


def _replay_pre_submission(env: Environment, episode: dict[str, Any]) -> None:
    """Recreate the source-independent baseline evidence without its answer."""

    for event in episode.get("events") or []:
        action = event.get("action")
        if action == "submit":
            break
        if action == "move":
            result = env.move(event["direction"], int(event["steps"]))
        elif action == "analyze":
            result = env.analyze()
        elif action == "infeasible" and event.get("requested_action") == "move":
            arguments = event.get("arguments") or {}
            result = env.move(arguments["direction"], int(arguments["steps"]))
        elif action == "infeasible" and event.get("requested_action") == "analyze":
            result = env.analyze()
        elif action == "invalid":
            result = env.record_invalid(event.get("code") or "E_SCHEMA", event.get("detail") or "", event.get("requested_action"))
        else:
            continue
        if stable_hash(_rounded_floats(result)) != stable_hash(_rounded_floats(event)):
            raise ValueError(
                f"baseline evidence replay mismatch for {episode['instance_id']} "
                f"at event {len(env.events) - 1}"
            )


def build_frozen_evidence_snapshots(
    store: DataStore,
    baseline_logs: Path,
    *,
    actor: str = "budgeted_serpentine",
) -> list[dict[str, Any]]:
    """Build truth-blind paired renderings from one deterministic evidence path."""

    selected = [row for row in jsonl(baseline_logs) if row.get("actor") == actor]
    by_task: dict[str, dict[str, Any]] = {}
    for row in selected:
        task_id = str(row["instance_id"])
        if task_id in by_task:
            raise ValueError(f"duplicate {actor} baseline evidence path for {task_id}")
        by_task[task_id] = row
    expected = {task["instance_id"] for task in store.tasks()}
    if set(by_task) != expected:
        missing = sorted(expected - set(by_task))
        extra = sorted(set(by_task) - expected)
        raise ValueError(f"{actor} baseline coverage mismatch; missing={missing}, extra={extra}")

    snapshots = []
    for task_id in sorted(by_task):
        episode = by_task[task_id]
        env = Environment(store, task_id, int(episode["seed"]))
        _replay_pre_submission(env, episode)
        canonical = env.observation()
        rows = canonical["cumulative_measurements"]
        evidence = {
            "cumulative_measurements": rows,
            "candidates": canonical["candidates"],
            "available_channels": canonical["available_channels"],
        }
        rendered = {
            representation: _present_observation(canonical, mode, [])
            for representation, mode in PRESENTATION_MODE.items()
        }
        forbidden = {"source_nodes", "L2_plume_positions", "exam_truth", "ground_truth"}
        leaked = forbidden & set(_keys_recursive(rendered))
        if leaked:
            raise ValueError(f"truth-derived keys detected in snapshot {task_id}: {sorted(leaked)}")
        canonical_hash = stable_hash(evidence)
        snapshot_id = stable_hash({
            "protocol": SNAPSHOT_PROTOCOL_VERSION,
            "task_id": task_id,
            "baseline_actor": actor,
            "baseline_event_chain_sha256": episode.get("event_chain_sha256"),
            "canonical_evidence_hash": canonical_hash,
        })
        snapshots.append({
            "snapshot_protocol_version": SNAPSHOT_PROTOCOL_VERSION,
            "snapshot_id": snapshot_id,
            "task_id": task_id,
            "baseline_actor": actor,
            "baseline_run_id": episode.get("run_id"),
            "baseline_seed": int(episode["seed"]),
            "baseline_event_chain_sha256": episode.get("event_chain_sha256"),
            "baseline_pre_submit_turns": env.turn,
            "canonical_evidence_hash": canonical_hash,
            "visited_mask_hash": stable_hash(sorted(int(row["node_index"]) for row in rows)),
            "measurement_history_count": sum(int(row["sample_count"]) for row in rows),
            "visited_nodes_count": len(rows),
            "rendered_observations": rendered,
            "rendered_observation_hashes": {
                representation: stable_hash(value) for representation, value in rendered.items()
            },
            "truth_fields_present": False,
        })
    return snapshots


class _ScriptedReadoutProvider:
    def __init__(self, config: dict[str, Any]):
        self.model = str(config.get("model") or "scripted-readout-v16.1")

    def complete(self, _messages: list[dict[str, Any]], _tools: list[dict[str, Any]]) -> ProviderResponse:
        call = {
            "id": "scripted-readout",
            "name": "readout_submit",
            "arguments": json.dumps({
                "peak_x": None, "peak_y": None, "peak_z": None,
                "source_x": None, "source_y": None, "source_z": None,
                "confidence_l2": 0.0, "confidence_l4": 0.0,
            }),
        }
        return ProviderResponse([call], None, {"input_tokens": 0, "output_tokens": 0}, self.model)


def run_snapshot_readouts(
    snapshots: Iterable[dict[str, Any]],
    config: dict[str, Any],
    *,
    representation: str,
    repeats: int,
    output_path: Path,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Run one non-navigating terminal readout; evaluator truth is not loaded."""

    validate_model_config(config)
    if representation not in SNAPSHOT_REPRESENTATIONS:
        raise ValueError(f"representation must be one of {SNAPSHOT_REPRESENTATIONS}")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    provider = _ScriptedReadoutProvider(config) if config["adapter"] == "scripted" else _provider(config)
    existing = list(jsonl(output_path)) if resume and output_path.is_file() else []
    completed = {
        (row.get("snapshot_id"), int(row.get("run_seed", -1)), row.get("representation"))
        for row in existing
        if not row.get("infrastructure_failure")
    }
    results = list(existing)
    if not resume:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
    prompt_id = "laketrace-v16.1-frozen-evidence-readout"
    for run_seed in range(repeats):
        for snapshot in snapshots:
            key = (snapshot["snapshot_id"], run_seed, representation)
            if key in completed:
                continue
            observation = snapshot["rendered_observations"][representation]
            messages = [
                {"role": "system", "content": READOUT_PROMPT},
                {"role": "user", "content": "FROZEN EVIDENCE SNAPSHOT\n" + json.dumps(observation, ensure_ascii=False, separators=(",", ":"))},
            ]
            provider_error = None
            try:
                response = provider.complete(messages, READOUT_TOOL)
            except RuntimeError as exc:
                response = ProviderResponse([], None, {"input_tokens": 0, "output_tokens": 0}, None)
                provider_error = str(exc)
            answer = None
            parse_error = None
            if not provider_error:
                if len(response.tool_calls) != 1:
                    parse_error = "expected exactly one readout_submit tool call"
                else:
                    try:
                        answer = _parse_readout(response.tool_calls[0])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        parse_error = str(exc)
            record = {
                "schema_version": SNAPSHOT_PROTOCOL_VERSION,
                "readout_id": f"{config['id']}::{snapshot['task_id']}::{run_seed}::{representation}",
                "model_id": config["id"],
                "model_snapshot": config["model"],
                "prompt_id": prompt_id,
                "run_seed": run_seed,
                "representation": representation,
                "snapshot_protocol_version": SNAPSHOT_PROTOCOL_VERSION,
                "snapshot_id": snapshot["snapshot_id"],
                "task_id": snapshot["task_id"],
                "canonical_evidence_hash": snapshot["canonical_evidence_hash"],
                "visited_mask_hash": snapshot["visited_mask_hash"],
                "measurement_history_count": snapshot["measurement_history_count"],
                "rendered_observation_hash": snapshot["rendered_observation_hashes"][representation],
                "messages_hash": stable_hash(messages),
                "tool_schema_hash": stable_hash(READOUT_TOOL),
                "answer": answer,
                "valid_submission": answer is not None,
                "infrastructure_failure": provider_error is not None,
                "parse_error": parse_error,
                "provider_error": provider_error,
                "usage": response.usage,
                "reported_model": response.raw_model,
                "model_identity_match": _identity_matches(str(config["model"]), [response.raw_model] if response.raw_model else []),
                "content": None,
                "content_sha256": stable_hash(response.content) if response.content else None,
                "content_char_count": len(response.content or ""),
                "chain_of_thought_requested": False,
                "chain_of_thought_stored": False,
            }
            record["log_hash"] = stable_hash(record)
            append_jsonl(output_path, record)
            results.append(record)
    return results


def score_snapshot_readouts(store: DataStore, raw_paths: Iterable[Path], output_path: Path) -> list[dict[str, Any]]:
    """Score frozen readout answers offline on the evaluator side."""

    rows = []
    for raw_path in raw_paths:
        for record in jsonl(raw_path):
            answer = record.get("answer") or {}
            truth = store.truth(record["task_id"])
            tolerance = store.protocol["tolerances"][truth["waterbody"]]
            score = score_submission(answer, truth, tolerance) if record.get("valid_submission") else {}
            peak = answer.get("peak_position") or {}
            source = answer.get("source_position") or {}
            rows.append({
                "readout_id": record["readout_id"],
                "model_id": record["model_id"],
                "model_snapshot": record["model_snapshot"],
                "task_id": record["task_id"],
                "run_seed": record["run_seed"],
                "representation": record["representation"],
                "snapshot_protocol_version": record["snapshot_protocol_version"],
                "snapshot_id": record["snapshot_id"],
                "canonical_evidence_hash": record["canonical_evidence_hash"],
                "visited_mask_hash": record["visited_mask_hash"],
                "measurement_history_count": record["measurement_history_count"],
                "rendered_observation_hash": record["rendered_observation_hash"],
                "valid_submission": record["valid_submission"],
                "infrastructure_failure": record["infrastructure_failure"],
                "peak_x_m": peak.get("x"), "peak_y_m": peak.get("y"), "peak_z_m": peak.get("z"),
                "source_x_m": source.get("x"), "source_y_m": source.get("y"), "source_z_m": source.get("z"),
                "confidence_l2": answer.get("confidence_l2"), "confidence_l4": answer.get("confidence_l4"),
                "l2_applicable": score.get("L2_applicable"), "l2_pass": score.get("L2_pass"),
                "l2_peak_distance_m": score.get("L2_peak_distance_m"),
                "l2_plume_distance_m": score.get("L2_plume_distance_m"),
                "l4_applicable": score.get("L4_applicable"), "l4_pass": score.get("L4_pass"),
                "l4_source_distance_m": score.get("L4_source_distance_m"),
                "input_tokens": (record.get("usage") or {}).get("input_tokens", 0),
                "output_tokens": (record.get("usage") or {}).get("output_tokens", 0),
                "reported_model": record.get("reported_model"),
                "model_identity_match": record.get("model_identity_match"),
                "content_sha256": record.get("content_sha256"),
                "content_char_count": record.get("content_char_count", 0),
                "provider_error": record.get("provider_error") or record.get("parse_error"),
                "chain_of_thought_requested": record.get("chain_of_thought_requested", False),
                "chain_of_thought_stored": record.get("chain_of_thought_stored", False),
                "raw_output_path": str(raw_path.resolve()),
                "log_hash": record.get("log_hash") or stable_hash(record),
            })
    if not rows:
        raise ValueError("no snapshot readout rows found")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, READOUT_FIELDS, rows)
    return rows
