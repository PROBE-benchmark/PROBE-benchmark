from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

from laketrace.agents.model import (
    SYSTEM_PROMPT,
    TOOLS,
    _identity_matches,
    _present_observation,
    _submission,
    _validate_arguments,
    validate_model_config,
)
from laketrace.contracts.v16_1_runner import InstrumentedEnvironment, behavior_event
from laketrace.contracts.v18_provider import AuditedOpenAICompatibleProvider
from laketrace.perception import PRESENTATION_MODES
from laketrace.providers.openai_compatible import ProviderResponse
from laketrace.util import stable_hash


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "v19" / "integrated_decision_protocol.json"
DATA_CONTRACT_VERSION = "v19.1.integrated-decision-record.1"
PROMPT_ID = "laketrace-v4-actions-plus-v19.1-integrated-decision-record"
DR_PREFIX = "dr_"
DR_FIELDS = (
    "objective",
    "evidence_refs",
    "source_x",
    "source_y",
    "source_z",
    "source_class",
    "action_rationale",
    "expected_evidence",
    "confidence",
    "main_uncertainty",
)


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_id") != "laketrace-v19.1-integrated-decision-record-v1":
        raise ValueError("unexpected v19 integrated decision-record protocol id")
    return value


PROTOCOL = load_protocol()


def _record_properties() -> dict[str, dict[str, Any]]:
    short = {"type": "string", "maxLength": 240}
    return {
        "dr_objective": dict(short),
        "dr_evidence_refs": {
            "type": "string",
            "maxLength": 160,
            "description": "Comma-separated measured node identifiers only, or an empty string.",
        },
        "dr_source_x": {"type": ["number", "null"]},
        "dr_source_y": {"type": ["number", "null"]},
        "dr_source_z": {"type": ["number", "null"]},
        "dr_source_class": {
            "type": ["string", "null"],
            "enum": [
                "external_point",
                "external_diffuse",
                "internal_release",
                "physical_resuspension",
                None,
            ],
        },
        "dr_action_rationale": dict(short),
        "dr_expected_evidence": dict(short),
        "dr_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "dr_main_uncertainty": dict(short),
    }


def _tools_v19() -> list[dict[str, Any]]:
    tools = copy.deepcopy(TOOLS)
    record_properties = _record_properties()
    record_required = [DR_PREFIX + field for field in DR_FIELDS]
    for item in tools:
        function = item["function"]
        parameters = function["parameters"]
        parameters["properties"].update(copy.deepcopy(record_properties))
        parameters["required"] = list(parameters.get("required") or []) + record_required
        function["description"] += (
            " Include the registered flat dr_* externalized decision record in the same call. "
            "The record is audit-only and never changes action execution or score."
        )
    return tools


TOOLS_V19 = _tools_v19()
SYSTEM_PROMPT_V19 = SYSTEM_PROMPT + """

Audit-record addendum (v19): every move, analyze, or submit tool call must also
include all ten flat dr_* fields. These are a concise externalized record, not
private chain-of-thought: state the current objective; cite only measured node
identifiers; give each source coordinate you currently support and use null for
each unsupported axis; one listed class or null; a short action rationale; expected observable evidence; confidence from
0 to 1; and the main uncertainty. Keep each text field to one short sentence.
The dr_* fields do not change action execution, budget, scoring, or later
observations, and are never returned to you as feedback."""


_DECIMAL = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_RECORD_STRING_FIELDS = {
    "objective",
    "evidence_refs",
    "action_rationale",
    "expected_evidence",
    "main_uncertainty",
}
_SOURCE_CLASSES = {
    None,
    "external_point",
    "external_diffuse",
    "internal_release",
    "physical_resuspension",
}


def _record_number(value: Any, field: str, normalized: list[str]) -> float:
    if isinstance(value, str) and _DECIMAL.fullmatch(value.strip()):
        value = float(value)
        normalized.append(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def split_integrated_arguments(arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    core = {key: value for key, value in arguments.items() if not key.startswith(DR_PREFIX)}
    raw_record = {
        field: arguments[DR_PREFIX + field]
        for field in DR_FIELDS
        if DR_PREFIX + field in arguments
    }
    return core, raw_record


def validate_integrated_record(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if set(raw) != set(DR_FIELDS):
        missing = sorted(set(DR_FIELDS) - set(raw))
        extra = sorted(set(raw) - set(DR_FIELDS))
        raise ValueError(f"integrated decision-record fields mismatch; missing={missing}, extra={extra}")
    output = dict(raw)
    for field in _RECORD_STRING_FIELDS:
        value = raw[field]
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string")
        if len(value) > 1200:
            raise ValueError(f"{field} exceeds the hard audit limit")
    normalized: list[str] = []
    for axis in "xyz":
        field = f"source_{axis}"
        value = output[field]
        if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
            output[field] = None
            normalized.append(field)
    if isinstance(output["source_class"], str) and output["source_class"].strip().lower() in {"none", "null"}:
        output["source_class"] = None
        normalized.append("source_class")
    coordinates = [output[f"source_{axis}"] for axis in "xyz"]
    for axis, value in zip("xyz", coordinates):
        if value is not None:
            field = f"source_{axis}"
            output[field] = _record_number(value, field, normalized)
    if output["source_class"] not in _SOURCE_CLASSES:
        raise ValueError("source_class is outside the registered enum")
    output["confidence"] = _record_number(raw["confidence"], "confidence", normalized)
    if not 0.0 <= output["confidence"] <= 1.0:
        raise ValueError("confidence must be from 0 to 1")
    return output, normalized


def _provider(config: dict[str, Any]):
    if config["adapter"] == "scripted":
        return _ScriptedProviderV19(config)
    return AuditedOpenAICompatibleProvider(config)


class _ScriptedProviderV19:
    """Zero-cost producer for integrated-record contract and replay tests."""

    def __init__(self, config: dict[str, Any]):
        self.turn = 0
        self.model = str(config.get("model") or "scripted-v19")
        self.fail_record = bool(config.get("scripted_integrated_record_failure", False))
        self.last_call_audit = None

    @staticmethod
    def _record() -> dict[str, Any]:
        return {
            "dr_objective": "collect one additional measured point",
            "dr_evidence_refs": "",
            "dr_source_x": None,
            "dr_source_y": None,
            "dr_source_z": None,
            "dr_source_class": None,
            "dr_action_rationale": "use one legal action to extend measured coverage",
            "dr_expected_evidence": "one additional measured node or an explicit analysis result",
            "dr_confidence": 0.1,
            "dr_main_uncertainty": "insufficient measured coverage",
        }

    def complete(self, messages: list[dict[str, Any]], _tools: list[dict[str, Any]]) -> ProviderResponse:
        self.turn += 1
        observation = json.loads(messages[-1]["content"].split("\n", 1)[-1])
        record = self._record()
        if self.fail_record and self.turn == 1:
            record.pop("dr_confidence")
        if self.turn == 1:
            available = [name for name, edge in observation["neighbors"].items() if edge["reachable"]]
            if available:
                arguments = {"direction": available[0], "steps": 1, **record}
                call = {"id": "v19-move", "name": "move", "arguments": json.dumps(arguments)}
            else:
                self.turn += 1
                call = {"id": "v19-analyze", "name": "analyze", "arguments": json.dumps(record)}
        elif self.turn == 2:
            call = {"id": "v19-analyze", "name": "analyze", "arguments": json.dumps(record)}
        else:
            arguments = {
                "anomaly": "no_anomaly",
                "peak_x": None,
                "peak_y": None,
                "peak_z": None,
                "mechanism": None,
                "source_x": None,
                "source_y": None,
                "source_z": None,
                "source_candidate_id": None,
                "decision": "cannot_determine",
                **record,
            }
            call = {"id": "v19-submit", "name": "submit", "arguments": json.dumps(arguments)}
        return ProviderResponse([call], None, {"input_tokens": 0, "output_tokens": 0}, self.model)


def _state_record_count(observation: dict[str, Any]) -> int:
    state = observation.get("cumulative_measurement_state") or {}
    mode = state.get("mode")
    if mode == "records":
        return len(state.get("rows") or [])
    if mode == "columnar":
        return int(state.get("row_count") or 0)
    if mode == "sparse_map":
        return len(state.get("nodes") or {})
    return 0


def _decision_record(
    *,
    raw_record: dict[str, Any],
    observation_hash: str,
    core_action: dict[str, Any],
    main_round_index: int,
    tool_call_index: int,
    event_index_before_action: int,
    reported_model: str | None,
    expected_model: str,
) -> dict[str, Any]:
    status = "ok"
    error = None
    normalized_record = None
    normalized_fields: list[str] = []
    try:
        normalized_record, normalized_fields = validate_integrated_record(raw_record)
    except ValueError as exc:
        status, error = "schema_failure", str(exc)
    normalized_numeric_fields = [
        field for field in normalized_fields
        if isinstance(raw_record.get(field), str)
        and _DECIMAL.fullmatch(str(raw_record[field]).strip())
    ]
    return {
        "decision_record_index": None,
        "main_provider_round_index": main_round_index,
        "tool_call_index": tool_call_index,
        "event_index_before_action": event_index_before_action,
        "basis_observation_hash": observation_hash,
        "chosen_action_name": core_action.get("name"),
        "chosen_action_arguments": core_action.get("arguments"),
        "chosen_action_hash": stable_hash(core_action),
        "readout_prompt_hash": None,
        "readout_schema_hash": stable_hash(_record_properties()),
        "timing": "same_provider_response_as_action",
        "fed_back_to_agent": False,
        "truth_visible": False,
        "status": status,
        "error": error,
        "reported_model": reported_model,
        "model_identity_match": _identity_matches(expected_model, [reported_model] if reported_model else []),
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "record": normalized_record,
        "raw_record": raw_record,
        "normalized_fields": normalized_fields,
        "normalized_numeric_fields": normalized_numeric_fields,
        "readout_attempts": [],
        "readout_attempt_count": 1,
        "record_transport": "integrated_flat_tool_arguments",
        "unexpected_content_sha256": None,
        "unexpected_content_char_count": 0,
    }


def run_model_v19(env: InstrumentedEnvironment, config: dict[str, Any]) -> dict[str, Any]:
    """Run the frozen core actions with one-call, non-blocking flat records."""

    validate_model_config(config)
    provider = _provider(config)
    presentation_mode = str(config.get("presentation_mode", "records"))
    if presentation_mode not in PRESENTATION_MODES:
        raise ValueError(f"presentation_mode must be one of {PRESENTATION_MODES}")
    prompt = str(config.get("system_prompt_v19") or SYSTEM_PROMPT_V19)
    action_schema_hash = stable_hash(TOOLS_V19)
    invalid_streak = 0
    main_calls: list[dict[str, Any]] = []
    decision_records: list[dict[str, Any]] = []
    last_results: list[dict[str, Any]] = []
    totals = {"input_tokens": 0, "output_tokens": 0}
    while not env.terminated:
        engineering_cap = config.get("engineering_max_provider_rounds")
        if engineering_cap is not None and len(main_calls) >= int(engineering_cap):
            if bool(config.get("science_eligible", False)):
                raise ValueError("engineering_max_provider_rounds is forbidden for science-eligible models")
            env.terminated, env.termination_reason = True, "engineering_round_cap"
            if env.events:
                env.events[-1]["termination_after_step"] = env.termination_reason
            break
        canonical = env.observation(first=not main_calls)
        canonical_rows = canonical["cumulative_measurements"]
        evidence_content_hash = stable_hash({
            "cumulative_measurements": canonical_rows,
            "candidates": canonical["candidates"],
            "available_channels": canonical["available_channels"],
        })
        visited_mask_hash = stable_hash(sorted(int(row["node_index"]) for row in canonical_rows))
        measurement_history_count = sum(int(row["sample_count"]) for row in canonical_rows)
        observation = _present_observation(canonical, presentation_mode, last_results)
        observation_json = json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
        observation_hash = stable_hash(observation)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "CURRENT OBSERVATION\n" + observation_json},
        ]
        messages_hash = stable_hash(messages)
        round_index = len(main_calls)
        try:
            response = provider.complete(messages, TOOLS_V19)
        except RuntimeError as exc:
            env.terminated, env.termination_reason = True, "infra_failure"
            main_calls.append({
                "provider_round_index": round_index,
                "tool_calls": [],
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "reported_model": None,
                "provider_error": str(exc),
                "observation_json": observation_json,
                "observation_hash": observation_hash,
                "messages_hash": messages_hash,
                "tool_schema_hash": action_schema_hash,
                "evidence_content_hash": evidence_content_hash,
                "visited_mask_hash": visited_mask_hash,
                "measurement_history_count": measurement_history_count,
                "state_record_count": _state_record_count(observation),
                "transport_audit": copy.deepcopy(getattr(provider, "last_call_audit", None)),
            })
            break
        usage = {
            "input_tokens": int(response.usage.get("input_tokens", 0)),
            "output_tokens": int(response.usage.get("output_tokens", 0)),
        }
        totals = {key: totals[key] + usage[key] for key in totals}
        main_calls.append({
            "provider_round_index": round_index,
            "tool_calls": response.tool_calls,
            "content": None,
            "content_sha256": stable_hash(response.content) if response.content else None,
            "content_char_count": len(response.content or ""),
            "usage": usage,
            "reported_model": response.raw_model,
            "provider_error": None,
            "observation_json": observation_json,
            "observation_hash": observation_hash,
            "messages_hash": messages_hash,
            "tool_schema_hash": action_schema_hash,
            "evidence_content_hash": evidence_content_hash,
            "visited_mask_hash": visited_mask_hash,
            "measurement_history_count": measurement_history_count,
            "state_record_count": _state_record_count(observation),
            "transport_audit": copy.deepcopy(getattr(provider, "last_call_audit", None)),
        })
        round_results: list[dict[str, Any]] = []
        if not response.tool_calls:
            invalid_streak += 1
            result = env.record_invalid("E_NO_TOOL_CALL", "provider returned no tool call")
            env.annotate_event(
                result,
                provider_round_index=round_index,
                tool_call_index=0,
                arguments={},
                observation_json=observation_json,
                observation_hash=observation_hash,
                messages_hash=messages_hash,
                tool_schema_hash=action_schema_hash,
                evidence_content_hash=evidence_content_hash,
                visited_mask_hash=visited_mask_hash,
                measurement_history_count=measurement_history_count,
                state_record_count=_state_record_count(observation),
                context_tokens_to_model=usage["input_tokens"],
            )
            round_results.append(result)
        else:
            basis_event_index = len(env.events)
            for tool_index, raw in enumerate(response.tool_calls):
                event_index_before = len(env.events)
                raw_arguments: dict[str, Any] = {}
                core_arguments: dict[str, Any] = {}
                raw_record: dict[str, Any] = {}
                name = str(raw.get("name"))
                try:
                    raw_arguments = json.loads(raw["arguments"])
                    if not isinstance(raw_arguments, dict):
                        raise ValueError("arguments not an object")
                    core_arguments, raw_record = split_integrated_arguments(raw_arguments)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    pass
                core_action = {
                    "name": name,
                    "arguments": json.dumps(core_arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                }
                decision = _decision_record(
                    raw_record=raw_record,
                    observation_hash=observation_hash,
                    core_action=core_action,
                    main_round_index=round_index,
                    tool_call_index=tool_index,
                    event_index_before_action=basis_event_index,
                    reported_model=response.raw_model,
                    expected_model=str(config["model"]),
                )
                decision["decision_record_index"] = len(decision_records)
                decision["event_index_before_execution"] = event_index_before
                decision_records.append(decision)
                try:
                    if not isinstance(raw_arguments, dict):
                        raise ValueError("arguments not an object")
                    _validate_arguments(name, core_arguments)
                    if name == "move":
                        result = env.move(core_arguments["direction"], core_arguments["steps"])
                    elif name == "analyze":
                        result = env.analyze()
                    elif name == "submit":
                        result = env.submit(_submission(core_arguments))
                    else:  # pragma: no cover
                        raise ValueError("unknown tool")
                    invalid_streak = 0
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    result = env.record_invalid("E_SCHEMA", str(exc), name or "unknown")
                    invalid_streak += 1
                env.annotate_event(
                    result,
                    provider_round_index=round_index,
                    tool_call_index=tool_index,
                    arguments=core_arguments,
                    observation_json=observation_json,
                    observation_hash=observation_hash,
                    messages_hash=messages_hash,
                    tool_schema_hash=action_schema_hash,
                    evidence_content_hash=evidence_content_hash,
                    visited_mask_hash=visited_mask_hash,
                    measurement_history_count=measurement_history_count,
                    state_record_count=_state_record_count(observation),
                    context_tokens_to_model=usage["input_tokens"],
                )
                decision["linked_step_index"] = len(env.events) - 1 if len(env.events) > event_index_before else None
                decision["action_valid"] = bool(result.get("ok"))
                decision["action_result_code"] = result.get("code")
                round_results.append(result)
                if not result.get("ok", False) or env.terminated:
                    break
        last_results = round_results
        if invalid_streak >= 3 and not env.terminated:
            env.terminated, env.termination_reason = True, "consecutive_invalid"
        if env.turn >= env.max_turns and not env.terminated:
            env.terminated, env.termination_reason = True, "turn_limit"
        if env.terminated and env.events:
            env.events[-1]["termination_after_step"] = env.termination_reason
    identities = [item["reported_model"] for item in main_calls if item.get("reported_model")]
    return {
        "data_contract_version": DATA_CONTRACT_VERSION,
        "prompt_id": str(config.get("prompt_id_v19") or PROMPT_ID),
        "decision_record_protocol_id": PROTOCOL["protocol_id"],
        "decision_record_protocol_hash": stable_hash(PROTOCOL),
        "decision_record_provider_settings_hash": stable_hash({
            "transport": "integrated_flat_tool_arguments",
            "model": config.get("model"),
            "tool_schema_hash": action_schema_hash,
        }),
        "main_action_schema_hash": action_schema_hash,
        "provider_calls": main_calls,
        "decision_record_calls": decision_records,
        "main_token_usage": totals,
        "decision_record_token_usage": {"input_tokens": 0, "output_tokens": 0},
        "token_usage": totals,
        "reported_models": sorted(set(identities)),
        "decision_record_reported_models": sorted(set(identities)),
        "model_identity_match": _identity_matches(str(config["model"]), identities),
        "decision_record_model_identity_match": _identity_matches(str(config["model"]), identities),
        "presentation_mode": presentation_mode,
        "decision_record_requested": True,
        "decision_record_fed_back": False,
        "decision_record_integrated": True,
        "chain_of_thought_requested": False,
        "chain_of_thought_stored": False,
        "behavior_chain_sha256": stable_hash([behavior_event(event) for event in env.events]),
    }
