from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from laketrace.agents.model import (
    SYSTEM_PROMPT,
    TOOLS,
    _identity_matches,
    _present_observation,
    _provider,
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
PROTOCOL_PATH = ROOT / "configs" / "v18" / "decision_record_protocol_v4.json"
DATA_CONTRACT_VERSION = "v18.decision-record.4"
PROMPT_ID = "laketrace-v4-actions-plus-v18-isolated-decision-record"


def load_decision_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_id") != "laketrace-v18-decision-record-v4":
        raise ValueError("unexpected decision-record protocol id")
    return value


DECISION_PROTOCOL = load_decision_protocol()
DECISION_TOOL = DECISION_PROTOCOL["decision_tool"]
DECISION_RECORD_FIELDS = tuple(
    DECISION_TOOL["function"]["parameters"]["required"]
)
DECISION_PROMPT = """You are recording a concise externalized decision state for an action
that has already been selected. Use only the decision_record tool. Do not choose
or change the navigation action. Use only the supplied pre-action observation and
chosen action. Put only comma-separated measured node identifiers in evidence_refs. A source coordinate must be
three finite numbers or three nulls. Summarize the current conclusion and evidence;
do not provide private chain-of-thought or hidden step-by-step reasoning."""


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{field} must be a finite number")
    return output


def validate_decision_record(arguments: dict[str, Any]) -> dict[str, Any]:
    if set(arguments) != set(DECISION_RECORD_FIELDS):
        raise ValueError("decision_record fields do not match the registered flat schema")
    output = dict(arguments)
    for field in ("objective", "evidence_refs", "action_rationale", "expected_evidence", "main_uncertainty"):
        value = arguments[field]
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string")
    coordinates = [arguments[f"source_{axis}"] for axis in "xyz"]
    if not all(value is None for value in coordinates):
        if any(value is None for value in coordinates):
            raise ValueError("source coordinate must be three numbers or three nulls")
        for axis, value in zip("xyz", coordinates):
            output[f"source_{axis}"] = _finite_number(value, f"source_{axis}")
    allowed_classes = {
        None,
        "external_point",
        "external_diffuse",
        "internal_release",
        "physical_resuspension",
    }
    if arguments["source_class"] not in allowed_classes:
        raise ValueError("source_class is outside the registered enum")
    confidence = _finite_number(arguments["confidence"], "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be from 0 to 1")
    output["confidence"] = confidence
    return output


class _ScriptedActionProvider:
    """Zero-cost three-action producer used by contract tests."""

    def __init__(self, config: dict[str, Any]):
        self.turn = 0
        self.model = str(config.get("model") or "scripted-v18")
        self.multi_call = bool(config.get("scripted_multi_call", False))

    def complete(self, messages: list[dict[str, Any]], _tools: list[dict[str, Any]]) -> ProviderResponse:
        self.turn += 1
        observation = json.loads(messages[-1]["content"].split("\n", 1)[-1])
        if self.turn == 1:
            available = [name for name, edge in observation["neighbors"].items() if edge["reachable"]]
            call = {
                "id": "action-move",
                "name": "move",
                "arguments": json.dumps({"direction": available[0], "steps": 1}),
            }
            if self.multi_call:
                return ProviderResponse(
                    [call, {"id": "action-analyze-same-round", "name": "analyze", "arguments": "{}"}],
                    None, {"input_tokens": 0, "output_tokens": 0}, self.model,
                )
        elif self.turn == 2:
            call = {"id": "action-analyze", "name": "analyze", "arguments": "{}"}
        else:
            call = {
                "id": "action-submit",
                "name": "submit",
                "arguments": json.dumps({
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
                }),
            }
        return ProviderResponse([call], None, {"input_tokens": 0, "output_tokens": 0}, self.model)


class _ScriptedDecisionProvider:
    def __init__(self, config: dict[str, Any]):
        self.model = str(config.get("model") or "scripted-v18")
        self.fail = bool(config.get("scripted_decision_failure", False))

    def complete(self, messages: list[dict[str, Any]], _tools: list[dict[str, Any]]) -> ProviderResponse:
        if self.fail:
            raise RuntimeError("scripted decision-record failure")
        user = json.loads(messages[-1]["content"].split("\n", 1)[-1])
        observation = user["pre_action_observation"]
        node_id = int(observation["position"]["node_index"])
        record = {
            "objective": "execute the already selected action",
            "evidence_refs": str(node_id),
            "source_x": None,
            "source_y": None,
            "source_z": None,
            "source_class": None,
            "action_rationale": "the current measured state supports this selected action",
            "expected_evidence": "obtain the registered action outcome",
            "confidence": 0.5,
            "main_uncertainty": "insufficient measured spatial coverage",
        }
        call = {"id": "decision-record", "name": "decision_record", "arguments": json.dumps(record)}
        return ProviderResponse([call], None, {"input_tokens": 0, "output_tokens": 0}, self.model)


def decision_record_provider(config: dict[str, Any]):
    """Build only the isolated readout provider from registered YAML fields."""

    if config["adapter"] == "scripted":
        return _ScriptedDecisionProvider(config)
    readout_config = copy.deepcopy(config)
    readout_config["max_output_tokens"] = int(config.get("decision_record_max_output_tokens", 384))
    if config.get("decision_record_extra_body") is not None:
        if not isinstance(config["decision_record_extra_body"], dict):
            raise ValueError("decision_record_extra_body must be an object")
        readout_config["extra_body"] = copy.deepcopy(config["decision_record_extra_body"])
    return AuditedOpenAICompatibleProvider(readout_config)


def _providers(config: dict[str, Any]):
    if config["adapter"] == "scripted":
        return _ScriptedActionProvider(config), _ScriptedDecisionProvider(config)
    main = AuditedOpenAICompatibleProvider(config)
    return main, decision_record_provider(config)


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


def _collect_decision_record(
    provider: Any,
    *,
    observation: dict[str, Any],
    observation_hash: str,
    action_call: dict[str, Any],
    main_round_index: int,
    tool_call_index: int,
    event_index_before_action: int,
    expected_model: str,
    schema_retries: int,
) -> dict[str, Any]:
    payload = {
        "pre_action_observation": observation,
        "already_chosen_action": {
            "name": action_call.get("name"),
            "arguments": action_call.get("arguments"),
        },
    }
    messages = [
        {"role": "system", "content": DECISION_PROMPT},
        {"role": "user", "content": "DECISION RECORD INPUT\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]
    base = {
        "decision_record_index": None,
        "main_provider_round_index": main_round_index,
        "tool_call_index": tool_call_index,
        "event_index_before_action": event_index_before_action,
        "basis_observation_hash": observation_hash,
        "chosen_action_name": action_call.get("name"),
        "chosen_action_arguments": action_call.get("arguments"),
        "chosen_action_hash": stable_hash(action_call),
        "readout_prompt_hash": stable_hash(messages),
        "readout_schema_hash": stable_hash(DECISION_TOOL),
        "timing": "after_action_choice_before_action_outcome",
        "fed_back_to_agent": False,
        "truth_visible": False,
        "status": None,
        "error": None,
        "reported_model": None,
        "model_identity_match": False,
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "record": None,
        "raw_tool_calls": [],
        "readout_attempts": [],
        "readout_attempt_count": 0,
        "unexpected_content_sha256": None,
        "unexpected_content_char_count": 0,
    }
    identities: list[str] = []
    failures: list[str] = []
    for attempt_index in range(max(0, int(schema_retries)) + 1):
        attempt = {
            "attempt_index": attempt_index,
            "status": None,
            "error": None,
            "reported_model": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "raw_tool_calls": [],
            "unexpected_content_sha256": None,
            "unexpected_content_char_count": 0,
        }
        try:
            response = provider.complete(messages, [DECISION_TOOL])
            attempt["transport_audit"] = copy.deepcopy(getattr(provider, "last_call_audit", None))
            attempt["reported_model"] = response.raw_model
            if response.raw_model:
                identities.append(response.raw_model)
            attempt["usage"] = {
                "input_tokens": int(response.usage.get("input_tokens", 0)),
                "output_tokens": int(response.usage.get("output_tokens", 0)),
            }
            base["usage"] = {
                key: int(base["usage"][key]) + int(attempt["usage"][key])
                for key in base["usage"]
            }
            attempt["raw_tool_calls"] = response.tool_calls
            attempt["unexpected_content_sha256"] = stable_hash(response.content) if response.content else None
            attempt["unexpected_content_char_count"] = len(response.content or "")
            candidates = [call for call in response.tool_calls if call.get("name") == "decision_record"]
            local_failures = []
            for candidate in candidates:
                try:
                    raw_arguments = candidate.get("arguments") or "{}"
                    arguments = json.loads(raw_arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("decision_record arguments are not an object")
                    base["record"] = validate_decision_record(arguments)
                    base["status"] = "ok"
                    base["selected_readout_attempt_index"] = attempt_index
                    base["selected_raw_tool_call_index"] = response.tool_calls.index(candidate)
                    attempt["status"] = "ok"
                    break
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    local_failures.append(str(exc))
            if base["status"] == "ok":
                base["raw_tool_calls"] = response.tool_calls
                base["unexpected_content_sha256"] = attempt["unexpected_content_sha256"]
                base["unexpected_content_char_count"] = attempt["unexpected_content_char_count"]
                base["readout_attempts"].append(attempt)
                break
            detail = "; ".join(local_failures) if local_failures else "no decision_record tool call"
            attempt["status"] = "schema_failure"
            attempt["error"] = f"no schema-valid decision_record call: {detail}"
            failures.append(attempt["error"])
        except RuntimeError as exc:
            attempt["transport_audit"] = copy.deepcopy(getattr(provider, "last_call_audit", None))
            attempt["status"] = "infra_failure"
            attempt["error"] = str(exc)
            failures.append(str(exc))
            base["readout_attempts"].append(attempt)
            break
        base["readout_attempts"].append(attempt)
    base["readout_attempt_count"] = len(base["readout_attempts"])
    base["reported_model"] = identities[-1] if identities else None
    base["model_identity_match"] = _identity_matches(expected_model, identities)
    if base["status"] != "ok":
        last = base["readout_attempts"][-1] if base["readout_attempts"] else {}
        base["status"] = last.get("status") or "schema_failure"
        base["error"] = "; ".join(failures) if failures else "decision record failed"
        base["raw_tool_calls"] = last.get("raw_tool_calls") or []
        base["unexpected_content_sha256"] = last.get("unexpected_content_sha256")
        base["unexpected_content_char_count"] = last.get("unexpected_content_char_count", 0)
    return base


def run_model_v18(env: InstrumentedEnvironment, config: dict[str, Any]) -> dict[str, Any]:
    """Run frozen v4 actions plus a non-blocking isolated per-action readout."""

    validate_model_config(config)
    try:
        schema_retries = int(config.get("decision_record_schema_retries", 2))
    except (TypeError, ValueError) as exc:
        raise ValueError("decision_record_schema_retries must equal the registered value 2") from exc
    if schema_retries != 2:
        raise ValueError("decision_record_schema_retries must equal the registered value 2")
    main_provider, readout_provider = _providers(config)
    presentation_mode = str(config.get("presentation_mode", "records"))
    if presentation_mode not in PRESENTATION_MODES:
        raise ValueError(f"presentation_mode must be one of {PRESENTATION_MODES}")
    prompt = str(config.get("system_prompt") or SYSTEM_PROMPT)
    action_schema_hash = stable_hash(TOOLS)
    invalid_streak = 0
    main_calls: list[dict[str, Any]] = []
    decision_records: list[dict[str, Any]] = []
    last_results: list[dict[str, Any]] = []
    main_totals = {"input_tokens": 0, "output_tokens": 0}
    readout_totals = {"input_tokens": 0, "output_tokens": 0}
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
            response = main_provider.complete(messages, TOOLS)
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
                "transport_audit": copy.deepcopy(getattr(main_provider, "last_call_audit", None)),
            })
            break
        usage = {
            "input_tokens": int(response.usage.get("input_tokens", 0)),
            "output_tokens": int(response.usage.get("output_tokens", 0)),
        }
        main_totals = {key: main_totals[key] + usage[key] for key in main_totals}
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
            "transport_audit": copy.deepcopy(getattr(main_provider, "last_call_audit", None)),
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
                decision = _collect_decision_record(
                    readout_provider,
                    observation=observation,
                    observation_hash=observation_hash,
                    action_call=raw,
                    main_round_index=round_index,
                    tool_call_index=tool_index,
                    event_index_before_action=basis_event_index,
                    expected_model=str(config["model"]),
                    schema_retries=schema_retries,
                )
                decision["decision_record_index"] = len(decision_records)
                decision["event_index_before_execution"] = event_index_before
                decision_records.append(decision)
                readout_totals = {
                    key: readout_totals[key] + int(decision["usage"].get(key, 0))
                    for key in readout_totals
                }
                arguments: dict[str, Any] = {}
                try:
                    arguments = json.loads(raw["arguments"])
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments not an object")
                    name = str(raw.get("name"))
                    _validate_arguments(name, arguments)
                    if name == "move":
                        result = env.move(arguments["direction"], arguments["steps"])
                    elif name == "analyze":
                        result = env.analyze()
                    elif name == "submit":
                        result = env.submit(_submission(arguments))
                    else:  # pragma: no cover
                        raise ValueError("unknown tool")
                    invalid_streak = 0
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    result = env.record_invalid("E_SCHEMA", str(exc), str(raw.get("name") or "unknown"))
                    invalid_streak += 1
                env.annotate_event(
                    result,
                    provider_round_index=round_index,
                    tool_call_index=tool_index,
                    arguments=arguments,
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
    main_identities = [item["reported_model"] for item in main_calls if item.get("reported_model")]
    readout_identities = [item["reported_model"] for item in decision_records if item.get("reported_model")]
    total_usage = {
        key: main_totals[key] + readout_totals[key]
        for key in main_totals
    }
    return {
        "data_contract_version": DATA_CONTRACT_VERSION,
        "prompt_id": str(config.get("prompt_id") or PROMPT_ID),
        "decision_record_protocol_id": DECISION_PROTOCOL["protocol_id"],
        "decision_record_protocol_hash": stable_hash(DECISION_PROTOCOL),
        "decision_record_provider_settings_hash": stable_hash({
            "model": config.get("model"),
            "max_output_tokens": int(config.get("decision_record_max_output_tokens", 384)),
            "extra_body": config.get("decision_record_extra_body", config.get("extra_body") or {}),
        }),
        "main_action_schema_hash": action_schema_hash,
        "provider_calls": main_calls,
        "decision_record_calls": decision_records,
        "main_token_usage": main_totals,
        "decision_record_token_usage": readout_totals,
        "token_usage": total_usage,
        "reported_models": sorted(set(main_identities)),
        "decision_record_reported_models": sorted(set(readout_identities)),
        "model_identity_match": _identity_matches(str(config["model"]), main_identities),
        "decision_record_model_identity_match": _identity_matches(str(config["model"]), readout_identities),
        "presentation_mode": presentation_mode,
        "decision_record_requested": True,
        "decision_record_fed_back": False,
        "chain_of_thought_requested": False,
        "chain_of_thought_stored": False,
        "behavior_chain_sha256": stable_hash([behavior_event(event) for event in env.events]),
    }
