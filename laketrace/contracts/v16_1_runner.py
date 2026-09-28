from __future__ import annotations

import copy
import json
import math
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
from laketrace.env.core import Environment
from laketrace.perception import PRESENTATION_MODES
from laketrace.providers.openai_compatible import ProviderResponse
from laketrace.util import stable_hash


DATA_CONTRACT_VERSION = "v16.1"
PROMPT_ID = "laketrace-v4-plus-v16.1-structured-state"
HYPOTHESIS_CLASSES = {
    None,
    "external_point",
    "external_diffuse",
    "internal_release",
    "physical_resuspension",
}
CONFIDENCE_FIELDS = tuple(f"confidence_l{level}" for level in range(1, 6))


def _tools_v16_1() -> list[dict[str, Any]]:
    """Extend only the payload needed for registered data collection.

    The action names, move schema, action costs, environment semantics, and
    scoring payload are inherited unchanged from the frozen v4 protocol.
    """

    tools = copy.deepcopy(TOOLS)
    by_name = {item["function"]["name"]: item["function"] for item in tools}
    analyze = by_name["analyze"]
    analyze["description"] = (
        "Calculate observed maxima over cumulative measurements and record the "
        "agent's current structured source hypothesis. This is not a request "
        "for natural-language reasoning."
    )
    analyze["parameters"] = {
        "type": "object",
        "properties": {
            "hypothesis_source_x": {"type": ["number", "null"]},
            "hypothesis_source_y": {"type": ["number", "null"]},
            "hypothesis_source_z": {"type": ["number", "null"]},
            "hypothesis_source_class": {
                "type": ["string", "null"],
                "enum": [
                    "external_point",
                    "external_diffuse",
                    "internal_release",
                    "physical_resuspension",
                    None,
                ],
            },
            "hypothesis_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": [
            "hypothesis_source_x",
            "hypothesis_source_y",
            "hypothesis_source_z",
            "hypothesis_source_class",
            "hypothesis_confidence",
        ],
        "additionalProperties": False,
    }
    submit = by_name["submit"]
    submit["description"] = (
        "Submit the five independent exam answers, plus one self-reported "
        "0-to-1 confidence for each level, and end the episode. Confidence is "
        "recorded for calibration analysis and does not change scoring."
    )
    properties = submit["parameters"]["properties"]
    for field in CONFIDENCE_FIELDS:
        properties[field] = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    submit["parameters"]["required"] = list(submit["parameters"]["required"]) + list(CONFIDENCE_FIELDS)
    return tools


TOOLS_V16_1 = _tools_v16_1()
SYSTEM_PROMPT_V16_1 = SYSTEM_PROMPT + """

Data-recording addendum (v16.1): analyze requires a compact, machine-readable
current source hypothesis (three coordinates or three nulls, one listed class
or null, and confidence from 0 to 1). submit requires one confidence from 0 to
1 for each scored level. These values report your present belief only; they do
not change the action budget or scoring. Do not provide or reveal chain-of-thought
or a natural-language rationale."""


def _position_or_null(arguments: dict[str, Any], prefix: str) -> dict[str, float] | None:
    values = [arguments.get(f"{prefix}_{axis}") for axis in "xyz"]
    if all(value is None for value in values):
        return None
    if any(value is None or isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"{prefix} coordinate must be three finite numbers or three nulls")
    values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{prefix} coordinate must be three finite numbers or three nulls")
    return dict(zip("xyz", values))


def _confidence(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number from 0 to 1")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be a number from 0 to 1")
    return result


def _validate_arguments_v16_1(name: str, arguments: dict[str, Any]) -> None:
    if name == "analyze":
        required = {
            "hypothesis_source_x",
            "hypothesis_source_y",
            "hypothesis_source_z",
            "hypothesis_source_class",
            "hypothesis_confidence",
        }
        if set(arguments) != required:
            raise ValueError("analyze fields do not match the v16.1 structured-state schema")
        _position_or_null(arguments, "hypothesis_source")
        if arguments["hypothesis_source_class"] not in HYPOTHESIS_CLASSES:
            raise ValueError("hypothesis_source_class is outside the registered enum")
        _confidence(arguments["hypothesis_confidence"], "hypothesis_confidence")
        return
    if name == "submit":
        if not set(CONFIDENCE_FIELDS) <= set(arguments):
            raise ValueError("submit requires five v16.1 confidence fields")
        base = {key: value for key, value in arguments.items() if key not in CONFIDENCE_FIELDS}
        _validate_arguments(name, base)
        for field in CONFIDENCE_FIELDS:
            _confidence(arguments[field], field)
        return
    _validate_arguments(name, arguments)


def _submission_v16_1(arguments: dict[str, Any]) -> dict[str, Any]:
    base = {key: value for key, value in arguments.items() if key not in CONFIDENCE_FIELDS}
    answer = _submission(base)
    answer["confidence"] = {
        f"L{level}": _confidence(arguments[f"confidence_l{level}"], f"confidence_l{level}")
        for level in range(1, 6)
    }
    return answer


def _hypothesis(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_position": _position_or_null(arguments, "hypothesis_source"),
        "source_class": arguments["hypothesis_source_class"],
        "confidence": _confidence(arguments["hypothesis_confidence"], "hypothesis_confidence"),
    }


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


AUDIT_EVENT_FIELDS = {
    "provider_round_index",
    "tool_call_index",
    "step_index",
    "sim_time_s",
    "node_id",
    "x_m",
    "y_m",
    "z_m",
    "action_type",
    "action_payload_json",
    "action_valid",
    "invalid_reason",
    "observation_json",
    "observation_hash",
    "state_hash",
    "messages_hash",
    "tool_schema_hash",
    "evidence_content_hash",
    "visited_mask_hash",
    "measurement_history_count",
    "budget_endurance_remaining_s",
    "budget_distance_remaining_m",
    "budget_turns_remaining",
    "visited_nodes_count",
    "state_record_count",
    "context_tokens_to_model",
    "hypothesis_source_x_m",
    "hypothesis_source_y_m",
    "hypothesis_source_z_m",
    "hypothesis_source_class",
    "hypothesis_confidence",
    "hypothesis_basis_observation_hash",
    "is_repeated_action",
    "termination_after_step",
}


def behavior_event(event: dict[str, Any]) -> dict[str, Any]:
    """Remove v16.1 audit-only fields for deterministic legacy replay."""

    return {key: value for key, value in event.items() if key not in AUDIT_EVENT_FIELDS}


class InstrumentedEnvironment(Environment):
    """Behavior-identical environment that annotates, but never changes, actions."""

    def _detect_repetition(self, result: dict[str, Any]) -> None:
        signature = self._result_signature(result)
        result["is_repeated_action"] = signature == self._last_result_signature
        super()._detect_repetition(result)

    def annotate_event(
        self,
        result: dict[str, Any],
        *,
        provider_round_index: int,
        tool_call_index: int,
        arguments: dict[str, Any],
        observation_json: str,
        observation_hash: str,
        messages_hash: str,
        tool_schema_hash: str,
        evidence_content_hash: str,
        visited_mask_hash: str,
        measurement_history_count: int,
        state_record_count: int,
        context_tokens_to_model: int,
        hypothesis: dict[str, Any] | None = None,
    ) -> None:
        if result not in self.events:
            return
        position = self.graph.positions[self.current]
        state = self.observation()["cumulative_measurements"]
        source = (hypothesis or {}).get("source_position") or {}
        raw_action = result.get("action")
        requested = result.get("requested_action")
        action = requested if raw_action in {"invalid", "infeasible"} and requested else raw_action
        result.update({
            "provider_round_index": provider_round_index,
            "tool_call_index": tool_call_index,
            "step_index": len(self.events) - 1,
            "sim_time_s": self.budget.used_s,
            "node_id": self.current,
            "x_m": position[0],
            "y_m": position[1],
            "z_m": position[2],
            "action_type": "analyse" if action == "analyze" else (action or "invalid"),
            "action_payload_json": json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "action_valid": bool(result.get("ok")),
            "invalid_reason": result.get("code") or result.get("detail"),
            "observation_json": observation_json,
            "observation_hash": observation_hash,
            "state_hash": stable_hash(state),
            "messages_hash": messages_hash,
            "tool_schema_hash": tool_schema_hash,
            "evidence_content_hash": evidence_content_hash,
            "visited_mask_hash": visited_mask_hash,
            "measurement_history_count": measurement_history_count,
            "budget_endurance_remaining_s": self.budget.remaining_s,
            "budget_distance_remaining_m": self.budget.remaining_m,
            "budget_turns_remaining": max(0, self.max_turns - self.turn),
            "visited_nodes_count": len(self.measurements),
            "state_record_count": state_record_count,
            "context_tokens_to_model": int(context_tokens_to_model),
            "hypothesis_source_x_m": source.get("x"),
            "hypothesis_source_y_m": source.get("y"),
            "hypothesis_source_z_m": source.get("z"),
            "hypothesis_source_class": (hypothesis or {}).get("source_class"),
            "hypothesis_confidence": (hypothesis or {}).get("confidence"),
            "hypothesis_basis_observation_hash": observation_hash if hypothesis is not None else None,
            "is_repeated_action": bool(result.get("is_repeated_action", False)),
            "termination_after_step": self.termination_reason if self.terminated else None,
        })


class _ScriptedProviderV16_1:
    """Three-round zero-cost producer for contract and replay acceptance."""

    def __init__(self, config: dict[str, Any]):
        self.turn = 0
        self.model = str(config.get("model") or "scripted-contract-v16.1")

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ProviderResponse:
        self.turn += 1
        observation = json.loads(messages[-1]["content"].split("\n", 1)[-1])
        if self.turn == 1:
            available = [name for name, item in observation["neighbors"].items() if item["reachable"]]
            if available:
                call = {"id": "v16-move", "name": "move", "arguments": json.dumps({"direction": available[0], "steps": 1})}
            else:
                self.turn += 1
                call = {"id": "v16-analyse", "name": "analyze", "arguments": json.dumps({
                    "hypothesis_source_x": None,
                    "hypothesis_source_y": None,
                    "hypothesis_source_z": None,
                    "hypothesis_source_class": None,
                    "hypothesis_confidence": 0.0,
                })}
        elif self.turn == 2:
            call = {"id": "v16-analyse", "name": "analyze", "arguments": json.dumps({
                "hypothesis_source_x": None,
                "hypothesis_source_y": None,
                "hypothesis_source_z": None,
                "hypothesis_source_class": None,
                "hypothesis_confidence": 0.0,
            })}
        else:
            call = {"id": "v16-submit", "name": "submit", "arguments": json.dumps({
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
                "confidence_l1": 0.5,
                "confidence_l2": 0.0,
                "confidence_l3": 0.0,
                "confidence_l4": 0.0,
                "confidence_l5": 0.5,
            })}
        return ProviderResponse([call], None, {"input_tokens": 0, "output_tokens": 0}, self.model)


def run_model_v16_1(env: InstrumentedEnvironment, config: dict[str, Any]) -> dict[str, Any]:
    """Run the frozen action environment with v16.1 audit-only collection."""

    validate_model_config(config)
    provider = _ScriptedProviderV16_1(config) if config["adapter"] == "scripted" else _provider(config)
    presentation_mode = str(config.get("presentation_mode", "records"))
    if presentation_mode not in PRESENTATION_MODES:
        raise ValueError(f"presentation_mode must be one of {PRESENTATION_MODES}")
    invalid_streak = 0
    calls: list[dict[str, Any]] = []
    last_results: list[dict[str, Any]] = []
    totals = {"input_tokens": 0, "output_tokens": 0}
    prompt = str(config.get("system_prompt_v16_1") or SYSTEM_PROMPT_V16_1)
    tools_hash = stable_hash(TOOLS_V16_1)
    while not env.terminated:
        engineering_cap = config.get("engineering_max_provider_rounds")
        if engineering_cap is not None and len(calls) >= int(engineering_cap):
            if bool(config.get("science_eligible", False)):
                raise ValueError("engineering_max_provider_rounds is forbidden for science-eligible models")
            env.terminated, env.termination_reason = True, "engineering_round_cap"
            if env.events:
                env.events[-1]["termination_after_step"] = env.termination_reason
            break
        canonical_observation = env.observation(first=not calls)
        canonical_rows = canonical_observation["cumulative_measurements"]
        evidence_content_hash = stable_hash({
            "cumulative_measurements": canonical_rows,
            "candidates": canonical_observation["candidates"],
            "available_channels": canonical_observation["available_channels"],
        })
        visited_mask_hash = stable_hash(sorted(int(row["node_index"]) for row in canonical_rows))
        measurement_history_count = sum(int(row["sample_count"]) for row in canonical_rows)
        observation = _present_observation(canonical_observation, presentation_mode, last_results)
        observation_json = json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "CURRENT OBSERVATION\n" + observation_json},
        ]
        observation_hash = stable_hash(observation)
        messages_hash = stable_hash(messages)
        round_index = len(calls)
        try:
            response = provider.complete(messages, TOOLS_V16_1)
        except RuntimeError as exc:
            env.terminated, env.termination_reason = True, "infra_failure"
            calls.append({
                "provider_round_index": round_index,
                "tool_calls": [],
                "content": None,
                "content_sha256": None,
                "content_char_count": 0,
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "reported_model": None,
                "provider_error": str(exc),
                "observation_json": observation_json,
                "observation_hash": observation_hash,
                "messages_hash": messages_hash,
                "tool_schema_hash": tools_hash,
                "evidence_content_hash": evidence_content_hash,
                "visited_mask_hash": visited_mask_hash,
                "measurement_history_count": measurement_history_count,
                "state_record_count": _state_record_count(observation),
            })
            if env.events:
                env.events[-1]["termination_after_step"] = env.termination_reason
            break
        totals = {key: totals[key] + int(response.usage.get(key, 0)) for key in totals}
        provider_record = {
            "provider_round_index": round_index,
            "tool_calls": response.tool_calls,
            # Natural-language model text is not retained.  Tool calls and the
            # registered structured hypothesis are sufficient for auditing;
            # keeping only a digest/length detects unexpected prose without
            # turning it into a chain-of-thought dataset.
            "content": None,
            "content_sha256": stable_hash(response.content) if response.content else None,
            "content_char_count": len(response.content or ""),
            "usage": response.usage,
            "reported_model": response.raw_model,
            "observation_json": observation_json,
            "observation_hash": observation_hash,
            "messages_hash": messages_hash,
            "tool_schema_hash": tools_hash,
            "evidence_content_hash": evidence_content_hash,
            "visited_mask_hash": visited_mask_hash,
            "measurement_history_count": measurement_history_count,
            "state_record_count": _state_record_count(observation),
        }
        calls.append(provider_record)
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
                tool_schema_hash=tools_hash,
                evidence_content_hash=evidence_content_hash,
                visited_mask_hash=visited_mask_hash,
                measurement_history_count=measurement_history_count,
                state_record_count=_state_record_count(observation),
                context_tokens_to_model=int(response.usage.get("input_tokens", 0)),
            )
            round_results.append(result)
        else:
            for tool_index, raw in enumerate(response.tool_calls):
                arguments: dict[str, Any] = {}
                hypothesis = None
                try:
                    arguments = json.loads(raw["arguments"])
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments not an object")
                    name = str(raw.get("name"))
                    _validate_arguments_v16_1(name, arguments)
                    if name == "move":
                        result = env.move(arguments["direction"], arguments["steps"])
                    elif name == "analyze":
                        hypothesis = _hypothesis(arguments)
                        result = env.analyze()
                    elif name == "submit":
                        result = env.submit(_submission_v16_1(arguments))
                    else:  # pragma: no cover - validation already rejects it
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
                    tool_schema_hash=tools_hash,
                    evidence_content_hash=evidence_content_hash,
                    visited_mask_hash=visited_mask_hash,
                    measurement_history_count=measurement_history_count,
                    state_record_count=_state_record_count(observation),
                    context_tokens_to_model=int(response.usage.get("input_tokens", 0)),
                    hypothesis=hypothesis,
                )
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
    identities = [item["reported_model"] for item in calls if item.get("reported_model")]
    return {
        "data_contract_version": DATA_CONTRACT_VERSION,
        "prompt_id": str(config.get("prompt_id") or PROMPT_ID),
        "provider_calls": calls,
        "token_usage": totals,
        "reported_models": sorted(set(identities)),
        "model_identity_match": _identity_matches(str(config["model"]), identities),
        "presentation_mode": presentation_mode,
        "behavior_chain_sha256": stable_hash([behavior_event(event) for event in env.events]),
        "chain_of_thought_requested": False,
        "chain_of_thought_stored": False,
    }


def replay_episode_v16_1(store: Any, episode: dict[str, Any]) -> bool:
    """Replay action semantics while ignoring audit-only annotations."""

    env = Environment(store, episode["instance_id"], int(episode["seed"]))
    actual = []
    for event in episode.get("events") or []:
        action = event.get("action")
        if action == "move":
            result = env.move(event["direction"], int(event["steps"]))
        elif action == "analyze":
            result = env.analyze()
        elif action == "submit":
            result = env.submit(event["answer"])
        elif action == "infeasible" and event.get("requested_action") == "move":
            arguments = event.get("arguments") or {}
            result = env.move(arguments["direction"], int(arguments["steps"]))
        elif action == "infeasible" and event.get("requested_action") == "analyze":
            result = env.analyze()
        elif action == "invalid":
            result = env.record_invalid(
                event.get("code") or "E_SCHEMA",
                event.get("detail") or "",
                event.get("requested_action"),
            )
        else:
            continue
        actual.append(result)
    expected = [
        behavior_event(event)
        for event in episode.get("events") or []
        if event.get("action") in {"move", "analyze", "submit", "invalid"}
        or (event.get("action") == "infeasible" and event.get("requested_action") in {"move", "analyze"})
    ]
    return stable_hash(actual) == stable_hash(expected)
