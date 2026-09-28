from __future__ import annotations

import json
import math
import re
from typing import Any, Callable

from laketrace.agents.model import _submission, validate_model_config
from laketrace.env.v20 import EnvironmentV20
from laketrace.perception.v20_state import derived_summary, render_measurements
from laketrace.providers.openai_compatible import OpenAICompatibleProvider, ProviderResponse
from laketrace.util import stable_hash


RECORD_FIELDS = (
    "intent",
    "evidence_node_ids_csv",
    "confidence",
    "decision_note",
)


def exact_model_identity_matches(expected: str, identities: list[str]) -> bool:
    """V20 binds every successful response to the exact registered snapshot."""

    return bool(identities) and all(value == expected for value in identities)
RECORD_PROPERTIES: dict[str, Any] = {
    "intent": {
        "type": "string",
        "enum": ["explore", "inspect", "revisit", "analyze", "submit"],
    },
    "evidence_node_ids_csv": {
        "type": "string",
        "maxLength": 160,
        "description": (
            "IDs of already measured nodes separated by commas, semicolons, or spaces; "
            "use an empty string for no citation."
        ),
    },
    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "decision_note": {
        "type": "string",
        "maxLength": 240,
        "description": "One short audit note; do not provide private chain-of-thought.",
    },
}


def _parameters(
    properties: dict[str, Any], required: list[str], *, all_of: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {**properties, **RECORD_PROPERTIES},
        "required": [*required, *RECORD_FIELDS],
        "additionalProperties": False,
    }
    if all_of:
        schema["allOf"] = all_of
    return schema


def _all_or_none_coordinate_schema(prefix: str) -> dict[str, Any]:
    """Express the public flat triplet invariant in provider-visible JSON Schema."""

    fields = [f"{prefix}_{axis}" for axis in "xyz"]
    return {
        "oneOf": [
            {
                "properties": {field: {"type": "number"} for field in fields},
                "required": fields,
            },
            {
                "properties": {field: {"type": "null"} for field in fields},
                "required": fields,
            },
        ],
    }


TOOLS_V20 = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move across exactly one currently reachable 3-D neighbor edge and measure on arrival.",
            "parameters": _parameters({
                "direction": {
                    "type": "string",
                    "enum": ["north", "south", "east", "west", "up", "down"],
                },
            }, ["direction"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze",
            "description": "Calculate deterministic summaries over your measured samples only.",
            "parameters": _parameters({}, []),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit the five independent exam answers and end the episode.",
            "parameters": _parameters({
                "anomaly": {"type": "string", "enum": ["anomaly", "no_anomaly"]},
                "plume_x": {
                    "type": ["number", "null"],
                    "description": "A measured anomaly-region x in metres; plume x/y/z are all numbers or all JSON null.",
                },
                "plume_y": {
                    "type": ["number", "null"],
                    "description": "A measured anomaly-region y in metres; plume x/y/z are all numbers or all JSON null.",
                },
                "plume_z": {
                    "type": ["number", "null"],
                    "description": (
                        "Depth-positive-down anomaly-region z; plume x/y/z are all numbers or all JSON null, "
                        "never the text 'None'."
                    ),
                },
                "mechanism": {
                    "type": ["string", "null"],
                    "enum": [
                        "external_point", "external_diffuse", "internal_release",
                        "physical_resuspension", None,
                    ],
                    "description": "Use JSON null if unanswered; never use the text 'None'.",
                },
                "source_x": {
                    "type": ["number", "null"],
                    "description": "Source x in metres; source x/y/z are all numbers or all JSON null.",
                },
                "source_y": {
                    "type": ["number", "null"],
                    "description": "Source y in metres; source x/y/z are all numbers or all JSON null.",
                },
                "source_z": {
                    "type": ["number", "null"],
                    "description": (
                        "Depth-positive-down source z; source x/y/z are all numbers or all JSON null, "
                        "never the text 'None'."
                    ),
                },
                "source_candidate_id": {
                    "type": ["string", "null"],
                    "description": "Optional listed candidate ID; use JSON null if unanswered.",
                },
                "decision": {"type": "string", "enum": ["declare", "cannot_determine"]},
            }, [
                "anomaly", "plume_x", "plume_y", "plume_z", "mechanism",
                "source_x", "source_y", "source_z", "decision",
            ], all_of=[
                _all_or_none_coordinate_schema("plume"),
                _all_or_none_coordinate_schema("source"),
            ]),
        },
    },
]
SUBMIT_TOOL_V20 = [item for item in TOOLS_V20 if item["function"]["name"] == "submit"]


SYSTEM_PROMPT_V20 = """You operate a mobile measurement platform on a discrete native 3-D lake grid.
Use exactly one tool call per response. A move crosses exactly one listed neighbor
edge, then measures at the destination. The next action is chosen only after you
receive the updated observation. Coordinates use depth-positive-down z. Neighbor
metadata gives navigation, cost, and visit history but never an unseen reading.

The navigation topology is static and source-independent. The 3-D visited mask and
measurement state contain only your own samples. Measurement age is elapsed mission
time, not a claim that the field changes. Empty cells have never been measured; no
unvisited value is inferred. Revisits are legal. The main condition is static F1.
The candidates are the complete domain catalog in a source-independent unranked
order; list position carries no priority.
Every exact measured row includes factual bed_depth_m and its surface/middle/bottom
depth_layer so the matching public calibration entry is unambiguous.

Five levels are scored independently: L1 anomaly/no anomaly; L2 one coordinate in
any registered principal-tracer anomaly region; L3 one listed mechanism; L4 a source
coordinate scored by horizontal and vertical distance to the nearest injection node;
L5 declare/cannot_determine. Unsubmitted levels score zero. Wrong and explicit
cannot-determine outcomes are recorded separately. Remaining endurance has no score
value. For both plume and source, provide all three coordinates as JSON numbers or all
three as JSON null; never use the quoted text "None". When episode_mode is submit_only,
submit is the only legal action.

Every tool call also includes four short audit-only fields. intent is generic;
evidence_node_ids_csv may cite measured node IDs only; confidence is 0 to 1;
decision_note is one short conclusion, not private chain-of-thought. These fields do
not change execution, observations, budget, or score."""


class _ScriptedV20Provider:
    def __init__(self, config: dict[str, Any]):
        self.turn = 0
        self.model = str(config.get("model") or "scripted-v20")

    @staticmethod
    def _record(intent: str) -> dict[str, Any]:
        return {
            "intent": intent,
            "evidence_node_ids_csv": "",
            "confidence": 0.1,
            "decision_note": "contract-only deterministic action",
        }

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ProviderResponse:
        self.turn += 1
        observation = json.loads(messages[-1]["content"].split("\n", 1)[-1])
        if observation.get("episode_mode") == "submit_only" or self.turn >= 3:
            arguments = {
                "anomaly": "no_anomaly",
                "plume_x": None, "plume_y": None, "plume_z": None,
                "mechanism": None,
                "source_x": None, "source_y": None, "source_z": None,
                "source_candidate_id": None,
                "decision": "cannot_determine",
                **self._record("submit"),
            }
            call = {"id": "v20-submit", "name": "submit", "arguments": json.dumps(arguments)}
        elif self.turn == 2:
            call = {
                "id": "v20-analyze", "name": "analyze",
                "arguments": json.dumps(self._record("analyze")),
            }
        else:
            direction = next(
                name for name, value in observation["neighbors"].items()
                if value["reachable"] and value.get("affordable", True)
            )
            call = {
                "id": "v20-move", "name": "move",
                "arguments": json.dumps({"direction": direction, **self._record("explore")}),
            }
        return ProviderResponse(
            [call], None, {"input_tokens": 1, "output_tokens": 1}, self.model,
        )


def _provider(config: dict[str, Any]):
    return _ScriptedV20Provider(config) if config["adapter"] == "scripted" else OpenAICompatibleProvider(config)


def _split_arguments(arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {key: value for key, value in arguments.items() if key not in RECORD_FIELDS},
        {key: arguments[key] for key in RECORD_FIELDS if key in arguments},
    )


def _record(
    raw: dict[str, Any], measured_nodes: set[int], *, round_index: int,
    observation_hash: str, action_name: str,
) -> dict[str, Any]:
    status, error = "ok", None
    cited: list[int] = []
    try:
        if set(raw) != set(RECORD_FIELDS):
            raise ValueError("audit fields are missing or extra")
        if raw["intent"] not in {"explore", "inspect", "revisit", "analyze", "submit"}:
            raise ValueError("intent outside registered enum")
        if not isinstance(raw["evidence_node_ids_csv"], str):
            raise ValueError("evidence_node_ids_csv must be a string")
        if len(raw["evidence_node_ids_csv"]) > 160:
            raise ValueError("evidence_node_ids_csv exceeds 160 characters")
        text = raw["evidence_node_ids_csv"].strip()
        parsed = [] if not text else [
            int(value) for value in re.split(r"[,;，；\s]+", text) if value
        ]
        # Citation order is preserved, while repeated mentions of the same
        # measured node carry no extra evidence and are normalized away.
        cited = list(dict.fromkeys(parsed))
        if any(value not in measured_nodes for value in cited):
            raise ValueError("evidence IDs must identify measured nodes")
        confidence = raw["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence outside [0,1]")
        if not isinstance(raw["decision_note"], str) or len(raw["decision_note"]) > 240:
            raise ValueError("decision_note must be a string of at most 240 characters")
    except (TypeError, ValueError) as exc:
        status, error = "schema_failure", str(exc)
    normalized_record = dict(raw)
    if status == "ok":
        normalized_record["evidence_node_ids_csv"] = ",".join(str(value) for value in cited)
    return {
        "provider_round_index": round_index,
        "basis_observation_hash": observation_hash,
        "chosen_action_name": action_name,
        "status": status,
        "error": error,
        "record": normalized_record if status == "ok" else None,
        "raw_record": raw,
        "cited_measured_node_ids": cited,
        "fed_back_to_agent": False,
        "truth_visible": False,
    }


def _submission_v20(arguments: dict[str, Any]) -> dict[str, Any]:
    """Map the truthful public L2 plume wording to the legacy score object."""

    for prefix in ("plume", "source"):
        fields = [f"{prefix}_{axis}" for axis in "xyz"]
        values = [arguments.get(field) for field in fields]
        all_null = all(value is None for value in values)
        all_finite_numbers = all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in values
        )
        if not (all_null or all_finite_numbers):
            names = "/".join(fields)
            raise ValueError(
                f"{names} must be all three finite numbers or all three JSON null"
            )

    translated = {
        key: value for key, value in arguments.items()
        if key not in {"plume_x", "plume_y", "plume_z"}
    }
    translated.update({
        f"peak_{axis}": arguments.get(f"plume_{axis}") for axis in "xyz"
    })
    return _submission(translated)


def _validate_core(name: str, arguments: dict[str, Any]) -> None:
    if name == "move":
        if set(arguments) != {"direction"}:
            raise ValueError("move requires exactly direction")
        if arguments["direction"] not in {"north", "south", "east", "west", "up", "down"}:
            raise ValueError("direction outside six-neighbor enum")
        return
    if name == "analyze":
        if arguments:
            raise ValueError("analyze takes no core arguments")
        return
    if name == "submit":
        required = {
            "anomaly", "plume_x", "plume_y", "plume_z", "mechanism",
            "source_x", "source_y", "source_z", "decision",
        }
        allowed = required | {"source_candidate_id"}
        if not required <= set(arguments) or not set(arguments) <= allowed:
            raise ValueError("submit fields do not match the flat public schema")
        if arguments["anomaly"] not in {"anomaly", "no_anomaly"}:
            raise ValueError("submit anomaly is outside the registered enum")
        if arguments["mechanism"] not in {
            None, "external_point", "external_diffuse", "internal_release",
            "physical_resuspension",
        }:
            raise ValueError("submit mechanism is outside the registered enum")
        if arguments["decision"] not in {"declare", "cannot_determine"}:
            raise ValueError("submit decision is outside the registered enum")
        candidate = arguments.get("source_candidate_id")
        if candidate is not None and not isinstance(candidate, str):
            raise ValueError("source_candidate_id must be a string or null")
        _submission_v20(arguments)
        return
    raise ValueError("unknown tool")


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    # A tuple is required here.  A set makes JSON key order depend on the
    # process hash seed, so a run and an independent replay can contain the
    # same values yet disagree on the exact bytes shown to the next model turn.
    fields = (
        "ok", "action", "requested_action", "code", "detail", "direction",
        "arrival_node", "cost_s", "path_length_m", "physical_cost_s",
        "basis_state_version", "new_samples_since_last_analyze", "no_new_evidence",
        "observed_maxima", "transition_after_action", "submit_only_reason",
        "same_action_same_state", "same_action_same_state_count",
        "same_result_no_progress", "same_result_no_progress_count",
    )
    return {key: result[key] for key in fields if key in result}


def present_observation(
    env: EnvironmentV20, mode: str, last_results: list[dict[str, Any]],
) -> dict[str, Any]:
    value = env.observation()
    rows = list(value.pop("cumulative_measurements"))
    value["measurement_presentation_mode"] = mode
    value["cumulative_measurement_state"] = render_measurements(
        rows,
        mode,
        bounds=value["domain_bounds"],
        current_position=value["position"],
        raw_ledger=env.sample_ledger,
        current_mission_time_s=env.mission_time_s,
        z_layers=sorted({position[2] for position in env.graph.positions}),
    )
    value["derived_measurement_summary"] = derived_summary(
        rows,
        value["position"],
        value.pop("directions_to_unvisited_neighbors"),
        {},
        int(value["n_total_nodes"]),
    )
    value["last_tool_results"] = [_public_result(result) for result in last_results]
    # Serialize immutable instance content before changing mission state.  This
    # changes no value, but lets providers with prefix caching reuse topology,
    # calibration, and candidates across every single-edge round.
    static_fields = (
        "schema_version", "protocol_id", "instance_id", "phase",
        "navigation_topology", "domain_bounds", "n_total_nodes", "depth_axis",
        "calibration", "candidate_catalog_semantics", "candidates", "available_channels",
        "measurement_presentation_mode",
    )
    ordered = {field: value[field] for field in static_fields if field in value}
    ordered.update({field: item for field, item in value.items() if field not in ordered})
    return ordered


def run_model_v20(
    env: EnvironmentV20,
    config: dict[str, Any],
    *,
    resume_state: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    validate_model_config(config)
    if config.get("science_eligible") is True and "system_prompt_v20" in config:
        raise ValueError(
            "science-eligible v20 runs use the frozen protocol prompt; "
            "system_prompt_v20 overrides are forbidden"
        )
    provider = None
    mode = str(config.get("presentation_mode", "sparse_map"))
    prompt = str(config.get("system_prompt_v20") or SYSTEM_PROMPT_V20)
    resume_state = resume_state or {}
    calls: list[dict[str, Any]] = list(resume_state.get("provider_calls") or [])
    decisions: list[dict[str, Any]] = list(resume_state.get("decision_record_calls") or [])
    last_results: list[dict[str, Any]] = list(resume_state.get("last_results") or [])
    totals = dict(resume_state.get("token_usage") or {"input_tokens": 0, "output_tokens": 0})
    invalid_submit_streak = int(resume_state.get("invalid_submit_streak", 0))
    # The scripted adapter is used only for deterministic contract tests.  Its
    # internal counter must follow the durable provider-round journal on resume.
    while not env.terminated:
        if len(calls) > env.max_turns or env.turn > env.max_turns:
            raise ValueError("resumed provider-call state exceeds the protocol guard")
        if len(calls) >= env.max_turns or env.turn >= env.max_turns:
            env.exhaust_provider_round_guard()
            break
        if provider is None:
            provider = _provider(config)
            if hasattr(provider, "turn"):
                provider.turn = len(calls)
        engineering_cap = config.get("engineering_max_provider_rounds")
        if (
            engineering_cap is not None
            and len(calls) >= int(engineering_cap)
            and env.mode == "active"
        ):
            env.enter_submit_only("engineering_provider_round_cap")
        round_started_submit_only = env.mode == "submit_only"
        observation = present_observation(env, mode, last_results)
        observation_json = json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
        observation_hash = stable_hash(observation)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "CURRENT OBSERVATION\n" + observation_json},
        ]
        tools = SUBMIT_TOOL_V20 if env.mode == "submit_only" else TOOLS_V20
        round_index = len(calls)
        try:
            response = provider.complete(messages, tools)
            response_usage = response.usage
            input_tokens = response_usage.get("input_tokens")
            output_tokens = response_usage.get("output_tokens")
            if (
                isinstance(input_tokens, bool)
                or not isinstance(input_tokens, int)
                or input_tokens <= 0
                or isinstance(output_tokens, bool)
                or not isinstance(output_tokens, int)
                or output_tokens <= 0
            ):
                raise RuntimeError(
                    "provider response lacks positive prompt/completion token usage"
                )
        except RuntimeError as exc:
            env.abort_infrastructure(str(exc))
            calls.append({
                "provider_round_index": round_index,
                "tool_calls": [],
                "chosen_tool_call": None,
                "cancelled_tool_calls": [],
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "reported_model": None,
                "provider_error": str(exc),
                "observation_hash": observation_hash,
                "observation_json": observation_json,
                "messages_hash": stable_hash(messages),
                "tool_schema_hash": stable_hash(tools),
                "unexpected_content_sha256": None,
                "unexpected_content_char_count": 0,
            })
            if checkpoint is not None:
                checkpoint({
                    "stage": "committed",
                    "provider_call": calls[-1],
                    "decision_record": None,
                    "event": env.events[-1],
                    "token_usage": dict(totals),
                    "last_results": [],
                    "invalid_submit_streak": invalid_submit_streak,
                    "round_started_submit_only": round_started_submit_only,
                    "termination_reason": env.termination_reason,
                })
            break
        usage = {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
        }
        totals = {key: totals[key] + usage[key] for key in totals}
        chosen = response.tool_calls[:1]
        cancelled = response.tool_calls[1:]
        call_row = {
            "provider_round_index": round_index,
            "tool_calls": response.tool_calls,
            "chosen_tool_call": chosen[0] if chosen else None,
            "cancelled_tool_calls": [
                {**item, "status": "cancelled_unobserved"} for item in cancelled
            ],
            "usage": usage,
            "reported_model": response.raw_model,
            "provider_error": None,
            "observation_hash": observation_hash,
            "observation_json": observation_json,
            "messages_hash": stable_hash(messages),
            "tool_schema_hash": stable_hash(tools),
            "unexpected_content_sha256": stable_hash(response.content) if response.content else None,
            "unexpected_content_char_count": len(response.content or ""),
        }
        calls.append(call_row)
        if not chosen:
            decision = {
                "provider_round_index": round_index,
                "basis_observation_hash": observation_hash,
                "chosen_action_name": None,
                "decision_record_index": len(decisions),
                "status": "missing_no_tool_call",
                "error": "provider returned no executable tool call",
                "record": None,
                "raw_record": {},
                "cited_measured_node_ids": [],
                "fed_back_to_agent": False,
                "truth_visible": False,
            }
            decisions.append(decision)
            if checkpoint is not None:
                checkpoint({
                    "stage": "received",
                    "provider_call": calls[-1],
                    "decision_record": decision,
                    "token_usage": dict(totals),
                    "invalid_submit_streak": invalid_submit_streak,
                    "round_started_submit_only": round_started_submit_only,
                })
            result = env.record_invalid("E_NO_TOOL_CALL", "provider returned no tool call")
            if result in env.events:
                result.update({
                    "provider_round_index": round_index,
                    "tool_call_index": None,
                    "cancelled_tool_call_count": 0,
                    "basis_observation_hash": observation_hash,
                    "decision_record_index": decision["decision_record_index"],
                })
            last_results = [result]
        else:
            raw = chosen[0]
            name = str(raw.get("name") or "unknown")
            raw_arguments: dict[str, Any] = {}
            core_arguments: dict[str, Any] = {}
            raw_record: dict[str, Any] = {}
            try:
                raw_arguments = json.loads(raw["arguments"])
                if not isinstance(raw_arguments, dict):
                    raise ValueError("arguments not an object")
                core_arguments, raw_record = _split_arguments(raw_arguments)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
            decision = _record(
                raw_record,
                set(env.measurements),
                round_index=round_index,
                observation_hash=observation_hash,
                action_name=name,
            )
            decision["decision_record_index"] = len(decisions)
            decisions.append(decision)
            if checkpoint is not None:
                checkpoint({
                    "stage": "received",
                    "provider_call": calls[-1],
                    "decision_record": decision,
                    "token_usage": dict(totals),
                    "invalid_submit_streak": invalid_submit_streak,
                    "round_started_submit_only": round_started_submit_only,
                })
            try:
                if not isinstance(raw_arguments, dict):
                    raise ValueError("arguments not an object")
                _validate_core(name, core_arguments)
                if name == "move":
                    result = env.move(str(core_arguments["direction"]))
                elif name == "analyze":
                    result = env.analyze()
                else:
                    result = env.submit(_submission_v20(core_arguments))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                result = env.record_invalid("E_SCHEMA", str(exc), name)
            if result in env.events:
                result.update({
                    "provider_round_index": round_index,
                    "tool_call_index": 0,
                    "cancelled_tool_call_count": len(cancelled),
                    "basis_observation_hash": observation_hash,
                    "decision_record_index": decision["decision_record_index"],
                })
            last_results = [result]
        submit_schema_exhausted = False
        if round_started_submit_only:
            submitted = bool(last_results and last_results[0].get("action") == "submit")
            if submitted:
                invalid_submit_streak = 0
            elif not env.terminated:
                invalid_submit_streak += 1
                limit = int(env.protocol["termination"]["submit_only_schema_retries"])
                submit_schema_exhausted = invalid_submit_streak >= limit
        if (
            not env.terminated
            and (len(calls) >= env.max_turns or env.turn >= env.max_turns)
        ):
            env.exhaust_provider_round_guard()
        elif not env.terminated and submit_schema_exhausted:
            env.exhaust_submit_schema_retries()
        if checkpoint is not None:
            event = next(
                (
                    item for item in reversed(env.events)
                    if item.get("provider_round_index") == round_index
                ),
                None,
            )
            checkpoint({
                "stage": "committed",
                "provider_call": calls[-1],
                "decision_record": decisions[-1] if decisions and decisions[-1].get("provider_round_index") == round_index else None,
                "event": event,
                "token_usage": dict(totals),
                "last_results": list(last_results),
                "invalid_submit_streak": invalid_submit_streak,
                "round_started_submit_only": round_started_submit_only,
                "termination_reason": env.termination_reason,
            })
    if len(calls) > env.max_turns or env.turn > env.max_turns:
        raise ValueError("provider-call guard exceeded before episode finalization")
    completed_responses = [row for row in calls if row.get("provider_error") is None]
    identities = [row.get("reported_model") for row in completed_responses]
    # The audit record is part of a tool call's arguments.  A provider response
    # that violates ``tool_choice=required`` and contains no tool call has no
    # executable decision to which such a record could be attached.  Preserve
    # that response, its token cost, invalid event, and no-tool counter, but do
    # not invent a decision record or count it in the tool-record denominator.
    # This keeps coverage strict for every action that can actually execute.
    expected_record_count = sum(
        row.get("chosen_tool_call") is not None for row in completed_responses
    )
    captured_records = sum(
        row.get("chosen_action_name") is not None for row in decisions
    )
    valid_records = sum(row.get("status") == "ok" for row in decisions)
    return {
        "data_contract_version": "v20.3d-single-edge.1",
        "prompt_id": "laketrace-v20-3d-single-edge",
        "provider_calls": calls,
        "decision_record_calls": decisions,
        "decision_record_coverage": (
            valid_records / expected_record_count if expected_record_count else 1.0
        ),
        "decision_record_capture_coverage": (
            captured_records / expected_record_count if expected_record_count else 1.0
        ),
        "decision_record_valid_coverage": (
            valid_records / expected_record_count if expected_record_count else 1.0
        ),
        "decision_record_expected_count": expected_record_count,
        "decision_record_captured_count": captured_records,
        "decision_record_valid_count": valid_records,
        "no_tool_response_count": sum(
            row.get("chosen_tool_call") is None for row in completed_responses
        ),
        "token_usage": totals,
        "token_usage_status": (
            "infrastructure_terminal_charge_unknown"
            if any(row.get("provider_error") is not None for row in calls)
            else "complete_provider_reported"
        ),
        "reported_models": sorted({str(value) for value in identities if value}),
        "model_identity_match": (
            bool(completed_responses)
            and all(isinstance(value, str) and value for value in identities)
            and exact_model_identity_matches(
                str(config["model"]), [str(value) for value in identities]
            )
        ),
        "presentation_mode": mode,
        "multi_call_cancelled_count": sum(len(row["cancelled_tool_calls"]) for row in calls),
        "chain_of_thought_requested": False,
        "chain_of_thought_stored": False,
        "provider_reasoning_enabled": bool(
            (config.get("extra_body") or {}).get("enable_thinking")
        ),
    }
