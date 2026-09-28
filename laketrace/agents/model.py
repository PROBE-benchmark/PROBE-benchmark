from __future__ import annotations

import json
import math
import re
from typing import Any

from laketrace.env.core import Environment
from laketrace.perception import PRESENTATION_MODES, derived_summary, render_measurements
from laketrace.providers.openai_compatible import OpenAICompatibleProvider
from laketrace.providers.scripted import ScriptedProvider


TOOLS = [
    {"type": "function", "function": {"name": "move", "description": "Move along adjacent field nodes; every traversed node is measured.", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["north", "south", "east", "west", "up", "down"]}, "steps": {"type": "integer", "minimum": 1, "maximum": 32}}, "required": ["direction", "steps"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "analyze", "description": "Calculate observed maxima over your cumulative measurements.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "submit", "description": "Submit the five independent exam answers and end the episode.", "parameters": {"type": "object", "properties": {
        "anomaly": {"type": "string", "enum": ["anomaly", "no_anomaly"]},
        "peak_x": {"type": ["number", "null"]}, "peak_y": {"type": ["number", "null"]}, "peak_z": {"type": ["number", "null"]},
        "mechanism": {"type": ["string", "null"], "enum": ["external_point", "external_diffuse", "internal_release", "physical_resuspension", None]},
        "source_x": {"type": ["number", "null"]}, "source_y": {"type": ["number", "null"]}, "source_z": {"type": ["number", "null"]},
        "source_candidate_id": {"type": ["string", "null"]},
        "decision": {"type": "string", "enum": ["declare", "cannot_determine"]},
    }, "required": ["anomaly", "peak_x", "peak_y", "peak_z", "mechanism", "source_x", "source_y", "source_z", "decision"], "additionalProperties": False}}},
]


SYSTEM_PROMPT = """You operate a mobile measurement platform on a discrete 3-D lake grid.
Every response must contain at least one tool call; do not answer with prose.
Use only move, analyze, and submit. A move measures every traversed node. Neighbor
metadata gives reachability and cost, never unseen readings. A derived summary
lists which reachable directions lead to nodes absent from your own measurements.
Your cumulative measurement state includes measurement age as a factual timestamp.
It retains exact nodes while compact, then uses a registered spatial aggregate,
current-column profile, and recent detail instead of truncation. No unvisited
value is inferred. last_tool_results reports the
outcome of the preceding provider round, including infeasible or invalid calls.

Five levels are scored independently: L1 anomaly/no anomaly; L2 one coordinate inside
any registered above-threshold principal-tracer anomaly region in the observed phase; L3 one of the
four listed mechanisms; L4 a source coordinate scored by horizontal and vertical distance
to the nearest registered injection node; L5 a binary declare/cannot_determine decision. An unsubmitted
level scores zero. A wrong answer and an explicit inability to determine are
recorded separately. Remaining endurance has no score value. You must submit to
receive any score. These are scoring rules, not navigation advice."""


def validate_model_config(config: dict[str, Any]) -> None:
    """Fail early with actionable errors for the public YAML model contract."""

    required = {"schema_version", "id", "adapter", "model", "science_eligible"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"model configuration missing required fields: {missing}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(config["id"])):
        raise ValueError("model id may contain only letters, digits, dot, underscore, and hyphen")
    if config["adapter"] not in {"openai_compatible", "scripted"}:
        raise ValueError("adapter must be openai_compatible or scripted")
    if config["adapter"] == "openai_compatible" and not config.get("base_url"):
        raise ValueError("openai_compatible model configuration requires base_url")
    if any(key in config for key in ("api_key", "access_token", "secret")):
        raise ValueError("credentials must not be stored in model YAML; use api_key_env")
    key_env = config.get("api_key_env")
    if key_env is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key_env)):
        raise ValueError("api_key_env must be an environment-variable name, not a credential")
    try:
        temperature = float(config.get("temperature", 1.0))
        timeout = float(config.get("timeout_s", 180))
        retries = int(config.get("retries", 3))
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature, timeout_s, and retries must be numeric") from exc
    if not math.isfinite(temperature) or timeout <= 0 or retries < 0:
        raise ValueError("temperature must be finite, timeout_s positive, and retries nonnegative")
    for field in ("max_output_tokens", "engineering_max_provider_rounds"):
        if config.get(field) is not None:
            try:
                value = int(config[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be a positive integer") from exc
            if value < 1:
                raise ValueError(f"{field} must be a positive integer")
    presentation = str(config.get("presentation_mode", "records"))
    if presentation not in PRESENTATION_MODES:
        raise ValueError(f"presentation_mode must be one of {PRESENTATION_MODES}")
    if config.get("science_eligible") is True and config.get("engineering_max_provider_rounds") is not None:
        raise ValueError("engineering_max_provider_rounds is forbidden for science-eligible models")
    if config.get("extra_body") is not None and not isinstance(config["extra_body"], dict):
        raise ValueError("extra_body must be an object")


def _provider(config: dict[str, Any]):
    adapter = config["adapter"]
    if adapter == "openai_compatible":
        return OpenAICompatibleProvider(config)
    if adapter == "scripted":
        return ScriptedProvider(config)
    raise ValueError(f"unsupported adapter {adapter!r}; choose a documented generic adapter")


def _submission(arguments: dict[str, Any]) -> dict[str, Any]:
    def position(prefix: str) -> dict[str, float] | None:
        values = [arguments.get(f"{prefix}_{axis}") for axis in "xyz"]
        if all(value is None for value in values):
            return None
        if any(value is None or isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError(f"{prefix} coordinate must be three numbers or three nulls")
        coordinates = list(map(float, values))
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"{prefix} coordinate must contain only finite numbers")
        return dict(zip("xyz", coordinates))
    return {
        "anomaly": arguments["anomaly"], "peak_position": position("peak"),
        "mechanism": arguments["mechanism"], "source_position": position("source"),
        "source_candidate_id": arguments.get("source_candidate_id"),
        "decision": arguments["decision"],
    }


def _validate_arguments(name: str, arguments: dict[str, Any]) -> None:
    """Enforce the public tool schema when a compatible provider does not."""

    if name == "move":
        required = {"direction", "steps"}
        if set(arguments) != required:
            raise ValueError("move requires exactly direction and steps")
        if arguments["direction"] not in {"north", "south", "east", "west", "up", "down"}:
            raise ValueError("move direction is not in the six-direction enum")
        if isinstance(arguments["steps"], bool) or not isinstance(arguments["steps"], int):
            raise ValueError("move steps must be an integer")
        return
    if name == "analyze":
        if arguments:
            raise ValueError("analyze takes no arguments")
        return
    if name == "submit":
        required = {
            "anomaly", "peak_x", "peak_y", "peak_z", "mechanism",
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
        candidate_id = arguments.get("source_candidate_id")
        if candidate_id is not None and not isinstance(candidate_id, str):
            raise ValueError("source_candidate_id must be a string or null")
        return
    raise ValueError("unknown tool")


def _public_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "ok", "action", "requested_action", "code", "detail", "direction", "steps",
        "cost_s", "path_length_m", "arrival_node", "visited_nodes", "observed_maxima",
        "budget_after",
    }
    value = {key: result[key] for key in allowed if key in result}
    if result.get("action") == "move":
        value["readings_returned"] = len(result.get("readings") or [])
    return value


def _present_observation(observation: dict[str, Any], mode: str, last_results: list[dict[str, Any]]) -> dict[str, Any]:
    value = dict(observation)
    rows = list(value.pop("cumulative_measurements"))
    value["measurement_presentation_mode"] = mode
    value["cumulative_measurement_state"] = render_measurements(
        rows, mode, bounds=value["domain_bounds"], current_position=value["position"]
    )
    value["derived_measurement_summary"] = derived_summary(
        rows, value["position"], value.pop("directions_to_unvisited_neighbors"),
        value.pop("observed_local_gradients"),
        int(value["n_total_nodes"])
    )
    value["last_tool_results"] = [_public_tool_result(result) for result in last_results]
    return value


def _identity_matches(expected: str, identities: list[str]) -> bool:
    """Require positive provider identity evidence, not vacuous agreement."""

    return bool(identities) and all(
        value == expected or value.startswith(expected + ":") for value in identities
    )


def run_model(env: Environment, config: dict[str, Any]) -> dict[str, Any]:
    validate_model_config(config)
    provider = _provider(config)
    presentation_mode = str(config.get("presentation_mode", "records"))
    if presentation_mode not in PRESENTATION_MODES:
        raise ValueError(f"presentation_mode must be one of {PRESENTATION_MODES}")
    invalid_streak = 0
    calls = []
    last_results: list[dict[str, Any]] = []
    totals = {"input_tokens": 0, "output_tokens": 0}
    while not env.terminated:
        engineering_cap = config.get("engineering_max_provider_rounds")
        if engineering_cap is not None and len(calls) >= int(engineering_cap):
            if bool(config.get("science_eligible", False)):
                raise ValueError("engineering_max_provider_rounds is forbidden for science-eligible models")
            env.terminated, env.termination_reason = True, "engineering_round_cap"
            break
        observation = _present_observation(env.observation(first=not calls), presentation_mode, last_results)
        system_prompt = config.get("system_prompt", SYSTEM_PROMPT)
        try:
            response = provider.complete(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": "CURRENT OBSERVATION\n" + json.dumps(observation, ensure_ascii=False, separators=(",", ":"))}],
                TOOLS,
            )
        except RuntimeError as exc:
            env.terminated, env.termination_reason = True, "infra_failure"
            identities = [item["reported_model"] for item in calls if item.get("reported_model")]
            return {
                "provider_error": str(exc),
                "provider_calls": calls,
                "token_usage": totals,
                "reported_models": sorted(set(identities)),
                "model_identity_match": _identity_matches(str(config["model"]), identities),
                "presentation_mode": presentation_mode,
            }
        totals = {key: totals[key] + response.usage[key] for key in totals}
        provider_record = {"tool_calls": response.tool_calls, "content": response.content, "usage": response.usage, "reported_model": response.raw_model}
        calls.append(provider_record)
        round_results: list[dict[str, Any]] = []
        if not response.tool_calls:
            invalid_streak += 1
            round_results.append(env.record_invalid("E_NO_TOOL_CALL", "provider returned no tool call"))
        else:
            for raw in response.tool_calls:
                try:
                    arguments = json.loads(raw["arguments"])
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments not an object")
                    _validate_arguments(str(raw.get("name")), arguments)
                    if raw["name"] == "move":
                        result = env.move(arguments["direction"], arguments["steps"])
                    elif raw["name"] == "analyze":
                        result = env.analyze()
                    elif raw["name"] == "submit":
                        result = env.submit(_submission(arguments))
                    else:  # pragma: no cover - guarded by _validate_arguments
                        raise ValueError("unknown tool")
                    invalid_streak = 0
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    result = env.record_invalid("E_SCHEMA", str(exc), str(raw.get("name") or "unknown"))
                    invalid_streak += 1
                round_results.append(result)
                if not result.get("ok", False):
                    break
                if env.terminated:
                    break
        last_results = round_results
        if invalid_streak >= 3 and not env.terminated:
            env.terminated, env.termination_reason = True, "consecutive_invalid"
        if env.turn >= env.max_turns:
            env.terminated, env.termination_reason = True, env.termination_reason or "turn_limit"
    identities = [item["reported_model"] for item in calls if item.get("reported_model")]
    expected = str(config["model"])
    identity_match = _identity_matches(expected, identities)
    return {
        "provider_calls": calls,
        "token_usage": totals,
        "reported_models": sorted(set(identities)),
        "model_identity_match": identity_match,
        "presentation_mode": presentation_mode,
    }
