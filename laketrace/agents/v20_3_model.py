"""V20.3 model contract: bounded memory, geometry-only routing, partial answers."""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Callable

from laketrace.agents.orchestrator import run_agent_loop
from laketrace.agents.v20_1_model import validate_model_config_v20_1
from laketrace.env.graph import DIRECTIONS
from laketrace.env.v20_3 import EnvironmentV20_3
from laketrace.perception.v20_2_state import present_observation_v20_2
from laketrace.util import stable_hash


PROTOCOL_ID = "laketrace-v20.3-closed-loop-3d-v1"
SCRATCHPAD_MAX_CHARS = 2048


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            **properties,
            "scratchpad": {
                "type": "string",
                "maxLength": SCRATCHPAD_MAX_CHARS,
                "description": (
                    "Your own compact mission memory. It is returned verbatim next round; "
                    "the harness never edits or adds content."
                ),
            },
        },
        # Updating mission memory is optional.  Requiring it on every action
        # would add a schema-failure surface without adding evidence; when it
        # is omitted the orchestrator returns the previous value unchanged.
        "required": list(required),
        "additionalProperties": False,
    }


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": _schema(properties, required),
        },
    }


GOTO_TOOL = _tool(
    "goto",
    (
        "Travel toward one chosen 3-D coordinate on a deterministic shortest "
        "geometry-only route, at most 32 measured native edges."
    ),
    {axis: {"type": "number", "description": f"Target {axis} in metres."} for axis in "xyz"},
    ["x", "y", "z"],
)
MOVE_ONE_TOOL = _tool(
    "move_one",
    "Move exactly one measured native 3-D neighbor edge.",
    {"direction": {"type": "string", "enum": list(DIRECTIONS)}},
    ["direction"],
)
MOVE_TOOL = _tool(
    "move", "Move 1--32 native edges on one chosen axis; measure each arrival.",
    {
        "direction": {"type": "string", "enum": list(DIRECTIONS)},
        "steps": {"type": "integer", "minimum": 1, "maximum": 32},
    },
    ["direction", "steps"],
)
SUBMIT_L1_TOOL = _tool(
    "submit_l1", "Save or replace the independently scored L1 answer.",
    {"anomaly": {"type": "string", "enum": ["anomaly", "no_anomaly"]}},
    ["anomaly"],
)
SUBMIT_L2_TOOL = _tool(
    "submit_l2", "Save or replace the independently scored L2 measured coordinate.",
    {axis: {"type": "number"} for axis in ("plume_x", "plume_y", "plume_z")},
    ["plume_x", "plume_y", "plume_z"],
)
SUBMIT_L3_TOOL = _tool(
    "submit_l3", "Save or replace the independently scored L3 mechanism.",
    {"mechanism": {
        "type": "string",
        "enum": [
            "external_point", "external_diffuse", "internal_release",
            "physical_resuspension",
        ],
    }},
    ["mechanism"],
)
SUBMIT_L4_TOOL = _tool(
    "submit_l4", "Save or replace the independently scored L4 source coordinate.",
    {
        "source_x": {"type": "number"},
        "source_y": {"type": "number"},
        "source_z": {"type": "number"},
        "source_candidate_id": {"type": ["string", "null"]},
    },
    ["source_x", "source_y", "source_z", "source_candidate_id"],
)
SUBMIT_L5_TOOL = _tool(
    "submit_l5", "Save or replace the independently scored L5 decision.",
    {"decision": {"type": "string", "enum": ["declare", "cannot_determine"]}},
    ["decision"],
)
FINALIZE_TOOL = _tool(
    "finalize_submission",
    "End the episode and submit all saved levels; missing levels score zero.",
    {},
    [],
)

PARTIAL_TOOLS = [
    SUBMIT_L1_TOOL, SUBMIT_L2_TOOL, SUBMIT_L3_TOOL, SUBMIT_L4_TOOL,
    SUBMIT_L5_TOOL, FINALIZE_TOOL,
]
TOOLS_V20_3 = [GOTO_TOOL, MOVE_ONE_TOOL, *PARTIAL_TOOLS]


SYSTEM_PROMPT_V20_3 = """MISSION

You control an autonomous water-quality sensor on a public six-neighbor 3-D lake grid.
Your task is to investigate a possible water-quality anomaly and trace its source using
only measurements collected along the route that you choose.

You must produce five independently scored answers:
L1 -- decide whether a detectable anomaly is present: anomaly or no_anomaly.
L2 -- report one 3-D location that you physically measured and judge to be inside the
detectable anomalous region.
L3 -- classify the source mechanism as external_point, external_diffuse,
internal_release, or physical_resuspension.
L4 -- report your best estimate of the 3-D source location and, optionally, a candidate ID.
L5 -- choose declare if the available evidence supports a source conclusion, or
cannot_determine if it does not.

MOVEMENT AND OBSERVATION

Every response must contain exactly one currently available tool call and no plain text.
Do not call a tool that is absent from the current tool list. goto executes a deterministic
geometry-only route toward your chosen x/y/z for at most 32 native edges; move_one executes
one chosen edge. Every arrival is charged and measured. Routing never inspects sensor
values, truth, scores, or candidates. z is depth-positive-down. Neighbor rows expose
reachability, cost, and visit history but never unseen readings.

The STATIC TASK CONTEXT is byte-stable. DYNAMIC MISSION STATE contains the current
position, measured-only cumulative map, factual measurement age, budgets, and a
measured-only arithmetic summary. Empty cells are unmeasured; no value is interpolated.
scratchpad is your own bounded mission memory and is returned verbatim next round.

SUBMISSION

Use movement tools to collect evidence. partial_submission.missing_levels lists unanswered
levels. Save a missing level with its available submit_l* tool, then call
finalize_submission, which permanently ends the episode. A malformed or missing level
cannot erase valid saved levels. Missing levels score zero. Unused endurance has no score
value. No audit note or private reasoning is requested. In SUBMIT_ONLY, do not attempt
movement: save the remaining levels and finalize."""


STATIC_FIELDS = (
    "schema_version", "protocol_id", "instance_id", "navigation_topology",
    "domain_bounds", "n_total_nodes", "depth_axis", "calibration",
    "candidate_catalog_semantics", "candidates", "measurement_presentation_mode",
)


def build_messages_v20_3(
    prompt: str, observation: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    static = {key: observation[key] for key in STATIC_FIELDS if key in observation}
    dynamic = {key: value for key, value in observation.items() if key not in static}
    static_text = json.dumps(static, ensure_ascii=False, separators=(",", ":"))
    dynamic_text = json.dumps(dynamic, ensure_ascii=False, separators=(",", ":"))
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "STATIC TASK CONTEXT\n" + static_text},
        {"role": "user", "content": "DYNAMIC MISSION STATE\n" + dynamic_text},
    ]
    return messages, {
        "static_prefix_sha256": stable_hash(messages[:2]),
        "static_prefix_utf8_bytes": len(
            json.dumps(messages[:2], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ),
        "dynamic_state_utf8_bytes": len(messages[-1]["content"].encode("utf-8")),
    }


def validate_model_config_v20_3(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("v20.3 model config must bind the v20.3 protocol id")
    profile = value.get("harness_profile", "legacy")
    if profile not in {"legacy", "batched_v1"}:
        raise ValueError("unknown v20.3 harness_profile")
    if profile == "batched_v1" and value.get("science_eligible") is True:
        raise ValueError("batched_v1 is an unfrozen engineering profile")
    physical_budget_enabled = value.get("physical_budget_enabled", True)
    if type(physical_budget_enabled) is not bool:
        raise ValueError("physical_budget_enabled must be a boolean")
    if not physical_budget_enabled and profile != "batched_v1":
        raise ValueError("unlimited travel requires the new batched_v1 profile")
    value["presentation_mode"] = "hierarchical_measured_map_v1"
    value["system_prompt_v20_1"] = str(
        value.get("system_prompt_v20_3") or SYSTEM_PROMPT_V20_3
    )
    if profile == "batched_v1":
        value["system_prompt_v20_1"] = value["system_prompt_v20_1"].replace(
            "move_one executes\none chosen edge.",
            "move(direction, steps) executes\n1--32 edges on one chosen axis, stopping at a boundary or physical budget limit.\nsteps=1 selects one edge; goto can include turns. Each movement call uses one action allowance.",
        )
    if not physical_budget_enabled:
        value["system_prompt_v20_1"] = value["system_prompt_v20_1"].replace(
            "or physical budget limit", "or the chosen segment end"
        ).replace(
            "Unused endurance has no score\nvalue.",
            "Travel time and distance are recorded without a physical cutoff.\nAction and monetary limits still apply.",
        )
    science_eligible = value.get("science_eligible") is True
    validation_copy = copy.deepcopy(value)
    validation_copy["science_eligible"] = False
    effective = validate_model_config_v20_1(validation_copy)
    effective["science_eligible"] = science_eligible
    effective["presentation_mode"] = "hierarchical_measured_map_v2"
    effective.setdefault("hard_cost_ceiling_cny", 4.0)
    effective.setdefault("submit_reserve_cny", 0.4)
    effective.setdefault("transport_max_retries", 3)
    effective.setdefault("transport_backoff_initial_s", 1.0)
    effective.setdefault("transport_backoff_multiplier", 2.0)
    effective.setdefault("transport_backoff_max_s", 8.0)
    if effective.get("science_eligible") is True and effective.get("allow_cost_guard_bypass"):
        raise ValueError("science-eligible v20.3 forbids cost-guard bypass")
    return effective


def _validate_core(name: str, arguments: dict[str, Any]) -> None:
    if name == "move":
        if set(arguments) != {"direction", "steps"}:
            raise ValueError("move requires direction and integer steps")
        if arguments["direction"] not in DIRECTIONS:
            raise ValueError("move requires a six-neighbor direction")
        if type(arguments["steps"]) is not int:
            raise ValueError("move steps must be an integer")
        # Range is an infeasible action, handled by the environment, not JSON repair.
        return
    if name == "goto":
        if set(arguments) != {"x", "y", "z"}:
            raise ValueError("goto requires exactly x, y, z")
        if any(isinstance(arguments[key], bool) or not isinstance(arguments[key], (int, float)) for key in "xyz"):
            raise ValueError("goto coordinates must be finite numbers")
        if any(not math.isfinite(float(arguments[key])) for key in "xyz"):
            raise ValueError("goto coordinates must be finite numbers")
        return
    if name == "move_one":
        if set(arguments) != {"direction"} or arguments["direction"] not in DIRECTIONS:
            raise ValueError("move_one requires one six-neighbor direction")
        return
    expected = {
        "submit_l1": {"anomaly"},
        "submit_l2": {"plume_x", "plume_y", "plume_z"},
        "submit_l3": {"mechanism"},
        "submit_l4": {"source_x", "source_y", "source_z", "source_candidate_id"},
        "submit_l5": {"decision"},
        "finalize_submission": set(),
    }
    if name not in expected or set(arguments) != expected[name]:
        raise ValueError(f"{name} fields do not match its flat registered schema")


def run_model_v20_3(
    env: EnvironmentV20_3,
    config: dict[str, Any],
    *,
    resume_state: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    move_executor: Callable[[EnvironmentV20_3, list[str], dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    effective = validate_model_config_v20_3(config)
    if env.physical_budget_enabled != effective.get("physical_budget_enabled", True):
        raise ValueError("environment and model physical-budget modes differ")
    batched = effective.get("harness_profile") == "batched_v1"
    movement_tools = [GOTO_TOOL, MOVE_TOOL if batched else MOVE_ONE_TOOL]

    def present(target, last_results):
        value = present_observation_v20_2(target, last_results)
        if batched:
            value["partial_submission"]["saved_answers"] = copy.deepcopy(target.partial_answer)
        return value

    partial_by_level = {
        "L1": SUBMIT_L1_TOOL,
        "L2": SUBMIT_L2_TOOL,
        "L3": SUBMIT_L3_TOOL,
        "L4": SUBMIT_L4_TOOL,
        "L5": SUBMIT_L5_TOOL,
    }

    def select_tools(
        target: EnvironmentV20_3, active: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        limit = int(target.control_protocol["submission"]["max_updates_per_level"])
        answer_tools = [
            partial_by_level[level]
            for level in ("L1", "L2", "L3", "L4", "L5")
            if (
                level not in target.completed_levels
                if target.mode == "submit_only"
                else target.partial_update_counts[level] < limit
            )
        ]
        return (
            [*movement_tools, *answer_tools, FINALIZE_TOOL]
            if target.mode != "submit_only"
            else [*answer_tools, FINALIZE_TOOL]
        )

    def route_builder(name: str, arguments: dict[str, Any], target: EnvironmentV20_3) -> list[str]:
        if name == "goto":
            return target.plan_goto(arguments["x"], arguments["y"], arguments["z"])[0]
        if name == "move_one":
            return [str(arguments["direction"])]
        if name == "move":
            # Avoid allocating an unbounded list for a typed but infeasible count.
            steps = arguments["steps"]
            return [str(arguments["direction"])] * min(target.max_steps + 1, max(0, steps))
        return []

    def execute(name: str, arguments: dict[str, Any], target: EnvironmentV20_3, context: dict[str, Any]) -> dict[str, Any]:
        if name in {"goto", "move_one", "move"}:
            directions = list(context["directions"])
            if name == "move" and not 1 <= len(directions) <= target.max_steps:
                return target.move(directions)
            if not directions:
                return target._record_infeasible(
                    "E_ALREADY_AT_TARGET", name, copy.deepcopy(arguments)
                )
            result = context["move_executor"](target, directions, context)
            if name == "goto":
                resolved = target.resolve_goto_target(
                    arguments["x"], arguments["y"], arguments["z"]
                )
                result.update({
                    "requested_action": "goto",
                    "requested_target_xyz": [
                        float(arguments["x"]), float(arguments["y"]),
                        float(arguments["z"]),
                    ],
                    "resolved_target_node": int(resolved),
                    "resolved_target_xyz": list(target.graph.positions[resolved]),
                    "geometry_only_route": True,
                })
            return result
        if name == "submit_l1":
            return target.record_partial_answer("L1", arguments)
        if name == "submit_l2":
            return target.record_partial_answer("L2", arguments)
        if name == "submit_l3":
            return target.record_partial_answer("L3", arguments)
        if name == "submit_l4":
            return target.record_partial_answer("L4", arguments)
        if name == "submit_l5":
            return target.record_partial_answer("L5", arguments)
        return target.finalize_submission()

    metadata = run_agent_loop(
        env,
        effective,
        resume_state=resume_state,
        checkpoint=checkpoint,
        move_executor=move_executor,
        active_tools=[*movement_tools, *PARTIAL_TOOLS],
        core_validator=_validate_core,
        route_builder=route_builder,
        observation_presenter=present,
        config_validator=lambda value: copy.deepcopy(value),
        message_builder=build_messages_v20_3,
        tool_selector=select_tools,
        custom_action_executor=execute,
        scratchpad_limit_chars=SCRATCHPAD_MAX_CHARS,
        decision_record_required=False,
        submit_retry_invalid_only=True,
        active_invalid_to_submit_only=True,
    )
    metadata.update({
        "data_contract_version": "v20.3.closed-loop-3d.1",
        "prompt_id": "laketrace-v20.3-closed-loop-3d",
        "protocol_id": PROTOCOL_ID,
        "analysis_action_removed": True,
        "partial_submission_enabled": True,
        "static_prefix_enabled": True,
    })
    if batched:
        metadata["harness_profile"] = "batched_v1"
        metadata["presentation_mode"] = "hierarchical_measured_map_v2"
    return metadata


__all__ = [
    "FINALIZE_TOOL", "GOTO_TOOL", "MOVE_ONE_TOOL", "PARTIAL_TOOLS",
    "PROTOCOL_ID", "SCRATCHPAD_MAX_CHARS", "SYSTEM_PROMPT_V20_3",
    "TOOLS_V20_3", "build_messages_v20_3", "run_model_v20_3",
    "validate_model_config_v20_3",
]
