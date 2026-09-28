"""Model-facing contract for the v20.2 resource-bounded control overlay."""

from __future__ import annotations

import copy
from typing import Any, Callable

from laketrace.agents.v20_1_model import (
    MOVE_TOOL_V20_1,
    TOOLS_V20_1,
    run_model_v20_1,
    validate_model_config_v20_1,
)
from laketrace.agents.v20_model import _parameters
from laketrace.env.graph import DIRECTIONS
from laketrace.env.v20_2 import EnvironmentV20_2
from laketrace.perception.v20_2_state import (
    PRESENTATION_MODE_V20_2,
    present_observation_v20_2,
)


PROTOCOL_ID = "laketrace-v20.2-resource-bounded-3d-v1"

MOVE_TO_BOUNDARY_TOOL_V20_2 = {
    "type": "function",
    "function": {
        "name": "move_to_boundary",
        "description": "Move on one axis to boundary/budget, at most 32 edges.",
        "parameters": _parameters({
            "direction": {
                "type": "string",
                "enum": ["north", "south", "east", "west", "up", "down"],
                "description": "Model-chosen axis.",
            },
        }, ["direction"]),
    },
}

TOOLS_V20_2 = [
    copy.deepcopy(MOVE_TOOL_V20_1),
    MOVE_TO_BOUNDARY_TOOL_V20_2,
    *[
        copy.deepcopy(item) for item in TOOLS_V20_1
        if item["function"]["name"] not in {"move"}
    ],
]
# Keep the wire schema concise; the registered system prompt carries the shared
# edge semantics once rather than duplicating them in each movement property.
TOOLS_V20_2[0]["function"]["description"] = "Move 1--32 nodes on one chosen 3-D axis; measure each arrival."
TOOLS_V20_2[0]["function"]["parameters"]["properties"]["steps"]["description"] = "Native edge count, 1--32."

SYSTEM_PROMPT_V20_2 = """Control a sensor on a six-neighbor 3-D lake grid; make one tool call.
move chooses an axis and 1--32 edges; move_to_boundary runs on one axis up to 32 edges
to a boundary/budget limit. Every arrival is charged and measured. The executor never
chooses, turns, reacts to readings, or estimates unmeasured nodes.
z is depth-positive-down. Neighbors show reachability, cost, and visits, not readings.

EXPLORE and REFINE have identical tools/data; labels depend only on resources.
SUBMIT_ONLY exposes submit only. Unsubmitted levels score zero; unused endurance has
no value. The map contains samples only; empty cells are unmeasured. Measurement age
means elapsed mission time, not change. Candidates are the
complete source-independent, unranked domain catalog.

Independent levels: L1 anomaly/no_anomaly; L2 a measured coordinate in a registered
principal-tracer anomaly region; L3 one listed mechanism; L4 a source coordinate scored
to the nearest injection node; L5 declare/cannot_determine.
Each plume/source xyz is three numbers or three JSON nulls.

Each call also gives audit-only intent, measured evidence IDs, confidence, and a short
decision_note; audit never affects execution or score."""


def validate_model_config_v20_2(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("v20.2 model config must bind the v20.2 protocol id")
    presentation = str(value.get("presentation_mode") or PRESENTATION_MODE_V20_2)
    if presentation != PRESENTATION_MODE_V20_2:
        raise ValueError(f"v20.2 presentation_mode must be {PRESENTATION_MODE_V20_2!r}")
    value["presentation_mode"] = "hierarchical_measured_map_v1"
    value["system_prompt_v20_1"] = str(
        value.get("system_prompt_v20_2") or SYSTEM_PROMPT_V20_2
    )
    # Reuse v20.1's provider/schema validation while keeping the v20.1 frozen
    # prompt override rule scoped to v20.1 itself. V20.2 owns a new registered
    # prompt and separately restores the science-eligible flag below.
    science_eligible = value.get("science_eligible") is True
    validation_copy = copy.deepcopy(value)
    validation_copy["science_eligible"] = False
    effective = validate_model_config_v20_1(validation_copy)
    effective["science_eligible"] = science_eligible
    effective["presentation_mode"] = PRESENTATION_MODE_V20_2
    required = {
        "hard_cost_ceiling_cny": 4.0,
        "submit_reserve_cny": 0.40,
        "transport_max_retries": 3,
        "transport_backoff_initial_s": 1.0,
        "transport_backoff_multiplier": 2.0,
        "transport_backoff_max_s": 8.0,
    }
    for field, expected in required.items():
        if field not in effective or float(effective[field]) != expected:
            raise ValueError(f"v20.2 requires {field}={expected}")
    if effective.get("science_eligible") is True and effective.get("allow_cost_guard_bypass"):
        raise ValueError("science-eligible v20.2 forbids cost-guard bypass")
    return effective


def run_model_v20_2(
    env: EnvironmentV20_2,
    config: dict[str, Any],
    *,
    resume_state: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    move_executor: Callable[[EnvironmentV20_2, list[str], dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    effective = validate_model_config_v20_2(config)

    def validate_core(name: str, arguments: dict[str, Any]) -> None:
        if name == "move":
            if set(arguments) != {"direction", "steps"}:
                raise ValueError("move requires exactly direction and steps")
            if arguments["direction"] not in DIRECTIONS:
                raise ValueError("direction is outside the six-neighbor enum")
            steps = arguments["steps"]
            if isinstance(steps, bool) or not isinstance(steps, int):
                raise ValueError("steps must be an integer")
            # Integer range is an environment feasibility constraint. Keeping
            # it out of schema-failure accounting preserves the registered
            # parse-vs-action distinction.
            return
        if name == "move_to_boundary":
            if set(arguments) != {"direction"}:
                raise ValueError("move_to_boundary requires exactly direction")
            if arguments["direction"] not in DIRECTIONS:
                raise ValueError("direction is outside the six-neighbor enum")
            return
        from laketrace.agents.v20_1_model import _validate_core
        _validate_core(name, arguments)

    def route_builder(
        name: str, arguments: dict[str, Any], target: EnvironmentV20_2,
    ) -> list[str]:
        if name == "move":
            return [str(arguments["direction"])] * int(arguments["steps"])
        if name == "move_to_boundary":
            return [str(arguments["direction"])] * int(target.max_steps)
        return []

    metadata = run_model_v20_1(
        env,
        effective,
        resume_state=resume_state,
        checkpoint=checkpoint,
        move_executor=move_executor,
        active_tools=TOOLS_V20_2,
        core_validator=validate_core,
        route_builder=route_builder,
        observation_presenter=present_observation_v20_2,
        config_validator=lambda value: copy.deepcopy(value),
    )
    metadata.update({
        "data_contract_version": "v20.2.resource-bounded-3d.1",
        "prompt_id": "laketrace-v20.2-resource-bounded-3d",
        "resource_control_protocol_id": PROTOCOL_ID,
        "presentation_mode": PRESENTATION_MODE_V20_2,
    })
    return metadata


__all__ = [
    "PROTOCOL_ID", "SYSTEM_PROMPT_V20_2",
    "MOVE_TO_BOUNDARY_TOOL_V20_2", "TOOLS_V20_2",
    "run_model_v20_2", "validate_model_config_v20_2",
]
