"""Model-facing adapter for the preregistered v20.1 axis-run protocol.

The frozen v20 adapter is intentionally not modified.  This successor changes
only the control frequency and measured-state encoding: a model can authorize
one straight run of up to 32 native 3-D edges, while the environment still
validates, debits, and samples every edge independently.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Callable

from laketrace.agents.model import validate_model_config
from laketrace.agents.v20_model import (
    RECORD_FIELDS,
    SUBMIT_TOOL_V20,
    TOOLS_V20,
    _parameters,
    _record,
    _split_arguments,
    _submission_v20,
    _validate_core as _validate_core_v20,
    exact_model_identity_matches,
)
from laketrace.env.graph import DIRECTIONS
from laketrace.env.v20_1 import EnvironmentV20_1
from laketrace.perception.v20_1_state import (
    compact_route_measurements,
    render_compact_measurements,
)
from laketrace.perception.v20_state import derived_summary
from laketrace.providers.openai_compatible import ProviderResponse
from laketrace.providers.openai_compatible_v20_1 import (
    OpenAICompatibleProviderV20_1,
    RetriableTransportInterruption,
    build_request_payload,
    encode_request_payload,
)
from laketrace.util import stable_hash


PRESENTATION_MODE_V20_1 = "hierarchical_measured_map_v1"
MAX_ROUTE_EDGES = 32
MAX_OUTPUT_TOKENS = 1000
MAX_REQUEST_UTF8_BYTES = 36000
ACTIVE_INVALID_STREAK_LIMIT = 3


MOVE_TOOL_V20_1 = {
    "type": "function",
    "function": {
        "name": "move",
        "description": (
            "Move 1 to 32 adjacent nodes along one native 3-D axis; measure and "
            "debit each completed edge and retain any feasible prefix."
        ),
        "parameters": _parameters({
            "direction": {
                "type": "string",
                "enum": ["north", "south", "east", "west", "up", "down"],
                "description": "One native 3-D axis direction.",
            },
            "steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_ROUTE_EDGES,
                "description": (
                    "Number of adjacent nodes to traverse. No harness path "
                    "planning or sensor-triggered interruption occurs."
                ),
            },
        }, ["direction", "steps"]),
    },
}


TOOLS_V20_1 = [
    MOVE_TOOL_V20_1,
    *[
        copy.deepcopy(item)
        for item in TOOLS_V20
        if item["function"]["name"] != "move"
    ],
]
SUBMIT_TOOL_V20_1 = copy.deepcopy(SUBMIT_TOOL_V20)


SYSTEM_PROMPT_V20_1 = """You control a mobile sensor on a native 3-D lake grid. Make exactly one tool
call. move(direction, steps) traverses 1--32 adjacent nodes on one north/south/
east/west/up/down axis. The executor never plans for you: it charges and measures
each completed edge, stops before an unreachable or unaffordable edge, and retains
the completed prefix. You then receive all arrival readings in order. z is depth-
positive-down. Neighbors give reachability, cost, and visit history, never unseen
readings.

Topology is static and source-independent. State and last_route_measurements contain
only your samples. Measurement age is elapsed mission time, not a claim of change;
empty cells are unmeasured and never inferred. Revisits are legal. The main condition
is static F1. The complete candidate catalog is source-independent, unranked, and
order-neutral. Exact rows include factual bed_depth_m and depth_layer for calibration.

Levels are independent: L1 anomaly/no_anomaly; L2 a coordinate in a registered
principal-tracer anomaly region; L3 one listed mechanism; L4 a source coordinate
scored to the nearest injection node; L5 declare/cannot_determine. Unsubmitted levels
score zero; wrong and cannot_determine are distinct; unused endurance has no value.
Each plume/source coordinate triplet must be three JSON numbers or three JSON nulls,
never text "None". In submit_only mode only submit is legal.

Every call also includes audit-only intent, evidence_node_ids_csv, confidence, and
decision_note. Cite only node IDs already visible; confidence is 0--1; decision_note
is one short conclusion, not private reasoning. Audit fields affect no execution,
observation, budget, or score."""


class RequestPayloadTooLarge(RuntimeError):
    """Raised locally before transport when the complete request is oversized."""

    def __init__(self, actual_bytes: int, limit_bytes: int):
        super().__init__(
            f"complete provider request exceeds registered UTF-8 byte ceiling: "
            f"{actual_bytes} > {limit_bytes}"
        )
        self.actual_bytes = int(actual_bytes)
        self.limit_bytes = int(limit_bytes)


def _effective_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value.setdefault("max_output_tokens", MAX_OUTPUT_TOKENS)
    value.setdefault("max_request_utf8_bytes", MAX_REQUEST_UTF8_BYTES)
    return value


def _thinking_is_explicitly_disabled(config: dict[str, Any]) -> bool:
    if config.get("adapter") == "scripted":
        return True
    extra = config.get("extra_body") or {}
    return (
        extra.get("enable_thinking") is False
        or extra.get("think") is False
        or str(extra.get("reasoning_effort", "")).lower() == "none"
    )


def validate_model_config_v20_1(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the effective v20.1 provider configuration."""

    effective = _effective_config(config)
    # The generic public validator knows the legacy presentation names.  V20.1
    # has one fixed renderer, so validate all other generic fields on a copy.
    generic = copy.deepcopy(effective)
    presentation = str(
        generic.get("presentation_mode", PRESENTATION_MODE_V20_1)
    )
    if presentation != PRESENTATION_MODE_V20_1:
        raise ValueError(
            f"v20.1 presentation_mode must be {PRESENTATION_MODE_V20_1!r}"
        )
    generic["presentation_mode"] = "sparse_map"
    validate_model_config(generic)

    if effective.get("science_eligible") is True and (
        "temperature" not in config or float(effective["temperature"]) != 1.0
    ):
        raise ValueError(
            "science-eligible v20.1 configs must explicitly set temperature: 1.0"
        )
    if int(effective.get("retries", 0)) != 0:
        raise ValueError("v20.1 forbids hidden provider retries")
    output_limit = int(effective["max_output_tokens"])
    if not 1 <= output_limit <= MAX_OUTPUT_TOKENS:
        raise ValueError(
            f"max_output_tokens must be between 1 and {MAX_OUTPUT_TOKENS}"
        )
    request_limit = int(effective["max_request_utf8_bytes"])
    if not 1 <= request_limit <= MAX_REQUEST_UTF8_BYTES:
        raise ValueError(
            "max_request_utf8_bytes must be positive and no greater than "
            f"{MAX_REQUEST_UTF8_BYTES}"
        )
    if not _thinking_is_explicitly_disabled(effective):
        raise ValueError(
            "v20.1 openai_compatible configs must explicitly disable provider reasoning"
        )
    if effective.get("science_eligible") is True and "system_prompt_v20_1" in effective:
        raise ValueError(
            "science-eligible v20.1 runs use the frozen protocol prompt; "
            "system_prompt_v20_1 overrides are forbidden"
        )
    return effective


class _ScriptedV20_1Provider:
    """Small deterministic adapter used only by local contract tests."""

    def __init__(self, config: dict[str, Any]):
        self.turn = 0
        self.model = str(config.get("model") or "scripted-v20.1")

    @staticmethod
    def _audit(intent: str) -> dict[str, Any]:
        return {
            "intent": intent,
            "evidence_node_ids_csv": "",
            "confidence": 0.1,
            "decision_note": "contract-only deterministic action",
        }

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        self.turn += 1
        observation = json.loads(messages[-1]["content"].split("\n", 1)[-1])
        tool_names = [item["function"]["name"] for item in tools]
        if "finalize_submission" in tool_names:
            scratchpad = "scripted contract memory"
            if self.turn == 1 and "move_one" in tool_names:
                direction = next(
                    name for name, value in _compact_neighbor_mapping(
                        observation["neighbors"]
                    ).items()
                    if value["reachable"] and value.get("affordable", True)
                )
                call = {
                    "id": "v20.3-move-one",
                    "name": "move_one",
                    "arguments": json.dumps({
                        "direction": direction, "scratchpad": scratchpad,
                    }),
                }
            elif self.turn == 2:
                call = {
                    "id": "v20.3-l1", "name": "submit_l1",
                    "arguments": json.dumps({
                        "anomaly": "no_anomaly", "scratchpad": scratchpad,
                    }),
                }
            elif self.turn == 3:
                call = {
                    "id": "v20.3-l5", "name": "submit_l5",
                    "arguments": json.dumps({
                        "decision": "cannot_determine", "scratchpad": scratchpad,
                    }),
                }
            else:
                call = {
                    "id": "v20.3-finalize", "name": "finalize_submission",
                    "arguments": json.dumps({"scratchpad": scratchpad}),
                }
            return ProviderResponse(
                [call], None, {"input_tokens": 1, "output_tokens": 1}, self.model
            )
        if observation.get("episode_mode") == "submit_only" or self.turn >= 3:
            arguments = {
                "anomaly": "no_anomaly",
                "plume_x": None,
                "plume_y": None,
                "plume_z": None,
                "mechanism": None,
                "source_x": None,
                "source_y": None,
                "source_z": None,
                "source_candidate_id": None,
                "decision": "cannot_determine",
                **self._audit("submit"),
            }
            call = {
                "id": "v20.1-submit",
                "name": "submit",
                "arguments": json.dumps(arguments),
            }
        elif self.turn == 2:
            call = {
                "id": "v20.1-analyze",
                "name": "analyze",
                "arguments": json.dumps(self._audit("analyze")),
            }
        else:
            direction = next(
                name
                for name, value in _compact_neighbor_mapping(
                    observation["neighbors"]
                ).items()
                if value["reachable"] and value.get("affordable", True)
            )
            call = {
                "id": "v20.1-move",
                "name": "move",
                "arguments": json.dumps({
                    "direction": direction,
                    "steps": 1,
                    **self._audit("explore"),
                }),
            }
        return ProviderResponse(
            [call], None, {"input_tokens": 1, "output_tokens": 1}, self.model
        )


def _provider(config: dict[str, Any]):
    if config["adapter"] == "scripted":
        return _ScriptedV20_1Provider(config)
    return OpenAICompatibleProviderV20_1(config)


def _validate_core(name: str, arguments: dict[str, Any]) -> None:
    if name != "move":
        _validate_core_v20(name, arguments)
        return
    if set(arguments) != {"direction", "steps"}:
        raise ValueError("move requires exactly direction and steps")
    direction = arguments["direction"]
    steps = arguments["steps"]
    if not isinstance(direction, str) or direction not in DIRECTIONS:
        raise ValueError("direction is outside the six-neighbor enum")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise ValueError("steps must be an integer")
    if not 1 <= steps <= MAX_ROUTE_EDGES:
        raise ValueError("steps must be between 1 and 32")


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return route outcome metadata without duplicating measured readings."""

    fields = (
        "ok",
        "action",
        "requested_action",
        "requested_target_xyz",
        "resolved_target_node",
        "resolved_target_xyz",
        "geometry_only_route",
        "code",
        "detail",
        "directions",
        "requested_edges",
        "completed_edges",
        "route_complete",
        "route_stop_reason",
        "route_intervention_event",
        "next_inference_event",
        "origin_node",
        "arrival_node",
        "cost_s",
        "path_length_m",
        "physical_cost_s",
        "basis_state_version",
        "new_samples_since_last_analyze",
        "no_new_evidence",
        "observed_maxima",
        "transition_after_action",
        "submit_only_reason",
        "same_action_same_state",
        "same_action_same_state_count",
        "same_result_no_progress",
        "same_result_no_progress_count",
    )
    value = {key: result[key] for key in fields if key in result}
    if result.get("action") == "move" or result.get("requested_action") == "move":
        registered_event = {
            "segment_complete": "segment_complete",
            "native_boundary": "native_boundary",
            "physical_budget_unaffordable": "physical_budget_unaffordable",
        }.get(result.get("route_stop_reason"))
        if registered_event is not None:
            value.setdefault("route_intervention_event", registered_event)
            value.setdefault("next_inference_event", registered_event)
    return value


def _compact_navigation_topology(value: dict[str, Any]) -> dict[str, Any]:
    """Re-encode the complete static topology without repeated field names."""

    axes = dict(value["axes"])
    axis_order = ["x", "y", "z"]
    axis_value_order = [
        "origin_m", "nominal_spacing_m", "count", "index_min", "index_max",
        "max_coordinate_rounding_error_m", "coordinate_rule",
    ]
    directions = dict(value["directions_as_index_deltas"])
    direction_order = list(DIRECTIONS)
    return {
        "encoding": value["encoding"],
        "axis_order": axis_order,
        "axis_value_order": axis_value_order,
        "axis_values": [
            [axes[axis][field] for field in axis_value_order]
            for axis in axis_order
        ],
        "column_depth_count_rows_base36": value[
            "column_depth_count_rows_base36"
        ],
        "column_depth_count_legend": value["column_depth_count_legend"],
        "direction_order": direction_order,
        "direction_index_deltas": [directions[name] for name in direction_order],
        "depth_axis": value["depth_axis"],
        "node_count": value["node_count"],
        "coordinate_sha256": value["coordinate_sha256"],
    }


def _compact_neighbors(value: dict[str, Any]) -> dict[str, Any]:
    """Encode all six exact action-boundary records as one schema table."""

    schema = [
        "direction", "reachable", "reason", "target_node_index",
        "target_x_index", "target_y_index", "target_z_index",
        "target_x", "target_y", "target_z", "one_step_cost_s",
        "one_step_distance_m", "affordable", "visit_count",
        "last_sample_time_s",
    ]
    rows = []
    for direction in DIRECTIONS:
        row = dict(value[direction])
        target = dict(row.get("target_xyz") or {})
        rows.append([
            direction,
            row.get("reachable"),
            row.get("reason"),
            row.get("target_node_index"),
            row.get("target_x_index"),
            row.get("target_y_index"),
            row.get("target_z_index"),
            target.get("x"),
            target.get("y"),
            target.get("z"),
            row.get("one_step_cost_s"),
            row.get("one_step_distance_m"),
            row.get("affordable"),
            row.get("visit_count"),
            row.get("last_sample_time_s"),
        ])
    return {
        "encoding": "schema_rows_v1",
        "schema": schema,
        "rows": rows,
    }


def _compact_neighbor_mapping(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Decode the compact table for deterministic local contract adapters."""

    if value.get("encoding") != "schema_rows_v1":
        raise ValueError("unknown neighbor encoding")
    schema = list(value.get("schema") or [])
    if not schema or "direction" not in schema:
        raise ValueError("compact neighbor schema is incomplete")
    decoded: dict[str, dict[str, Any]] = {}
    for raw in value.get("rows") or []:
        if len(raw) != len(schema):
            raise ValueError("compact neighbor row length mismatch")
        row = dict(zip(schema, raw))
        direction = str(row.pop("direction"))
        decoded[direction] = row
    if tuple(decoded) != DIRECTIONS:
        raise ValueError("compact neighbor direction order mismatch")
    return decoded


def _compact_candidate_catalog(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-encode the unchanged, ordered catalog without repeated field names."""

    encoded = []
    for row in rows:
        raw = row["coordinates"]
        coordinates = [raw] if isinstance(raw, dict) else list(raw)
        encoded.append([
            row["candidate_id"],
            row["geometry"],
            [[point["x"], point["y"]] for point in coordinates],
        ])
    return {
        "encoding": "schema_rows_v1",
        "schema": ["candidate_id", "geometry", "coordinates_xy"],
        "coordinate_order": ["x", "y"],
        "point_rule": "one coordinate pair; line/polygon retain listed pair order",
        "row_count": len(encoded),
        "rows": encoded,
    }


def _compact_calibration(value: dict[str, Any]) -> dict[str, Any]:
    """Re-encode the complete calibration table as a fixed value matrix."""

    channels = sorted(value)
    depth_layers = ["surface", "middle", "bottom"]
    return {
        "encoding": "channel_depth_matrix_v1",
        "channel_order": channels,
        "depth_layer_order": depth_layers,
        "value_order": ["background_median", "declared_lod"],
        "values": [
            [
                [
                    value[channel][layer]["background_median"],
                    value[channel][layer]["declared_lod"],
                ]
                for layer in depth_layers
            ]
            for channel in channels
        ],
    }


def _compact_derived_summary(value: dict[str, Any]) -> dict[str, Any]:
    """Remove repeated channel field names from unchanged arithmetic summaries."""

    ranges = dict(value.get("observed_channel_ranges") or {})
    maxima = dict(value.get("observed_channel_maxima") or {})
    channels = sorted(set(ranges) | set(maxima))
    maximum_schema = [
        "value", "node_index", "x", "y", "z", "relative_dx_m",
        "relative_dy_m", "relative_dz_m", "distance_m",
    ]
    return {
        "visited_nodes": value.get("visited_nodes"),
        "total_nodes": value.get("total_nodes"),
        "visited_fraction": value.get("visited_fraction"),
        "channel_order": channels,
        "range_min_max": [
            [ranges[channel]["min"], ranges[channel]["max"]]
            if channel in ranges else [None, None]
            for channel in channels
        ],
        "maximum_schema": maximum_schema,
        "maxima": [
            [maxima[channel].get(field) for field in maximum_schema]
            if channel in maxima else [None for _ in maximum_schema]
            for channel in channels
        ],
        "directions_to_unvisited_neighbors": value.get(
            "directions_to_unvisited_neighbors", []
        ),
    }


def present_observation(
    env: EnvironmentV20_1,
    last_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Render the fixed compact measured-only state and exact recent route."""

    value = env.observation()
    rows = list(value.pop("cumulative_measurements"))
    directions_to_unvisited = list(
        value.pop("directions_to_unvisited_neighbors")
    )
    value["last_route_measurements"] = compact_route_measurements(
        list(value.get("last_route_measurements") or [])
    )
    value["navigation_topology"] = _compact_navigation_topology(
        dict(value["navigation_topology"])
    )
    value["neighbors"] = _compact_neighbors(dict(value["neighbors"]))
    value["candidates"] = _compact_candidate_catalog(list(value["candidates"]))
    value["calibration"] = _compact_calibration(dict(value["calibration"]))
    value["measurement_presentation_mode"] = PRESENTATION_MODE_V20_1
    value["cumulative_measurement_state"] = render_compact_measurements(
        rows,
        bounds=value["domain_bounds"],
        current_position=value["position"],
        raw_ledger=env.sample_ledger,
        current_mission_time_s=env.mission_time_s,
        z_layers=sorted({position[2] for position in env.graph.positions}),
        total_node_count=int(value["n_total_nodes"]),
    )
    value["derived_measurement_summary"] = _compact_derived_summary(derived_summary(
        rows,
        value["position"],
        directions_to_unvisited,
        {},
        int(value["n_total_nodes"]),
    ))
    # Exact duplicates of the compact map or frozen prompt are removed from the
    # per-round payload. No measurement, topology, candidate, or budget value is
    # removed.
    for redundant in (
        "sample_ledger_count",
        "sample_ledger_sha256",
        "control_granularity",
        "available_channels",
    ):
        value.pop(redundant, None)
    value["last_tool_results"] = [
        _public_result(result) for result in last_results
    ]
    # Keep immutable public task content in a stable prefix for providers that
    # support prefix caching.  No conversation or private reasoning is retained.
    static_fields = (
        "schema_version",
        "protocol_id",
        "instance_id",
        "phase",
        "navigation_topology",
        "domain_bounds",
        "n_total_nodes",
        "depth_axis",
        "calibration",
        "candidate_catalog_semantics",
        "candidates",
        "measurement_presentation_mode",
    )
    ordered = {field: value[field] for field in static_fields if field in value}
    ordered.update({field: item for field, item in value.items() if field not in ordered})
    return ordered


def request_payload_evidence(
    config: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure the exact complete request and fail before any network call."""

    payload = build_request_payload(config, messages, tools)
    encoded = encode_request_payload(payload)
    limit = int(config.get("max_request_utf8_bytes", MAX_REQUEST_UTF8_BYTES))
    if len(encoded) > limit:
        raise RequestPayloadTooLarge(len(encoded), limit)
    return {
        "request_utf8_bytes": len(encoded),
        "request_sha256": stable_hash(payload),
    }


def _terminate_active_invalid_guard(
    env: EnvironmentV20_1, streak: int,
) -> None:
    """Stop an unusable active-state interface loop without assigning a score."""

    if env.terminated:
        return
    env.terminated = True
    env.mode = "terminated"
    env.termination_reason = "active_invalid_schema_guard"
    env.events.append({
        "ok": False,
        "action": "engineering_safety",
        "code": "E_ACTIVE_INVALID_SCHEMA_GUARD",
        "detail": f"{streak} consecutive active-state schema/no-tool failures",
        "scoreable": False,
        "budget_after": env._budget_snapshot(),
    })


def run_model_v20_1(
    env: EnvironmentV20_1,
    config: dict[str, Any],
    *,
    resume_state: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    move_executor: Callable[
        [EnvironmentV20_1, list[str], dict[str, Any]], dict[str, Any]
    ] | None = None,
    known_cost_ceiling_cny: float | None = None,
    active_tools: list[dict[str, Any]] | None = None,
    core_validator: Callable[[str, dict[str, Any]], None] | None = None,
    route_builder: Callable[[str, dict[str, Any], EnvironmentV20_1], list[str]] | None = None,
    observation_presenter: Callable[[EnvironmentV20_1, list[dict[str, Any]]], dict[str, Any]] | None = None,
    config_validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    message_builder: Callable[
        [str, dict[str, Any]],
        tuple[list[dict[str, Any]], dict[str, Any]],
    ] | None = None,
    tool_selector: Callable[
        [EnvironmentV20_1, list[dict[str, Any]]], list[dict[str, Any]]
    ] | None = None,
    custom_action_executor: Callable[
        [str, dict[str, Any], EnvironmentV20_1, dict[str, Any]], dict[str, Any]
    ] | None = None,
    scratchpad_limit_chars: int = 0,
    decision_record_required: bool = True,
    submit_retry_invalid_only: bool = False,
    active_invalid_to_submit_only: bool = False,
) -> dict[str, Any]:
    """Run one v20.1 episode without retaining private model reasoning."""

    config = (config_validator or validate_model_config_v20_1)(config)
    if config.get("science_eligible") is True and move_executor is None:
        raise ValueError(
            "science-eligible v20.1 runs require the formal contract move_executor"
        )
    if move_executor is None:
        move_executor = lambda target, directions, _context: target.move(directions)
    active_tools = copy.deepcopy(active_tools or TOOLS_V20_1)
    core_validator = core_validator or _validate_core
    route_builder = route_builder or (
        lambda name, arguments, _target: (
            [str(arguments["direction"])] * int(arguments["steps"])
            if name == "move" else []
        )
    )
    observation_presenter = observation_presenter or present_observation
    provider = None
    prompt = str(config.get("system_prompt_v20_1") or SYSTEM_PROMPT_V20_1)
    resume_state = resume_state or {}
    calls: list[dict[str, Any]] = list(resume_state.get("provider_calls") or [])
    decisions: list[dict[str, Any]] = list(
        resume_state.get("decision_record_calls") or []
    )
    last_results: list[dict[str, Any]] = list(resume_state.get("last_results") or [])
    totals = dict(
        resume_state.get("token_usage")
        or {"input_tokens": 0, "output_tokens": 0}
    )
    invalid_submit_streak = int(resume_state.get("invalid_submit_streak", 0))
    active_invalid_streak = int(resume_state.get("active_invalid_streak", 0))
    agent_scratchpad = str(resume_state.get("agent_scratchpad") or "")
    if scratchpad_limit_chars < 0:
        raise ValueError("scratchpad_limit_chars must be nonnegative")
    if len(agent_scratchpad) > scratchpad_limit_chars:
        raise ValueError("resumed agent scratchpad exceeds the registered limit")
    unknown_charge_upper_cny = float(
        resume_state.get("unknown_charge_upper_cny", 0.0)
    )
    configured_cost_ceiling = config.get("hard_cost_ceiling_cny")
    effective_cost_ceiling = (
        float(known_cost_ceiling_cny)
        if known_cost_ceiling_cny is not None
        else float(configured_cost_ceiling)
        if configured_cost_ceiling is not None
        else None
    )
    submit_reserve_cny = float(config.get("submit_reserve_cny", 0.0))
    if effective_cost_ceiling is not None and not (
        0.0 <= submit_reserve_cny < effective_cost_ceiling
    ):
        raise ValueError("submit_reserve_cny must be nonnegative and below the hard cost ceiling")

    def checkpoint_system_events(events: list[dict[str, Any]]) -> None:
        if checkpoint is None:
            return
        for system_event in events:
            checkpoint({
                "stage": "system_event_committed",
                "event": copy.deepcopy(system_event),
                "token_usage": dict(totals),
                "last_results": list(last_results),
                "invalid_submit_streak": invalid_submit_streak,
                "active_invalid_streak": active_invalid_streak,
                "agent_scratchpad": agent_scratchpad,
                "termination_reason": env.termination_reason,
                "episode_mode": env.mode,
                "submit_only_reason": env.submit_only_reason,
            })

    while not env.terminated:
        set_provider_call_count = getattr(env, "set_provider_call_count", None)
        if callable(set_provider_call_count):
            set_provider_call_count(len(calls))
        pre_stage_event_count = len(env.events)
        apply_resource_stage = getattr(env, "apply_resource_stage", None)
        if callable(apply_resource_stage):
            apply_resource_stage()
        checkpoint_system_events(env.events[pre_stage_event_count:])
        known_cost_cny: float | None = None
        if effective_cost_ceiling is not None:
            known_cost_cny = (
                3.0 * int(totals["input_tokens"])
                + 9.0 * int(totals["output_tokens"])
            ) / 1_000_000 + unknown_charge_upper_cny
            if known_cost_cny >= effective_cost_ceiling:
                env.terminated = True
                env.mode = "terminated"
                legacy_guard = known_cost_ceiling_cny is not None and configured_cost_ceiling is None
                env.termination_reason = (
                    "engineering_known_cost_guard" if legacy_guard
                    else "engineering_hard_cost_guard"
                )
                env.events.append({
                    "ok": False,
                    "action": "engineering_safety",
                    "code": "E_KNOWN_COST_GUARD" if legacy_guard else "E_HARD_COST_GUARD",
                    "detail": (
                        f"conservative public-price cost CNY {known_cost_cny:.6f} "
                        f"reached hard ceiling CNY {effective_cost_ceiling:.6f}"
                    ),
                    "scoreable": False,
                    "conservative_public_price_cost_cny": round(known_cost_cny, 6),
                    "budget_after": env._budget_snapshot(),
                })
                checkpoint_system_events(env.events[-1:])
                break
            if (
                env.mode == "active"
                and known_cost_cny >= effective_cost_ceiling - submit_reserve_cny
            ):
                before_transition = len(env.events)
                env.enter_submit_only("cost_submit_reserve")
                checkpoint_system_events(env.events[before_transition:])
        if len(calls) > env.max_turns or env.turn > env.max_turns:
            raise ValueError("resumed provider-call state exceeds the protocol guard")
        if len(calls) >= env.max_turns or env.turn >= env.max_turns:
            before_transition = len(env.events)
            env.exhaust_provider_round_guard()
            checkpoint_system_events(env.events[before_transition:])
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
            before_transition = len(env.events)
            env.enter_submit_only("engineering_provider_round_cap")
            checkpoint_system_events(env.events[before_transition:])

        round_started_submit_only = env.mode == "submit_only"
        observation = observation_presenter(env, last_results)
        if scratchpad_limit_chars:
            observation["agent_scratchpad"] = agent_scratchpad
        observation_json = json.dumps(
            observation, ensure_ascii=False, separators=(",", ":")
        )
        observation_hash = stable_hash(observation)
        # Freeze the evidence set before the provider response.  In particular,
        # a route's audit record cannot cite a node discovered by that route.
        basis_measured_nodes = set(env.measurements)
        if message_builder is None:
            messages = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": "CURRENT OBSERVATION\n" + observation_json,
                },
            ]
            message_evidence: dict[str, Any] = {}
        else:
            messages, message_evidence = message_builder(prompt, observation)
        tools = (
            tool_selector(env, active_tools)
            if tool_selector is not None
            else SUBMIT_TOOL_V20_1 if env.mode == "submit_only" else active_tools
        )
        round_index = len(calls)
        payload_evidence: dict[str, Any] = {}
        try:
            payload_evidence = request_payload_evidence(config, messages, tools)
            request_cost_upper_cny = (
                3.0 * int(payload_evidence["request_utf8_bytes"])
                + 9.0 * int(config["max_output_tokens"])
            ) / 1_000_000
            if (
                configured_cost_ceiling is not None
                and
                effective_cost_ceiling is not None
                and known_cost_cny is not None
                and known_cost_cny + request_cost_upper_cny > effective_cost_ceiling
            ):
                if env.mode == "active":
                    before_transition = len(env.events)
                    env.enter_submit_only("cost_preflight_submit_reserve")
                    checkpoint_system_events(env.events[before_transition:])
                    continue
                env.terminated = True
                env.mode = "terminated"
                env.termination_reason = "engineering_hard_cost_preflight_guard"
                env.events.append({
                    "ok": False,
                    "action": "engineering_safety",
                    "code": "E_HARD_COST_PREFLIGHT_GUARD",
                    "detail": (
                        "conservative next-request upper bound would exceed "
                        "the registered hard cost ceiling"
                    ),
                    "scoreable": False,
                    "known_cost_cny": round(known_cost_cny, 6),
                    "request_cost_upper_cny": round(request_cost_upper_cny, 6),
                    "budget_after": env._budget_snapshot(),
                })
                checkpoint_system_events(env.events[-1:])
                break
            transport_retry = 0
            if callable(set_provider_call_count):
                # One logical provider round may contain transport retries, but
                # it remains one control decision and one retained call row.
                set_provider_call_count(len(calls) + 1)
            transport_max_retries = int(config.get("transport_max_retries", 0))
            backoff_s = float(config.get("transport_backoff_initial_s", 1.0))
            backoff_multiplier = float(config.get("transport_backoff_multiplier", 2.0))
            backoff_max_s = float(config.get("transport_backoff_max_s", 8.0))
            while True:
                try:
                    response = provider.complete(messages, tools)
                    break
                except RetriableTransportInterruption as exc:
                    # Treat the interrupted request as potentially billable.
                    # UTF-8 bytes are a conservative upper bound on input tokens.
                    unknown_charge_upper_cny += (
                        3.0 * int(payload_evidence["request_utf8_bytes"])
                        + 9.0 * int(config["max_output_tokens"])
                    ) / 1_000_000
                    if checkpoint is not None:
                        checkpoint({
                            "stage": "transport_interrupted",
                            "provider_round_index": round_index,
                            "transport_retry_index": transport_retry,
                            "transport_interruption": dict(exc.detail),
                            "unknown_charge_upper_cny": round(unknown_charge_upper_cny, 6),
                            "observation_hash": observation_hash,
                            "messages_hash": stable_hash(messages),
                            "tool_schema_hash": stable_hash(tools),
                            **payload_evidence,
                        })
                    if configured_cost_ceiling is not None and effective_cost_ceiling is not None:
                        interrupted_total = (
                            3.0 * int(totals["input_tokens"])
                            + 9.0 * int(totals["output_tokens"])
                        ) / 1_000_000 + unknown_charge_upper_cny
                        if interrupted_total + request_cost_upper_cny > effective_cost_ceiling:
                            raise
                    if transport_retry >= transport_max_retries:
                        raise
                    delay = min(backoff_s * (backoff_multiplier ** transport_retry), backoff_max_s)
                    time.sleep(max(0.0, delay))
                    transport_retry += 1
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
        except RetriableTransportInterruption:
            # No provider response is committed and no environment counter,
            # sample, or physical budget has advanced.  The durable contract
            # owns an explicit retry/resume decision.
            raise
        except RuntimeError as exc:
            if isinstance(exc, RequestPayloadTooLarge):
                payload_evidence = {
                    "request_utf8_bytes": exc.actual_bytes,
                    "request_sha256": None,
                    "request_utf8_byte_limit": exc.limit_bytes,
                }
            env.abort_infrastructure(str(exc))
            call_row = {
                "provider_round_index": round_index,
                "tool_calls": [],
                "chosen_tool_call": None,
                "cancelled_tool_calls": [],
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "reported_model": None,
                "provider_error": str(exc),
                "provider_error_detail": copy.deepcopy(
                    getattr(exc, "detail", None)
                ),
                "observation_hash": observation_hash,
                "observation_json": observation_json,
                "messages_hash": stable_hash(messages),
                "tool_schema_hash": stable_hash(tools),
                "unexpected_content_sha256": None,
                "unexpected_content_char_count": 0,
                **message_evidence,
                **payload_evidence,
            }
            calls.append(call_row)
            if checkpoint is not None:
                checkpoint({
                    "stage": "committed",
                    "provider_call": call_row,
                    "decision_record": None,
                    "event": env.events[-1],
                    "token_usage": dict(totals),
                    "last_results": [],
                    "invalid_submit_streak": invalid_submit_streak,
                    "active_invalid_streak": active_invalid_streak,
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
                {**item, "status": "cancelled_unobserved"}
                for item in cancelled
            ],
            "usage": usage,
            "reported_model": response.raw_model,
            "provider_error": None,
            "provider_error_detail": None,
            "observation_hash": observation_hash,
            "observation_json": observation_json,
            "messages_hash": stable_hash(messages),
            "tool_schema_hash": stable_hash(tools),
            "unexpected_content_sha256": (
                stable_hash(response.content) if response.content else None
            ),
            "unexpected_content_char_count": len(response.content or ""),
            **message_evidence,
            **payload_evidence,
        }
        calls.append(call_row)

        round_event_count_start = len(env.events)
        active_invalid_this_round = False
        if not chosen:
            active_invalid_this_round = not round_started_submit_only
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
                    "provider_call": call_row,
                    "decision_record": decision,
                    "token_usage": dict(totals),
                    "invalid_submit_streak": invalid_submit_streak,
                    "active_invalid_streak": active_invalid_streak,
                    "round_started_submit_only": round_started_submit_only,
                })
            result = env.record_invalid(
                "E_NO_TOOL_CALL", "provider returned no tool call"
            )
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
                if scratchpad_limit_chars:
                    raw_scratchpad = raw_arguments.pop("scratchpad", None)
                    if raw_scratchpad is not None:
                        if not isinstance(raw_scratchpad, str):
                            raise ValueError("scratchpad must be a string when present")
                        if len(raw_scratchpad) > scratchpad_limit_chars:
                            raise ValueError(
                                "scratchpad exceeds the registered character limit"
                            )
                        agent_scratchpad = raw_scratchpad
                core_arguments, raw_record = _split_arguments(raw_arguments)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
            decision = (
                _record(
                    raw_record,
                    basis_measured_nodes,
                    round_index=round_index,
                    observation_hash=observation_hash,
                    action_name=name,
                )
                if decision_record_required
                else {
                    "provider_round_index": round_index,
                    "basis_observation_hash": observation_hash,
                    "chosen_action_name": name,
                    "status": "not_requested",
                    "error": None,
                    "record": None,
                    "raw_record": {},
                    "cited_measured_node_ids": [],
                    "fed_back_to_agent": False,
                    "truth_visible": False,
                }
            )
            decision["decision_record_index"] = len(decisions)
            decisions.append(decision)
            if checkpoint is not None:
                checkpoint({
                    "stage": "received",
                    "provider_call": call_row,
                    "decision_record": decision,
                    "token_usage": dict(totals),
                    "invalid_submit_streak": invalid_submit_streak,
                    "active_invalid_streak": active_invalid_streak,
                    "round_started_submit_only": round_started_submit_only,
                })
            try:
                if not isinstance(raw_arguments, dict):
                    raise ValueError("arguments not an object")
                offered_tool_names = {
                    str(item.get("function", {}).get("name") or "")
                    for item in tools
                }
                if name not in offered_tool_names:
                    raise ValueError(f"{name} is unavailable in the current mode")
                core_validator(name, core_arguments)
                directions = route_builder(name, core_arguments, env)
                if custom_action_executor is not None:
                    result = custom_action_executor(
                        name,
                        core_arguments,
                        env,
                        {
                            "round_index": round_index,
                            "raw_provider_call": copy.deepcopy(raw),
                            "decision": copy.deepcopy(decision),
                            "observation_hash": observation_hash,
                            "move_executor": move_executor,
                            "directions": directions,
                        },
                    )
                elif name in {"move", "move_to_boundary"}:
                    if not 1 <= len(directions) <= env.max_steps:
                        # A typed but out-of-range route is semantically
                        # infeasible, not an interface parse failure. No native
                        # edge exists to add to the route transaction journal.
                        result = env.move(directions)
                    else:
                        result = move_executor(
                            env,
                            directions,
                            {
                                "round_index": round_index,
                                "raw_provider_call": copy.deepcopy(raw),
                                "decision": copy.deepcopy(decision),
                                "observation_hash": observation_hash,
                            },
                        )
                    if not isinstance(result, dict):
                        raise RuntimeError(
                            "formal move_executor must return one environment result object"
                        )
                elif name == "analyze":
                    result = env.analyze()
                else:
                    result = env.submit(_submission_v20(core_arguments))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                result = env.record_invalid("E_SCHEMA", str(exc), name)
                active_invalid_this_round = not round_started_submit_only
            if result in env.events:
                result.update({
                    "provider_round_index": round_index,
                    "tool_call_index": 0,
                    "cancelled_tool_call_count": len(cancelled),
                    "basis_observation_hash": observation_hash,
                    "decision_record_index": decision["decision_record_index"],
                })
            last_results = [result]

        if round_started_submit_only:
            active_invalid_streak = 0
        elif active_invalid_this_round:
            active_invalid_streak += 1
        else:
            active_invalid_streak = 0

        submit_schema_exhausted = False
        if round_started_submit_only:
            submitted = bool(
                last_results and last_results[0].get("action") == "submit"
            )
            if submitted:
                invalid_submit_streak = 0
            elif not env.terminated:
                if (
                    not submit_retry_invalid_only
                    or not last_results
                    or last_results[0].get("action") == "invalid"
                ):
                    invalid_submit_streak += 1
                control = env.control_protocol
                termination_policy = (
                    control.get("termination") or control.get("resource_control") or {}
                )
                limit = int(termination_policy.get(
                    "submit_only_schema_retries",
                    termination_policy.get("repairs_per_decision_point", 3),
                ))
                submit_schema_exhausted = invalid_submit_streak >= limit
        if (
            not env.terminated
            and (len(calls) >= env.max_turns or env.turn >= env.max_turns)
        ):
            env.exhaust_provider_round_guard()
        elif not env.terminated and submit_schema_exhausted:
            env.exhaust_submit_schema_retries()
        elif (
            not env.terminated
            and active_invalid_streak >= ACTIVE_INVALID_STREAK_LIMIT
        ):
            if active_invalid_to_submit_only:
                env.enter_submit_only("active_invalid_repair_exhausted")
                active_invalid_streak = 0
            else:
                _terminate_active_invalid_guard(env, active_invalid_streak)
        if checkpoint is not None:
            event = next(
                (
                    item
                    for item in reversed(env.events)
                    if item.get("provider_round_index") == round_index
                ),
                None,
            )
            checkpoint({
                "stage": "committed",
                "provider_call": call_row,
                "decision_record": (
                    decisions[-1]
                    if decisions
                    and decisions[-1].get("provider_round_index") == round_index
                    else None
                ),
                "event": event,
                "token_usage": dict(totals),
                "last_results": list(last_results),
                "invalid_submit_streak": invalid_submit_streak,
                "active_invalid_streak": active_invalid_streak,
                "agent_scratchpad": agent_scratchpad,
                "round_started_submit_only": round_started_submit_only,
                "termination_reason": env.termination_reason,
            })
            # Harness-originated transitions are durable events in their own
            # right.  They must never be inferred later from termination
            # metadata, because that made v20.2 replay one event short.
            checkpoint_system_events([
                item for item in env.events[round_event_count_start:]
                if item is not event
                and item.get("action") in {
                    "engineering_safety", "control_transition"
                }
            ])

    if len(calls) > env.max_turns or env.turn > env.max_turns:
        raise ValueError("provider-call guard exceeded before episode finalization")
    completed_responses = [
        row for row in calls if row.get("provider_error") is None
    ]
    identities = [row.get("reported_model") for row in completed_responses]
    expected_record_count = sum(
        row.get("chosen_tool_call") is not None for row in completed_responses
    )
    captured_records = sum(
        row.get("chosen_action_name") is not None for row in decisions
    )
    valid_records = sum(
        row.get("status") in {"ok", "not_requested"} for row in decisions
    )
    return {
        "data_contract_version": "v20.1.3d-axis-run.1",
        "prompt_id": "laketrace-v20.1-3d-axis-run",
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
        "unknown_charge_upper_cny": round(unknown_charge_upper_cny, 6),
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
        "presentation_mode": PRESENTATION_MODE_V20_1,
        "multi_call_cancelled_count": sum(
            len(row["cancelled_tool_calls"]) for row in calls
        ),
        "request_utf8_bytes": [
            int(row["request_utf8_bytes"])
            for row in calls
            if row.get("request_utf8_bytes") is not None
        ],
        "chain_of_thought_requested": False,
        "chain_of_thought_stored": False,
        "provider_reasoning_enabled": False,
        "decision_record_required": decision_record_required,
        "submit_retry_invalid_only": submit_retry_invalid_only,
        "max_output_tokens": int(config["max_output_tokens"]),
        "max_request_utf8_bytes": int(config["max_request_utf8_bytes"]),
        "active_invalid_streak_limit": ACTIVE_INVALID_STREAK_LIMIT,
        "active_invalid_streak_final": active_invalid_streak,
        "active_invalid_schema_guard_triggered": (
            env.termination_reason == "active_invalid_schema_guard"
        ),
        "active_invalid_submit_only_transition_count": sum(
            event.get("action") == "control_transition"
            and event.get("reason") == "active_invalid_repair_exhausted"
            for event in env.events
        ),
        "agent_scratchpad_final": agent_scratchpad,
        "agent_scratchpad_sha256": stable_hash(agent_scratchpad),
        "agent_scratchpad_limit_chars": scratchpad_limit_chars,
    }


__all__ = [
    "ACTIVE_INVALID_STREAK_LIMIT",
    "MAX_OUTPUT_TOKENS",
    "MAX_REQUEST_UTF8_BYTES",
    "MAX_ROUTE_EDGES",
    "MOVE_TOOL_V20_1",
    "PRESENTATION_MODE_V20_1",
    "RequestPayloadTooLarge",
    "SUBMIT_TOOL_V20_1",
    "SYSTEM_PROMPT_V20_1",
    "TOOLS_V20_1",
    "present_observation",
    "request_payload_evidence",
    "run_model_v20_1",
    "validate_model_config_v20_1",
]
