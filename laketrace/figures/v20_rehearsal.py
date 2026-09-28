"""Deterministic capability-informed rehearsal data for v20 paper figures.

The generator uses the registered 84 F1 tasks, evaluator truth and the real
six-axis :class:`~laketrace.env.graph.NodeGraph`.  Every synthetic move follows
one native adjacent edge.  Answers are passed through the production
``score_submission`` function; no L1--L5 result is assigned directly.

These are *layout rehearsal scenarios*, not model measurements.  They contain
no provider response, provider identity claim, replay verdict or provenance
claim.  The companion data layer stamps every row and artifact accordingly.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from laketrace.data.store import DataStore, Frame
from laketrace.env.graph import Edge, NodeGraph
from laketrace.scoring.scorer import score_submission
from laketrace.util import load_config, stable_hash
from laketrace.figures.v20_paper_data import (
    DEFAULT_CONTROL_ROOT,
    PAPER_SCHEMA_VERSION,
    PROTOCOL_ID,
    SYNTHETIC_KIND,
    SYNTHETIC_WATERMARK,
    PaperDataBundle,
    attach_model_behavior_tertiles,
    attach_storyboard_selection,
    build_task_geometry,
    decision_rows_from_episode,
    load_registered_controls,
    outcome_from_episode,
    step_rows_from_episode,
    task_dstar,
    write_bundle,
)


CAPABILITY_FIELDS = (
    "detection", "footprint_localization", "source_inference", "mechanism",
    "decision_calibration", "navigation", "efficiency", "analysis_use",
    "tool_stability", "vertical_reasoning",
)


DEFAULT_SCENARIO: dict[str, Any] = {
    "schema_version": "laketrace-v20-rehearsal-scenario-v1",
    "scenario_id": "v20-figure-rehearsal-s1",
    "dataset_kind": SYNTHETIC_KIND,
    "watermark": SYNTHETIC_WATERMARK,
    "random_seed": 20260830,
    "seeds": [0, 1, 2],
    "max_moves_per_episode": 140,
    "research_status": "neutral weak-prior fallback for layout testing only",
    "weak_prior_script": {
        "task_effect_scale": .12,
        "seed_effect_scale": .06,
        "maximum_absolute_model_main_effect": .025,
        "fixed_total_rank": None,
    },
    "models": [
        {"id": "gpt-5.6-sol", "actor_key": "gpt_5_6_sol", "display_name": "GPT-5.6 Sol"},
        {"id": "deepseek-v4-pro-0813", "actor_key": "deepseek_v4_pro_0813", "display_name": "DeepSeek V4 Pro 0813"},
        {"id": "claude-opus-5", "actor_key": "claude_opus_5", "display_name": "Claude Opus 5"},
        {"id": "kimi-k3", "actor_key": "kimi_k3", "display_name": "Kimi K3"},
        {"id": "glm-5.3", "actor_key": "glm_5_3", "display_name": "GLM 5.3"},
        {"id": "gemini-3.7-flash", "actor_key": "gemini_3_7_flash", "display_name": "Gemini 3.7 Flash"},
    ],
}


def _unit(*parts: Any) -> float:
    digest = stable_hash({"domain": "v20-paper-rehearsal-v1", "parts": parts})
    return int(digest[:16], 16) / float(2**64)


def _trial(probability: float, *parts: Any) -> bool:
    return _unit(*parts) < max(0.0, min(1.0, probability))


def _pick(values: Sequence[Any], *parts: Any) -> Any:
    if not values:
        raise ValueError("cannot choose from an empty sequence")
    index = int(stable_hash({"domain": "v20-paper-choice-v1", "parts": parts})[:16], 16)
    return values[index % len(values)]


def _weak_capability_profiles(
    models: Sequence[Mapping[str, Any]], *, random_seed: int, maximum_absolute: float,
) -> dict[str, dict[str, float]]:
    """Create symmetric, zero-centred rehearsal offsets without model stereotypes."""

    maximum_absolute = max(0.0, min(float(maximum_absolute), .025))
    profiles = {str(model["id"]): {} for model in models}
    for field in CAPABILITY_FIELDS:
        raw = [
            2.0 * _unit(
                random_seed, str(model.get("actor_key") or model["id"]), field,
                "weak_model_main_effect",
            ) - 1.0
            for model in models
        ]
        mean = sum(raw) / len(raw)
        centred = [value - mean for value in raw]
        span = max((abs(value) for value in centred), default=0.0)
        offsets = [
            value / span * maximum_absolute if span else 0.0 for value in centred
        ]
        for model, offset in zip(models, offsets):
            profiles[str(model["id"])][field] = .5 + offset
    return profiles


def _capabilities(model: Mapping[str, Any]) -> dict[str, float]:
    raw = model.get("capabilities") or model.get("behavior_profile") or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"model {model.get('id')} capabilities must be a mapping")
    aliases = {
        "footprint_localization": ("footprint_localization", "anomaly_mapping", "mapping"),
        "source_inference": ("source_inference", "transport_reasoning", "source_reasoning"),
        "decision_calibration": ("decision_calibration", "calibration"),
        "navigation": ("navigation", "spatial_navigation"),
        "analysis_use": ("analysis_use", "analysis"),
        "tool_stability": ("tool_stability", "tool_use"),
        "vertical_reasoning": ("vertical_reasoning", "three_d_reasoning"),
        "detection": ("detection",),
        "mechanism": ("mechanism", "mechanism_classification"),
        "efficiency": ("efficiency", "navigation_efficiency"),
    }
    output: dict[str, float] = {}
    defaults = model.get("capability_defaults") or {}
    for field in CAPABILITY_FIELDS:
        value = next((raw[name] for name in aliases[field] if name in raw), defaults.get(field))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"model {model.get('id')} lacks numeric capability {field}")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"model {model.get('id')} capability {field} outside [0,1]")
        output[field] = number
    return output


def load_scenario(value: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load and normalize the research scenario expected under configs/figures."""

    source_path: str | None = None
    if value is None:
        scenario = copy.deepcopy(DEFAULT_SCENARIO)
    elif isinstance(value, Mapping):
        scenario = copy.deepcopy(dict(value))
    else:
        source_path = str(Path(value))
        scenario = load_config(value)

    # Direct adapter for configs/figures/v20_rehearsal_s1.json.  Keeping the
    # publication contract as the single source avoids a second shadow config.
    contract_shape = (
        scenario.get("mode") == "synthetic_rehearsal"
        and isinstance(scenario.get("model_order"), Mapping)
        and isinstance(scenario.get("watermark"), Mapping)
    )
    if contract_shape:
        if scenario.get("synthetic") is not True or scenario.get("rehearsal") is not True:
            raise ValueError("figure rehearsal contract must be explicitly synthetic")
        if (scenario.get("formal_use") or {}).get("eligible") is not False:
            raise ValueError("figure rehearsal contract must be ineligible for formal use")
        watermark = scenario.get("watermark") or {}
        if not isinstance(watermark, Mapping) or watermark.get("exact_text") != SYNTHETIC_WATERMARK:
            raise ValueError("figure rehearsal contract lacks the exact watermark")
        story = scenario.get("protocol_story") or {}
        if story.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("figure rehearsal contract uses a different protocol")
        order = scenario.get("model_order") or {}
        source_models = order.get("models") if isinstance(order, Mapping) else None
        if not isinstance(source_models, list):
            raise ValueError("figure rehearsal contract lacks model_order.models")
        scenario["models"] = [
            {
                **dict(model),
                "id": str(model.get("official_id") or model.get("official_api_id") or ""),
                "actor_key": str(model.get("actor_key") or ""),
                "display_name": str(model.get("display_name") or ""),
                "snapshot": "synthetic-rehearsal-no-provider-snapshot",
            }
            for model in source_models
        ]
        scenario["dataset_kind"] = SYNTHETIC_KIND
        scenario["scenario_id"] = str(scenario.get("rehearsal_id") or "v20-figure-rehearsal")
        scenario["seeds"] = list(range(int(story.get("seed_count_for_layout", 3))))
        scenario["max_moves_per_episode"] = int(scenario.get("max_moves_per_episode", 140))
        scenario["research_status"] = str(scenario.get("status") or "layout_only_never_formal")
        scenario["research_manifest"] = source_path

    if scenario.get("dataset_kind") != SYNTHETIC_KIND:
        raise ValueError("rehearsal scenario must declare synthetic_layout_rehearsal")
    watermark_value = scenario.get("watermark")
    if isinstance(watermark_value, Mapping):
        watermark_value = watermark_value.get("exact_text")
    if watermark_value not in (None, SYNTHETIC_WATERMARK):
        raise ValueError("rehearsal watermark is not the registered synthetic label")
    scenario["watermark"] = SYNTHETIC_WATERMARK
    seeds = [int(value) for value in scenario.get("seeds", [])]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("rehearsal requires exactly three distinct seeds")
    scenario["seeds"] = seeds
    raw_models = scenario.get("models")
    if isinstance(raw_models, Mapping):
        models = [dict(value, id=key) for key, value in raw_models.items()]
    elif isinstance(raw_models, list):
        models = [dict(model) for model in raw_models]
    else:
        raise ValueError("rehearsal models must be a list or mapping")
    if len(models) != 6:
        raise ValueError(f"rehearsal requires six models; observed {len(models)}")
    ids = [str(model.get("id") or model.get("model_id") or model.get("actor") or "") for model in models]
    if any(not value for value in ids) or len(set(ids)) != 6:
        raise ValueError("rehearsal model IDs must be six non-empty unique values")
    weak_prior = scenario.get("weak_prior_script") or {}
    if not isinstance(weak_prior, Mapping):
        raise ValueError("weak_prior_script must be a mapping")
    task_scale = float(weak_prior.get("task_effect_scale", .12))
    seed_scale = float(weak_prior.get("seed_effect_scale", .06))
    maximum_model = float(weak_prior.get("maximum_absolute_model_main_effect", .025))
    if not (0.0 < maximum_model < seed_scale < task_scale):
        raise ValueError(
            "rehearsal variance hierarchy must satisfy model < seed < task"
        )
    prepared = [
        {
            **model,
            "id": actor,
            "actor_key": str(model.get("actor_key") or actor),
        }
        for actor, model in zip(ids, models)
    ]
    derived_profiles = _weak_capability_profiles(
        prepared,
        random_seed=int(scenario.get("random_seed", 20260830)),
        maximum_absolute=maximum_model,
    )
    normalized = []
    for actor, model in zip(ids, prepared):
        explicit = model.get("capabilities") or model.get("behavior_profile")
        capabilities = (
            _capabilities(model)
            if isinstance(explicit, Mapping) and not contract_shape
            else derived_profiles[actor]
        )
        normalized.append({
            **model,
            "id": actor,
            "display_name": str(model.get("display_name") or model.get("name") or actor),
            "snapshot": str(model.get("snapshot") or model.get("model_snapshot") or "research-scenario"),
            "capabilities": capabilities,
        })
    scenario["models"] = normalized
    scenario["scenario_id"] = str(scenario.get("scenario_id") or "v20-figure-rehearsal")
    scenario["max_moves_per_episode"] = int(scenario.get("max_moves_per_episode", 140))
    if scenario["max_moves_per_episode"] < 1:
        raise ValueError("max_moves_per_episode must be positive")
    scenario["task_effect_scale"] = task_scale
    scenario["seed_effect_scale"] = seed_scale
    scenario["maximum_absolute_model_main_effect"] = maximum_model
    scenario["overall_rank_defined"] = False
    return scenario


def _graph_index(graph: NodeGraph, value: Mapping[str, Any]) -> int:
    return graph.index_by_xyz[(float(value["x"]), float(value["y"]), float(value["z"]))]


def _shortest(graph: NodeGraph, start: int, targets: set[int]) -> tuple[int, ...]:
    if not targets:
        return (start,)
    try:
        return graph.shortest_path(start, targets)[1]
    except ValueError:
        return (start,)


def _source_normalized_distance(
    point: tuple[float, float, float], truth: Mapping[str, Any], tolerance: Mapping[str, Any],
) -> float:
    values = []
    for source in truth.get("source_nodes") or []:
        horizontal = math.dist(point[:2], (float(source["x"]), float(source["y"])))
        vertical = abs(point[2] - float(source["z"]))
        values.append(math.sqrt(
            (horizontal / float(tolerance["horizontal_m"])) ** 2
            + (vertical / float(tolerance["vertical_m"])) ** 2
        ))
    return min(values, default=float("inf"))


def _route_assets(
    store: DataStore, frame: Frame, graph: NodeGraph, task: Mapping[str, Any],
    truth: Mapping[str, Any],
) -> dict[str, Any]:
    access = store.access_points["access_points"][task["domain_id"]]
    start = graph.index_by_xyz[(float(access["x"]), float(access["y"]), float(access["z"]))]
    plume_targets = {_graph_index(graph, row) for row in truth.get("L2_plume_positions") or []}
    source_targets = {_graph_index(graph, row) for row in truth.get("source_nodes") or []}
    peak_targets = {_graph_index(graph, row) for row in truth.get("L2_peak_positions") or []}
    plume_path = _shortest(graph, start, plume_targets)
    source_path = _shortest(graph, start, source_targets)
    peak_path = _shortest(graph, start, peak_targets)
    tolerance = store.protocol["tolerances"][truth["waterbody"]]
    if plume_targets and source_targets:
        far_plume = max(
            plume_targets,
            key=lambda node: (
                _source_normalized_distance(graph.positions[node], truth, tolerance), -node,
            ),
        )
        footprint_path = _shortest(graph, start, {far_plume})
    else:
        footprint_path = plume_path
    first_plume = next(
        (index for index, node in enumerate(plume_path) if node in plume_targets), len(plume_path)
    )
    avoid_path = plume_path[:max(1, first_plume)]
    target = int(stable_hash({"task": task["instance_id"], "purpose": "explore"})[:16], 16) % len(graph.positions)
    if target == start:
        target = (target + max(1, len(graph.positions) // 2)) % len(graph.positions)
    explore_path = _shortest(graph, start, {target})
    wrong_l2 = next(
        (node for node in (start, *reversed(explore_path)) if node not in plume_targets), start,
    )
    wrong_l4 = max(
        {start, wrong_l2, *(explore_path[-4:] if explore_path else ())},
        key=lambda node: (_source_normalized_distance(graph.positions[node], truth, tolerance), -node),
    )
    return {
        "start": start,
        "plume_targets": plume_targets,
        "source_targets": source_targets,
        "peak_targets": peak_targets,
        "plume_path": plume_path,
        "source_path": source_path,
        "peak_path": peak_path,
        "footprint_path": footprint_path,
        "avoid_path": avoid_path,
        "explore_path": explore_path,
        "wrong_l2": wrong_l2,
        "wrong_l4": wrong_l4,
    }


def _edge(graph: NodeGraph, source: int, target: int) -> Edge:
    for value in graph.neighbor_map[source].values():
        if value.target == target:
            return value
    raise ValueError(f"synthetic route contains a non-adjacent edge: {source}->{target}")


def _add_detours(
    route: Sequence[int], graph: NodeGraph, probability: float, *parts: Any,
) -> tuple[int, ...]:
    if len(route) < 2:
        return tuple(route)
    output = [route[0]]
    for index, node in enumerate(route[1:], 1):
        previous = output[-1]
        _edge(graph, previous, node)
        output.append(node)
        if index < len(route) - 1 and _trial(probability, *parts, "detour", index):
            output.extend((previous, node))
    for left, right in zip(output, output[1:]):
        _edge(graph, left, right)
    return tuple(output)


def _trim_route(
    route: Sequence[int], graph: NodeGraph, task: Mapping[str, Any], max_moves: int,
) -> tuple[int, ...]:
    if not route:
        raise ValueError("synthetic route is empty")
    budget = task["budget"]
    range_limit = float(budget.get("range_m", float("inf"))) * .92
    endurance_limit = float(budget["endurance_s"]) * .92
    distance = duration = 0.0
    output = [route[0]]
    for source, target in zip(route, route[1:]):
        if len(output) - 1 >= max_moves:
            break
        edge = _edge(graph, source, target)
        dz = graph.positions[target][2] - graph.positions[source][2]
        speed = float(
            budget["descend_ms"] if dz > 0
            else budget["ascend_ms"] if dz < 0
            else budget["cruise_ms"]
        )
        candidate_distance = distance + edge.distance_m
        candidate_duration = duration + edge.distance_m / speed
        if candidate_distance > range_limit or candidate_duration > endurance_limit:
            break
        distance, duration = candidate_distance, candidate_duration
        output.append(target)
    return tuple(output)


def _reading(
    frame: Frame, graph: NodeGraph, node_index: int, *, sequence: int, visit: int,
    mission_time_s: float, mode: str,
) -> dict[str, Any]:
    node = frame.by_xyz[graph.positions[node_index]]
    return {
        "node_index": node_index,
        "frame_node_id": node.index,
        "x": node.x, "y": node.y, "z": node.z,
        "bed_depth_m": node.bed_depth,
        "channels": dict(node.channels), "flow": {
            "u_m_s": node.flow["u_m_s"],
            "v_m_s": node.flow["v_m_s"],
            "w_m_s": node.flow["w_m_s"],
        },
        "phase": "F1", "mode": mode, "mission_time_s": mission_time_s,
        "sample_sequence": sequence, "visit_index": visit,
    }


def _budget_snapshot(
    task: Mapping[str, Any], *, used_s: float, used_m: float, turns: int,
) -> dict[str, Any]:
    raw = task["budget"]
    endurance = float(raw["endurance_s"])
    range_limit = float(raw.get("range_m", endurance * float(raw["cruise_ms"])))
    return {
        "endurance": {"limit": endurance, "used": used_s, "remaining": max(0.0, endurance - used_s)},
        "range": {"limit": range_limit, "used": used_m, "remaining": max(0.0, range_limit - used_m)},
        "provider_round_guard": {"limit": 7201, "used": turns, "remaining": 7201 - turns},
    }


def _decision_record(
    *, episode_id: str, index: int, action: str, current: int, confidence: float,
) -> dict[str, Any]:
    intent = "explore" if action == "move" else "analyze" if action == "analyze" else "submit"
    note = {
        "move": "synthetic rehearsal route follows currently represented evidence",
        "analyze": "synthetic rehearsal consolidates measured samples",
        "submit": "synthetic rehearsal records the terminal answer",
    }[action]
    record = {
        "intent": intent,
        "evidence_node_ids_csv": str(current),
        "confidence": round(max(0.0, min(1.0, confidence)), 6),
        "decision_note": note,
    }
    return {
        "basis_observation_hash": "synthetic:" + stable_hash({
            "episode_id": episode_id, "round": index, "node": current,
        }),
        "chosen_action_name": action,
        "decision_record_index": index,
        "provider_round_index": index,
        "record": record,
        "raw_record": record,
        "status": "synthetic_rehearsal",
        "fed_back_to_agent": False,
        "truth_visible": None,
    }


def _make_episode(
    *, store: DataStore, frame: Frame, graph: NodeGraph, task: Mapping[str, Any],
    truth: Mapping[str, Any], assets: Mapping[str, Any], model: Mapping[str, Any],
    seed: int, scenario: Mapping[str, Any], mechanisms: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    actor = str(model["id"])
    instance_id = str(task["instance_id"])
    episode_id = f"SYN-EPISODE::{actor}::{instance_id}::{seed}"
    capabilities = model["capabilities"]
    rehearsal_behavior = model.get("rehearsal_behavior") or {}
    if not isinstance(rehearsal_behavior, Mapping):
        raise ValueError(f"model {actor} rehearsal_behavior must be a mapping")
    probability_offsets = rehearsal_behavior.get("probability_offsets") or {}
    if not isinstance(probability_offsets, Mapping):
        raise ValueError(f"model {actor} probability_offsets must be a mapping")

    def probability_offset(name: str) -> float:
        """Return an explicit, bounded rehearsal-only scenario adjustment.

        These offsets exist solely to exercise figure layouts against visibly
        different model profiles.  They never enter the formal harness and are
        retained verbatim in the synthetic scenario configuration.
        """

        return max(-.35, min(.35, float(probability_offsets.get(name, 0.0))))
    key = scenario["scenario_id"], scenario.get("random_seed"), instance_id, seed
    tier = int(truth.get("difficulty_tier") or 1)
    dstar = task_dstar(store, truth) or 0.0
    difficulty = .055 * max(0, tier - 1)
    displacement = min(.24, math.log1p(dstar) * .065)
    detectable = bool(truth.get("detectable_source"))
    task_effect = float(scenario["task_effect_scale"]) * (
        2.0 * _unit(scenario.get("random_seed"), instance_id, "task_effect") - 1.0
    )
    seed_effect = float(scenario["seed_effect_scale"]) * (
        2.0 * _unit(scenario.get("random_seed"), seed, "seed_effect") - 1.0
    )

    def variation(metric: str) -> float:
        # Actor interactions change sign across task/seed/metric; there is no
        # actor-wide advantage and their amplitude stays below the seed effect.
        interaction = .018 * (
            2.0 * _unit(
                scenario.get("random_seed"), actor, instance_id, seed, metric,
                "actor_interaction",
            ) - 1.0
        )
        return task_effect + seed_effect + interaction

    if detectable:
        p_detect = (
            .60 + .30 * (capabilities["detection"] - .5)
            - difficulty + variation("detect") + probability_offset("detect")
        )
        detected = _trial(p_detect, *key, "detect")
        p_enter = (
            .54 + .18 * (capabilities["navigation"] - .5)
            + .18 * (capabilities["footprint_localization"] - .5)
            - difficulty + variation("enter_footprint")
            + probability_offset("enter_footprint")
        )
        enter_footprint = detected and _trial(p_enter, *key, "enter_footprint")
        p_source_approach = (
            .48 + .12 * (capabilities["navigation"] - .5)
            + .18 * (capabilities["source_inference"] - .5)
            + .10 * (capabilities["vertical_reasoning"] - .5)
            - difficulty - displacement * .55 + variation("approach_source")
            + probability_offset("approach_source")
        )
        approach_source = enter_footprint and _trial(
            p_source_approach, *key, "approach_source",
        )
        base_route = (
            assets["source_path"] if approach_source
            else assets["footprint_path"] if enter_footprint
            else assets["avoid_path"]
        )
    else:
        false_positive = _trial(
            .10 - .12 * (capabilities["detection"] - .5)
            - variation("false_positive") + probability_offset("false_positive"),
            *key, "false_positive",
        )
        detected = false_positive
        base_route = assets["explore_path"]

    default_detour_probability = max(
        .02,
        .10 - .10 * (capabilities["efficiency"] - .5) - variation("efficiency") * .35,
    )
    detour_probability = max(0.0, min(
        .45, float(rehearsal_behavior.get("detour_probability", default_detour_probability)),
    ))
    route = _add_detours(base_route, graph, detour_probability, *key)
    route = _trim_route(
        route, graph, task, int(scenario["max_moves_per_episode"]),
    )
    if not route:
        route = (assets["start"],)

    events: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    visits: dict[int, int] = {}
    used_s = used_m = 0.0
    sample_sequence = 1
    visits[route[0]] = 1
    ledger.append(_reading(
        frame, graph, route[0], sequence=sample_sequence, visit=1,
        mission_time_s=used_s, mode="initial",
    ))
    default_analysis_interval = max(6, round(24 - 16 * capabilities["analysis_use"]))
    analysis_interval = max(2, min(
        40, int(rehearsal_behavior.get("analysis_interval", default_analysis_interval)),
    ))
    current = route[0]
    for move_index, target in enumerate(route[1:], 1):
        edge = _edge(graph, current, target)
        dz = graph.positions[target][2] - graph.positions[current][2]
        budget = task["budget"]
        speed = float(
            budget["descend_ms"] if dz > 0
            else budget["ascend_ms"] if dz < 0
            else budget["cruise_ms"]
        )
        cost_s = edge.distance_m / speed
        used_s += cost_s
        used_m += edge.distance_m
        sample_sequence += 1
        visits[target] = visits.get(target, 0) + 1
        reading = _reading(
            frame, graph, target, sequence=sample_sequence, visit=visits[target],
            mission_time_s=used_s, mode="transit",
        )
        ledger.append(reading)
        round_index = len(events)
        event = {
            "action": "move", "ok": True, "provider_round_index": round_index,
            "origin_node": current, "arrival_node": target, "direction": edge.direction,
            "cost_s": cost_s, "path_length_m": edge.distance_m, "reading": reading,
            "same_result_no_progress": visits[target] > 1,
            "cancelled_tool_call_count": int(_trial(
                .035 * (1 - capabilities["tool_stability"]), *key, "cancelled", move_index,
            )),
            "budget_after": _budget_snapshot(
                task, used_s=used_s, used_m=used_m, turns=round_index + 1,
            ),
        }
        events.append(event)
        progress = move_index / max(1, len(route) - 1)
        confidence = .08 + progress * (.45 + .35 * capabilities["decision_calibration"])
        decisions.append(_decision_record(
            episode_id=episode_id,
            index=round_index, action="move", current=target, confidence=confidence,
        ))
        current = target
        if move_index % analysis_interval == 0:
            used_s += 30.0
            round_index = len(events)
            events.append({
                "action": "analyze", "ok": True, "provider_round_index": round_index,
                "physical_cost_s": 30.0,
                "new_samples_since_last_analyze": analysis_interval,
                "no_new_evidence": False, "visited_nodes": len(visits),
                "cancelled_tool_call_count": 0,
                "budget_after": _budget_snapshot(
                    task, used_s=used_s, used_m=used_m, turns=round_index + 1,
                ),
            })
            decisions.append(_decision_record(
                episode_id=episode_id,
                index=round_index, action="analyze", current=current,
                confidence=.22 + .60 * capabilities["decision_calibration"] * progress,
            ))

    plume_positions = truth.get("L2_plume_positions") or []
    source_nodes = truth.get("source_nodes") or []
    if detectable and detected:
        p_l2 = (
            .55 + .22 * (capabilities["footprint_localization"] - .5)
            + (.16 if any(
                (round(row["x"], 9), round(row["y"], 9), round(row["z"], 9))
                in {(round(value["x"], 9), round(value["y"], 9), round(value["z"], 9)) for value in plume_positions}
                for row in ledger
            ) else 0.0) - difficulty + variation("l2_answer")
            + probability_offset("l2_answer")
        )
        l2_correct = _trial(p_l2, *key, "l2_answer")
        plume_answer = _pick(plume_positions, *key, "plume_position") if l2_correct else None
        if plume_answer is None:
            wrong = graph.positions[int(assets["wrong_l2"])]
            plume_answer = {"x": wrong[0], "y": wrong[1], "z": wrong[2]}
        p_l3 = (
            .56 + .24 * (capabilities["mechanism"] - .5)
            - difficulty + variation("mechanism") + probability_offset("mechanism")
        )
        mechanism_correct = _trial(p_l3, *key, "mechanism")
        mechanism = truth.get("L3_mechanism") if mechanism_correct else _pick(
            [value for value in mechanisms if value != truth.get("L3_mechanism")],
            *key, "wrong_mechanism",
        )
        p_l4 = (
            .43 + .16 * (capabilities["source_inference"] - .5)
            + .08 * (capabilities["vertical_reasoning"] - .5)
            + .06 * (capabilities["navigation"] - .5)
            - difficulty - displacement + variation("source_answer")
            + probability_offset("source_answer")
        )
        source_correct = _trial(p_l4, *key, "source_answer")
        if source_correct:
            source_answer = _pick(source_nodes, *key, "source_position")
            candidate = truth.get("L4_candidate_id")
        elif truth.get("L2_peak_positions") and _trial(.72, *key, "peak_as_source"):
            source_answer = _pick(truth["L2_peak_positions"], *key, "peak_source_error")
            candidate = None
        else:
            wrong = graph.positions[int(assets["wrong_l4"])]
            source_answer = {"x": wrong[0], "y": wrong[1], "z": wrong[2]}
            candidate = None
        answer: dict[str, Any] = {
            "anomaly": "anomaly",
            "peak_position": {axis: float(plume_answer[axis]) for axis in "xyz"},
            "mechanism": mechanism,
            "source_position": {axis: float(source_answer[axis]) for axis in "xyz"},
            "source_candidate_id": candidate,
            "decision": "cannot_determine",
        }
    elif detectable:
        answer = {
            "anomaly": "no_anomaly", "peak_position": None, "mechanism": None,
            "source_position": None, "source_candidate_id": None,
            "decision": "cannot_determine",
        }
    elif detected:
        wrong = graph.positions[int(assets["wrong_l4"])]
        answer = {
            "anomaly": "anomaly",
            "peak_position": {"x": wrong[0], "y": wrong[1], "z": wrong[2]},
            "mechanism": _pick(mechanisms, *key, "false_positive_mechanism"),
            "source_position": {"x": wrong[0], "y": wrong[1], "z": wrong[2]},
            "source_candidate_id": None, "decision": "cannot_determine",
        }
    else:
        answer = {
            "anomaly": "no_anomaly", "peak_position": None, "mechanism": None,
            "source_position": None, "source_candidate_id": None,
            "decision": "cannot_determine",
        }

    tolerance = store.protocol["tolerances"][truth["waterbody"]]
    provisional = score_submission(answer, truth, tolerance)
    expected_decision = "declare" if detectable and provisional.get("L4_pass") is True else "cannot_determine"
    decision_correct = _trial(
        .75 + .12 * (capabilities["decision_calibration"] - .5)
        + variation("terminal_decision") + probability_offset("terminal_decision"),
        *key, "terminal_decision",
    )
    answer["decision"] = expected_decision if decision_correct else (
        "cannot_determine" if expected_decision == "declare" else "declare"
    )
    default_submission_probability = (
        .98 + .02 * (capabilities["tool_stability"] - .5)
        + variation("valid_submission") * .15
    )
    submission_probability = max(0.05, min(
        .999,
        float(rehearsal_behavior.get("submission_probability", default_submission_probability)),
    ))
    submitted = _trial(submission_probability, *key, "valid_submission")
    if submitted:
        round_index = len(events)
        events.append({
            "action": "submit", "ok": True, "provider_round_index": round_index,
            "answer": answer, "cancelled_tool_call_count": 0,
            "budget_after": _budget_snapshot(
                task, used_s=used_s, used_m=used_m, turns=round_index + 1,
            ),
        })
        confidence = (
            .20 + .72 * capabilities["decision_calibration"]
            + (_unit(*key, "confidence_noise") - .5) * .14
        )
        decisions.append(_decision_record(
            episode_id=episode_id, index=round_index, action="submit", current=current,
            confidence=confidence,
        ))
        termination_reason = "submitted"
        episode_answer: dict[str, Any] | None = answer
    else:
        termination_reason = str(
            rehearsal_behavior.get("unsubmitted_terminal", "submit_schema_exhausted")
        )
        episode_answer = None
    tokenless_profile_hash = stable_hash({
        "actor": actor, "snapshot": model["snapshot"], "capabilities": capabilities,
    })
    budget = task["budget"]
    endurance_limit = float(budget["endurance_s"])
    range_limit = float(budget.get("range_m", endurance_limit * float(budget["cruise_ms"])))
    task_workload = _unit(scenario.get("random_seed"), instance_id, "operational_workload")
    seed_workload = _unit(scenario.get("random_seed"), seed, "operational_workload")
    actor_jitter = _unit(
        scenario.get("random_seed"), actor, instance_id, seed, "operational_jitter",
    ) - .5
    input_tokens = int(1350 + 13 * len(events) + 720 * task_workload + 280 * seed_workload + 45 * actor_jitter)
    output_tokens = int(210 + 5 * len(events) + 230 * task_workload + 95 * seed_workload + 18 * actor_jitter)
    reasoning_tokens = int(320 + 8 * len(events) + 410 * task_workload + 180 * seed_workload + 24 * actor_jitter)
    cached_tokens = int(max(0, input_tokens * (.12 + .18 * seed_workload)))
    latency_s = round(
        4.0 + .42 * len(events) + 13.0 * task_workload
        + 6.0 * seed_workload + 1.2 * actor_jitter,
        6,
    )
    cost_public_price = round(
        (input_tokens - cached_tokens) * .000002
        + cached_tokens * .0000005
        + (output_tokens + reasoning_tokens) * .000008,
        8,
    )
    episode: dict[str, Any] = {
        "schema_version": "synthetic-figure-rehearsal-v1",
        "data_contract_version": "synthetic-figure-rehearsal-v1",
        "dataset_kind": SYNTHETIC_KIND,
        "synthetic": True,
        "protocol_id": PROTOCOL_ID,
        "run_id": f"SYN-RUN::{scenario['scenario_id']}",
        "episode_id": episode_id,
        "actor": actor,
        "model_display_name": model["display_name"],
        "configured_model": model["snapshot"],
        "model_config_hash": tokenless_profile_hash,
        "presentation_mode": "sparse_map",
        "instance_id": instance_id,
        "seed": seed,
        "attempt_index": 0,
        "science_eligible": False,
        "valid_for_scoring": False,
        "events": events,
        "decision_record_calls": decisions,
        "sample_ledger": ledger,
        "answer": episode_answer,
        "termination_reason": termination_reason,
        "submit_only_reason": "engineering_provider_round_cap" if submitted else "synthetic_submit_failure",
        "turns": len(events),
        "endurance_used_s": used_s,
        "endurance_used_fraction": used_s / endurance_limit,
        "range_used_m": used_m,
        "range_used_fraction": used_m / range_limit,
        "visited_nodes": len(visits),
        "sample_count": len(ledger),
        "n_total_nodes": len(graph.positions),
        "multi_call_cancelled_count": sum(
            int(event.get("cancelled_tool_call_count", 0)) for event in events
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "cost_public_price": cost_public_price,
        "latency_s": latency_s,
        "provider_attempts": len(events),
        "identity_failures": int(_trial(
            .015 + .02 * seed_workload, *key, "synthetic_identity_failure_layout",
        )),
        "operational_metric_origin": "synthetic_layout_only",
        "watermark": SYNTHETIC_WATERMARK,
        "provider_calls": None,
        "replay_verified": None,
    }
    score = score_submission(episode_answer or {}, truth, tolerance)
    return episode, score


def generate_rehearsal_bundle(
    *,
    store: DataStore,
    scenario: str | Path | Mapping[str, Any] | None = None,
    instance_ids: Iterable[str] | None = None,
    require_full_matrix: bool = True,
    control_root: str | Path = DEFAULT_CONTROL_ROOT,
) -> PaperDataBundle:
    """Generate the deterministic six-model, 84-task, three-seed rehearsal."""

    if not store.evaluator:
        raise PermissionError("rehearsal generation requires evaluator truth")
    config = load_scenario(scenario)
    registered = [str(row["instance_id"]) for row in store.tasks()]
    selected = sorted(set(instance_ids) if instance_ids is not None else set(registered))
    if not set(selected) <= set(registered):
        raise ValueError("rehearsal requested an unregistered task")
    if require_full_matrix and selected != sorted(registered):
        raise ValueError("production rehearsal requires the complete registered task set")
    mechanisms = sorted({
        str(store.truth(instance_id).get("L3_mechanism"))
        for instance_id in selected
        if store.truth(instance_id).get("L3_mechanism") is not None
    })
    if len(mechanisms) < 2:
        raise ValueError("rehearsal needs at least two registered mechanism labels")

    outcomes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for instance_id in selected:
        task = store.task(instance_id)
        truth = store.truth(instance_id)
        frame = store.frame(instance_id, "F1")
        graph = NodeGraph.from_frame(frame)
        assets = _route_assets(store, frame, graph, task, truth)
        for model in config["models"]:
            for seed in config["seeds"]:
                episode, score = _make_episode(
                    store=store, frame=frame, graph=graph, task=task, truth=truth,
                    assets=assets, model=model, seed=seed, scenario=config,
                    mechanisms=mechanisms,
                )
                outcome = outcome_from_episode(
                    store, episode, score, dataset_kind=SYNTHETIC_KIND,
                    replay_verified=None, scenario_id=config["scenario_id"],
                    model_display_name=model["display_name"],
                    model_snapshot=model["snapshot"],
                    source_episode_sha256="synthetic:" + stable_hash(episode),
                )
                outcomes.append(outcome)
                steps.extend(step_rows_from_episode(episode, dataset_kind=SYNTHETIC_KIND))
                decisions.extend(decision_rows_from_episode(episode, dataset_kind=SYNTHETIC_KIND))
        # Avoid retaining 84 large F1 frames after their graph-derived routes
        # have been materialized.  This changes no data or deterministic order.
        store.frame.cache_clear()

    behavior_tertiles = attach_model_behavior_tertiles(outcomes)
    storyboard = attach_storyboard_selection(
        outcomes,
        steps,
        selection_seed=config.get("random_seed", config["scenario_id"]),
        require_contrast=require_full_matrix,
    )
    outcomes.sort(key=lambda row: (str(row["actor"]), str(row["instance_id"]), int(row["seed"])))
    steps.sort(key=lambda row: (str(row["actor"]), str(row["instance_id"]), int(row["seed"]), int(row["step_index"])))
    decisions.sort(key=lambda row: (str(row["actor"]), str(row["instance_id"]), int(row["seed"]), int(row["decision_record_index"])))
    expected = len(config["models"]) * len(selected) * len(config["seeds"])
    if len(outcomes) != expected:
        raise AssertionError(f"synthetic matrix has {len(outcomes)} rows, expected {expected}")
    if require_full_matrix and expected != 1512:
        raise AssertionError(f"full rehearsal must contain 1512 episodes, observed {expected}")
    controls = load_registered_controls(store, control_root)
    geometry = build_task_geometry(
        store, dataset_kind=SYNTHETIC_KIND, instance_ids=selected,
    )
    representative_task = next(
        (str(row["instance_id"]) for row in geometry if row.get("representative") is True),
        None,
    )
    metadata = {
        "schema_version": PAPER_SCHEMA_VERSION,
        "dataset_kind": SYNTHETIC_KIND,
        "synthetic": True,
        "watermark": SYNTHETIC_WATERMARK,
        "protocol_id": PROTOCOL_ID,
        "scenario_id": config["scenario_id"],
        "scenario_sha256": stable_hash(config),
        "research_status": config.get("research_status"),
        "research_manifest": config.get("research_manifest"),
        "episodes": len(outcomes),
        "models": len(config["models"]),
        "tasks": len(selected),
        "seeds": len(config["seeds"]),
        "control_rows": len(controls),
        "registered_controls": sorted({str(row["control_kind"]) for row in controls}),
        "score_writer": "laketrace.scoring.scorer.score_submission",
        "route_contract": "real registered F1 NodeGraph; every move is one native adjacent edge",
        "provider_provenance_claimed": False,
        "replay_provenance_claimed": False,
        "operational_metrics": (
            "synthetic layout-only placeholders; never model cost, latency, identity, "
            "or provider-performance evidence"
        ),
        "variance_hierarchy_prior": str(config.get(
            "variance_hierarchy",
            "task_effect > seed_effect > model_main_effect",
        )),
        "overall_rank_defined": False,
        "behavior_tertiles": behavior_tertiles,
        "storyboard": storyboard,
        "representative_task": {
            "instance_id": representative_task,
            "selection_rule": (
                "outcome-blind minimum stable hash among registered detectable tasks "
                "with nonempty source and peak geometry"
            ),
        },
    }
    bundle = PaperDataBundle(
        metadata=metadata,
        episode_outcomes=outcomes,
        episode_steps=steps,
        task_geometry=geometry,
        decision_records=decisions,
        control_rows=controls,
    )
    bundle.validate()
    return bundle


def generate_rehearsal_to_directory(
    *,
    data_root: str | Path,
    output_dir: str | Path,
    scenario: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    store = DataStore(data_root, evaluator=True)
    bundle = generate_rehearsal_bundle(store=store, scenario=scenario)
    return write_bundle(bundle, output_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/laketrace_v2")
    parser.add_argument("--scenario")
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = generate_rehearsal_to_directory(
        data_root=args.data, output_dir=args.output, scenario=args.scenario,
    )
    print(json.dumps({
        "status": "SYNTHETIC_LAYOUT_REHEARSAL_COMPLETE",
        "episodes": manifest["episodes"],
        "models": manifest["models"],
        "tasks": manifest["tasks"],
        "seeds": manifest["seeds"],
        "output": str(Path(args.output).resolve()),
        "watermark": manifest["watermark"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAPABILITY_FIELDS", "DEFAULT_SCENARIO", "generate_rehearsal_bundle",
    "generate_rehearsal_to_directory", "load_scenario", "main",
]
