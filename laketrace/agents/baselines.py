from __future__ import annotations

import math
import random
from typing import Any

from laketrace.env.core import Environment, _depth_layer
from laketrace.perception import public_measurement


BASELINES = (
    "random", "lawnmower", "budgeted_serpentine", "greedy_gradient",
    "upstream_flow_heuristic", "adjoint_proxy", "detect_then_climb",
)
DETECTION_SCORE_THRESHOLD = 3.0
BUDGETED_SERPENTINE_STOP_FRACTION = 0.55


def _scores(env: Environment, row: dict[str, Any]) -> dict[str, float]:
    node = env._node(row["node_index"])
    layer = _depth_layer(node.z, node.bed_depth)
    output = {}
    for channel, value in row["channels"].items():
        median, lod = env.limits[(env.task["waterbody"], channel, layer)]
        output[channel] = abs(value - median) / max(lod, 1e-12)
    return output


def infer_submission(env: Environment) -> dict[str, Any]:
    public = [public_measurement(row, env.turn) for row in env.measurements.values()]
    scored = [(max(_scores(env, row).values()), row, _scores(env, row)) for row in public]
    strongest, strongest_row, channel_scores = max(scored, key=lambda item: item[0])
    primary = max(channel_scores, key=channel_scores.get)
    peak_row = max(public, key=lambda row: row["channels"].get(primary, -float("inf")))
    anomaly = strongest >= 3.0
    if primary == "turb":
        mechanism = "physical_resuspension" if strongest_row["z"] > 0.6 * env._node(strongest_row["node_index"]).bed_depth else "external_diffuse"
    elif primary == "fdom":
        mechanism = "external_point"
    elif primary == "cond":
        mechanism = "internal_release"
    else:
        mechanism = "external_diffuse"
    flow = peak_row["flow"]
    magnitude = math.hypot(flow["u_m_s"], flow["v_m_s"])
    offset = min(30.0, magnitude * 300.0)
    if magnitude > 1e-9:
        sx = peak_row["x"] - offset * flow["u_m_s"] / magnitude
        sy = peak_row["y"] - offset * flow["v_m_s"] / magnitude
    else:
        sx, sy = peak_row["x"], peak_row["y"]
    source_row = min(public, key=lambda row: math.hypot(row["x"] - sx, row["y"] - sy) + abs(row["z"] - peak_row["z"]))
    return {
        "anomaly": "anomaly" if anomaly else "no_anomaly",
        "peak_position": {key: peak_row[key] for key in ("x", "y", "z")} if anomaly else None,
        "mechanism": mechanism if anomaly else None,
        "source_position": {key: source_row[key] for key in ("x", "y", "z")} if anomaly else None,
        "decision": "declare" if anomaly else "cannot_determine",
    }


def _available(env: Environment) -> list[str]:
    return [name for name, edge in env.graph.neighbors(env.current).items() if edge is not None]


def _move(env: Environment, direction: str, preferred: int = 32) -> bool:
    steps = env.ray_length(direction, preferred)
    if steps < 1:
        return False
    return bool(env.move(direction, steps)["ok"])


def _node_score(env: Environment, index: int) -> float:
    return max(_scores(env, public_measurement(env.measurements[index], env.turn)).values())


def _greedy_climb(env: Environment) -> bool:
    """Probe unmeasured adjacent nodes, then take one strict uphill step.

    Neighbor values are never exposed for free: every missing comparison is
    obtained by a paid one-edge out-and-back movement.  The final move is to
    the highest measured neighbor only when it strictly improves on the
    current node.
    """

    origin = env.current
    neighbors = [(name, edge) for name, edge in env.graph.neighbors(origin).items() if edge is not None]
    for direction, edge in neighbors:
        if edge.target in env.measurements:
            continue
        if not _move(env, direction, 1) or env.terminated:
            return False
        return_direction = next(
            name for name, candidate in env.graph.neighbors(env.current).items()
            if candidate is not None and candidate.target == origin
        )
        if not _move(env, return_direction, 1) or env.terminated:
            return False
    if env.current != origin or not neighbors:
        return False
    best_direction, best_edge = max(
        neighbors,
        key=lambda item: (_node_score(env, item[1].target), -item[1].target),
    )
    if _node_score(env, best_edge.target) <= _node_score(env, origin) + 1e-9:
        return False
    return _move(env, best_direction, 1)


def _move_to_node(env: Environment, target: int) -> bool:
    _, path = env.graph.shortest_path(env.current, {target})
    cursor = 0
    while cursor + 1 < len(path):
        direction = next(
            name for name, edge in env.graph.neighbor_map[path[cursor]].items()
            if edge.target == path[cursor + 1]
        )
        steps = 1
        while (
            cursor + steps + 1 < len(path)
            and steps < env.max_steps
            and env.graph.neighbor_map[path[cursor + steps]].get(direction) is not None
            and env.graph.neighbor_map[path[cursor + steps]][direction].target == path[cursor + steps + 1]
        ):
            steps += 1
        if not _move(env, direction, steps) or env.terminated:
            return False
        cursor += steps
    return env.current == target


def _submit_peak_as_source(env: Environment) -> None:
    answer = infer_submission(env)
    if answer["anomaly"] == "anomaly":
        current = public_measurement(env.measurements[env.current], env.turn)
        position = {key: current[key] for key in ("x", "y", "z")}
        answer["peak_position"] = position
        answer["source_position"] = dict(position)
    env.submit(answer)


def run_baseline(env: Environment, baseline_id: str, seed: int) -> None:
    if baseline_id not in BASELINES:
        raise ValueError(f"unknown baseline: {baseline_id}")
    rng = random.Random(seed)
    horizontal = ["east", "west", "north", "south"]
    heading = "east"
    previous_best = -float("inf")
    stagnation = 0
    while not env.terminated and env.turn < env.max_turns - 1:
        used = env.budget.used_s / env.budget.endurance_s
        if baseline_id in {"upstream_flow_heuristic", "adjoint_proxy"} and (stagnation >= 8 or used >= 0.20):
            break
        if baseline_id == "budgeted_serpentine" and used >= BUDGETED_SERPENTINE_STOP_FRACTION:
            break
        if baseline_id == "random":
            choices = _available(env)
            direction = rng.choice(choices)
            _move(env, direction, rng.randint(1, max(1, min(8, env.ray_length(direction)))))
        elif baseline_id in {"lawnmower", "budgeted_serpentine"}:
            if not _move(env, heading, 32):
                if _move(env, "north", 1):
                    heading = "west" if heading == "east" else "east"
                elif _move(env, "down", 1):
                    heading = "east"
                else:
                    break
        elif baseline_id == "greedy_gradient":
            if not _greedy_climb(env):
                break
        elif baseline_id == "detect_then_climb":
            detected = [
                index for index in env.measurements
                if _node_score(env, index) >= DETECTION_SCORE_THRESHOLD
            ]
            if detected:
                strongest = max(detected, key=lambda index: (_node_score(env, index), -index))
                if strongest != env.current and not _move_to_node(env, strongest):
                    break
                if not _greedy_climb(env):
                    break
            elif not _move(env, heading, 32):
                if _move(env, "north", 1):
                    heading = "west" if heading == "east" else "east"
                elif _move(env, "down", 1):
                    heading = "east"
                else:
                    break
        else:
            current = public_measurement(env.measurements[env.current], env.turn)
            u, v = current["flow"]["u_m_s"], current["flow"]["v_m_s"]
            if abs(u) >= abs(v):
                preferred = "west" if u > 0 else "east"
            else:
                preferred = "south" if v > 0 else "north"
            if baseline_id == "adjoint_proxy" and rng.random() < 0.25:
                preferred = rng.choice(horizontal)
            if not _move(env, preferred, 1):
                alternatives = _available(env)
                if not alternatives:
                    break
                _move(env, rng.choice(alternatives), 1)
            score = max(_scores(env, public_measurement(env.measurements[env.current], env.turn)).values())
            if score > previous_best + 1e-9:
                previous_best, stagnation = score, 0
            else:
                stagnation += 1
        if env.budget.remaining_s < 100.0:
            break
    if not env.terminated:
        if baseline_id == "detect_then_climb":
            _submit_peak_as_source(env)
        else:
            env.submit(infer_submission(env))
