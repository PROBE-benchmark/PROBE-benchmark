from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from laketrace.data.store import DataStore, Frame, Node
from laketrace.env.graph import DIRECTIONS, NodeGraph
from laketrace.perception import public_measurement
from laketrace.util import stable_hash


def _depth_layer(z: float, bed: float) -> str:
    ratio = z / bed if bed > 0 else 0.0
    return "surface" if ratio < 1 / 3 else "middle" if ratio < 2 / 3 else "bottom"


def _load_limits(path: Path) -> dict[tuple[str, str, str], tuple[float, float]]:
    output = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            output[(row["waterbody"], row["channel"], row["depth_layer"])] = (
                float(row["background_median"]), float(row["declared_lod"])
            )
    return output


@dataclass
class Budget:
    endurance_s: float
    range_m: float
    cruise_m_s: float
    descend_m_s: float
    ascend_m_s: float
    used_s: float = 0.0
    used_m: float = 0.0

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.endurance_s - self.used_s)

    @property
    def remaining_m(self) -> float:
        return max(0.0, self.range_m - self.used_m)


class Environment:
    """Truth-blind static-F1 environment with cumulative observations."""

    def __init__(self, store: DataStore, instance_id: str, seed: int = 0):
        self.store = store
        self.task = store.task(instance_id)
        self.frame: Frame = store.frame(instance_id, self.task["active_phase"])
        self.graph = NodeGraph.from_frame(self.frame)
        self.domain_bounds = {
            axis: {"min": min(position[index] for position in self.graph.positions),
                   "max": max(position[index] for position in self.graph.positions)}
            for index, axis in enumerate("xyz")
        }
        access = store.access_points["access_points"][self.task["domain_id"]]
        self.current = self.graph.index_by_xyz[(float(access["x"]), float(access["y"]), float(access["z"]))]
        raw_budget = self.task["budget"]
        self.budget = Budget(
            float(raw_budget["endurance_s"]), float(raw_budget.get("range_m", float(raw_budget["endurance_s"]) * float(raw_budget["cruise_ms"]))), float(raw_budget["cruise_ms"]),
            float(raw_budget["descend_ms"]), float(raw_budget["ascend_ms"]),
        )
        self.max_turns = int(store.protocol["max_tool_turns"])
        self.max_steps = int(store.protocol["max_move_steps"])
        self.turn = 0
        self.terminated = False
        self.termination_reason: str | None = None
        self.answer: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.repetition_limit = int(store.protocol.get("repetition_identical_result_limit", 3))
        self._last_result_signature: str | None = None
        self._identical_result_count = 0
        self.measurements: dict[int, dict[str, Any]] = {}
        self.visit_count: dict[int, int] = {}
        self.limits = _load_limits(store.root / "config" / "detection_limits.csv")
        self.rng = random.Random(int(stable_hash({"instance": instance_id, "seed": seed})[:16], 16))
        self._measure(self.current, mode="station")

    def _node(self, index: int) -> Node:
        return self.frame.by_xyz[self.graph.positions[index]]

    def _measure(self, index: int, mode: str) -> dict[str, Any]:
        node = self._node(index)
        layer = _depth_layer(node.z, node.bed_depth)
        multiplier = 1.0 if mode == "station" else float(self.store.protocol["sensor_noise"]["transit_multiplier"])
        channels = {}
        for channel, raw in node.channels.items():
            _, lod = self.limits[(self.task["waterbody"], channel, layer)]
            channels[channel] = raw + self.rng.gauss(0.0, lod / 3.0 * multiplier)
        flow_sigma = float(self.store.protocol["sensor_noise"]["flow_sigma_m_s"])
        flow = {key: raw + self.rng.gauss(0.0, flow_sigma) for key, raw in node.flow.items()}
        self.visit_count[index] = self.visit_count.get(index, 0) + 1
        value = {
            "node_index": index, "x": node.x, "y": node.y, "z": node.z,
            "phase": self.task["active_phase"], "turn": self.turn,
            "visit_count": self.visit_count[index], "channels": channels, "flow": flow,
        }
        previous = self.measurements.get(index)
        if previous is None:
            aggregate = value | {"channel_sums": dict(channels), "flow_sums": dict(flow)}
        else:
            aggregate = dict(value)
            aggregate["channel_sums"] = {key: previous["channel_sums"][key] + channels[key] for key in channels}
            aggregate["flow_sums"] = {key: previous["flow_sums"][key] + flow[key] for key in flow}
        self.measurements[index] = aggregate
        return value

    def observation(self, first: bool = False) -> dict[str, Any]:
        neighbors = {}
        graph_neighbors = self.graph.neighbors(self.current)
        for direction, edge in graph_neighbors.items():
            neighbors[direction] = {
                "reachable": edge is not None,
                "one_step_cost_s": self._edge_time(edge) if edge else None,
            }
        current_measurement = self.measurements[self.current]
        public_current = public_measurement(current_measurement, self.turn)
        observed_gradients = {}
        for channel in self.task["available_channels"]:
            current_value = float(public_current["channels"][channel])
            comparisons = []
            for direction, edge in graph_neighbors.items():
                if edge is None or edge.target not in self.measurements:
                    continue
                neighbor = public_measurement(self.measurements[edge.target], self.turn)
                neighbor_value = float(neighbor["channels"][channel])
                comparisons.append((neighbor_value - current_value, direction))
            best = max(comparisons, default=None)
            observed_gradients[channel] = {
                "direction_of_largest_measured_increase": (
                    best[1] if best is not None and best[0] > 0 else None
                ),
                "largest_measured_delta": round(best[0], 4) if best is not None else None,
                "compared_measured_directions": sorted(direction for _, direction in comparisons),
            }
        position = self.graph.positions[self.current]
        return {
            "instance_id": self.task["instance_id"], "position": {"node_index": self.current, "x": position[0], "y": position[1], "z": position[2]},
            "turn": self.turn, "turns_remaining": max(0, self.max_turns - self.turn),
            "endurance_remaining_s": self.budget.remaining_s,
            "endurance_remaining_fraction": self.budget.remaining_s / self.budget.endurance_s,
            "budget": self._budget_snapshot(),
            "domain_bounds": self.domain_bounds,
            "n_total_nodes": len(self.frame.nodes),
            "neighbors": neighbors,
            "directions_to_unvisited_neighbors": sorted(
                direction for direction, edge in graph_neighbors.items()
                if edge is not None and edge.target not in self.measurements
            ),
            "observed_local_gradients": observed_gradients,
            "cumulative_measurements": [public_measurement(self.measurements[index], self.turn) for index in sorted(self.measurements)],
            # The model harness deliberately replaces full dialogue history with
            # the cumulative measured state.  The compact candidate registry is
            # immutable task input, so it must remain visible after round one.
            "candidates": self.task["candidates"],
            "available_channels": self.task["available_channels"],
        }

    def _edge_time(self, edge: Any) -> float:
        if edge.direction == "down":
            return edge.distance_m / self.budget.descend_m_s
        if edge.direction == "up":
            return edge.distance_m / self.budget.ascend_m_s
        return edge.distance_m / self.budget.cruise_m_s

    def _budget_snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            "endurance": {"used": self.budget.used_s, "limit": self.budget.endurance_s, "remaining": self.budget.remaining_s},
            "range": {"used": self.budget.used_m, "limit": self.budget.range_m, "remaining": self.budget.remaining_m},
            "tool_turns": {"used": self.turn, "limit": self.max_turns, "remaining": max(0, self.max_turns - self.turn)},
        }

    def _result_signature(self, result: dict[str, Any]) -> str:
        """Canonical public outcome, excluding parameters and turn count."""

        current = self.measurements[self.current]
        public = public_measurement(current, self.turn)
        public.pop("measurement_age_turns", None)
        public.pop("turn", None)
        return stable_hash({
            "action": result.get("action"),
            "requested_action": result.get("requested_action"),
            "ok": result.get("ok"),
            "code": result.get("code"),
            "observed_maxima": result.get("observed_maxima"),
            "node_index": self.current,
            "visited_nodes": len(self.measurements),
            "visit_count": current["visit_count"],
            "measurement": public,
        })

    def _detect_repetition(self, result: dict[str, Any]) -> None:
        signature = self._result_signature(result)
        if signature == self._last_result_signature:
            self._identical_result_count += 1
        else:
            self._last_result_signature = signature
            self._identical_result_count = 1
        if self._identical_result_count >= self.repetition_limit and not self.terminated:
            self.terminated, self.termination_reason = True, "repetition_loop"

    def _caps(self) -> None:
        if self.terminated:
            return
        if self.budget.remaining_s <= 1e-9:
            self.terminated, self.termination_reason = True, "endurance_exhausted"
        elif self.budget.remaining_m <= 1e-9:
            self.terminated, self.termination_reason = True, "range_exhausted"
        elif self.turn >= self.max_turns:
            self.terminated, self.termination_reason = True, "turn_limit"

    def ray_length(self, direction: str, cap: int | None = None) -> int:
        maximum = min(self.max_steps, cap or self.max_steps)
        current, count = self.current, 0
        while count < maximum:
            edge = self.graph.neighbor_map[current].get(direction)
            if edge is None:
                break
            current, count = edge.target, count + 1
        return count

    def move(self, direction: str, steps: int) -> dict[str, Any]:
        if self.terminated:
            return {"ok": False, "code": "E_TERMINATED"}
        self.turn += 1
        request = {"direction": direction, "steps": steps}
        if steps < 1 or steps > self.max_steps:
            return self._infeasible("E_STEPS_RANGE", "move", request)
        try:
            path = self.graph.trace(self.current, direction, steps)
        except ValueError as exc:
            return self._infeasible(str(exc), "move", request)
        total = sum(self._edge_time(edge) for edge in path)
        distance = sum(edge.distance_m for edge in path)
        if total > self.budget.remaining_s + 1e-9:
            return self._infeasible("E_ENDURANCE", "move", request)
        if distance > self.budget.remaining_m + 1e-9:
            return self._infeasible("E_RANGE", "move", request)
        readings = []
        for edge in path:
            self.budget.used_s += self._edge_time(edge)
            self.budget.used_m += edge.distance_m
            self.current = edge.target
            readings.append(self._measure(self.current, mode="transit"))
        result = {"ok": True, "action": "move", "direction": direction, "steps": steps, "cost_s": total, "path_length_m": distance, "arrival_node": self.current, "readings": readings, "budget_after": self._budget_snapshot()}
        self.events.append(result)
        self._detect_repetition(result)
        self._caps()
        return result

    def analyze(self) -> dict[str, Any]:
        if self.terminated:
            return {"ok": False, "code": "E_TERMINATED"}
        self.turn += 1
        cost = 30.0
        if cost > self.budget.remaining_s:
            return self._infeasible("E_ENDURANCE", "analyze", {})
        self.budget.used_s += cost
        maxima = {}
        public_rows = [
            public_measurement(row, self.turn) for row in self.measurements.values()
        ]
        for channel in self.task["available_channels"]:
            best = max(public_rows, key=lambda row: float(row["channels"][channel]))
            maxima[channel] = {
                "value": best["channels"][channel],
                "x": best["x"], "y": best["y"], "z": best["z"],
            }
        result = {"ok": True, "action": "analyze", "cost_s": cost, "observed_maxima": maxima, "visited_nodes": len(self.measurements), "budget_after": self._budget_snapshot()}
        self.events.append(result)
        self._detect_repetition(result)
        self._caps()
        return result

    def submit(self, answer: dict[str, Any]) -> dict[str, Any]:
        if self.terminated:
            return {"ok": False, "code": "E_TERMINATED"}
        self.turn += 1
        self.answer = answer
        self.terminated, self.termination_reason = True, "submitted"
        event = {"ok": True, "action": "submit", "answer": answer, "budget_after": self._budget_snapshot()}
        self.events.append(event)
        return event

    def record_invalid(self, code: str, detail: str, requested_action: str | None = None) -> dict[str, Any]:
        """Record a provider action that could not be parsed; it still uses one turn."""

        if self.terminated:
            return {"ok": False, "code": "E_TERMINATED"}
        self.turn += 1
        result = {
            "ok": False,
            "action": "invalid",
            "requested_action": requested_action,
            "code": code,
            "detail": detail,
            "budget_after": self._budget_snapshot(),
        }
        self.events.append(result)
        self._caps()
        return result

    def _infeasible(self, code: str, requested_action: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = {
            "ok": False,
            "action": "infeasible",
            "requested_action": requested_action,
            "arguments": arguments,
            "code": code,
            "budget_after": self._budget_snapshot(),
        }
        self.events.append(result)
        self._detect_repetition(result)
        self._caps()
        return result

    def log(self, run_id: str, actor: str, seed: int, science_eligible: bool) -> dict[str, Any]:
        return {
            "schema_version": "3.0", "run_id": run_id, "instance_id": self.task["instance_id"],
            "actor": actor, "seed": seed, "science_eligible": science_eligible,
            "phase": self.task["active_phase"], "events": self.events, "answer": self.answer,
            "termination_reason": self.termination_reason,
            "turns": self.turn, "endurance_used_s": self.budget.used_s,
            "endurance_used_fraction": self.budget.used_s / self.budget.endurance_s,
            "range_used_m": self.budget.used_m,
            "range_used_fraction": self.budget.used_m / self.budget.range_m,
            "budget": self._budget_snapshot(),
            "visited_nodes": len(self.measurements), "n_total_nodes": len(self.frame.nodes),
            "path_length_m": sum(float(event.get("path_length_m", 0.0)) for event in self.events),
            "event_chain_sha256": stable_hash(self.events),
        }
