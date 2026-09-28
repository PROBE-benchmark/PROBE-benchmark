from __future__ import annotations

import copy
import csv
import math
import random
from pathlib import Path
from typing import Any

from laketrace.data.store import DataStore, Frame, Node
from laketrace.env.core import Budget, _depth_layer
from laketrace.env.graph import DIRECTIONS, Edge, NodeGraph
from laketrace.util import load_config, stable_hash


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "configs" / "v20" / "protocol.json"
_BASE36_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _public_candidate_catalog(
    store: DataStore,
    *,
    domain_id: str,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the complete, truth-independent candidate catalog for a domain.

    The v2 task rows contain source-conditioned eight-item shortlists.  Their
    membership and first position therefore cannot be exposed to an actor.  V20
    instead takes the union of the public candidate registry across the domain,
    verifies it against the preregistered catalog digest, and applies a frozen
    deterministic permutation whose inputs contain no evaluator truth.
    """

    specification = protocol.get("candidate_presentation") or {}
    if specification.get("catalog") != "public_task_union_by_domain":
        raise ValueError("unexpected v20 candidate catalog rule")
    if specification.get("ordering") != "sha256_protocol_domain_candidate":
        raise ValueError("unexpected v20 candidate ordering rule")

    catalog: dict[str, dict[str, Any]] = {}
    for task in store.tasks():
        if str(task["domain_id"]) != str(domain_id):
            continue
        for raw in task.get("candidates") or ():
            candidate = copy.deepcopy(raw)
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id:
                raise ValueError("candidate catalog contains an empty ID")
            previous = catalog.get(candidate_id)
            if previous is not None and previous != candidate:
                raise ValueError(f"candidate geometry conflict for {candidate_id}")
            catalog[candidate_id] = candidate

    canonical = [catalog[key] for key in sorted(catalog)]
    registered_counts = specification.get("catalog_counts") or {}
    registered_hashes = specification.get("catalog_sha256") or {}
    if domain_id in registered_counts or domain_id in registered_hashes:
        expected_count = int(registered_counts.get(domain_id, -1))
        expected_hash = str(registered_hashes.get(domain_id, ""))
        if len(canonical) != expected_count or stable_hash(canonical) != expected_hash:
            raise ValueError(f"candidate catalog does not match frozen v20 registry for {domain_id}")

    separator = str(specification.get("order_domain_separator") or "")
    protocol_id = str(protocol["protocol_id"])

    def order_key(candidate: dict[str, Any]) -> str:
        return stable_hash({
            "domain_separator": separator,
            "protocol_id": protocol_id,
            "domain_id": str(domain_id),
            "candidate_id": str(candidate["candidate_id"]),
        })

    return sorted(canonical, key=lambda candidate: (order_key(candidate), candidate["candidate_id"]))


def _load_limits(path: Path) -> dict[tuple[str, str, str], tuple[float, float]]:
    output: dict[tuple[str, str, str], tuple[float, float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            output[(row["waterbody"], row["channel"], row["depth_layer"])] = (
                float(row["background_median"]),
                float(row["declared_lod"]),
            )
    return output


def _summary_mapping(
    sums: dict[str, float], squares: dict[str, float], latest: dict[str, float], count: int,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key in sorted(sums):
        mean = sums[key] / count
        variance = max(0.0, squares[key] / count - mean * mean)
        result[key] = {
            "latest": latest[key],
            "mean": mean,
            "std": math.sqrt(variance),
        }
    return result


class EnvironmentV20:
    """Truth-blind, single-edge 3-D environment for the v20 protocol.

    Provider turns never substitute for physical time.  Repetition is logged but
    never terminates an episode.  Once no physical move is affordable, the
    environment enters a budget-free submit-only state rather than discarding the
    agent's accumulated work.
    """

    def __init__(
        self,
        store: DataStore,
        instance_id: str,
        seed: int = 0,
        *,
        protocol_path: str | Path = DEFAULT_PROTOCOL,
    ):
        self.store = store
        self.task = store.task(instance_id)
        self.protocol = load_config(Path(protocol_path))
        if self.protocol.get("protocol_id") != "laketrace-v20-3d-single-edge-v1":
            raise ValueError("unexpected v20 protocol id")
        if int(self.protocol["termination"]["provider_round_guard"]) < 2:
            raise ValueError("provider_round_guard must reserve an active round and final submit")
        if self.task["active_phase"] != "F1":
            raise ValueError("v20 main condition requires static F1 tasks")
        self.public_candidates = _public_candidate_catalog(
            store,
            domain_id=str(self.task["domain_id"]),
            protocol=self.protocol,
        )
        self.frame: Frame = store.frame(instance_id, self.task["active_phase"])
        self.graph = NodeGraph.from_frame(self.frame)
        self._axis_values = tuple(
            tuple(sorted({position[axis] for position in self.graph.positions}))
            for axis in range(3)
        )
        self._axis_indices = tuple(
            {value: index for index, value in enumerate(values)}
            for values in self._axis_values
        )
        self._position_indices = tuple(
            tuple(
                self._axis_indices[axis][coordinate]
                for axis, coordinate in enumerate(position)
            )
            for position in self.graph.positions
        )
        self._column_z_indices: dict[tuple[int, int], tuple[int, ...]] = {}
        pending_columns: dict[tuple[int, int], list[int]] = {}
        for x_index, y_index, z_index in self._position_indices:
            pending_columns.setdefault((x_index, y_index), []).append(z_index)
        self._column_z_indices = {
            column: tuple(sorted(layers))
            for column, layers in pending_columns.items()
        }
        self._column_indices = frozenset(self._column_z_indices)
        self.domain_bounds = {
            axis: {
                "min": min(position[index] for position in self.graph.positions),
                "max": max(position[index] for position in self.graph.positions),
            }
            for index, axis in enumerate("xyz")
        }
        access = store.access_points["access_points"][self.task["domain_id"]]
        self.current = self.graph.index_by_xyz[
            (float(access["x"]), float(access["y"]), float(access["z"]))
        ]
        raw_budget = self.task["budget"]
        self.budget = Budget(
            float(raw_budget["endurance_s"]),
            float(raw_budget.get(
                "range_m",
                float(raw_budget["endurance_s"]) * float(raw_budget["cruise_ms"]),
            )),
            float(raw_budget["cruise_ms"]),
            float(raw_budget["descend_ms"]),
            float(raw_budget["ascend_ms"]),
        )
        self.seed = int(seed)
        self.turn = 0
        self.state_version = 0
        self.sample_sequence = 0
        self.phase = str(self.task["active_phase"])
        self.mode = "active"
        self.submit_only_reason: str | None = None
        self.terminated = False
        self.termination_reason: str | None = None
        self.answer: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.sample_ledger: list[dict[str, Any]] = []
        self.measurements: dict[int, dict[str, Any]] = {}
        self.visit_count: dict[int, int] = {}
        self.last_analyze_sample_sequence = 0
        self._last_result_signature: str | None = None
        self._same_result_no_progress_count = 0
        self.limits = _load_limits(store.root / "config" / "detection_limits.csv")
        self.navigation_topology = self._navigation_topology()
        self._measure(self.current, mode="initial")

    @property
    def mission_time_s(self) -> float:
        return self.budget.used_s

    @property
    def max_turns(self) -> int:
        """Maximum total provider rounds, including the reserved final submit."""

        return int(self.protocol["termination"]["provider_round_guard"])

    @property
    def max_steps(self) -> int:
        return 1

    def _node(self, index: int) -> Node:
        return self.frame.by_xyz[self.graph.positions[index]]

    def _counter_gauss(self, sigma: float, *parts: Any) -> float:
        if sigma <= 0.0:
            return 0.0
        digest = stable_hash({
            "protocol": self.protocol["protocol_id"],
            "instance": self.task["instance_id"],
            "seed": self.seed,
            "parts": parts,
        })
        return random.Random(int(digest[:16], 16)).gauss(0.0, sigma)

    def _measure(self, index: int, mode: str) -> dict[str, Any]:
        node = self._node(index)
        layer = _depth_layer(node.z, node.bed_depth)
        visit_index = self.visit_count.get(index, 0) + 1
        multiplier = 1.0 if mode == "initial" else float(
            self.protocol["sensor_noise"]["transit_multiplier"]
        )
        channels: dict[str, float] = {}
        for channel, raw in node.channels.items():
            _, lod = self.limits[(self.task["waterbody"], channel, layer)]
            channels[channel] = raw + self._counter_gauss(
                lod / 3.0 * multiplier,
                "channel", index, visit_index, channel, mode,
            )
        flow_sigma = float(self.protocol["sensor_noise"]["flow_sigma_m_s"])
        flow = {
            key: raw + self._counter_gauss(
                flow_sigma, "flow", index, visit_index, key, mode,
            )
            for key, raw in node.flow.items()
        }
        self.visit_count[index] = visit_index
        self.sample_sequence += 1
        sample = {
            "sample_sequence": self.sample_sequence,
            "provider_round": self.turn,
            "mission_time_s": self.mission_time_s,
            "node_index": index,
            "x": node.x,
            "y": node.y,
            "z": node.z,
            "bed_depth_m": node.bed_depth,
            "phase": self.phase,
            "mode": mode,
            "visit_index": visit_index,
            "channels": channels,
            "flow": flow,
        }
        self.sample_ledger.append(sample)
        previous = self.measurements.get(index)
        if previous is None:
            summary = {
                "node_index": index,
                "x": node.x,
                "y": node.y,
                "z": node.z,
                "bed_depth_m": node.bed_depth,
                "phase": self.phase,
                "turn": self.turn,
                "visit_count": 1,
                "sample_count": 1,
                "first_sample_time_s": self.mission_time_s,
                "last_sample_time_s": self.mission_time_s,
                "first_sample_sequence": self.sample_sequence,
                "last_sample_sequence": self.sample_sequence,
                "channels": dict(channels),
                "flow": dict(flow),
                "latest_channels": dict(channels),
                "latest_flow": dict(flow),
                "channel_sums": dict(channels),
                "flow_sums": dict(flow),
                "channel_sum_squares": {key: value * value for key, value in channels.items()},
                "flow_sum_squares": {key: value * value for key, value in flow.items()},
            }
        else:
            summary = dict(previous)
            count = int(previous["sample_count"]) + 1
            summary.update({
                "turn": self.turn,
                "visit_count": count,
                "sample_count": count,
                "last_sample_time_s": self.mission_time_s,
                "last_sample_sequence": self.sample_sequence,
                "channels": dict(channels),
                "flow": dict(flow),
                "latest_channels": dict(channels),
                "latest_flow": dict(flow),
                "channel_sums": {
                    key: previous["channel_sums"][key] + value
                    for key, value in channels.items()
                },
                "flow_sums": {
                    key: previous["flow_sums"][key] + value
                    for key, value in flow.items()
                },
                "channel_sum_squares": {
                    key: previous["channel_sum_squares"][key] + value * value
                    for key, value in channels.items()
                },
                "flow_sum_squares": {
                    key: previous["flow_sum_squares"][key] + value * value
                    for key, value in flow.items()
                },
            })
        self.measurements[index] = summary
        self.state_version += 1
        return sample

    def _public_summary(self, value: dict[str, Any]) -> dict[str, Any]:
        count = int(value["sample_count"])
        channel_stats = _summary_mapping(
            value["channel_sums"], value["channel_sum_squares"],
            value["latest_channels"], count,
        )
        flow_stats = _summary_mapping(
            value["flow_sums"], value["flow_sum_squares"],
            value["latest_flow"], count,
        )
        return {
            "node_index": int(value["node_index"]),
            "x": round(float(value["x"]), 3),
            "y": round(float(value["y"]), 3),
            "z": round(float(value["z"]), 3),
            "bed_depth_m": round(float(value["bed_depth_m"]), 3),
            "phase": value["phase"],
            "visit_count": count,
            "sample_count": count,
            "first_sample_time_s": round(float(value["first_sample_time_s"]), 3),
            "last_sample_time_s": round(float(value["last_sample_time_s"]), 3),
            "measurement_age_s": round(
                self.mission_time_s - float(value["last_sample_time_s"]), 3
            ),
            "first_sample_sequence": int(value["first_sample_sequence"]),
            "last_sample_sequence": int(value["last_sample_sequence"]),
            "channels": {
                key: round(float(stats["mean"]), 4)
                for key, stats in channel_stats.items()
            },
            "flow": {
                key: round(float(stats["mean"]), 5)
                for key, stats in flow_stats.items()
            },
            "channel_stats": {
                key: {name: round(float(number), 4) for name, number in stats.items()}
                for key, stats in channel_stats.items()
            },
            "flow_stats": {
                key: {name: round(float(number), 5) for name, number in stats.items()}
                for key, stats in flow_stats.items()
            },
        }

    def _edge_time(self, edge: Edge) -> float:
        if edge.direction == "down":
            return edge.distance_m / self.budget.descend_m_s
        if edge.direction == "up":
            return edge.distance_m / self.budget.ascend_m_s
        return edge.distance_m / self.budget.cruise_m_s

    def _edge_affordable(self, edge: Edge) -> bool:
        return (
            self._edge_time(edge) <= self.budget.remaining_s + 1e-9
            and edge.distance_m <= self.budget.remaining_m + 1e-9
        )

    def _boundary_reason(self, direction: str) -> str:
        x_index, y_index, z_index = self._position_indices[self.current]
        if direction == "up":
            return "surface_boundary" if z_index == 0 else "vertical_gap_boundary"
        if direction == "down":
            layers = self._column_z_indices[(x_index, y_index)]
            return "bed_boundary" if z_index == layers[-1] else "vertical_gap_boundary"
        axis, sign = {
            "east": (0, 1), "west": (0, -1),
            "north": (1, 1), "south": (1, -1),
        }[direction]
        target = [x_index, y_index]
        target[axis] += sign
        if not 0 <= target[axis] < len(self._axis_values[axis]):
            return "shoreline_boundary"
        return (
            "bathymetry_boundary"
            if tuple(target) in self._column_indices
            else "shoreline_boundary"
        )

    @staticmethod
    def _axis_rule(
        values: tuple[float, ...], nominal_spacing: float | None,
    ) -> tuple[dict[str, Any], bool]:
        if len(values) == 1:
            return ({
                "origin_m": round(values[0], 3),
                "nominal_spacing_m": None,
                "count": 1,
                "index_min": 0,
                "index_max": 0,
                "max_coordinate_rounding_error_m": 0.0,
                "coordinate_rule": "coordinate=origin (single layer)",
            }, True)
        if nominal_spacing is None or nominal_spacing <= 0.0:
            return ({"values_m": [round(value, 3) for value in values]}, False)
        errors = [
            abs(value - (values[0] + index * nominal_spacing))
            for index, value in enumerate(values)
        ]
        maximum_error = max(errors)
        # Source coordinates are stored to millimetres.  A half-centimetre
        # tolerance admits only rounding jitter, not a genuinely irregular axis.
        regular = maximum_error <= max(0.005, abs(nominal_spacing) * 1e-6)
        return ({
            "origin_m": round(values[0], 3),
            "nominal_spacing_m": round(nominal_spacing, 6),
            "count": len(values),
            "index_min": 0,
            "index_max": len(values) - 1,
            "max_coordinate_rounding_error_m": round(maximum_error, 6),
            "coordinate_rule": "coordinate~=origin+index*nominal_spacing",
        }, regular)

    def _exact_navigation_topology(
        self, axis_rules: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        columns = []
        for (x_index, y_index), layers in sorted(
            self._column_z_indices.items(), key=lambda item: (item[0][1], item[0][0])
        ):
            columns.append({
                "x_index": x_index,
                "y_index": y_index,
                "z_layer_indices": list(layers),
            })
        return {
            "encoding": "exact_axis_values_and_column_z_layer_indices",
            "axes": {
                axis: {
                    **axis_rules[axis],
                    "values_m": [round(value, 3) for value in self._axis_values[index]],
                }
                for index, axis in enumerate("xyz")
            },
            "columns": columns,
            "directions_as_index_deltas": {
                "east": [1, 0, 0], "west": [-1, 0, 0],
                "north": [0, 1, 0], "south": [0, -1, 0],
                "down": [0, 0, 1], "up": [0, 0, -1],
            },
            "node_count": len(self.graph.positions),
            "coordinate_sha256": stable_hash(self.graph.positions),
        }

    def _navigation_topology(self) -> dict[str, Any]:
        axis_pairs = [
            self._axis_rule(self._axis_values[index], self.graph.spacing[index])
            for index in range(3)
        ]
        axis_rules = {
            axis: axis_pairs[index][0] for index, axis in enumerate("xyz")
        }
        columns_are_surface_contiguous = all(
            layers == tuple(range(len(layers)))
            for layers in self._column_z_indices.values()
        )
        compact = (
            all(regular for _, regular in axis_pairs)
            and columns_are_surface_contiguous
            and len(self._axis_values[2]) < len(_BASE36_DIGITS)
        )
        if not compact:
            return self._exact_navigation_topology(axis_rules)

        x_count = len(self._axis_values[0])
        y_count = len(self._axis_values[1])
        rows = []
        for y_index in range(y_count):
            rows.append("".join(
                _BASE36_DIGITS[len(self._column_z_indices.get((x_index, y_index), ()))]
                for x_index in range(x_count)
            ))
        return {
            "encoding": "regular_axes_plus_base36_column_depth_counts",
            "axes": axis_rules,
            "column_depth_count_rows_base36": rows,
            "column_depth_count_legend": (
                "rows[y_index][x_index] is one base36 digit (0-9,A-Z); "
                "0 means no water column; N>0 means exactly z_index 0..N-1 exist"
            ),
            "directions_as_index_deltas": {
                "east": [1, 0, 0], "west": [-1, 0, 0],
                "north": [0, 1, 0], "south": [0, -1, 0],
                "down": [0, 0, 1], "up": [0, 0, -1],
            },
            "depth_axis": "positive_down",
            "node_count": len(self.graph.positions),
            "coordinate_sha256": stable_hash(self.graph.positions),
        }

    def _calibration(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for channel in self.task["available_channels"]:
            output[channel] = {}
            for layer in ("surface", "middle", "bottom"):
                median, lod = self.limits[(self.task["waterbody"], channel, layer)]
                output[channel][layer] = {
                    "background_median": median,
                    "declared_lod": lod,
                }
        return output

    def _budget_snapshot(self) -> dict[str, Any]:
        return {
            "endurance": {
                "used": self.budget.used_s,
                "limit": self.budget.endurance_s,
                "remaining": self.budget.remaining_s,
            },
            "range": {
                "used": self.budget.used_m,
                "limit": self.budget.range_m,
                "remaining": self.budget.remaining_m,
            },
            "provider_round_guard": {
                "used": self.turn,
                "limit": self.max_turns,
                "remaining": max(0, self.max_turns - self.turn),
                "scientific_budget": False,
            },
        }

    def _position_snapshot(self) -> dict[str, Any]:
        current = self._node(self.current)
        x_index, y_index, z_index = self._position_indices[self.current]
        return {
            "node_index": self.current,
            "x_index": x_index,
            "y_index": y_index,
            "z_index": z_index,
            "x": current.x,
            "y": current.y,
            "z": current.z,
            "bed_depth_m": current.bed_depth,
        }

    def _neighbors_snapshot(self) -> tuple[dict[str, Edge | None], dict[str, Any]]:
        graph_neighbors = self.graph.neighbors(self.current)
        neighbors: dict[str, Any] = {}
        for direction, edge in graph_neighbors.items():
            if edge is None:
                neighbors[direction] = {
                    "reachable": False,
                    "reason": self._boundary_reason(direction),
                    "target_node_index": None,
                    "target_x_index": None,
                    "target_y_index": None,
                    "target_z_index": None,
                    "target_xyz": None,
                    "one_step_cost_s": None,
                    "one_step_distance_m": None,
                    "visit_count": None,
                    "last_sample_time_s": None,
                }
                continue
            previous = self.measurements.get(edge.target)
            target_x, target_y, target_z = self._position_indices[edge.target]
            neighbors[direction] = {
                "reachable": True,
                "reason": None,
                "target_node_index": edge.target,
                "target_x_index": target_x,
                "target_y_index": target_y,
                "target_z_index": target_z,
                "target_xyz": dict(zip("xyz", self.graph.positions[edge.target])),
                "one_step_cost_s": self._edge_time(edge),
                "one_step_distance_m": edge.distance_m,
                "affordable": self._edge_affordable(edge),
                "visit_count": int(previous["sample_count"]) if previous else 0,
                "last_sample_time_s": previous["last_sample_time_s"] if previous else None,
            }
        return graph_neighbors, neighbors

    def _static_observation_fields(self) -> dict[str, Any]:
        return {
            "domain_bounds": self.domain_bounds,
            "navigation_topology": self.navigation_topology,
            "calibration": self._calibration(),
            "candidate_catalog_semantics": {
                "scope": "complete_domain_catalog",
                "ordering": "source_independent_unranked_permutation",
                "rank_has_priority": False,
                "candidate_count": len(self.public_candidates),
            },
            "candidates": copy.deepcopy(self.public_candidates),
            "available_channels": self.task["available_channels"],
        }

    def observation(self, first: bool = False) -> dict[str, Any]:
        graph_neighbors, neighbors = self._neighbors_snapshot()
        rows = [self._public_summary(self.measurements[index]) for index in sorted(self.measurements)]
        return {
            "schema_version": "5.0",
            "protocol_id": self.protocol["protocol_id"],
            "instance_id": self.task["instance_id"],
            "episode_mode": self.mode,
            "submit_only_reason": self.submit_only_reason,
            "position": self._position_snapshot(),
            "depth_axis": "positive_down",
            "provider_round": self.turn,
            "mission_time_s": self.mission_time_s,
            "state_version": self.state_version,
            "sample_sequence": self.sample_sequence,
            "phase": self.phase,
            "budget": self._budget_snapshot(),
            "domain_bounds": self.domain_bounds,
            "n_total_nodes": len(self.frame.nodes),
            "neighbors": neighbors,
            "directions_to_unvisited_neighbors": sorted(
                direction for direction, edge in graph_neighbors.items()
                if edge is not None and edge.target not in self.measurements
            ),
            "cumulative_measurements": rows,
            "sample_ledger_count": len(self.sample_ledger),
            "sample_ledger_sha256": stable_hash(self.sample_ledger),
            **self._static_observation_fields(),
        }

    def observation_delta(
        self, since_sample_sequence: int = 0, *, include_static: bool = False,
    ) -> dict[str, Any]:
        """Return an incremental public observation without rebuilding history.

        ``sample_sequence`` is contiguous and one-based, so the append-only
        ledger can be sliced in O(number of new samples).  The payload contains
        measured values only; callers may maintain their own presentation state
        without receiving truth or values for unvisited nodes.
        """

        if (
            isinstance(since_sample_sequence, bool)
            or not isinstance(since_sample_sequence, int)
            or not 0 <= since_sample_sequence <= self.sample_sequence
        ):
            raise ValueError("since_sample_sequence outside recorded range")
        graph_neighbors, neighbors = self._neighbors_snapshot()
        new_samples = [
            dict(sample) for sample in self.sample_ledger[since_sample_sequence:]
        ]
        changed_nodes = sorted({int(sample["node_index"]) for sample in new_samples})
        value = {
            "schema_version": "5.0-delta",
            "protocol_id": self.protocol["protocol_id"],
            "instance_id": self.task["instance_id"],
            "episode_mode": self.mode,
            "submit_only_reason": self.submit_only_reason,
            "position": self._position_snapshot(),
            "depth_axis": "positive_down",
            "provider_round": self.turn,
            "mission_time_s": self.mission_time_s,
            "state_version": self.state_version,
            "sample_sequence": self.sample_sequence,
            "since_sample_sequence": since_sample_sequence,
            "phase": self.phase,
            "budget": self._budget_snapshot(),
            "n_total_nodes": len(self.frame.nodes),
            "neighbors": neighbors,
            "directions_to_unvisited_neighbors": sorted(
                direction for direction, edge in graph_neighbors.items()
                if edge is not None and edge.target not in self.measurements
            ),
            "new_samples": new_samples,
            "changed_node_summaries": [
                self._public_summary(self.measurements[index]) for index in changed_nodes
            ],
        }
        if include_static:
            value.update(self._static_observation_fields())
        return value

    def _record_repetition(self, result: dict[str, Any], request: dict[str, Any]) -> None:
        """Record repeated outcomes, independent of surface parameters.

        R6 defines repetition by an identical evidence-bearing result, not by
        identical arguments.  ``request`` is intentionally retained only in
        the event itself and excluded from the signature, so two different
        boundary directions that leave the actor at the same state are
        recognized as the same no-progress outcome.
        """

        signature = stable_hash({
            "ok": result.get("ok"),
            "action": result.get("action"),
            "code": result.get("code"),
            "position_after": self.current,
            "state_version_after": self.state_version,
            "sample_sequence_after": self.sample_sequence,
            "new_samples_since_last_analyze": result.get(
                "new_samples_since_last_analyze"
            ),
            "no_new_evidence": result.get("no_new_evidence"),
        })
        repeated = signature == self._last_result_signature
        self._same_result_no_progress_count = (
            self._same_result_no_progress_count + 1 if repeated else 1
        )
        self._last_result_signature = signature
        result["same_result_no_progress"] = repeated
        result["same_result_no_progress_count"] = self._same_result_no_progress_count
        # Compatibility aliases retain old engineering readers while the v20
        # protocol and new reports use the result-based names above.
        result["same_action_same_state"] = repeated
        result["same_action_same_state_count"] = self._same_result_no_progress_count

    def _maybe_submit_only(self) -> None:
        if self.terminated or self.mode == "submit_only":
            return
        affordable = any(
            edge is not None and self._edge_affordable(edge)
            for edge in self.graph.neighbors(self.current).values()
        )
        if not affordable:
            self.mode = "submit_only"
            self.submit_only_reason = "physical_budget_exhausted"
        # One final provider round is reserved for the budget-free submission.
        # Without the subtraction, 7,201 active calls could be followed by a
        # 7,202nd submit despite the public guard claiming a 7,201-round limit.
        elif self.turn >= self.max_turns - 1:
            self.mode = "submit_only"
            self.submit_only_reason = "provider_round_guard"

    def enter_submit_only(self, reason: str) -> None:
        """Enter the final answer state without discarding accumulated work."""

        if not self.terminated and self.mode != "submit_only":
            self.mode = "submit_only"
            self.submit_only_reason = str(reason)

    def exhaust_provider_round_guard(self) -> None:
        """Terminate locally once the hard provider-call guard is exhausted.

        This transition emits no execution event: it happens *between* provider
        rounds and exists precisely to prevent another request from being sent.
        The existing submit-only reason is retained so a simultaneous physical
        exhaustion remains distinguishable from the engineering safety guard.
        """

        if self.terminated:
            return
        if self.submit_only_reason is None:
            self.submit_only_reason = "provider_round_guard"
        self.terminated = True
        self.mode = "terminated"
        self.termination_reason = "provider_round_guard_exhausted"

    def exhaust_submit_schema_retries(self) -> None:
        """Terminate after the registered submit-only schema retry cap."""

        if self.terminated:
            return
        self.terminated = True
        self.mode = "terminated"
        self.termination_reason = "submit_schema_exhausted"

    def _provider_round_available(self) -> bool:
        return self.turn < self.max_turns

    def move(self, direction: str, steps: int = 1) -> dict[str, Any]:
        if self.terminated:
            return {"ok": False, "code": "E_TERMINATED"}
        if not self._provider_round_available():
            self.exhaust_provider_round_guard()
            return {"ok": False, "code": "E_PROVIDER_ROUND_GUARD"}
        if self.mode == "submit_only":
            return self._record_infeasible("E_SUBMIT_ONLY", "move", {"direction": direction})
        self.turn += 1
        request = {"direction": direction}
        before = self.state_version
        origin = self.current
        if steps != 1:
            return self._record_infeasible("E_SINGLE_EDGE_ONLY", "move", request, turn_already_used=True)
        if direction not in DIRECTIONS:
            return self._record_infeasible("E_DIRECTION", "move", request, turn_already_used=True)
        edge = self.graph.neighbor_map[self.current].get(direction)
        if edge is None:
            return self._record_infeasible(
                "E_BOUNDARY", "move", request,
                turn_already_used=True, detail=self._boundary_reason(direction),
            )
        if not self._edge_affordable(edge):
            result = self._record_infeasible(
                "E_PHYSICAL_BUDGET", "move", request, turn_already_used=True,
            )
            self._maybe_submit_only()
            return result
        cost_s = self._edge_time(edge)
        self.budget.used_s += cost_s
        self.budget.used_m += edge.distance_m
        self.current = edge.target
        reading = self._measure(self.current, mode="transit")
        result = {
            "ok": True,
            "action": "move",
            "direction": direction,
            "steps": 1,
            "origin_node": origin,
            "arrival_node": self.current,
            "cost_s": cost_s,
            "path_length_m": edge.distance_m,
            "reading": reading,
            "state_version_before": before,
            "state_version_after": self.state_version,
            "budget_after": self._budget_snapshot(),
        }
        self._record_repetition(result, request)
        self.events.append(result)
        self._maybe_submit_only()
        if self.mode == "submit_only":
            result["transition_after_action"] = "submit_only"
            result["submit_only_reason"] = self.submit_only_reason
        return result

    def analyze(self) -> dict[str, Any]:
        if self.terminated:
            return {"ok": False, "code": "E_TERMINATED"}
        if not self._provider_round_available():
            self.exhaust_provider_round_guard()
            return {"ok": False, "code": "E_PROVIDER_ROUND_GUARD"}
        if self.mode == "submit_only":
            return self._record_infeasible("E_SUBMIT_ONLY", "analyze", {})
        self.turn += 1
        cost = float(self.protocol["analysis"]["physical_cost_s"])
        if cost > self.budget.remaining_s + 1e-9:
            result = self._record_infeasible(
                "E_PHYSICAL_BUDGET", "analyze", {}, turn_already_used=True,
            )
            self._maybe_submit_only()
            return result
        self.budget.used_s += cost
        public = [self._public_summary(value) for value in self.measurements.values()]
        maxima = {}
        for channel in self.task["available_channels"]:
            best = max(public, key=lambda row: float(row["channels"][channel]))
            maxima[channel] = {
                "value": best["channels"][channel],
                "node_index": best["node_index"],
                "x": best["x"],
                "y": best["y"],
                "z": best["z"],
            }
        new_samples = self.sample_sequence - self.last_analyze_sample_sequence
        result = {
            "ok": True,
            "action": "analyze",
            "cost_s": cost,
            "physical_cost_s": cost,
            "basis_state_version": self.state_version,
            "basis_sample_sequence": self.sample_sequence,
            "new_samples_since_last_analyze": new_samples,
            "no_new_evidence": new_samples == 0,
            "observed_maxima": maxima,
            "visited_nodes": len(self.measurements),
            "budget_after": self._budget_snapshot(),
        }
        self.last_analyze_sample_sequence = self.sample_sequence
        self._record_repetition(result, {"action": "analyze"})
        self.events.append(result)
        self._maybe_submit_only()
        return result

    def submit(self, answer: dict[str, Any]) -> dict[str, Any]:
        if self.terminated:
            return {"ok": False, "code": "E_TERMINATED"}
        if not self._provider_round_available():
            self.exhaust_provider_round_guard()
            return {"ok": False, "code": "E_PROVIDER_ROUND_GUARD"}
        self.turn += 1
        self.answer = answer
        self.terminated = True
        self.mode = "terminated"
        self.termination_reason = "submitted"
        result = {
            "ok": True,
            "action": "submit",
            "answer": answer,
            "budget_after": self._budget_snapshot(),
        }
        self.events.append(result)
        return result

    def record_invalid(
        self, code: str, detail: str, requested_action: str | None = None,
    ) -> dict[str, Any]:
        if self.terminated:
            return {"ok": False, "code": "E_TERMINATED"}
        if not self._provider_round_available():
            self.exhaust_provider_round_guard()
            return {"ok": False, "code": "E_PROVIDER_ROUND_GUARD"}
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
        self._maybe_submit_only()
        return result

    def _record_infeasible(
        self,
        code: str,
        requested_action: str,
        arguments: dict[str, Any],
        *,
        turn_already_used: bool = False,
        detail: str | None = None,
    ) -> dict[str, Any]:
        if not turn_already_used:
            if not self._provider_round_available():
                self.exhaust_provider_round_guard()
                return {"ok": False, "code": "E_PROVIDER_ROUND_GUARD"}
            self.turn += 1
        result = {
            "ok": False,
            "action": "infeasible",
            "requested_action": requested_action,
            "arguments": arguments,
            "code": code,
            "detail": detail,
            "origin_node": self.current,
            "state_version_before": self.state_version,
            "state_version_after": self.state_version,
            "budget_after": self._budget_snapshot(),
        }
        self._record_repetition(result, {"action": requested_action, **arguments})
        self.events.append(result)
        self._maybe_submit_only()
        return result

    def abort_infrastructure(self, detail: str) -> None:
        self.terminated = True
        self.mode = "terminated"
        self.termination_reason = "infra_failure"
        self.events.append({
            "ok": False,
            "action": "infrastructure",
            "code": "E_INFRASTRUCTURE",
            "detail": detail,
            "budget_after": self._budget_snapshot(),
        })

    def log(self, run_id: str, actor: str, seed: int, science_eligible: bool) -> dict[str, Any]:
        return {
            "schema_version": "5.0",
            "protocol_id": self.protocol["protocol_id"],
            "run_id": run_id,
            "instance_id": self.task["instance_id"],
            "actor": actor,
            "seed": seed,
            "science_eligible": science_eligible,
            "phase": self.phase,
            "events": self.events,
            "sample_ledger": self.sample_ledger,
            "sample_ledger_sha256": stable_hash(self.sample_ledger),
            "answer": self.answer,
            "termination_reason": self.termination_reason,
            "submit_only_reason": self.submit_only_reason,
            "turns": self.turn,
            "state_version": self.state_version,
            "endurance_used_s": self.budget.used_s,
            "endurance_used_fraction": self.budget.used_s / self.budget.endurance_s,
            "range_used_m": self.budget.used_m,
            "range_used_fraction": self.budget.used_m / self.budget.range_m,
            "budget": self._budget_snapshot(),
            "visited_nodes": len(self.measurements),
            "sample_count": len(self.sample_ledger),
            "n_total_nodes": len(self.frame.nodes),
            "path_length_m": sum(
                float(event.get("path_length_m", 0.0)) for event in self.events
            ),
            "event_chain_sha256": stable_hash(self.events),
        }
