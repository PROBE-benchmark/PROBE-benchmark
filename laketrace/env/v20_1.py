"""V20.1 bounded-route control over the frozen V20 physical environment.

This successor module intentionally leaves V20 untouched. A provider action may
contain a short route, but the route is still executed as an ordered sequence of
native six-neighbour 3-D edges. Every successful edge is budgeted and sampled
before the next edge is considered. Batching therefore removes unnecessary
provider round-trips without adding path planning or unseen sensor values.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

from laketrace.env.graph import DIRECTIONS
from laketrace.env.v20 import DEFAULT_PROTOCOL as V20_PROTOCOL
from laketrace.env.v20 import EnvironmentV20
from laketrace.util import load_config, stable_hash


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "configs" / "v20_1" / "protocol.json"


class EnvironmentV20_1(EnvironmentV20):
    """Truth-blind V20 physics with one-to-32-edge provider actions.

    ``turn`` remains the provider-round counter. ``edge_sequence`` is the
    independent physical movement counter. Multiple arrival samples may share
    one provider round, while each keeps a distinct physical mission timestamp
    and sample sequence.
    """

    def __init__(
        self,
        store: Any,
        instance_id: str,
        seed: int = 0,
        *,
        protocol_path: str | Path = DEFAULT_PROTOCOL,
        physics_protocol_path: str | Path = V20_PROTOCOL,
    ):
        self.control_protocol_path = Path(protocol_path)
        self.control_protocol = load_config(self.control_protocol_path)
        if self.control_protocol.get("protocol_id") != "laketrace-v20.1-3d-axis-run-v1":
            raise ValueError("unexpected v20.1 protocol id")
        if self.control_protocol.get("parent_protocol_id") != "laketrace-v20-3d-single-edge-v1":
            raise ValueError("v20.1 requires the frozen v20 physical protocol")
        movement = self.control_protocol.get("movement") or {}
        if movement.get("adjacency") != "six_axis_aligned_native_grid":
            raise ValueError("v20.1 requires native six-neighbour movement")
        if movement.get("command") != "flat_direction_and_steps":
            raise ValueError("unexpected v20.1 route command")
        self.edge_sequence = 0
        super().__init__(
            store,
            instance_id,
            seed,
            protocol_path=physics_protocol_path,
        )

    @property
    def protocol_id(self) -> str:
        return str(self.control_protocol["protocol_id"])

    @property
    def max_turns(self) -> int:
        return int(self.control_protocol["termination"]["provider_round_guard"])

    @property
    def max_steps(self) -> int:
        return int(self.control_protocol["movement"]["max_edges_per_action"])

    def _measure(self, index: int, mode: str) -> dict[str, Any]:
        sample = super()._measure(index, mode)
        # ``sample`` is the same object retained by the append-only ledger.
        sample["edge_sequence"] = self.edge_sequence
        return sample

    def _budget_snapshot(self) -> dict[str, Any]:
        value = super()._budget_snapshot()
        value["physical_edges"] = {
            "used": self.edge_sequence,
            "scientific_budget": False,
        }
        return value

    @staticmethod
    def _public_route_sample(sample: dict[str, Any]) -> dict[str, Any]:
        """Return every route-arrival sample at registered public precision."""

        return {
            "sample_sequence": int(sample["sample_sequence"]),
            "edge_sequence": int(sample["edge_sequence"]),
            "provider_round": int(sample["provider_round"]),
            "mission_time_s": round(float(sample["mission_time_s"]), 3),
            "node_index": int(sample["node_index"]),
            "x": round(float(sample["x"]), 3),
            "y": round(float(sample["y"]), 3),
            "z": round(float(sample["z"]), 3),
            "bed_depth_m": round(float(sample["bed_depth_m"]), 3),
            "phase": str(sample["phase"]),
            "visit_index": int(sample["visit_index"]),
            "channels": {
                key: round(float(value), 4)
                for key, value in sample["channels"].items()
            },
            "flow": {
                key: round(float(value), 5)
                for key, value in sample["flow"].items()
            },
        }

    def _last_route_measurements(self) -> list[dict[str, Any]]:
        return [
            self._public_route_sample(sample)
            for sample in self.sample_ledger
            if int(sample["provider_round"]) == self.turn
            and int(sample["edge_sequence"]) > 0
        ]

    def observation(self, first: bool = False) -> dict[str, Any]:
        value = super().observation(first=first)
        value.update({
            "schema_version": "5.1",
            "protocol_id": self.protocol_id,
            "control_granularity": {
                "provider_action": "flat_direction_and_steps",
                "native_edges_per_action_max": self.max_steps,
                "each_native_edge_measured": True,
                "automatic_path_planning": False,
            },
            "physical_edge_sequence": self.edge_sequence,
            "last_route_measurements": self._last_route_measurements(),
        })
        return value

    def observation_delta(
        self, since_sample_sequence: int = 0, *, include_static: bool = False,
    ) -> dict[str, Any]:
        value = super().observation_delta(
            since_sample_sequence,
            include_static=include_static,
        )
        value.update({
            "schema_version": "5.1-delta",
            "protocol_id": self.protocol_id,
            "physical_edge_sequence": self.edge_sequence,
            "last_route_measurements": self._last_route_measurements(),
        })
        return value

    def move(
        self,
        directions: list[str],
        *,
        edge_checkpoint: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Execute the reachable and affordable prefix of an explicit route.

        Boundary and budget checks occur before each edge. A failed later edge
        never rolls back earlier physical movement; the remaining suffix is
        cancelled. Sensor values never cause an early stop, so the executor
        supplies no hidden task policy.  ``edge_checkpoint`` is an audit-only
        hook called immediately after each physical edge and arrival sample;
        it cannot choose, skip, or alter an edge.  A durable route journal uses
        the hook to make crash recovery exact at native-edge granularity.
        """

        if self.terminated:
            return {"ok": False, "code": "E_TERMINATED"}
        if not self._provider_round_available():
            self.exhaust_provider_round_guard()
            return {"ok": False, "code": "E_PROVIDER_ROUND_GUARD"}
        if self.mode == "submit_only":
            return self._record_infeasible(
                "E_SUBMIT_ONLY", "move", {"directions": directions},
            )

        self.turn += 1
        request = {"directions": directions}
        before = self.state_version
        origin = self.current
        if (
            not isinstance(directions, list)
            or not 1 <= len(directions) <= self.max_steps
        ):
            return self._record_infeasible(
                "E_ROUTE_LENGTH", "move", request, turn_already_used=True,
            )
        if any(not isinstance(direction, str) or direction not in DIRECTIONS for direction in directions):
            return self._record_infeasible(
                "E_DIRECTION", "move", request, turn_already_used=True,
            )

        edge_results: list[dict[str, Any]] = []
        total_cost_s = 0.0
        total_distance_m = 0.0
        stop_code: str | None = None
        stop_detail: str | None = None
        for route_index, direction in enumerate(directions):
            edge = self.graph.neighbor_map[self.current].get(direction)
            if edge is None:
                stop_code = "E_BOUNDARY"
                stop_detail = self._boundary_reason(direction)
                break
            if not self._edge_affordable(edge):
                stop_code = "E_PHYSICAL_BUDGET"
                break

            edge_origin = self.current
            cost_s = self._edge_time(edge)
            self.budget.used_s += cost_s
            self.budget.used_m += edge.distance_m
            self.current = edge.target
            self.edge_sequence += 1
            reading = self._measure(self.current, mode="transit")
            edge_results.append({
                "route_index": route_index,
                "edge_sequence": self.edge_sequence,
                "direction": direction,
                "origin_node": edge_origin,
                "arrival_node": self.current,
                "cost_s": cost_s,
                "path_length_m": edge.distance_m,
                "reading": reading,
            })
            if edge_checkpoint is not None:
                # Do not expose the live sample-ledger object to audit code.
                edge_checkpoint(copy.deepcopy(edge_results[-1]))
            total_cost_s += cost_s
            total_distance_m += edge.distance_m

        completed = len(edge_results)
        if completed == 0:
            result = {
                "ok": False,
                "action": "infeasible",
                "requested_action": "move",
                "arguments": request,
                "code": stop_code,
                "detail": stop_detail,
                "origin_node": origin,
                "state_version_before": before,
                "state_version_after": self.state_version,
                "requested_edges": len(directions),
                "completed_edges": 0,
                "route_complete": False,
                "route_stop_reason": (
                    "physical_budget_unaffordable"
                    if stop_code == "E_PHYSICAL_BUDGET"
                    else "native_boundary"
                ),
                "edge_transitions": [],
                "budget_after": self._budget_snapshot(),
            }
        else:
            result = {
                "ok": True,
                "action": "move",
                "directions": list(directions),
                "requested_edges": len(directions),
                "completed_edges": completed,
                "route_complete": completed == len(directions),
                "route_stop_reason": (
                    "segment_complete"
                    if completed == len(directions)
                    else "physical_budget_unaffordable"
                    if stop_code == "E_PHYSICAL_BUDGET"
                    else "native_boundary"
                ),
                "code": stop_code,
                "detail": stop_detail,
                "origin_node": origin,
                "arrival_node": self.current,
                "cost_s": total_cost_s,
                "path_length_m": total_distance_m,
                "edge_transitions": edge_results,
                "readings": [item["reading"] for item in edge_results],
                "state_version_before": before,
                "state_version_after": self.state_version,
                "budget_after": self._budget_snapshot(),
            }

        self._record_repetition(result, request)
        self.events.append(result)
        self._maybe_submit_only()
        result["route_intervention_event"] = result["route_stop_reason"]
        result["next_inference_event"] = result["route_stop_reason"]
        if self.mode == "submit_only":
            result["transition_after_action"] = "submit_only"
            result["submit_only_reason"] = self.submit_only_reason
            result["next_inference_event"] = "submit_only"
        return result

    def log(
        self, run_id: str, actor: str, seed: int, science_eligible: bool,
    ) -> dict[str, Any]:
        value = super().log(run_id, actor, seed, science_eligible)
        value.update({
            "schema_version": "5.1",
            "protocol_id": self.protocol_id,
            "parent_protocol_id": str(self.protocol["protocol_id"]),
            "control_protocol_sha256": stable_hash(self.control_protocol),
            "physical_edge_count": self.edge_sequence,
        })
        return value


__all__ = ["DEFAULT_PROTOCOL", "EnvironmentV20_1"]
