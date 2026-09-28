"""V20.3 closed-loop control over unchanged six-neighbour 3-D physics."""

from __future__ import annotations

import copy
import math
from collections import deque
from pathlib import Path
from typing import Any

from laketrace.env.v20 import DEFAULT_PROTOCOL as V20_PHYSICS_PROTOCOL
from laketrace.env.v20_1 import EnvironmentV20_1
from laketrace.env.v20_2 import EnvironmentV20_2
from laketrace.util import load_config, stable_hash


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "configs" / "v20_3" / "protocol.json"
GOTO_DIRECTION_ORDER = ("up", "down", "east", "west", "north", "south")


class EnvironmentV20_3(EnvironmentV20_2):
    """Count only successfully executed actions toward the 96-action guard.

    Provider/schema repair attempts remain visible append-only events, but they
    do not consume the scientific action budget or any physical budget.
    """

    def __init__(
        self,
        store: Any,
        instance_id: str,
        seed: int = 0,
        *,
        protocol_path: str | Path = DEFAULT_PROTOCOL,
        physics_protocol_path: str | Path = V20_PHYSICS_PROTOCOL,
        physical_budget_enabled: bool = True,
    ) -> None:
        # These counters are deliberately independent. ``turn`` is retained
        # only for inherited execution mechanics and must not be presented as
        # either provider calls or successful active actions.
        self.provider_call_count = 0
        self.physical_budget_enabled = physical_budget_enabled
        self._successful_action_count = 0
        super().__init__(
            store,
            instance_id,
            seed,
            physics_protocol_path=physics_protocol_path,
        )
        self.control_protocol_path = Path(protocol_path)
        self.control_protocol = load_config(self.control_protocol_path)
        if self.control_protocol.get("protocol_id") != "laketrace-v20.3-closed-loop-3d-v1":
            raise ValueError("unexpected v20.3 protocol id")
        if not self.physical_budget_enabled:
            self.control_protocol["resource_control"]["physical_budget_enabled"] = False
        self.partial_answer: dict[str, Any] = {}
        self.partial_update_counts = {
            level: 0 for level in ("L1", "L2", "L3", "L4", "L5")
        }

    @property
    def max_turns(self) -> int:
        return int(self.control_protocol["resource_control"]["provider_call_hard_guard"])

    @property
    def max_steps(self) -> int:
        return int(self.control_protocol["movement"]["goto_max_edges_per_action"])

    @property
    def successful_action_count(self) -> int:
        return int(self._successful_action_count)

    def set_provider_call_count(self, value: int) -> None:
        if value < 0:
            raise ValueError("provider call count must be nonnegative")
        self.provider_call_count = int(value)

    def _budget_snapshot(self) -> dict[str, Any]:
        value = super()._budget_snapshot()
        if not self.physical_budget_enabled:
            for name in ("endurance", "range"):
                value[name].update({"limit": None, "remaining": None, "limited": False})
        value["successful_actions"] = {
            "used": self.successful_action_count,
            "limit": int(
                self.control_protocol["resource_control"]["max_successful_active_actions"]
            ),
            "scientific_budget": False,
        }
        value["provider_round_guard"] = {
            "used": int(self.provider_call_count),
            "limit": self.max_turns,
            "remaining": max(0, self.max_turns - int(self.provider_call_count)),
            "scientific_budget": False,
        }
        return value

    def _measure(self, index: int, mode: str) -> dict[str, Any]:
        sample = super()._measure(index, mode)
        # Provider rounds are zero-based in retained call records.
        sample["provider_round"] = max(0, int(self.provider_call_count) - 1)
        return sample

    def _last_route_measurements(self) -> list[dict[str, Any]]:
        round_index = max(0, int(self.provider_call_count) - 1)
        return [
            self._public_route_sample(sample)
            for sample in self.sample_ledger
            if int(sample["provider_round"]) == round_index
            and int(sample["edge_sequence"]) > 0
        ]

    def apply_resource_stage(self) -> None:
        if self.terminated or self.mode == "submit_only":
            return
        policy = self.control_protocol["resource_control"]
        fraction = self.budget.used_s / self.budget.endurance_s
        if self.physical_budget_enabled and fraction >= float(policy["submit_only_endurance_fraction"]):
            self.enter_submit_only("endurance_submit_reserve")
        elif self.successful_action_count >= int(policy["max_successful_active_actions"]):
            self.enter_submit_only("successful_action_submit_reserve")

    def enter_submit_only(self, reason: str) -> None:
        before = self.mode
        super().enter_submit_only(reason)
        if before != self.mode:
            self.events.append({
                "ok": True,
                "action": "control_transition",
                "from_mode": before,
                "to_mode": self.mode,
                "reason": str(reason),
                "budget_after": self._budget_snapshot(),
            })

    def exhaust_provider_round_guard(self) -> None:
        if self.terminated:
            return
        before = self.mode
        super().exhaust_provider_round_guard()
        self.events.append({
            "ok": False,
            "action": "control_transition",
            "from_mode": before,
            "to_mode": self.mode,
            "reason": self.termination_reason,
            "scoreable": False,
            "budget_after": self._budget_snapshot(),
        })

    def exhaust_submit_schema_retries(self) -> None:
        if self.terminated:
            return
        before = self.mode
        super().exhaust_submit_schema_retries()
        self.events.append({
            "ok": False,
            "action": "control_transition",
            "from_mode": before,
            "to_mode": self.mode,
            "reason": self.termination_reason,
            "scoreable": False,
            "budget_after": self._budget_snapshot(),
        })

    @property
    def resource_stage(self) -> str:
        if self.mode == "submit_only":
            return "SUBMIT_ONLY"
        if self.terminated:
            return "TERMINATED"
        fraction = self.budget.used_s / self.budget.endurance_s
        return (
            "REFINE"
            if self.physical_budget_enabled and fraction >= float(self.control_protocol["resource_control"]["refine_endurance_fraction"])
            else "EXPLORE"
        )

    def _edge_affordable(self, edge) -> bool:
        return not self.physical_budget_enabled or super()._edge_affordable(edge)

    def record_invalid(
        self, code: str, detail: str, requested_action: str | None = None,
    ) -> dict[str, Any]:
        if self.terminated:
            return {"ok": False, "code": "E_TERMINATED"}
        result = {
            "ok": False,
            "action": "invalid",
            "requested_action": requested_action,
            "code": code,
            "detail": detail,
            "successful_action_counted": False,
            "budget_after": self._budget_snapshot(),
        }
        self.events.append(result)
        return result

    def move(self, directions: list[str], **kwargs: Any) -> dict[str, Any]:
        before = self.turn
        result = super().move(directions, **kwargs)
        if not result.get("ok"):
            self.turn = before
            result["successful_action_counted"] = False
            if "budget_after" in result:
                result["budget_after"] = self._budget_snapshot()
        else:
            self._successful_action_count += 1
            result["successful_action_counted"] = True
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
        before = self.turn
        result = super()._record_infeasible(
            code,
            requested_action,
            arguments,
            turn_already_used=turn_already_used,
            detail=detail,
        )
        self.turn = before
        result["successful_action_counted"] = False
        result["budget_after"] = self._budget_snapshot()
        return result

    def resolve_goto_target(self, x: float, y: float, z: float) -> int:
        """Snap to the nearest reachable node; cache only immutable geometry."""
        requested = (float(x), float(y), float(z))
        reachable = getattr(self, "_goto_component", None)
        if reachable is None:
            reachable = {self.current}
            queue = deque([self.current])
            while queue:
                for edge in self.graph.neighbor_map[queue.popleft()].values():
                    if edge is not None and edge.target not in reachable:
                        reachable.add(edge.target)
                        queue.append(edge.target)
            self._goto_component = frozenset(reachable)
        exact = self.graph.index_by_xyz.get(requested)
        if exact in reachable:
            return int(exact)
        return min(reachable, key=lambda node: (
            math.dist(self.graph.positions[node], requested), node,
        ))

    def plan_goto(self, x: float, y: float, z: float) -> tuple[list[str], int]:
        """Return a deterministic, measurement-blind route toward ``xyz``.

        The planner uses only the public navigation graph and coordinates. It
        performs no sensor, source, score, threshold, or candidate query.
        """

        target = self.resolve_goto_target(x, y, z)
        parents: dict[int, tuple[int, str] | None] = {self.current: None}
        queue: deque[int] = deque([self.current])
        while queue and target not in parents:
            node = queue.popleft()
            for direction in GOTO_DIRECTION_ORDER:
                edge = self.graph.neighbor_map[node].get(direction)
                if edge is None or edge.target in parents:
                    continue
                parents[edge.target] = (node, direction)
                queue.append(edge.target)
        reversed_route: list[str] = []
        cursor = target
        while parents[cursor] is not None:
            parent, direction = parents[cursor]  # type: ignore[misc]
            reversed_route.append(direction)
            cursor = parent
        route = list(reversed(reversed_route))[: self.max_steps]
        return route, target

    def record_partial_answer(self, level: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.terminated:
            return {"ok": False, "code": "E_TERMINATED"}
        if level not in {"L1", "L2", "L3", "L4", "L5"}:
            raise ValueError("unknown answer level")
        limit = int(self.control_protocol["submission"]["max_updates_per_level"])
        if self.partial_update_counts[level] >= limit:
            return self._record_infeasible(
                "E_LEVEL_UPDATE_LIMIT",
                f"submit_{level.lower()}",
                {"level": level},
                detail=f"{level} already used its {limit} registered updates",
            )
        canonical_payload = copy.deepcopy(payload)
        if level == "L2":
            canonical_payload = {
                "peak_position": {
                    "x": float(payload["plume_x"]),
                    "y": float(payload["plume_y"]),
                    "z": float(payload["plume_z"]),
                },
            }
        elif level == "L4":
            canonical_payload = {
                "source_position": {
                    "x": float(payload["source_x"]),
                    "y": float(payload["source_y"]),
                    "z": float(payload["source_z"]),
                },
                "source_candidate_id": payload.get("source_candidate_id"),
            }
        self.partial_answer.update(canonical_payload)
        self.partial_update_counts[level] += 1
        result = {
            "ok": True,
            "action": "submit_partial",
            "level": level,
            "accepted_fields": sorted(canonical_payload),
            "completed_levels": sorted(self.completed_levels),
            "update_index": self.partial_update_counts[level],
            "successful_action_counted": False,
            "budget_after": self._budget_snapshot(),
        }
        self.events.append(result)
        return result

    @property
    def completed_levels(self) -> set[str]:
        levels = set()
        if "anomaly" in self.partial_answer:
            levels.add("L1")
        if "peak_position" in self.partial_answer:
            levels.add("L2")
        if "mechanism" in self.partial_answer:
            levels.add("L3")
        if "source_position" in self.partial_answer:
            levels.add("L4")
        if "decision" in self.partial_answer:
            levels.add("L5")
        return levels

    def finalize_submission(self) -> dict[str, Any]:
        return self.submit(copy.deepcopy(self.partial_answer))

    def observation(self, first: bool = False) -> dict[str, Any]:
        # Bypass the v20.2 presentation overlay: it names provider rounds as
        # the active budget, while v20.3 deliberately counts only successful
        # actions.  Reusing that inherited text would make the public contract
        # contradict the executor even though the state transition is correct.
        self.apply_resource_stage()
        value = EnvironmentV20_1.observation(self, first=first)
        policy = self.control_protocol["resource_control"]
        value.update({
            "schema_version": "5.3",
            "protocol_id": self.protocol_id,
            "control_granularity": {
                "provider_actions": ["goto", "move_one"],
                "goto_native_edges_per_action_max": self.max_steps,
                "goto_path_rule": self.control_protocol["movement"]["goto_path_rule"],
                "each_native_edge_measured": True,
                "planner_information": "public_geometry_only",
            },
            "successful_action_count": self.successful_action_count,
            "provider_round": int(self.provider_call_count),
            "resource_stage_policy": {
                "signals": [
                    "endurance_fraction", "successful_action_count",
                    "known_public_price_cost",
                ],
                "refine_endurance_fraction": policy["refine_endurance_fraction"],
                "submit_only_endurance_fraction": policy["submit_only_endurance_fraction"],
                "max_successful_active_actions": policy["max_successful_active_actions"],
            },
            "partial_submission": {
                "completed_levels": sorted(self.completed_levels),
                "missing_levels": sorted({"L1", "L2", "L3", "L4", "L5"} - self.completed_levels),
                "update_counts": copy.deepcopy(self.partial_update_counts),
                "max_updates_per_level": int(
                    self.control_protocol["submission"]["max_updates_per_level"]
                ),
            },
        })
        if not self.physical_budget_enabled:
            value["physical_budget_enabled"] = False
            value["resource_stage_policy"] = {
                "signals": ["successful_action_count", "known_public_price_cost"],
                "max_successful_active_actions": policy["max_successful_active_actions"],
            }
        return value

    def log(self, run_id: str, actor: str, seed: int, science_eligible: bool) -> dict[str, Any]:
        value = super().log(run_id, actor, seed, science_eligible)
        value.update({
            "schema_version": "5.3",
            "protocol_id": self.protocol_id,
            "control_protocol_sha256": stable_hash(self.control_protocol),
            "successful_action_count": self.successful_action_count,
            "turns": int(self.provider_call_count),
            "partial_answer": copy.deepcopy(self.partial_answer),
            "partial_update_counts": copy.deepcopy(self.partial_update_counts),
        })
        if not self.physical_budget_enabled:
            value.update({
                "physical_budget_enabled": False,
                "endurance_used_fraction": None,
                "range_used_fraction": None,
            })
        return value


__all__ = ["DEFAULT_PROTOCOL", "EnvironmentV20_3", "GOTO_DIRECTION_ORDER"]
