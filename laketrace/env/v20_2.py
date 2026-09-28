"""Resource-bounded control overlay over the verified v20.1 3-D physics."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from laketrace.env.v20 import DEFAULT_PROTOCOL as V20_PHYSICS_PROTOCOL
from laketrace.env.v20_1 import DEFAULT_PROTOCOL as V20_1_PROTOCOL
from laketrace.env.v20_1 import EnvironmentV20_1
from laketrace.util import load_config, stable_hash


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "configs" / "v20_2" / "protocol.json"


class EnvironmentV20_2(EnvironmentV20_1):
    """V20.1 edge physics with task-neutral resource stages.

    Stage transitions inspect only physical endurance and provider-round count.
    They never inspect measurements, evaluator truth, candidate distance, or score.
    """

    def __init__(
        self,
        store: Any,
        instance_id: str,
        seed: int = 0,
        *,
        protocol_path: str | Path = DEFAULT_PROTOCOL,
        physics_protocol_path: str | Path = V20_PHYSICS_PROTOCOL,
    ):
        super().__init__(
            store,
            instance_id,
            seed,
            protocol_path=V20_1_PROTOCOL,
            physics_protocol_path=physics_protocol_path,
        )
        self.control_protocol_path = Path(protocol_path)
        self.control_protocol = load_config(self.control_protocol_path)
        if self.control_protocol.get("protocol_id") != "laketrace-v20.2-resource-bounded-3d-v1":
            raise ValueError("unexpected v20.2 protocol id")
        if self.control_protocol.get("parent_protocol_id") != "laketrace-v20.1-3d-axis-run-v1":
            raise ValueError("v20.2 requires the verified v20.1 control parent")
        self._analysis_cache: dict[str, Any] | None = None

    @property
    def max_turns(self) -> int:
        policy = self.control_protocol["resource_control"]
        return int(policy["max_active_provider_rounds"]) + int(
            policy["submit_only_schema_retries"]
        )

    @property
    def max_steps(self) -> int:
        return int(self.control_protocol["movement"]["max_edges_per_action"])

    @property
    def resource_stage(self) -> str:
        policy = self.control_protocol["resource_control"]
        fraction = self.budget.used_s / self.budget.endurance_s
        if self.mode == "submit_only" or self.terminated:
            return "SUBMIT_ONLY" if self.submit_only_reason else "TERMINATED"
        if fraction >= float(policy["refine_endurance_fraction"]):
            return "REFINE"
        return "EXPLORE"

    def apply_resource_stage(self) -> None:
        if self.terminated or self.mode == "submit_only":
            return
        policy = self.control_protocol["resource_control"]
        fraction = self.budget.used_s / self.budget.endurance_s
        if fraction >= float(policy["submit_only_endurance_fraction"]):
            self.enter_submit_only("endurance_submit_reserve")
        elif self.turn >= int(policy["max_active_provider_rounds"]):
            self.enter_submit_only("provider_round_submit_reserve")

    def _maybe_submit_only(self) -> None:
        if self.terminated or self.mode == "submit_only":
            return
        self.apply_resource_stage()
        if self.mode == "submit_only":
            return
        affordable = any(
            edge is not None and self._edge_affordable(edge)
            for edge in self.graph.neighbors(self.current).values()
        )
        if not affordable:
            self.enter_submit_only("physical_budget_exhausted")

    def observation(self, first: bool = False) -> dict[str, Any]:
        self.apply_resource_stage()
        value = super().observation(first=first)
        policy = self.control_protocol["resource_control"]
        value.update({
            "schema_version": "5.2",
            "protocol_id": self.protocol_id,
            "resource_stage": self.resource_stage,
            "resource_stage_policy": {
                "signals": ["endurance_fraction", "provider_round_count", "known_public_price_cost"],
                "refine_endurance_fraction": policy["refine_endurance_fraction"],
                "submit_only_endurance_fraction": policy["submit_only_endurance_fraction"],
                "max_active_provider_rounds": policy["max_active_provider_rounds"],
            },
        })
        return value

    def analyze(self) -> dict[str, Any]:
        if (
            self._analysis_cache is not None
            and self.sample_sequence == self.last_analyze_sample_sequence
            and not self.terminated
            and self.mode != "submit_only"
        ):
            self.turn += 1
            result = copy.deepcopy(self._analysis_cache)
            result.update({
                "cost_s": 0.0,
                "physical_cost_s": 0.0,
                "new_samples_since_last_analyze": 0,
                "no_new_evidence": True,
                "cache_hit": True,
                "budget_after": self._budget_snapshot(),
            })
            self._record_repetition(result, {"action": "analyze"})
            self.events.append(result)
            self._maybe_submit_only()
            return result
        result = super().analyze()
        if result.get("ok") and result.get("action") == "analyze":
            result["cache_hit"] = False
            self._analysis_cache = copy.deepcopy(result)
        return result

    def log(self, run_id: str, actor: str, seed: int, science_eligible: bool) -> dict[str, Any]:
        value = super().log(run_id, actor, seed, science_eligible)
        value.update({
            "schema_version": "5.2",
            "protocol_id": self.protocol_id,
            "parent_protocol_id": str(self.control_protocol["parent_protocol_id"]),
            "control_protocol_sha256": stable_hash(self.control_protocol),
            "resource_stage_final": self.resource_stage,
        })
        return value


__all__ = ["DEFAULT_PROTOCOL", "EnvironmentV20_2"]
