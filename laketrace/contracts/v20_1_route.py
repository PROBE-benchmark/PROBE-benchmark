#!/usr/bin/env python3
"""Append-only, native-edge route transactions for LakeTrace v20.1.

This module is deliberately smaller than the full frozen v20 run contract.  It
provides the part that changes in v20.1: one durable provider authorization may
expand into several native 3-D edges.  The journal makes that expansion
recoverable without asking the provider again and replayable down to every
arrival sample and physical budget debit.

Durability boundary
-------------------
``provider_received`` is fsynced before any physical edge.  ``route_started``
binds the authorization to the exact pre-route environment.  Every arrival is
then fsynced as ``edge_committed`` before the next edge can execute, followed by
one ``route_committed`` aggregate event.  A process crash may leave only a tail
prefix.  Recovery reconstructs the committed prefix from the frozen simulator
inputs, verifies it byte-for-byte, and appends only the missing suffix.  It never
calls a provider.

Like any local append-only log, deletion of a complete tail is indistinguishable
from a crash unless an external seal has already been anchored.  Interior edits,
deletions, insertions, and reordering are detected by the hash chain and lifecycle
checks.  Completed attempts should additionally anchor ``head_sha256`` in the
ordinary v20.1 episode seal.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from laketrace.env.graph import DIRECTIONS
from laketrace.env.v20_1 import EnvironmentV20_1
from laketrace.util import append_jsonl, jsonl, recover_jsonl, stable_hash


PROTOCOL_ID = "laketrace-v20.1-3d-axis-run-v1"
_ROUTE_KINDS = {
    "provider_received", "route_started", "edge_committed", "route_committed",
}


class JournalIntegrityError(ValueError):
    """Raised when retained evidence is not an unambiguous append-only prefix."""


class SimulatedProcessCrash(RuntimeError):
    """Test-only crash injected immediately after a durable edge commit."""


@dataclass(frozen=True)
class ReplayResult:
    environment: EnvironmentV20_1
    completed_route_count: int
    recovered_edge_count: int
    appended_edge_count: int
    pending_route_recovered: bool
    head_sha256: str


@dataclass(frozen=True)
class RouteExecutionResult:
    event: dict[str, Any]
    recovered_edge_count: int
    appended_edge_count: int
    head_sha256: str


def environment_checkpoint(env: EnvironmentV20_1) -> dict[str, Any]:
    """Return the exact mutable-state checkpoint needed for deterministic replay."""

    value = {
        "current_node": int(env.current),
        "physical_edges_used": int(env.edge_sequence),
        "state_version": int(env.state_version),
        "sample_sequence": int(env.sample_sequence),
        "budget": copy.deepcopy(env._budget_snapshot()),
        "episode_mode": str(env.mode),
        "submit_only_reason": env.submit_only_reason,
        "terminated": bool(env.terminated),
        "termination_reason": env.termination_reason,
        "sample_ledger_sha256": stable_hash(env.sample_ledger),
        "event_chain_sha256": stable_hash(env.events),
    }
    if str(env.protocol_id).startswith("laketrace-v20.3"):
        value.update({
            "successful_actions_used": int(env.turn),
            "control_counter_semantics": "successful_actions_only",
        })
    else:
        value["provider_rounds_used"] = int(env.turn)
    return value


def _entry_digest(entry: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in entry.items() if key != "entry_sha256"})


def _validate_directions(directions: Any) -> list[str]:
    if not isinstance(directions, list) or not 1 <= len(directions) <= 32:
        raise ValueError("directions must be a list containing 1..32 native edges")
    if any(not isinstance(item, str) or item not in DIRECTIONS for item in directions):
        raise ValueError("directions contains a non-native 3-D edge")
    return list(directions)


def _route_id(
    attempt_id: str,
    provider_round_index: int,
    directions: list[str],
    provider_payload_sha256: str,
) -> str:
    return stable_hash({
        "attempt_id": attempt_id,
        "provider_round_index": provider_round_index,
        "directions": directions,
        "provider_payload_sha256": provider_payload_sha256,
    })


class AppendOnlyRouteJournal:
    """A hash-chained sequence of provider-route transactions."""

    def __init__(
        self,
        path: str | Path,
        *,
        attempt_id: str,
        binding: dict[str, Any],
        resume: bool = False,
    ) -> None:
        self.path = Path(path)
        self.attempt_id = str(attempt_id)
        if not self.attempt_id:
            raise ValueError("attempt_id must be non-empty")
        self.binding = copy.deepcopy(binding)
        if resume:
            self.entries = self.load(self.path)
            self._validate_start_binding()
        else:
            if self.path.is_file() and self.path.stat().st_size:
                raise ValueError("new route journal refuses a non-empty path")
            self.entries: list[dict[str, Any]] = []
            self._append({
                "kind": "attempt_started",
                "attempt_id": self.attempt_id,
                "protocol_id": PROTOCOL_ID,
                "binding": self.binding,
                "binding_sha256": stable_hash(self.binding),
            })
        self.validate_lifecycle(self.entries)

    @staticmethod
    def load(path: str | Path) -> list[dict[str, Any]]:
        target = Path(path)
        if not target.is_file():
            raise JournalIntegrityError("route journal does not exist")
        # A torn last line is a crash artifact, not a committed record.
        recover_jsonl(target)
        entries = list(jsonl(target))
        if not entries:
            raise JournalIntegrityError("route journal is empty")
        previous: str | None = None
        for index, entry in enumerate(entries):
            if entry.get("journal_index") != index:
                raise JournalIntegrityError(f"non-contiguous journal index at row {index}")
            if entry.get("previous_entry_sha256") != previous:
                raise JournalIntegrityError(f"predecessor hash mismatch at row {index}")
            digest = _entry_digest(entry)
            if entry.get("entry_sha256") != digest:
                raise JournalIntegrityError(f"entry hash mismatch at row {index}")
            previous = digest
        AppendOnlyRouteJournal.validate_lifecycle(entries)
        return entries

    def _validate_start_binding(self) -> None:
        start = self.entries[0]
        expected = {
            "kind": "attempt_started",
            "attempt_id": self.attempt_id,
            "protocol_id": PROTOCOL_ID,
            "binding": self.binding,
            "binding_sha256": stable_hash(self.binding),
        }
        for key, value in expected.items():
            if start.get(key) != value:
                raise JournalIntegrityError(f"route journal binding mismatch: {key}")

    def _append(self, payload: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "journal_index": len(self.entries),
            "previous_entry_sha256": (
                self.entries[-1]["entry_sha256"] if self.entries else None
            ),
            **copy.deepcopy(payload),
        }
        entry["entry_sha256"] = _entry_digest(entry)
        append_jsonl(self.path, entry)
        self.entries.append(entry)
        return entry

    @property
    def head_sha256(self) -> str:
        return str(self.entries[-1]["entry_sha256"])

    @staticmethod
    def validate_lifecycle(entries: list[dict[str, Any]]) -> None:
        if not entries or entries[0].get("kind") != "attempt_started":
            raise JournalIntegrityError("journal must start with attempt_started")
        if sum(entry.get("kind") == "attempt_started" for entry in entries) != 1:
            raise JournalIntegrityError("journal contains multiple attempt_started entries")
        if entries[0].get("protocol_id") != PROTOCOL_ID:
            raise JournalIntegrityError("attempt protocol id is not v20.1")
        if entries[0].get("binding_sha256") != stable_hash(entries[0].get("binding")):
            raise JournalIntegrityError("attempt binding digest mismatch")
        attempt_id = entries[0].get("attempt_id")
        pending: dict[str, Any] | None = None
        last_round = -1
        committed_route_ids: set[str] = set()
        for row, entry in enumerate(entries[1:], 1):
            kind = entry.get("kind")
            if kind not in _ROUTE_KINDS:
                raise JournalIntegrityError(f"unknown route journal kind at row {row}: {kind}")
            if entry.get("attempt_id") != attempt_id:
                raise JournalIntegrityError(f"attempt id mismatch at row {row}")
            route_id = entry.get("route_id")
            round_index = entry.get("provider_round_index")
            if not isinstance(route_id, str) or not route_id:
                raise JournalIntegrityError(f"missing route id at row {row}")
            if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
                raise JournalIntegrityError(f"invalid provider round at row {row}")

            if kind == "provider_received":
                if pending is not None:
                    raise JournalIntegrityError("a second provider response follows an open route")
                if round_index <= last_round:
                    raise JournalIntegrityError("provider route rounds are not strictly increasing")
                if route_id in committed_route_ids:
                    raise JournalIntegrityError("duplicate route id")
                directions = _validate_directions(entry.get("directions"))
                provider_payload = entry.get("provider_payload")
                if not isinstance(provider_payload, dict):
                    raise JournalIntegrityError("provider payload is not an object")
                if entry.get("provider_payload_sha256") != stable_hash(provider_payload):
                    raise JournalIntegrityError("provider payload digest mismatch")
                if entry.get("decision_record_sha256") != stable_hash(
                    entry.get("decision_record")
                ):
                    raise JournalIntegrityError("decision record digest mismatch")
                expected_id = _route_id(
                    str(attempt_id), round_index, directions,
                    str(entry.get("provider_payload_sha256") or ""),
                )
                if route_id != expected_id:
                    raise JournalIntegrityError("provider route id does not match its authorization")
                pending = {
                    "route_id": route_id,
                    "round": round_index,
                    "started": False,
                    "edge_count": 0,
                    "directions": directions,
                }
                last_round = round_index
                continue

            if pending is None or route_id != pending["route_id"] or round_index != pending["round"]:
                raise JournalIntegrityError(f"orphan or mismatched route record at row {row}")
            if kind == "route_started":
                if pending["started"] or pending["edge_count"]:
                    raise JournalIntegrityError("route_started is duplicated or out of order")
                pre_state = entry.get("pre_state")
                if not isinstance(pre_state, dict):
                    raise JournalIntegrityError("route pre-state is not an object")
                if entry.get("pre_state_sha256") != stable_hash(pre_state):
                    raise JournalIntegrityError("route pre-state digest mismatch")
                if entry.get("directions_sha256") != stable_hash(pending["directions"]):
                    raise JournalIntegrityError("route directions digest mismatch")
                pending["started"] = True
            elif kind == "edge_committed":
                if not pending["started"]:
                    raise JournalIntegrityError("edge committed before route_started")
                if entry.get("route_index") != pending["edge_count"]:
                    raise JournalIntegrityError("edge route indices are not contiguous")
                transition = entry.get("transition")
                if not isinstance(transition, dict):
                    raise JournalIntegrityError("edge transition is not an object")
                if entry.get("transition_sha256") != stable_hash(transition):
                    raise JournalIntegrityError("edge transition digest mismatch")
                state = entry.get("state_checkpoint")
                if not isinstance(state, dict):
                    raise JournalIntegrityError("edge state checkpoint is not an object")
                if entry.get("state_checkpoint_sha256") != stable_hash(state):
                    raise JournalIntegrityError("edge state checkpoint digest mismatch")
                if pending["edge_count"] >= len(pending["directions"]):
                    raise JournalIntegrityError("journal contains more edges than authorized")
                if transition.get("route_index") != pending["edge_count"]:
                    raise JournalIntegrityError("transition route index is inconsistent")
                if transition.get("direction") != pending["directions"][pending["edge_count"]]:
                    raise JournalIntegrityError("transition direction was not authorized")
                pending["edge_count"] += 1
            else:
                if not pending["started"]:
                    raise JournalIntegrityError("route committed before route_started")
                event = entry.get("event")
                if not isinstance(event, dict) or entry.get("event_sha256") != stable_hash(event):
                    raise JournalIntegrityError("route event digest mismatch")
                if entry.get("completed_edges") != pending["edge_count"]:
                    raise JournalIntegrityError("route event/edge count mismatch")
                final_state = entry.get("final_state")
                if not isinstance(final_state, dict):
                    raise JournalIntegrityError("route final state is not an object")
                if entry.get("final_state_sha256") != stable_hash(final_state):
                    raise JournalIntegrityError("route final state digest mismatch")
                if event.get("completed_edges") != pending["edge_count"]:
                    raise JournalIntegrityError("aggregate event completed-edge count mismatch")
                if not isinstance(event.get("route_complete"), bool):
                    raise JournalIntegrityError("aggregate route-complete flag is not boolean")
                if event["route_complete"] != (
                    pending["edge_count"] == len(pending["directions"])
                ):
                    raise JournalIntegrityError("aggregate route-complete flag mismatch")
                committed_route_ids.add(route_id)
                pending = None

    def route_records(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for entry in self.entries[1:]:
            kind = entry["kind"]
            if kind == "provider_received":
                current = {
                    "provider": entry,
                    "started": None,
                    "edges": [],
                    "committed": None,
                }
                output.append(current)
            elif current is None:
                raise JournalIntegrityError("route record precedes provider authorization")
            elif kind == "route_started":
                current["started"] = entry
            elif kind == "edge_committed":
                current["edges"].append(entry)
            elif kind == "route_committed":
                current["committed"] = entry
                current = None
        return output

    def receive_route(
        self,
        *,
        provider_round_index: int,
        directions: list[str],
        provider_payload: dict[str, Any],
        decision_record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Durably retain one already-received provider route authorization."""

        self.entries = self.load(self.path)
        self._validate_start_binding()
        records = self.route_records()
        if records and records[-1]["committed"] is None:
            raise JournalIntegrityError("cannot receive a provider response while a route is open")
        directions = _validate_directions(directions)
        if isinstance(provider_round_index, bool) or not isinstance(provider_round_index, int):
            raise ValueError("provider_round_index must be an integer")
        provider_payload = copy.deepcopy(provider_payload)
        payload_sha = stable_hash(provider_payload)
        route_id = _route_id(
            self.attempt_id, provider_round_index, directions, payload_sha,
        )
        entry = self._append({
            "kind": "provider_received",
            "attempt_id": self.attempt_id,
            "provider_round_index": provider_round_index,
            "route_id": route_id,
            "directions": directions,
            "provider_payload": provider_payload,
            "provider_payload_sha256": payload_sha,
            "decision_record": copy.deepcopy(decision_record),
            "decision_record_sha256": stable_hash(decision_record),
        })
        self.validate_lifecycle(self.entries)
        return entry

    def _start_route_if_needed(
        self, env: EnvironmentV20_1, record: dict[str, Any], *,
        provider_round_is_action_counter: bool = True,
    ) -> None:
        if record["started"] is not None:
            if record["started"].get("pre_state") != environment_checkpoint(env):
                raise JournalIntegrityError("pending route pre-state does not replay")
            return
        provider = record["provider"]
        if (
            provider_round_is_action_counter
            and provider["provider_round_index"] != env.turn
        ):
            raise JournalIntegrityError("provider authorization round does not match environment")
        pre_state = environment_checkpoint(env)
        self._append({
            "kind": "route_started",
            "attempt_id": self.attempt_id,
            "provider_round_index": provider["provider_round_index"],
            "route_id": provider["route_id"],
            "directions_sha256": stable_hash(provider["directions"]),
            "pre_state": pre_state,
            "pre_state_sha256": stable_hash(pre_state),
        })

    @staticmethod
    def _verify_started(env: EnvironmentV20_1, record: dict[str, Any]) -> None:
        started = record["started"]
        if not isinstance(started, dict):
            raise JournalIntegrityError("route lacks route_started")
        pre_state = environment_checkpoint(env)
        if started.get("pre_state") != pre_state:
            raise JournalIntegrityError("route pre-state does not replay")
        if started.get("pre_state_sha256") != stable_hash(pre_state):
            raise JournalIntegrityError("route pre-state digest mismatch")

    def _execute_record(
        self,
        env: EnvironmentV20_1,
        record: dict[str, Any],
        *,
        append_missing: bool,
        crash_after_new_edges: int | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        provider = record["provider"]
        directions = list(provider["directions"])
        durable_edges = list(record["edges"])
        recovered = 0
        appended = 0

        def checkpoint(transition: dict[str, Any]) -> None:
            nonlocal recovered, appended
            route_index = int(transition["route_index"])
            state = environment_checkpoint(env)
            if route_index < len(durable_edges):
                expected = durable_edges[route_index]
                if expected.get("transition") != transition:
                    raise JournalIntegrityError(
                        f"edge {route_index} transition does not replay"
                    )
                if expected.get("state_checkpoint") != state:
                    raise JournalIntegrityError(
                        f"edge {route_index} state checkpoint does not replay"
                    )
                recovered += 1
                return
            if not append_missing:
                raise JournalIntegrityError("replay produced an unjournaled edge")
            edge_entry = self._append({
                "kind": "edge_committed",
                "attempt_id": self.attempt_id,
                "provider_round_index": provider["provider_round_index"],
                "route_id": provider["route_id"],
                "route_index": route_index,
                "transition": transition,
                "transition_sha256": stable_hash(transition),
                "state_checkpoint": state,
                "state_checkpoint_sha256": stable_hash(state),
            })
            durable_edges.append(edge_entry)
            appended += 1
            if crash_after_new_edges is not None and appended >= crash_after_new_edges:
                raise SimulatedProcessCrash(
                    f"injected crash after durable edge {route_index}"
                )

        result = env.move(directions, edge_checkpoint=checkpoint)
        if len(durable_edges) != int(result.get("completed_edges", -1)):
            raise JournalIntegrityError("journal edge count does not match executed route prefix")
        return result, recovered, appended

    def execute_pending_on_environment(
        self,
        env: EnvironmentV20_1,
        *,
        crash_after_new_edges: int | None = None,
        provider_round_is_action_counter: bool = True,
    ) -> RouteExecutionResult:
        """Complete the journal's open tail route on a reconstructed environment.

        This is the integration point for the full v20.1 run contract.  That
        outer contract first reconstructs all earlier move/analyze/submit rounds,
        then passes the exact pre-route environment here.  The retained provider
        authorization is used directly; no provider object or callback is accepted.
        """

        self.entries = self.load(self.path)
        self._validate_start_binding()
        records = self.route_records()
        if not records or records[-1]["committed"] is not None:
            raise JournalIntegrityError("route journal has no open tail route")
        record = records[-1]
        if record["started"] is None:
            self._start_route_if_needed(
                env,
                record,
                provider_round_is_action_counter=provider_round_is_action_counter,
            )
            self.entries = self.load(self.path)
            record = self.route_records()[-1]
        self._verify_started(env, record)
        event, recovered, appended = self._execute_record(
            env,
            record,
            append_missing=True,
            crash_after_new_edges=crash_after_new_edges,
        )
        final_state = environment_checkpoint(env)
        self._append({
            "kind": "route_committed",
            "attempt_id": self.attempt_id,
            "provider_round_index": record["provider"]["provider_round_index"],
            "route_id": record["provider"]["route_id"],
            "completed_edges": int(event["completed_edges"]),
            "event": copy.deepcopy(event),
            "event_sha256": stable_hash(event),
            "final_state": final_state,
            "final_state_sha256": stable_hash(final_state),
        })
        self.entries = self.load(self.path)
        return RouteExecutionResult(
            event=event,
            recovered_edge_count=recovered,
            appended_edge_count=appended,
            head_sha256=self.head_sha256,
        )

    def replay_and_resume(
        self,
        environment_factory: Callable[[], EnvironmentV20_1],
        *,
        crash_after_new_edges: int | None = None,
    ) -> ReplayResult:
        """Rebuild complete routes and finish at most one open tail route.

        The method has no provider callback by design.  A retained
        ``provider_received`` authorization is the sole authority for a resumed
        suffix, so recovery cannot accidentally rebill or change the route.
        """

        self.entries = self.load(self.path)
        self._validate_start_binding()
        records = self.route_records()
        env = environment_factory()
        completed = 0
        recovered = 0
        appended = 0
        pending_recovered = False

        for index, record in enumerate(records):
            is_pending = record["committed"] is None
            if is_pending and index != len(records) - 1:
                raise JournalIntegrityError("only the journal tail may contain an open route")
            if record["started"] is None:
                if not is_pending:
                    raise JournalIntegrityError("committed route lacks route_started")
                self._start_route_if_needed(env, record)
                self.entries = self.load(self.path)
                record = self.route_records()[index]
            self._verify_started(env, record)
            result, recovered_now, appended_now = self._execute_record(
                env,
                record,
                append_missing=is_pending,
                crash_after_new_edges=(crash_after_new_edges if is_pending else None),
            )
            recovered += recovered_now
            appended += appended_now

            if is_pending:
                event = copy.deepcopy(result)
                final_state = environment_checkpoint(env)
                self._append({
                    "kind": "route_committed",
                    "attempt_id": self.attempt_id,
                    "provider_round_index": record["provider"]["provider_round_index"],
                    "route_id": record["provider"]["route_id"],
                    "completed_edges": int(result["completed_edges"]),
                    "event": event,
                    "event_sha256": stable_hash(event),
                    "final_state": final_state,
                    "final_state_sha256": stable_hash(final_state),
                })
                pending_recovered = True
                completed += 1
            else:
                committed = record["committed"]
                if committed.get("event") != result:
                    raise JournalIntegrityError("route aggregate event does not replay")
                if committed.get("final_state") != environment_checkpoint(env):
                    raise JournalIntegrityError("route final state does not replay")
                completed += 1

        self.entries = self.load(self.path)
        return ReplayResult(
            environment=env,
            completed_route_count=completed,
            recovered_edge_count=recovered,
            appended_edge_count=appended,
            pending_route_recovered=pending_recovered,
            head_sha256=self.head_sha256,
        )


__all__ = [
    "AppendOnlyRouteJournal",
    "JournalIntegrityError",
    "PROTOCOL_ID",
    "ReplayResult",
    "RouteExecutionResult",
    "SimulatedProcessCrash",
    "environment_checkpoint",
]
