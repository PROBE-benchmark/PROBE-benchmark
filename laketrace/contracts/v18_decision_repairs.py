from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from laketrace.util import stable_hash


REPAIR_CONTRACT_VERSION = "v18.decision-record-repair.1"


def repair_key(episode_id: str, decision_record_index: int) -> tuple[str, int]:
    return str(episode_id), int(decision_record_index)


def index_repairs(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    indexed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("repair_contract_version") != REPAIR_CONTRACT_VERSION:
            continue
        indexed[repair_key(row["episode_id"], row["decision_record_index"])].append(row)
    return dict(indexed)


def repair_matches_original(repair: dict[str, Any], episode: dict[str, Any], original: dict[str, Any]) -> bool:
    """Verify that a repair is only a readout of the exact frozen action input."""

    return (
        repair.get("data_contract_version") == episode.get("data_contract_version")
        and repair.get("model_config_hash") == episode.get("model_config_hash")
        and repair.get("decision_record_protocol_hash") == episode.get("decision_record_protocol_hash")
        and repair.get("basis_observation_hash") == original.get("basis_observation_hash")
        and repair.get("chosen_action_hash") == original.get("chosen_action_hash")
        and repair.get("readout_prompt_hash") == original.get("readout_prompt_hash")
        and repair.get("readout_schema_hash") == original.get("readout_schema_hash")
        and repair.get("main_event_chain_sha256") == episode.get("event_chain_sha256")
    )


def merge_episode_decision_records(
    episode: dict[str, Any], repair_rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return inline records plus the first valid exact-input repair for each gap.

    Inline valid records are immutable and always win. Repair attempts remain in
    their own append-only log; this function only presents a deterministic view.
    """

    episode_id = str(episode.get("episode_id") or "")
    indexed = index_repairs(repair_rows)
    merged: list[dict[str, Any]] = []
    for position, original in enumerate(episode.get("decision_record_calls") or []):
        index = int(original.get("decision_record_index", position))
        selected = dict(original)
        selected["record_origin"] = "inline"
        selected["original_status"] = original.get("status")
        selected["repair_attempt_index"] = None
        if original.get("status") != "ok":
            for repair in indexed.get(repair_key(episode_id, index), []):
                candidate = repair.get("decision_record") or {}
                if (
                    candidate.get("status") == "ok"
                    and candidate.get("model_identity_match") is True
                    and repair_matches_original(repair, episode, original)
                ):
                    selected = dict(candidate)
                    selected["decision_record_index"] = index
                    selected["event_index_before_execution"] = original.get("event_index_before_execution")
                    selected["linked_step_index"] = original.get("linked_step_index")
                    selected["action_valid"] = original.get("action_valid")
                    selected["action_result_code"] = original.get("action_result_code")
                    selected["record_origin"] = "repair"
                    selected["original_status"] = original.get("status")
                    selected["repair_attempt_index"] = repair.get("repair_attempt_index")
                    selected["repair_id"] = repair.get("repair_id")
                    break
        merged.append(selected)
    return merged


def repair_usage(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0}
    for row in rows:
        usage = (row.get("decision_record") or {}).get("usage") or {}
        for key in totals:
            totals[key] += int(usage.get(key, 0))
    return totals


def make_repair_id(payload: dict[str, Any]) -> str:
    return "repair-" + stable_hash(payload)[:24]
