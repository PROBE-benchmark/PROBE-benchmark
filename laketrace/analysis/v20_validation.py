"""V20 design controls with an explicit oracle/public information boundary.

Only ``reachable_upper_episode`` and its path helpers may inspect evaluator
truth or environment internals. Every executable classical baseline is run
through a newly opened public ``DataStore`` and the observation/action adapter
in :mod:`laketrace.agents.v20_baselines`.
"""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Any

from laketrace.agents.v20_baselines import (
    BASELINES, BASELINE_INFORMATION_TIER, run_baseline,
)
from laketrace.data.store import DataStore
from laketrace.env.v20 import EnvironmentV20
from laketrace.scoring.scorer import score_submission
from laketrace.util import stable_hash, write_csv, write_jsonl


REACHABLE_UPPER_INFORMATION_TIER = "oracle_reachable_upper"


def _public_runtime_store(store: DataStore) -> DataStore:
    """Reopen a data root without evaluator truth capability."""

    public = DataStore(store.root, evaluator=False)
    if public.evaluator:  # pragma: no cover - constructor contract guard
        raise AssertionError("public baseline store unexpectedly has evaluator access")
    return public


def _validate_public_baseline_episode(episode: dict[str, Any]) -> None:
    if episode.get("actor") not in BASELINES:
        raise ValueError("public baseline episode has an unregistered actor")
    if episode.get("information_tier") != "public_observation_only":
        raise ValueError("public baseline episode is missing its information tier")
    if "upper_truth_access" in episode:
        raise ValueError("oracle provenance leaked into a public baseline episode")
    moves = [event for event in episode.get("events", ()) if event.get("action") == "move"]
    if any(int(event.get("steps", 0)) != 1 for event in moves):
        raise ValueError("v20 public baseline emitted a non-single-edge move")
    if episode.get("answer") is None:
        raise ValueError("v20 public baseline did not submit an answer")


def _public_baseline_episode(
    store: DataStore,
    instance_id: str,
    baseline_id: str,
    seed: int,
    run_id: str,
) -> dict[str, Any]:
    """Run one baseline cell through public observation and action methods."""

    if store.evaluator:
        raise ValueError("public baseline runner requires a truth-disabled DataStore")
    env = EnvironmentV20(store, instance_id, seed)
    run_baseline(env, baseline_id, seed)
    episode = env.log(run_id, baseline_id, seed, False) | {
        "information_tier": BASELINE_INFORMATION_TIER[baseline_id],
    }
    _validate_public_baseline_episode(episode)
    return episode


def replay_public_baseline_episode(
    store: DataStore, episode: dict[str, Any],
) -> dict[str, Any]:
    """Replay a public baseline log through only the registered v20 actions."""

    public = _public_runtime_store(store)
    env = EnvironmentV20(
        public, str(episode["instance_id"]), int(episode["seed"]),
    )
    stored_events = episode.get("events")
    action_journal = episode.get("action_journal")
    source_events = stored_events if isinstance(stored_events, list) else action_journal
    if not isinstance(source_events, list):
        raise ValueError("public baseline replay requires events or action_journal")
    replayed: list[dict[str, Any]] = []
    unsupported = []
    for event in source_events:
        action = event.get("action")
        if action == "move" and int(event.get("steps", 0)) == 1:
            result = env.move(str(event["direction"]))
        elif action == "submit":
            result = env.submit(dict(event["answer"]))
        else:
            unsupported.append({
                "action": action, "code": event.get("code"),
            })
            continue
        replayed.append(result)

    replay_log = env.log(
        str(episode.get("run_id") or "v20-public-baseline-replay"),
        str(episode["actor"]), int(episode["seed"]), False,
    )
    summary_fields = (
        "termination_reason", "submit_only_reason", "turns", "state_version",
        "endurance_used_s", "endurance_used_fraction", "range_used_m",
        "range_used_fraction", "budget", "visited_nodes", "sample_count",
        "n_total_nodes", "path_length_m",
    )
    stored_ledger = episode.get("sample_ledger")
    checks = {
        "supported_single_edge_actions": not unsupported,
        "events_exact_or_hash_equivalent": (
            replayed == stored_events if isinstance(stored_events, list)
            else stable_hash(replayed) == episode.get("event_chain_sha256")
        ),
        "stored_event_hash_valid": (
            episode.get("event_chain_sha256") == stable_hash(stored_events)
            if isinstance(stored_events, list) else True
        ),
        "replayed_event_hash_match": stable_hash(replayed)
        == episode.get("event_chain_sha256"),
        "stored_sample_ledger_hash_valid": (
            episode.get("sample_ledger_sha256") == stable_hash(stored_ledger)
            if isinstance(stored_ledger, list) else True
        ),
        "sample_ledger_exact_or_hash_equivalent": (
            env.sample_ledger == stored_ledger if isinstance(stored_ledger, list)
            else stable_hash(env.sample_ledger) == episode.get("sample_ledger_sha256")
        ),
        "summary_exact": all(
            replay_log.get(field) == episode.get(field) for field in summary_fields
        ),
        "answer_exact": replay_log.get("answer") == episode.get("answer"),
        "information_tier_valid": episode.get("information_tier")
        == "public_observation_only",
    }
    return {
        "instance_id": episode["instance_id"],
        "actor": episode["actor"],
        "seed": episode["seed"],
        "verified": all(checks.values()),
        "checks": checks,
        "unsupported_events": unsupported,
    }


def targets(env: EnvironmentV20, rows: list[dict[str, Any]]) -> set[int]:
    """Map registered truth coordinates for the oracle upper bound only."""

    return {
        env.graph.index_by_xyz[(float(row["x"]), float(row["y"]), float(row["z"]))]
        for row in rows
    }


def shortest_time_path(
    env: EnvironmentV20, start: int, target_nodes: set[int],
) -> tuple[float, tuple[int, ...]]:
    """Use privileged full topology for the explicitly labeled upper bound."""

    queue = [(0.0, start)]
    costs = {start: 0.0}
    previous: dict[int, int] = {}
    found: int | None = None
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != costs[node]:
            continue
        if node in target_nodes:
            found = node
            break
        for edge in env.graph.neighbor_map[node].values():
            candidate = cost + env._edge_time(edge)
            if candidate < costs.get(edge.target, float("inf")):
                costs[edge.target] = candidate
                previous[edge.target] = node
                heapq.heappush(queue, (candidate, edge.target))
    if found is None:
        raise ValueError("target unreachable")
    path = [found]
    while path[-1] != start:
        path.append(previous[path[-1]])
    return costs[found], tuple(reversed(path))


def execute_path(env: EnvironmentV20, path: tuple[int, ...]) -> None:
    """Execute an oracle witness as public single-edge movement actions."""

    for source, target in zip(path, path[1:]):
        direction = next(
            name for name, edge in env.graph.neighbor_map[source].items()
            if edge.target == target
        )
        result = env.move(direction)
        if not result.get("ok"):
            raise ValueError(f"single-edge witness failed: {result}")
        if int(result.get("steps", 0)) != 1:
            raise ValueError("oracle witness attempted a non-single-edge move")


def reachable_upper_episode(
    store: DataStore, instance_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an oracle-only feasibility witness, never a benchmark actor."""

    truth = store.truth(instance_id)
    env = EnvironmentV20(store, instance_id, 0)
    start = env.current
    source_path: tuple[int, ...] = ()
    source_time = None
    if truth.get("source_nodes"):
        source_time, source_path = shortest_time_path(
            env, start, targets(env, truth["source_nodes"]),
        )
    if truth["detectable_source"]:
        _, plume_path = shortest_time_path(
            env, env.current, targets(env, truth["L2_plume_positions"]),
        )
        execute_path(env, plume_path)
        plume_position = dict(zip("xyz", env.graph.positions[env.current]))
        _, to_source = shortest_time_path(
            env, env.current, targets(env, truth["source_nodes"]),
        )
        execute_path(env, to_source)
        source_position = dict(zip("xyz", env.graph.positions[env.current]))
        answer = {
            "anomaly": "anomaly",
            "peak_position": plume_position,
            "mechanism": truth["L3_mechanism"],
            "source_position": source_position,
            "source_candidate_id": truth.get("L4_candidate_id"),
            "decision": "declare",
        }
    else:
        answer = {
            "anomaly": "no_anomaly", "peak_position": None,
            "mechanism": None, "source_position": None,
            "source_candidate_id": None, "decision": "cannot_determine",
        }
    env.submit(answer)
    episode = env.log("v20-reachable-upper", "reachable_upper", 0, False)
    reach = {
        "instance_id": instance_id,
        "waterbody": truth["waterbody"],
        "domain_id": truth["domain_id"],
        "detectable_source": truth["detectable_source"],
        "source_reachable": source_time is not None,
        "minimum_source_time_s": source_time,
        "minimum_source_budget_fraction": (
            source_time / env.budget.endurance_s if source_time is not None else None
        ),
        "minimum_source_edges": len(source_path) - 1 if source_path else None,
        "upper_used_fraction": episode["endurance_used_fraction"],
        "upper_turns": episode["turns"],
    }
    episode["upper_truth_access"] = {
        "available_to_benchmark_agents": False,
        "route_fields": ["L2_plume_positions", "source_nodes"],
        "answer_fields": ["L3_mechanism", "L4_candidate_id"],
    }
    episode["information_tier"] = REACHABLE_UPPER_INFORMATION_TIER
    return episode, reach


def _scores(store: DataStore, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for episode in episodes:
        truth = store.truth(episode["instance_id"])
        output.append({
            "instance_id": episode["instance_id"],
            "actor": episode["actor"],
            "seed": episode["seed"],
            **score_submission(
                episode.get("answer") or {}, truth,
                store.protocol["tolerances"][truth["waterbody"]],
            ),
        })
    return output


def _rate(rows: list[dict[str, Any]], level: int) -> tuple[int, int, float]:
    values = [row[f"L{level}_pass"] for row in rows if row[f"L{level}_pass"] is not None]
    return sum(bool(value) for value in values), len(values), (
        sum(bool(value) for value in values) / len(values) if values else 0.0
    )


def _paired_exact_discrimination(
    upper_rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
    level: int,
) -> dict[str, int | float]:
    """One-sided exact sign test on paired task-level correctness outcomes."""

    field = f"L{level}_pass"
    upper_by_instance = {
        str(row["instance_id"]): row[field]
        for row in upper_rows if row.get(field) is not None
    }
    upper_only = public_only = ties = 0
    for row in public_rows:
        public_value = row.get(field)
        upper_value = upper_by_instance.get(str(row["instance_id"]))
        if public_value is None or upper_value is None:
            continue
        if bool(upper_value) and not bool(public_value):
            upper_only += 1
        elif bool(public_value) and not bool(upper_value):
            public_only += 1
        else:
            ties += 1
    discordant = upper_only + public_only
    if discordant == 0 or upper_only <= public_only:
        probability = 1.0
    else:
        numerator = sum(
            math.comb(discordant, successes)
            for successes in range(upper_only, discordant + 1)
        )
        probability = numerator / (2 ** discordant)
    return {
        "upper_only": upper_only,
        "public_only": public_only,
        "ties": ties,
        "discordant": discordant,
        "one_sided_exact_p": probability,
    }


def _l1_balanced(store: DataStore, episodes: list[dict[str, Any]]) -> dict[str, float]:
    sensitivity_values = []
    specificity_values = []
    for episode in episodes:
        positive = bool(store.truth(episode["instance_id"])["detectable_source"])
        predicted = (episode.get("answer") or {}).get("anomaly")
        if positive:
            sensitivity_values.append(predicted == "anomaly")
        else:
            specificity_values.append(predicted == "no_anomaly")
    sensitivity = sum(sensitivity_values) / len(sensitivity_values)
    specificity = sum(specificity_values) / len(specificity_values)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
    }


def _source_plume_class(truth: dict[str, Any]) -> str | None:
    if not truth["detectable_source"]:
        return None
    plume = {
        (float(row["x"]), float(row["y"]), float(row["z"]))
        for row in truth["L2_plume_positions"]
    }
    source = {
        (float(row["x"]), float(row["y"]), float(row["z"]))
        for row in truth["source_nodes"]
    }
    return "I" if plume & source else "II"


def validate_v20_design(
    store: DataStore,
    output: str | Path,
    *,
    random_seeds: int = 3,
) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    upper, reach = [], []
    for task in store.tasks():
        episode, row = reachable_upper_episode(store, task["instance_id"])
        upper.append(episode)
        reach.append(row)
    random_rows = []
    public = _public_runtime_store(store)
    for seed in range(random_seeds):
        for task in public.tasks():
            random_rows.append(_public_baseline_episode(
                public, task["instance_id"], "random", seed, "v20-random-gate",
            ))
    upper_scores, random_scores = _scores(store, upper), _scores(store, random_rows)
    levels: dict[str, Any] = {}
    for level in range(1, 6):
        up_pass, up_n, up_rate = _rate(upper_scores, level)
        rnd_pass, rnd_n, rnd_rate = _rate(random_scores, level)
        if level == 1:
            upper_metric = _l1_balanced(store, upper)
            random_metric = _l1_balanced(store, random_rows)
            up_rate = upper_metric["balanced_accuracy"]
            rnd_rate = random_metric["balanced_accuracy"]
        else:
            upper_metric = {"passed": up_pass, "applicable": up_n, "rate": up_rate}
            random_metric = {"passed": rnd_pass, "applicable": rnd_n, "rate": rnd_rate}
        paired = _paired_exact_discrimination(upper_scores, random_scores, level)
        exact_p = float(paired["one_sided_exact_p"])
        levels[f"L{level}"] = {
            "upper": upper_metric,
            "random": random_metric,
            "upper_rate": up_rate,
            "random_rate": rnd_rate,
            "difference": up_rate - rnd_rate,
            "one_sided_exact_p": exact_p,
            "paired_discordance": paired,
            "upper_gate": up_rate >= 0.90,
            "discrimination_gate": up_rate > rnd_rate and exact_p < 0.05,
        }
    phase_consistent = all(
        task["active_phase"] == store.truth(task["instance_id"])["phase"] == "F1"
        for task in store.tasks()
    )
    report = {
        "protocol_id": "laketrace-v20-3d-single-edge-v1",
        "instances": len(store.tasks()),
        "upper_episodes": len(upper),
        "random_episodes": len(random_rows),
        "information_tiers": {
            "reachable_upper": REACHABLE_UPPER_INFORMATION_TIER,
            "random_gate": BASELINE_INFORMATION_TIER["random"],
        },
        "phase_consistency": phase_consistent,
        "source_reachability": {
            "positive_instances": sum(row["detectable_source"] for row in reach),
            "reachable": sum(bool(row["source_reachable"]) for row in reach if row["detectable_source"]),
            "max_minimum_source_budget_fraction": max(
                float(row["minimum_source_budget_fraction"])
                for row in reach if row["detectable_source"]
            ),
            "max_minimum_source_edges": max(
                int(row["minimum_source_edges"])
                for row in reach if row["detectable_source"]
            ),
        },
        "levels": levels,
    }
    report["passed"] = (
        phase_consistent
        and report["source_reachability"]["reachable"]
        == report["source_reachability"]["positive_instances"]
        and all(row["upper_gate"] and row["discrimination_gate"] for row in levels.values())
    )
    write_jsonl(root / "upper_bound_logs.jsonl", upper)
    write_jsonl(root / "random_gate_logs.jsonl", random_rows)
    write_csv(root / "source_reachability.csv", list(reach[0]), reach)
    (root / "design_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return report


def run_all_v20_baselines(
    store: DataStore,
    output: str | Path,
    *,
    seeds: int = 1,
) -> list[dict[str, Any]]:
    rows = []
    public = _public_runtime_store(store)
    for baseline_id in BASELINES:
        for seed in range(seeds):
            for task in public.tasks():
                rows.append(_public_baseline_episode(
                    public, task["instance_id"], baseline_id, seed,
                    "v20-public-baselines",
                ))
    write_jsonl(Path(output), rows)
    return rows


def detect_then_climb_prediction_check(
    store: DataStore, episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for episode in episodes:
        if episode["actor"] != "detect_then_climb":
            continue
        truth = store.truth(episode["instance_id"])
        if not truth["detectable_source"]:
            continue
        stratum = _source_plume_class(truth)
        score = score_submission(
            episode.get("answer") or {}, truth,
            store.protocol["tolerances"][truth["waterbody"]],
        )
        rows.append({"stratum": stratum, **score})
    summary: dict[str, Any] = {}
    for stratum in ("I", "II"):
        subset = [row for row in rows if row["stratum"] == stratum]
        summary[stratum] = {
            "n": len(subset),
            "L2_rate": sum(row["L2_pass"] is True for row in subset) / len(subset) if subset else None,
            "L4_rate": sum(row["L4_pass"] is True for row in subset) / len(subset) if subset else None,
        }
    summary["preregistered_predictions"] = {
        "II_L4_zero": summary["II"]["L4_rate"] == 0.0,
        "I_L2_greater_than_L4": (
            summary["I"]["L2_rate"] is not None
            and summary["I"]["L2_rate"] > summary["I"]["L4_rate"]
        ),
    }
    return summary
