from __future__ import annotations

import heapq
import json
import math
import statistics
from pathlib import Path
from typing import Any

from laketrace.agents.baselines import run_baseline
from laketrace.data.store import DataStore
from laketrace.env.core import Environment
from laketrace.scoring.scorer import score_submission
from laketrace.util import write_csv, write_jsonl


def _source_targets(env: Environment, truth: dict[str, Any]) -> set[int]:
    return {
        env.graph.index_by_xyz[(float(item["x"]), float(item["y"]), float(item["z"]))]
        for item in truth["source_nodes"]
    }


def _plume_targets(env: Environment, truth: dict[str, Any]) -> set[int]:
    return {
        env.graph.index_by_xyz[(float(item["x"]), float(item["y"]), float(item["z"]))]
        for item in truth["L2_plume_positions"]
    }


def _shortest_time(env: Environment, start: int, targets: set[int]) -> tuple[float, tuple[int, ...]]:
    queue = [(0.0, start)]
    cost = {start: 0.0}
    previous: dict[int, int] = {}
    found = None
    while queue:
        value, node = heapq.heappop(queue)
        if value != cost[node]:
            continue
        if node in targets:
            found = node
            break
        for edge in env.graph.neighbor_map[node].values():
            candidate = value + env._edge_time(edge)
            if candidate < cost.get(edge.target, float("inf")):
                cost[edge.target] = candidate
                previous[edge.target] = node
                heapq.heappush(queue, (candidate, edge.target))
    if found is None:
        raise ValueError("truth target unreachable")
    path = [found]
    while path[-1] != start:
        path.append(previous[path[-1]])
    return cost[found], tuple(reversed(path))


def _execute_path(env: Environment, path: tuple[int, ...]) -> None:
    directions = []
    for source, target in zip(path, path[1:]):
        direction = next(name for name, edge in env.graph.neighbor_map[source].items() if edge.target == target)
        directions.append(direction)
    cursor = 0
    while cursor < len(directions):
        direction = directions[cursor]
        steps = 1
        while cursor + steps < len(directions) and directions[cursor + steps] == direction and steps < env.max_steps:
            steps += 1
        result = env.move(direction, steps)
        if not result["ok"]:
            raise ValueError(f"upper witness path failed: {result}")
        cursor += steps


def _upper_episode(store: DataStore, instance_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    truth = store.truth(instance_id)
    env = Environment(store, instance_id, 0)
    source_fraction = None
    source_path = ()
    if truth["registered_physical_source"]:
        targets = _source_targets(env, truth)
        source_time, source_path = _shortest_time(env, env.current, targets)
        source_fraction = source_time / env.budget.endurance_s
    if truth["detectable_source"]:
        _, plume_path = _shortest_time(env, env.current, _plume_targets(env, truth))
        _execute_path(env, plume_path)
        plume_xyz = env.graph.positions[env.current]
        _, path_to_source = _shortest_time(env, env.current, _source_targets(env, truth))
        _execute_path(env, path_to_source)
        source = dict(zip("xyz", env.graph.positions[env.current]))
        answer = {"anomaly": "anomaly", "peak_position": dict(zip("xyz", plume_xyz)), "mechanism": truth["L3_mechanism"], "source_position": source, "decision": "declare"}
    else:
        answer = {"anomaly": "no_anomaly", "peak_position": None, "mechanism": None, "source_position": None, "decision": "cannot_determine"}
    env.submit(answer)
    episode = env.log("reachable-upper-v4", "reachable_upper", 0, False)
    reach = {
        "instance_id": instance_id, "waterbody": truth["waterbody"], "domain_id": truth["domain_id"],
        "registered_physical_source": truth["registered_physical_source"],
        "scored_detectable_source": truth["detectable_source"],
        "minimum_source_time_s": source_fraction * env.budget.endurance_s if source_fraction is not None else None,
        "minimum_source_budget_fraction": source_fraction,
        "upper_witness_used_fraction": episode["endurance_used_fraction"],
        "upper_witness_turns": episode["turns"],
        "start_to_plume_boundary_m": truth.get("start_to_plume_boundary_m"),
        "start_exceeds_registered_detection_threshold": truth.get("start_exceeds_registered_detection_threshold"),
        "start_primary_tracer_robust_score": truth.get("start_primary_tracer_robust_score"),
        "start_to_source_m": truth.get("start_to_source_m"),
        "upper_truth_access": {
            "solvability_route": {"fields": ["source_node_ids"], "used_at": "shortest-time reachability route", "purpose": "prove physical reachability within registered budgets"},
            "L2_answer": {"fields": ["L2_plume_positions"], "used_at": "shortest route and submitted peak_position", "purpose": "construct a reachable correct L2 witness"},
            "L3_answer": {"fields": ["L3_mechanism"], "used_at": "submitted mechanism", "purpose": "construct the inference ceiling"},
            "L4_answer": {"fields": ["source_node_ids", "source_nodes"], "used_at": "shortest route and submitted source_position", "purpose": "construct a reachable correct L4 witness"},
            "L5_answer": {"fields": ["detectable_source", "source_nodes"], "used_at": "submitted terminal decision", "purpose": "construct a correct stop/abstain witness"},
        },
    }
    episode["upper_truth_access"] = reach["upper_truth_access"]
    return episode, reach


def _pass_rate(rows: list[dict[str, Any]], level: int) -> tuple[int, int, float]:
    values = [row[f"L{level}_pass"] for row in rows if row[f"L{level}_pass"] is not None]
    return sum(bool(value) for value in values), len(values), sum(bool(value) for value in values) / len(values) if values else 0.0


def _l1_balanced_metrics(episodes: list[dict[str, Any]], store: DataStore) -> dict[str, float | int | str]:
    positive_predictions = []
    negative_predictions = []
    correct = 0
    for episode in episodes:
        truth_positive = bool(store.truth(episode["instance_id"])["detectable_source"])
        answer = (episode.get("answer") or {}).get("anomaly")
        predicted_positive = answer == "anomaly"
        predicted_negative = answer == "no_anomaly"
        (positive_predictions if truth_positive else negative_predictions).append(
            predicted_positive if truth_positive else predicted_negative
        )
        correct += predicted_positive if truth_positive else predicted_negative
    sensitivity = sum(positive_predictions) / len(positive_predictions) if positive_predictions else 0.0
    specificity = sum(negative_predictions) / len(negative_predictions) if negative_predictions else 0.0
    return {
        "metric": "balanced_accuracy",
        "passes": correct,
        "n": len(episodes),
        "raw_accuracy": correct / len(episodes) if episodes else 0.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "rate": (sensitivity + specificity) / 2.0,
    }


def _one_sided_paired_exact(b: int, c: int) -> float:
    """Exact P(upper-minus-random is at least the observed discordance)."""

    discordant = b + c
    if not discordant:
        return 1.0
    return sum(math.comb(discordant, value) for value in range(b, discordant + 1)) / (2 ** discordant)


def _l1_stratified_exact(upper: list[dict[str, Any]], random_rows: list[dict[str, Any]], store: DataStore) -> dict[str, Any]:
    upper_by_instance = {row["instance_id"]: row for row in upper}
    strata = {}
    for label, truth_positive in (("positive", True), ("negative", False)):
        b = c = pairs = 0
        for random_episode in random_rows:
            truth = store.truth(random_episode["instance_id"])
            if bool(truth["detectable_source"]) != truth_positive:
                continue
            upper_episode = upper_by_instance[random_episode["instance_id"]]
            expected = "anomaly" if truth_positive else "no_anomaly"
            upper_correct = (upper_episode.get("answer") or {}).get("anomaly") == expected
            random_correct = (random_episode.get("answer") or {}).get("anomaly") == expected
            pairs += 1
            b += upper_correct and not random_correct
            c += not upper_correct and random_correct
        strata[label] = {
            "paired_n": pairs,
            "upper_only_correct": b,
            "random_only_correct": c,
            "one_sided_exact_p": _one_sided_paired_exact(b, c),
        }
    # Two class-specific tests are possible. Bonferroni correction keeps the
    # discrimination gate conservative while respecting the balanced metric.
    combined = min(1.0, 2.0 * min(item["one_sided_exact_p"] for item in strata.values()))
    return {"method": "class-stratified_exact_with_bonferroni", "strata": strata, "p": combined}


def validate_design(store: DataStore, output: str | Path, random_seeds: int = 3) -> dict[str, Any]:
    target = Path(output); target.mkdir(parents=True, exist_ok=True)
    upper, reach = [], []
    for task in store.tasks():
        episode, row = _upper_episode(store, task["instance_id"])
        upper.append(episode); reach.append(row)
    random_rows = []
    for seed in range(random_seeds):
        for task in store.tasks():
            env = Environment(store, task["instance_id"], seed)
            run_baseline(env, "random", seed)
            random_rows.append(env.log("random-design-gate-v4", "random", seed, False))
    write_jsonl(target / "upper_bound_logs.jsonl", upper)
    write_jsonl(target / "random_gate_logs.jsonl", random_rows)
    write_csv(target / "source_reachability.csv", list(reach[0]), reach)
    truth_access = {
        "schema": "laketrace-reachable-upper-truth-access-v1",
        "actor": "reachable_upper",
        "available_to_benchmark_agents": False,
        "entries": reach[0]["upper_truth_access"],
    }
    (target / "upper_truth_access.json").write_text(
        json.dumps(truth_access, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    def scored(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for episode in episodes:
            truth = store.truth(episode["instance_id"])
            rows.append(score_submission(episode["answer"] or {}, truth, store.protocol["tolerances"][truth["waterbody"]]))
        return rows
    upper_scores, random_scores = scored(upper), scored(random_rows)
    levels = {}
    for level in range(1, 6):
        up_pass, up_n, up_rate = _pass_rate(upper_scores, level)
        random_pass, random_n, random_rate = _pass_rate(random_scores, level)
        if level == 1:
            upper_metric = _l1_balanced_metrics(upper, store)
            random_metric = _l1_balanced_metrics(random_rows, store)
            up_rate, random_rate = float(upper_metric["rate"]), float(random_metric["rate"])
            exact = _l1_stratified_exact(upper, random_rows, store)
            p_exact = float(exact["p"])
        else:
            upper_metric = {"metric": "pass_rate", "passes": up_pass, "n": up_n, "rate": up_rate}
            random_metric = {"metric": "pass_rate", "passes": random_pass, "n": random_n, "rate": random_rate}
            # Pair the upper result with every seeded random observation. The
            # upper passes all applicable rows, so exact one-sided McNemar
            # reduces to 2^-b where b is the upper-only discordance count.
            discordant = random_n - random_pass
            p_exact = 2.0 ** (-discordant) if discordant < 1075 else 0.0
            exact = {"method": "paired_exact_one_sided", "p": p_exact}
        levels[f"L{level}"] = {
            "upper": upper_metric,
            "random": random_metric,
            "difference": up_rate - random_rate, "exact_one_sided_p": p_exact,
            "exact_test": exact,
            "upper_gate_pass": up_rate >= 0.90,
            "discrimination_gate_pass": random_rate < up_rate and p_exact < 0.05,
        }
    phase_ok = all(task["active_phase"] == store.truth(task["instance_id"])["phase"] == "F1" for task in store.tasks())
    physical = [row["minimum_source_budget_fraction"] for row in reach if row["scored_detectable_source"]]
    physical.sort()
    reachability = {
        "reachable": len(physical), "expected": sum(1 for task in store.tasks() if store.truth(task["instance_id"])["detectable_source"]),
        "minimum": min(physical), "median": physical[len(physical)//2],
        "p90": physical[math.ceil(0.9 * len(physical)) - 1], "maximum": max(physical),
    }
    start_rows = [row for row in reach if row["scored_detectable_source"]]
    start_signal = {
        "n": len(start_rows),
        "above_registered_detection_threshold": sum(
            row["start_exceeds_registered_detection_threshold"] is True
            for row in start_rows
        ),
        "start_to_plume_boundary_m_min": min(float(row["start_to_plume_boundary_m"]) for row in start_rows),
        "start_to_plume_boundary_m_median": statistics.median(
            float(row["start_to_plume_boundary_m"]) for row in start_rows
        ),
        "start_to_plume_boundary_m_max": max(float(row["start_to_plume_boundary_m"]) for row in start_rows),
        "start_to_source_m_median": statistics.median(
            float(row["start_to_source_m"]) for row in start_rows
        ),
    }
    report = {
        "schema_version": "3.0", "random_seeds": random_seeds,
        "upper_definition": "budgeted shortest route to any registered F1 anomalous-footprint node plus exact registered inference answer",
        "random_baseline": "seeded random legal moves with the same environment and budget",
        "levels": levels,
        "phase_consistency": {"matches": len(store.tasks()) if phase_ok else 0, "n": len(store.tasks()), "rate": 1.0 if phase_ok else 0.0, "gate_pass": phase_ok},
        "physical_source_reachability": reachability,
        "start_signal_condition": start_signal,
        "upper_truth_access_manifest": "upper_truth_access.json",
    }
    report["all_gates_pass"] = phase_ok and all(item["upper_gate_pass"] and item["discrimination_gate_pass"] for item in levels.values()) and reachability["reachable"] == reachability["expected"]
    (target / "design_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
