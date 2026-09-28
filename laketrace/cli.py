from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from laketrace import __version__
from laketrace.agents.baselines import BASELINES, run_baseline
from laketrace.agents.model import run_model
from laketrace.analysis.validation import validate_design
from laketrace.data.store import DataStore
from laketrace.env.core import Environment
from laketrace.figures.build import build_figures
from laketrace.scoring.geometry import horizontal_distance_to_geometry
from laketrace.scoring.scorer import score_submission
from laketrace.util import append_csv, append_jsonl, jsonl, load_config, read_csv, recover_jsonl, sha256_file, stable_hash, write_csv, write_jsonl


def _repository_root() -> Path:
    package_root = Path(__file__).resolve().parents[1]
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents, package_root]
    for candidate in candidates:
        if (candidate / "configs" / "FROZEN_PROTOCOL.json").is_file() and (candidate / "data" / "laketrace_v2").is_dir():
            return candidate
    return package_root


ROOT = _repository_root()
DEFAULT_DATA = ROOT / "data" / "laketrace_v2"


def _store(value: str | None, *, evaluator: bool = False) -> DataStore:
    return DataStore(Path(value).resolve() if value else DEFAULT_DATA, evaluator=evaluator)


def _model_path(value: str) -> Path:
    direct = Path(value)
    candidates = [direct, ROOT / "configs" / "models" / value, ROOT / "configs" / "models" / f"{value}.yaml"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"model configuration not found: {value}")


def _run_path(value: str | None, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if not value:
        raise ValueError("provide --logs or --run")
    direct = Path(value)
    candidates = [direct, ROOT / "results" / "runs" / value, ROOT / "results" / "runs" / f"{value}.jsonl"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ValueError(f"run log not found: {value}")


def _log_files(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_file():
        yield from jsonl(path)
    elif path.is_dir():
        for item in sorted(path.rglob("*.jsonl")):
            yield from jsonl(item)
    else:
        raise ValueError(f"log path not found: {path}")


def _valid_episode(episode: dict[str, Any]) -> bool:
    """Whether an episode is a completed, scoreable execution attempt."""

    return (
        episode.get("valid_for_scoring") is not False
        and episode.get("termination_reason") != "infra_failure"
    )


def _replay_episode(store: DataStore, episode: dict[str, Any]) -> bool:
    env = Environment(store, episode["instance_id"], int(episode["seed"]))
    expected = []
    for event in episode["events"]:
        action = event.get("action")
        if action == "move": result = env.move(event["direction"], int(event["steps"]))
        elif action == "analyze": result = env.analyze()
        elif action == "submit": result = env.submit(event["answer"])
        elif action == "infeasible" and event.get("requested_action") == "move":
            arguments = event.get("arguments") or {}
            result = env.move(arguments["direction"], int(arguments["steps"]))
        elif action == "infeasible" and event.get("requested_action") == "analyze":
            result = env.analyze()
        elif action == "invalid":
            result = env.record_invalid(event.get("code") or "E_SCHEMA", event.get("detail") or "", event.get("requested_action"))
        elif action == "infeasible": continue
        else: continue
        expected.append(result)
    comparable = [
        event for event in episode["events"]
        if event.get("action") in {"move", "analyze", "submit", "invalid"}
        or (event.get("action") == "infeasible" and event.get("requested_action") in {"move", "analyze"})
    ]
    return stable_hash(expected) == stable_hash(comparable)


def cmd_verify(args: argparse.Namespace) -> int:
    store = _store(args.data, evaluator=True)
    manifest = json.loads((store.root / "MANIFEST.json").read_text())
    failures = []
    for entry in manifest["files"]:
        path = store.root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            failures.append(entry["path"])
    phase = [row for row in store.tasks() if row["active_phase"] != store.truth(row["instance_id"])["phase"]]
    freeze = json.loads((ROOT / "configs" / "FROZEN_PROTOCOL.json").read_text())
    freeze_failures = [relative for relative, digest in freeze["files"].items() if not (ROOT / relative).is_file() or sha256_file(ROOT / relative) != digest]
    public_store = _store(args.data, evaluator=False)
    truth_isolated = False
    try:
        public_store.truth(public_store.tasks()[0]["instance_id"])
    except PermissionError:
        truth_isolated = True
    instance_id = public_store.tasks()[0]["instance_id"]
    env_left, env_right = Environment(public_store, instance_id, 20260826), Environment(public_store, instance_id, 20260826)
    direction = next(name for name, edge in env_left.graph.neighbors(env_left.current).items() if edge is not None)
    env_left.move(direction, 1); env_right.move(direction, 1)
    env_left.analyze(); env_right.analyze()
    environment_deterministic = stable_hash(env_left.events) == stable_hash(env_right.events)
    report = {
        "manifest_files": len(manifest["files"]), "hash_failures": failures,
        "freeze_hash_failures": freeze_failures, "instances": len(store.tasks()),
        "phase_mismatches": len(phase), "truth_isolated": truth_isolated,
        "environment_deterministic": environment_deterministic,
    }
    report["ok"] = not failures and not freeze_failures and not phase and len(store.tasks()) == 84 and truth_isolated and environment_deterministic
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def cmd_baseline(args: argparse.Namespace) -> int:
    store = _store(args.data, evaluator=False)
    ids = [item["instance_id"] for item in store.tasks()]
    if args.limit:
        ids = ids[:args.limit]
    names = list(BASELINES) if args.all or not args.baseline else args.baseline
    if not names:
        raise ValueError("select --all or at least one --baseline")
    run_id = args.run_id or "baseline-v4"
    output = Path(args.output or ROOT / "results" / "runs" / f"{run_id}.jsonl"); output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and output.is_file():
        recover_jsonl(output)
        rows = list(jsonl(output))
    else:
        rows = []
        write_jsonl(output, rows)
    completed = {(row["actor"], int(row["seed"]), row["instance_id"]) for row in rows}
    for name in names:
        for seed in range(args.seeds):
            for instance_id in ids:
                if (name, seed, instance_id) in completed:
                    continue
                env = Environment(store, instance_id, seed)
                run_baseline(env, name, seed)
                row = env.log(run_id, name, seed, False) | {"presentation_mode": "direct_state"}
                append_jsonl(output, row)
                rows.append(row)
    print(json.dumps({"run_id": run_id, "episodes": len(rows), "output": str(output)}, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    model_path = _model_path(args.model)
    config = load_config(model_path)
    config_hash = stable_hash(config)
    store = _store(args.data, evaluator=False)
    ids = [item["instance_id"] for item in store.tasks()]
    if args.instance:
        ids = args.instance
    if args.limit:
        ids = ids[:args.limit]
    output = Path(args.output or ROOT / "results" / "runs" / f"{config['id']}.jsonl"); output.parent.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"model-{config['id']}-{int(time.time())}"
    if args.resume and output.is_file():
        recover_jsonl(output)
        rows = list(jsonl(output))
        incompatible = [
            row for row in rows
            if row.get("actor") != config["id"]
            or row.get("run_id") != run_id
            or row.get("model_config_hash") != config_hash
        ]
        if incompatible:
            raise ValueError(
                "resume refused: existing rows do not match the requested "
                "run_id, actor, and model configuration hash"
            )
    else:
        rows = []
        write_jsonl(output, rows)
    # Infrastructure failures remain in the append-only audit log, but do not
    # block a later --resume from producing a valid replacement attempt.
    completed = {
        (row["instance_id"], int(row["seed"]))
        for row in rows
        if _valid_episode(row)
    }
    for seed in range(args.seeds):
        for instance_id in ids:
            if (instance_id, seed) in completed:
                continue
            env = Environment(store, instance_id, seed)
            try:
                metadata = run_model(env, config)
                valid = env.termination_reason != "infra_failure"
            except RuntimeError as exc:
                env.terminated, env.termination_reason = True, "infra_failure"
                metadata = {"provider_error": str(exc), "provider_calls": [], "token_usage": {"input_tokens": 0, "output_tokens": 0}, "model_identity_match": False}
                valid = False
            row = env.log(run_id, config["id"], seed, bool(config.get("science_eligible", False))) | metadata | {"model_config_hash": config_hash, "valid_for_scoring": valid}
            append_jsonl(output, row)
            rows.append(row)
    print(json.dumps({"run_id": run_id, "episodes": len(rows), "output": str(output), "science_eligible": bool(config.get("science_eligible", False))}, indent=2))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    store = _store(args.data, evaluator=True)
    replay_verdicts: dict[str, bool] = {}
    if args.replay_verdicts:
        replay_verdicts = {
            (row["instance_id"], row["actor"], int(row["seed"]), row["event_chain_sha256"]): row["replay_verified"] == "True"
            for row in read_csv(args.replay_verdicts)
        }
    rows = []
    logs = _run_path(args.run, args.logs)
    for episode in _log_files(logs):
        if not _valid_episode(episode):
            continue
        truth = store.truth(episode["instance_id"])
        tolerance = store.protocol["tolerances"][truth["waterbody"]]
        answer = episode.get("answer") or {}
        score = score_submission(answer, truth, tolerance)
        events = episode.get("events") or []
        invalid_calls = sum(event.get("action") == "invalid" for event in events)
        infeasible_calls = sum(event.get("action") == "infeasible" for event in events)
        valid_calls = sum(event.get("action") in {"move", "analyze", "submit", "infeasible"} for event in events)
        attempts = valid_calls + invalid_calls
        first_error = next((f"L{level}" for level in range(1, 6) if score[f"L{level}_pass"] is False), "none")
        token_usage = episode.get("token_usage") or {}
        peak = answer.get("peak_position") or {}
        source = answer.get("source_position") or {}
        access = store.access_points["access_points"][truth["domain_id"]]
        trajectory_positions = [
            (float(access["x"]), float(access["y"]), float(access["z"])),
            *[
                (float(reading["x"]), float(reading["y"]), float(reading["z"]))
                for event in events
                for reading in (event.get("readings") or [])
            ],
        ]
        trajectory_pairs = [
            (
                math.dist((position[0], position[1]), (float(node["x"]), float(node["y"]))),
                abs(position[2] - float(node["z"])),
            )
            for position in trajectory_positions
            for node in (truth.get("source_nodes") or [])
        ]
        if trajectory_pairs:
            trajectory_h, trajectory_v = min(
                trajectory_pairs, key=lambda item: (math.hypot(item[0], item[1]), item)
            )
            trajectory_distance = math.hypot(trajectory_h, trajectory_v)
            trajectory_entered_l4 = any(
                horizontal <= float(tolerance["horizontal_m"])
                and vertical <= float(tolerance["vertical_m"])
                for horizontal, vertical in trajectory_pairs
            )
        else:
            trajectory_h = trajectory_v = trajectory_distance = None
            trajectory_entered_l4 = None
        rows.append({
            "run_id": episode["run_id"], "actor": episode["actor"], "science_eligible": episode.get("science_eligible", False),
            "instance_id": episode["instance_id"], "seed": episode["seed"], "waterbody": truth["waterbody"], "domain_id": truth["domain_id"],
            "phase": episode["phase"], "difficulty_tier": truth.get("difficulty_tier"),
            "event_cluster_id": truth.get("event_cluster_id"), "background_id": truth.get("background_id"),
            "registered_physical_source": truth.get("registered_physical_source"), "detectable_source": truth.get("detectable_source"), "source_geometry_type": truth.get("source_geometry_type"),
            "source_in_plume": truth.get("source_in_plume"), "source_plume_class": truth.get("source_plume_class"),
            "candidate_ambiguous": truth.get("candidate_ambiguous"),
            "start_to_source_m": truth.get("start_to_source_m"), "peak_to_source_m": truth.get("peak_to_source_m"), "hydrodynamic_context": truth.get("hydrodynamic_context"),
            "termination_reason": episode.get("termination_reason"), "turns": episode.get("turns"),
            "endurance_used_s": episode.get("endurance_used_s"), "endurance_used_fraction": episode.get("endurance_used_fraction"),
            "path_length_m": episode.get("path_length_m", sum(float(event.get("path_length_m", 0.0)) for event in events)),
            "visited_nodes": episode.get("visited_nodes"), "n_total_nodes": episode.get("n_total_nodes"),
            "visited_fraction": (float(episode.get("visited_nodes", 0)) / float(episode.get("n_total_nodes", 1))) if episode.get("n_total_nodes") else None,
            "trajectory_source_closest_horizontal_m": trajectory_h,
            "trajectory_source_closest_vertical_m": trajectory_v,
            "trajectory_source_closest_distance_m": trajectory_distance,
            "trajectory_entered_L4_tolerance": trajectory_entered_l4,
            "invalid_calls": invalid_calls, "infeasible_calls": infeasible_calls,
            "action_parse_success_rate": valid_calls / attempts if attempts else 1.0,
            "replay_verified": replay_verdicts.get((episode["instance_id"], episode["actor"], int(episode["seed"]), episode.get("event_chain_sha256") or stable_hash(events))) if replay_verdicts else None,
            "event_chain_sha256": episode.get("event_chain_sha256") or stable_hash(events),
            "model_config_hash": episode.get("model_config_hash"), "model_identity_match": episode.get("model_identity_match"),
            "presentation_mode": episode.get("presentation_mode", "records"),
            "input_tokens": token_usage.get("input_tokens", 0), "output_tokens": token_usage.get("output_tokens", 0),
            "answer_anomaly": answer.get("anomaly"), "answer_peak_x": peak.get("x"), "answer_peak_y": peak.get("y"), "answer_peak_z": peak.get("z"),
            "answer_mechanism": answer.get("mechanism"), "answer_source_x": source.get("x"), "answer_source_y": source.get("y"), "answer_source_z": source.get("z"),
            "answer_candidate_id": answer.get("source_candidate_id"), "answer_decision": answer.get("decision"),
            "first_error_stage": first_error,
            **score,
        })
    if not rows:
        raise ValueError("no episodes found")
    fields = list(rows[0])
    output = Path(args.output or ROOT / "results" / "scores" / "benchmark_results.csv")
    if output.name != "benchmark_results.csv":
        raise ValueError("score output must be named benchmark_results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, fields, rows)
    print(json.dumps({"episodes": len(rows), "output": str(output)}, indent=2))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    results = Path(args.results or ROOT / "results" / "scores" / "benchmark_results.csv")
    rows = read_csv(results)
    grouped: dict[tuple[str, str], list[bool]] = {}
    for row in rows:
        for level in range(1, 6):
            value = row[f"L{level}_pass"]
            if value not in {"True", "False"}:
                continue
            grouped.setdefault((row["actor"], f"L{level}"), []).append(value == "True")
    output = []
    for (actor, level), values in sorted(grouped.items()):
        row = {
            "actor": actor, "level": level, "n": len(values), "passes": sum(values),
            "pass_rate": f"{sum(values)/len(values):.12f}",
            "sensitivity": "", "specificity": "", "balanced_accuracy": "",
        }
        if level == "L1":
            subset = [item for item in rows if item["actor"] == actor and item["L1_pass"] in {"True", "False"}]
            truth_positive = lambda item: item.get("detectable_source", item.get("registered_physical_source")) == "True"
            predicted_positive = lambda item: item.get("answer_anomaly") == "anomaly"
            predicted_negative = lambda item: item.get("answer_anomaly") == "no_anomaly"
            positives = [item for item in subset if truth_positive(item)]
            negatives = [item for item in subset if not truth_positive(item)]
            sensitivity = sum(predicted_positive(item) for item in positives) / len(positives) if positives else 0.0
            specificity = sum(predicted_negative(item) for item in negatives) / len(negatives) if negatives else 0.0
            row.update(
                sensitivity=f"{sensitivity:.12f}",
                specificity=f"{specificity:.12f}",
                balanced_accuracy=f"{(sensitivity + specificity) / 2.0:.12f}",
            )
        output.append(row)
    target = Path(args.output or ROOT / "results" / "scores" / "summary.csv"); target.parent.mkdir(parents=True, exist_ok=True)
    write_csv(target, ["actor", "level", "n", "passes", "pass_rate", "sensitivity", "specificity", "balanced_accuracy"], output)
    tier_rows = []
    for actor in sorted({row["actor"] for row in rows}):
        for tier in sorted({row["difficulty_tier"] for row in rows if row.get("difficulty_tier")}):
            subset = [row for row in rows if row["actor"] == actor and row.get("difficulty_tier") == tier]
            for level in range(1, 6):
                values = [row[f"L{level}_pass"] == "True" for row in subset if row[f"L{level}_pass"] in {"True", "False"}]
                if values:
                    tier_rows.append({"actor": actor, "difficulty_tier": tier, "level": f"L{level}", "n": len(values), "passes": sum(values), "pass_rate": f"{sum(values)/len(values):.12f}"})
    tier_path = target.with_name(target.stem + "_tier.csv")
    write_csv(tier_path, ["actor", "difficulty_tier", "level", "n", "passes", "pass_rate"], tier_rows)

    rng = random.Random(20260826)
    bootstrap_rows = []
    for (actor, level), _ in sorted(grouped.items()):
        subset = [row for row in rows if row["actor"] == actor and row[f"{level}_pass"] in {"True", "False"}]
        events = sorted({row.get("event_cluster_id") or row["instance_id"] for row in subset})
        backgrounds = sorted({row.get("background_id") or "unregistered" for row in subset})
        estimates = []
        for _replicate in range(1000):
            event_weights = Counter(rng.choice(events) for _ in events)
            background_weights = Counter(rng.choice(backgrounds) for _ in backgrounds)
            numerator = denominator = 0
            for row in subset:
                weight = event_weights[row.get("event_cluster_id") or row["instance_id"]] * background_weights[row.get("background_id") or "unregistered"]
                numerator += weight * (row[f"{level}_pass"] == "True")
                denominator += weight
            if denominator:
                estimates.append(numerator / denominator)
        estimates.sort()
        bootstrap_rows.append({
            "actor": actor, "level": level, "replicates": len(estimates),
            "event_clusters": len(events), "background_clusters": len(backgrounds),
            "mean": f"{statistics.mean(estimates):.12f}",
            "ci_low_2_5": f"{estimates[int(0.025 * (len(estimates)-1))]:.12f}",
            "ci_high_97_5": f"{estimates[int(0.975 * (len(estimates)-1))]:.12f}",
            "method": "two-way pigeonhole cluster bootstrap",
        })
    bootstrap_path = target.with_name(target.stem + "_cluster_bootstrap.csv")
    write_csv(bootstrap_path, list(bootstrap_rows[0]), bootstrap_rows)

    exact_rows = []
    reference = "greedy_gradient"
    by_actor = {actor: [row for row in rows if row["actor"] == actor] for actor in sorted({row["actor"] for row in rows})}
    reference_rows = {(row["instance_id"], row["seed"]): row for row in by_actor.get(reference, [])}
    for actor, actor_rows in by_actor.items():
        if actor == reference:
            continue
        actor_map = {(row["instance_id"], row["seed"]): row for row in actor_rows}
        for level in range(1, 6):
            b = c = pairs = 0
            for key in sorted(set(actor_map) & set(reference_rows)):
                left, right = actor_map[key][f"L{level}_pass"], reference_rows[key][f"L{level}_pass"]
                if left not in {"True", "False"} or right not in {"True", "False"}:
                    continue
                pairs += 1
                b += left == "True" and right == "False"
                c += left == "False" and right == "True"
            discordant = b + c
            tail = sum(math.comb(discordant, index) for index in range(0, min(b, c) + 1)) / (2 ** discordant) if discordant else 1.0
            exact_rows.append({"actor": actor, "reference": reference, "level": f"L{level}", "paired_n": pairs, "actor_only_pass": b, "reference_only_pass": c, "exact_two_sided_p": min(1.0, 2 * tail)})
    exact_path = target.with_name(target.stem + "_exact_tests.csv")
    write_csv(exact_path, list(exact_rows[0]), exact_rows)
    print(json.dumps({"summary_rows": len(output), "summary": str(target), "tier": str(tier_path), "cluster_bootstrap": str(bootstrap_path), "exact_tests": str(exact_path)}, indent=2))
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    results = Path(args.results or ROOT / "results" / "scores" / "benchmark_results.csv")
    output = Path(args.output or ROOT / "results" / "figures" / "latest")
    outputs = build_figures(results, output)
    print(json.dumps({"artifacts": [str(path) for path in outputs]}, indent=2))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    store = _store(args.data, evaluator=False)
    logs = _run_path(args.run, args.logs)
    output = Path(args.output) if args.output else Path(str(logs) + ".replay.csv")
    fields = ["event_chain_sha256", "instance_id", "actor", "seed", "replay_verified"]
    if args.resume and output.is_file():
        completed_rows = read_csv(output)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(output, fields, [])
        completed_rows = []
    completed = {
        (row["instance_id"], row["actor"], int(row["seed"]), row["event_chain_sha256"])
        for row in completed_rows
    }
    total = len(completed_rows)
    failures = sum(row["replay_verified"] != "True" for row in completed_rows)
    for episode in _log_files(logs):
        if not _valid_episode(episode):
            continue
        event_hash = episode.get("event_chain_sha256") or stable_hash(episode.get("events") or [])
        key = (episode["instance_id"], episode["actor"], int(episode["seed"]), event_hash)
        if key in completed:
            continue
        total += 1
        verified = _replay_episode(store, episode)
        verdict = {"event_chain_sha256": event_hash, "instance_id": episode["instance_id"], "actor": episode["actor"], "seed": episode["seed"], "replay_verified": verified}
        append_csv(output, fields, verdict)
        if not verified:
            failures += 1
    report = {"episodes": total, "replay_failures": failures, "verdicts": str(output), "ok": total > 0 and failures == 0}
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def cmd_validate(args: argparse.Namespace) -> int:
    report = validate_design(_store(args.data, evaluator=True), args.output, args.random_seeds)
    print(json.dumps(report, indent=2))
    return 0 if report["all_gates_pass"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laketrace")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify"); verify.add_argument("--data"); verify.set_defaults(func=cmd_verify)
    baseline = sub.add_parser("baseline"); baseline.add_argument("--data"); baseline.add_argument("--all", action="store_true"); baseline.add_argument("--baseline", action="append", choices=BASELINES); baseline.add_argument("--seeds", type=int, default=1); baseline.add_argument("--limit", type=int); baseline.add_argument("--run-id"); baseline.add_argument("--resume", action="store_true"); baseline.add_argument("--output"); baseline.set_defaults(func=cmd_baseline)
    run = sub.add_parser("run"); run.add_argument("--data"); run.add_argument("--model", required=True); run.add_argument("--instance", action="append"); run.add_argument("--seeds", type=int, default=1); run.add_argument("--limit", type=int); run.add_argument("--run-id"); run.add_argument("--resume", action="store_true"); run.add_argument("--output"); run.set_defaults(func=cmd_run)
    score = sub.add_parser("score"); score.add_argument("--data"); score.add_argument("--logs"); score.add_argument("--run"); score.add_argument("--replay-verdicts"); score.add_argument("--output"); score.set_defaults(func=cmd_score)
    analyze = sub.add_parser("analyze"); analyze.add_argument("--results"); analyze.add_argument("--output"); analyze.set_defaults(func=cmd_analyze)
    figures = sub.add_parser("figures"); figures.add_argument("--results"); figures.add_argument("--output"); figures.set_defaults(func=cmd_figures)
    replay = sub.add_parser("replay"); replay.add_argument("--data"); replay.add_argument("--logs"); replay.add_argument("--run"); replay.add_argument("--output"); replay.add_argument("--resume", action="store_true"); replay.set_defaults(func=cmd_replay)
    validate = sub.add_parser("validate"); validate.add_argument("--data"); validate.add_argument("--random-seeds", type=int, default=3); validate.add_argument("--output", required=True); validate.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return int(args.func(args))
    except (ValueError, RuntimeError, KeyError) as exc:
        print(f"laketrace: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
