#!/usr/bin/env python3
"""Run/replay the current development protocol. One local check precedes paid runs."""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from laketrace.data.store import DataStore
from laketrace.investigation.runner import load_settings, manifest, run_episode, verify_episode, read_trace
from laketrace.investigation.metrics import evaluate
from laketrace.util import dump_json, stable_hash

LOCAL_CASES = ["BF1-002D5D599A4D", "BF1-0187BBB3799B", "BF1-1B7012AB814D"]


def validate_local(directory, expected):
    """Check native execution/replay, not the small model's completion/accuracy ceiling."""
    receipt = json.loads((directory / "validation.json").read_text())
    local = json.loads((directory / "manifest.json").read_text())
    native_calls = sum(
        bool(row["response"].get("message", {}).get("tool_calls")) and
        row["response"].get("reported_model") == local["model_config"]["model"]
        for path in directory.glob("episodes/*/trace.jsonl") for row in read_trace(path)
        if row["kind"] == "response")
    # Launcher/gate/reporting edits do not change the validated agent interface.
    # All environment, loop, adapter, session and scoring code must still match.
    core = lambda m: {k:v for k,v in m["source_sha256"].items() if k.startswith("laketrace/")}
    if (receipt["manifest_sha256"] != stable_hash(local) or
            receipt["planned"] < 1 or receipt["finished"] != receipt["planned"] or
            receipt["replay_verified"] != receipt["planned"] or native_calls < 1 or
            local["model_config"]["provider"] != "ollama" or not local["model_config"]["thinking"] or
            local["protocol"] != expected["protocol"] or
            local["public_tasks_sha256"] != expected["public_tasks_sha256"] or
            (expected["protocol"].get("closure_version") and local["data_manifest_sha256"] != expected["data_manifest_sha256"]) or core(local) != core(expected)):
        raise ValueError("Local native execution and complete replay must cover the current engine/protocol")
    return {"engineering_passed": True, "local_reports_completed": receipt["completed_reports"],
            "local_planned": receipt["planned"], "replay_verified": receipt["replay_verified"], "native_tool_responses": native_calls,
            "local_manifest_sha256": stable_hash(local),
            "criterion": "all planned trajectories replay; real native tool responses; exact engine/protocol/public-data hashes; report completion is an outcome, not a gate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="configs/investigation/models/local_source_stable.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--local-cases", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--local-validation", type=Path)
    parser.add_argument("--key-stdin", action="store_true", help="Read the authorized credential without terminal echo; never save it")
    args = parser.parse_args()
    config, protocol, public = load_settings(args.model)
    public_tasks = public.get("tasks", public)
    store = DataStore(ROOT / config.get("data_root", "data/laketrace_v2"))
    instances = args.instances or ([i for i in LOCAL_CASES if i in public_tasks] if args.local_cases else sorted(public_tasks))
    if args.limit:
        instances = instances[:args.limit]
    if not instances or len(instances) != len(set(instances)) or not set(instances) <= set(public_tasks) or not set(instances) <= {t["instance_id"] for t in store.tasks()}:
        raise ValueError("invalid instance set")
    expected = manifest(config, protocol, public, instances, args.seed)
    out = Path(args.output).resolve()
    if out.exists():
        if not args.resume and not args.verify_only:
            raise ValueError("Existing output requires --resume or --verify-only")
        if json.loads((out / "manifest.json").read_text()) != expected:
            raise ValueError("Run definition changed; use a new output directory")
    else:
        if args.verify_only:
            raise ValueError("Missing output")
        out.mkdir(parents=True)
        dump_json(out / "manifest.json", expected)
        # Snapshot before any model call; later edits cannot replace provenance.
        for relative in expected['source_sha256']:
            source = ROOT / relative
            destination = out / 'code_snapshot' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    if not args.verify_only and config["provider"] == "bailian":
        if args.local_validation is None:
            raise ValueError("Paid execution requires the completed local validation artifact")
        engineering = validate_local(args.local_validation, expected)
        dump_json(out / "local_engineering_validation.json", engineering)
        if args.key_stdin:
            os.environ[config["api_key_env"]] = getpass.getpass("Authorized API key (not saved): ")
        if not os.environ.get(config["api_key_env"]):
            raise ValueError("Missing process credential")
    if not args.verify_only and config["provider"] == "ollama":
        with urllib.request.urlopen(config["base_url"] + "/api/tags", timeout=10) as response:
            models = json.load(response)["models"]
        matching = [m for m in models if m["name"] == config["model"] and m["digest"] == config["expected_digest"]]
        if len(matching) != 1:
            raise ValueError("Local model digest mismatch")
        dump_json(out / "local_model_identity.json", matching[0])
    if protocol.get('closure_version') and not args.verify_only:
        from laketrace.investigation.workbench import compute
        preflight = compute("print(float(np.mean([1.,2.,3.]))); assert len(df)==1", {'samples':[], 'target_nodes':[{'target_value':1.}]}, out/'runtime_preflight', 1, api=3)
        if not preflight['ok']:
            raise RuntimeError('Scientific runtime preflight failed before any model request: '+str(preflight.get('termination')))
    episodes = []
    passed = 0
    for index, iid in enumerate(instances):
        directory = out / "episodes" / iid
        completed = directory / "episode.json"
        if completed.exists():
            episode = json.loads(completed.read_text())
        elif args.verify_only:
            raise ValueError(f"Missing episode {iid}")
        else:
            print(json.dumps({"episode": index + 1, "total": len(instances), "instance": iid, "status": "starting"}), flush=True)
            episode = run_episode(store, iid, args.seed, config, protocol, public_tasks[iid], directory)
        verified = verify_episode(store, episode, protocol, public_tasks[iid], directory)
        if not verified:
            raise RuntimeError("Trajectory replay failed; remaining cases stopped")
        passed += verified
        episodes.append(episode)
        summary = evaluate(episodes, store.root, out, expected.get("evaluation_caveats"))
        if protocol.get('scientific_workbench'):
            summary.update(planned_episodes=len(instances), finished_episodes=len(episodes),
                           batch_complete=len(episodes)==len(instances))
            dump_json(out/'summary.json',summary)
        dump_json(out / "validation.json", {"passed": passed == len(instances) and all(e["status"] == "completed" for e in episodes),
                  "planned": len(instances), "finished": len(episodes), "replay_verified": passed,
                  "completed_reports": sum(e["status"] == "completed" for e in episodes), "manifest_sha256": stable_hash(expected)})
        print(json.dumps({"finished": len(episodes), "planned": len(instances), "status": episode["status"], "cost_cny": summary["cost_cny_proxy"]}), flush=True)
        from laketrace.investigation.stability import stops_run
        # A local Ollama transport failure has no external charge/usage ambiguity.
        # Keep the episode as an explicit failure and continue the batch; paid
        # provider transport uncertainty remains a run-level stop condition.
        if stops_run(episode['termination'], protocol) and config.get('provider') != 'ollama':
            raise RuntimeError('Run stopped: '+episode['termination']+'; remaining cases not executed')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if passed == len(instances) and all(e["status"] == "completed" for e in episodes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
