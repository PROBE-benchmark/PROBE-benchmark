# PROBE benchmark

PROBE is an interactive benchmark for water-environment source tracing. A model directs a virtual survey robot through a three-dimensional water-quality field, acquires local measurements, and submits judgments about source presence, affected water, release mechanism and source location. The runner records actions, observations, reports, scores and replay checks.

The package contains 48 fixed tasks across Sites A–C: 36 source-present tasks and 12 source-absent controls. The evaluator truth is available only to the scorer during an investigation. This repository contains the task environment and execution framework; model-cohort outcome tables are distributed separately.

Data provenance, field construction and usage scope are consolidated in [Data source and use statement](docs/Data_source_and_use_statement.docx), which accompanies this package. The distributed tasks are a reproducibility set; new leaderboard claims require a separate held-out evaluation.

## Install

Use Python 3.12 or newer. Scientific computation requires macOS with `/usr/bin/sandbox-exec`, or Linux with `bubblewrap` (`bwrap`) and permission to create its namespaces. The workbench does not support Windows. Install the Python dependencies in a virtual environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run an offline check

Check all task files, dependencies, content checksums and the isolated computation/replay interface:

```bash
.venv/bin/python scripts/check_environment.py --output ../probe_environment_check
```

The two transport-based controllers use the same observation and movement interface as model investigations. This one-case check requires no model provider or API key:

```bash
.venv/bin/python scripts/run_three_waterbody_ade_policies.py \
  --instances TW-9F1AE2240610 --seeds 0 --output ../probe_smoke
```

The command writes separate episode traces, scores and a `summary.json` containing replay status. Use a new output directory for each run.

Omit `--instances` to run both controllers on all 48 tasks. `--seeds 0` selects one repeat; omitting `--seeds` selects the registered three repeats. L2 and L4 scores are computed for source-present tasks; the controllers do not classify source presence or release mechanism.

## Run a model investigation

`scripts/run_investigation.py` accepts a model configuration, an optional list of task IDs and an output directory. Start Ollama locally and install `qwen3.5:4b`. The supplied configuration checks the installed model digest; if your model revision differs, verify its identity through Ollama's `/api/tags` endpoint and record that digest in your configuration. The three-call local configuration checks execution and replay, rather than model accuracy. A paid-provider run requires that local validation record; credentials are read with `--key-stdin` and are not stored in the repository.

```bash
.venv/bin/python scripts/run_investigation.py \
  --model configs/investigation/models/local_v22_survey_v6_gate3.json \
  --instances TW-9F1AE2240610 --output ../probe_local_check
```

Omitting `--instances` selects all 48 tasks. To verify saved trajectories without calling a provider, run the same command with the existing output directory and `--verify-only`.

## Package layout

- `data/three_waterbody_candidate_v22/tasks/instances.jsonl` defines the tasks; `truth/exam_truth.jsonl` is evaluator-only ground truth. Environment arrays and site background tables are in the same data directory.
- `configs/investigation/three_waterbody_v22_survey_v6.json` defines the shared survey protocol and call allowances.
- `laketrace/investigation/` implements the survey cockpit, provider loop, observation delivery and replay; `laketrace/scoring/` defines post-submission scoring.
- `laketrace/baselines/` and `scripts/run_three_waterbody_ade_policies.py` implement the offline transport controllers.
- `SHA256SUMS.txt` lists checksums for the committed package files.

The repository provides the benchmark and its task fields. It does not contain model-cohort outcome tables. A benchmark score should be reported with the task manifest, runner configuration, model identifier, complete episode traces and replay status.

## Connect another model

The hosted adapter sends OpenAI-style chat-completion requests. Copy the supplied hosted configuration, then set `id`, `base_url`, `model`, `api_key_env`, the provider-specific thinking format and the input/output price proxy. Keep the task paths and evaluation limits fixed for a controlled comparison. Configuration field `provider: "bailian"` selects this existing hosted adapter; it is an internal adapter name, not a requirement to use that vendor.

```bash
.venv/bin/python scripts/run_investigation.py \
  --model configs/investigation/models/mimo_v26_flash_tokenplan_v22_survey_v6_budget20.json \
  --instances TW-9F1AE2240610 --output ../probe_hosted_check \
  --local-validation ../probe_local_check --key-stdin
```

The example uses 20 calls and an 8,192-token output cap. Record any changed provider cap in the run configuration. API-key values and run outputs belong outside the repository. A local engineering check can finish without a scientifically correct report; inspect replay and delivery outcomes separately.

## Contract tests

```bash
.venv/bin/python -m unittest discover -s tests
```

Each result directory records `scores.csv`, `summary.json`, `validation.json` and `episodes/`. Recalculate/replay with `--verify-only`; it makes no provider requests. Cite the repository commit and the task-manifest hash when reporting results.
