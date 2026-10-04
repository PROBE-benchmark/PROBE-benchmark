# PROBE benchmark

PROBE evaluates model-driven water-environment investigations on 48 fixed tasks
(36 source-present tasks and 12 controls). The agent observes local measurements,
chooses survey actions and submits a report. The runner records scores and
replayable episode traces. Model-cohort results are not included.

## Install

Use Python 3.12+, macOS with `/usr/bin/sandbox-exec`, or Linux with `bubblewrap`
(`bwrap`) and permission to create its namespaces. Windows is not supported.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/check_environment.py --output ../probe_environment_check
```

Use a new output directory outside this repository for each run.

## Test without an API key

Run the two included baseline controllers through the survey and scoring interface:

```bash
.venv/bin/python scripts/run_three_waterbody_ade_policies.py \
  --instances TW-9F1AE2240610 --seeds 0 --output ../probe_baseline_check
```

Omit `--instances` for all 48 tasks. Baselines compare L2 and L4 on source-present
tasks; they do not evaluate L1 or L3. Omitting `--seeds` runs three registered seeds.

## Run a model

Start Ollama with `qwen3.5:4b`. Match the model digest in the local configuration
to the installed revision returned by Ollama's `/api/tags` endpoint.

```bash
.venv/bin/python scripts/run_investigation.py \
  --model configs/investigation/models/local_v22_survey_v6_gate3.json \
  --instances TW-9F1AE2240610 --output ../probe_local_check
```

The local check validates tool execution and replay, not scientific accuracy.
For a hosted model, copy the supplied hosted configuration and set `id`,
`base_url`, `model`, `api_key_env`, pricing and provider-specific thinking settings.
The adapter uses OpenAI-style chat completions; `provider: "bailian"` is its
internal name and does not require that vendor. Keep evaluation limits fixed
when comparing models. The supplied hosted configuration allows 20 calls and
8,192 output tokens per call.

```bash
.venv/bin/python scripts/run_investigation.py \
  --model configs/investigation/models/mimo_v26_flash_tokenplan_v22_survey_v6_budget20.json \
  --output ../probe_full48 --local-validation ../probe_local_check --key-stdin
```

Without `--instances`, all 48 tasks run. Add `--instances TW-...` for one task.
Enter the API key at the hidden prompt; do not commit credentials. Hosted runs
require a local validation record matching the current code, protocol and data.

## Results and replay

Model runs write `scores.csv`, `summary.json`, `validation.json` and `episodes/`.
To verify an existing model run without requesting the model again:

```bash
.venv/bin/python scripts/run_investigation.py \
  --model configs/investigation/models/mimo_v26_flash_tokenplan_v22_survey_v6_budget20.json \
  --output ../probe_full48 --verify-only
```

A completed replay and a correct scientific answer are separate outcomes.
Report the commit, data-manifest hash, model configuration and evaluation seed.

## Files

- `data/three_waterbody_candidate_v22/`: fixed observation fields, navigation,
  task definitions, calibration, evaluator truth and integrity manifest.
- `configs/`: model connections and execution/scoring protocols.
- `laketrace/`: agent, environment access, scoring, replay and baseline inference.
- `scripts/`: runtime checks and evaluation entry points.
- `tests/`: scoring and task-contract checks.

Positions use local east/north coordinates and positive-down depth in metres;
flow components are in metres per second. Water-quality values retain their
channel units and calibration. Task truth is withheld from the model-facing
interface; source geometry and thresholds are used by the evaluator.

The distributed fields are fixed, processed benchmark inputs and include
computed releases and condition transformations; they are not unmodified
snapshots of 48 observed pollution events. Evaluation reads these stored fields
rather than regenerating them. Runtime measurement noise remains part of the
registered protocol. The supplied tasks are a reproducibility set, not an unseen
held-out test. Data hashes and provenance identifiers remain in the manifest.
