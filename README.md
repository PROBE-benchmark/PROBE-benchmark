# PROBE research release candidate

This local candidate contains the interactive survey cockpit, 48 three-waterbody tasks and two observation-only source-inversion policies. It is a reproducibility candidate for author review and is not yet the published release underlying the manuscript's 31-model figures. The manuscript figure-source archive uses a different internal task/data cohort; reconcile those before publication.

## Setup

From this directory, create a Python 3.11+ environment and install `requirements.txt`. The data root is `data/three_waterbody_candidate_v22`. All outputs must use a new directory.

## No-model smoke test

`python scripts/run_three_waterbody_ade_policies.py --limit 1 --seeds 0 --output results/smoke_ade`

The script executes both survey and adaptive ADE policies through the same cockpit, records actions and observations, scores them after completion, and verifies exact replay. It does not call any provider or require an API key.

## Local-model engineering check

`python scripts/run_investigation.py --model configs/investigation/models/local_v22_survey_v6_gate3.json --instances TW-9F1AE2240610 --output results/local_gate`

The local model name and digest in its config must match an installed Ollama model. This checks execution and replay, not scientific correctness.

## Paid model

The included MiMo config is an example only. `run_investigation.py` requires a completed local engineering validation before paid execution. Enter the key through `--key-stdin`; do not commit keys or credentials.

## Data access

`tasks/instances.jsonl` is public task metadata; `truth/exam_truth.jsonl` is used only by the evaluator. Runtime observations are obtained via the cockpit, not by loading hidden fields into a model prompt. The 48 task fields are provided to make published results reproducible. Because the ground truth is present in this package, use a fresh held-out task set for future leaderboard claims.

## Verified control output

`examples/actual_v22_baseline/` records the full 48-task, three-seed run of both policies on this task release; all 288 episodes passed replay. These results belong to this release only and must not be substituted for figures from another task cohort.

## Scope and release status

The control policies produce L2/L4 spatial answers and do not implement L1/L3 classification. Current package licensing and data-rights confirmation are pending. Source hashes are in `SHA256SUMS.txt`.
