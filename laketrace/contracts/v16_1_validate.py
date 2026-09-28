from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from laketrace.util import jsonl, stable_hash, write_csv


REQUIRED_COLUMNS = {
    "model_registry.csv": {
        "model_id", "model_snapshot", "prompt_id", "representation",
        "primary_condition", "n_registered_repeats",
    },
    "task_registry.csv": {
        "task_id", "event_cluster_id", "background_id", "anomaly_present",
        "source_class", "peak_to_source_h_m", "peak_to_source_v_m",
        "source_tol_h_m", "source_tol_v_m", "domain_node_count",
        "solvability_pass", "task_version_hash", "source_truth_kind",
        "source_node_count", "ground_truth_closable",
    },
    "episode_results.csv": {
        "episode_id", "model_id", "task_id", "run_id", "run_seed",
        "is_primary_condition", "analysis_eligible", "infrastructure_failure",
        "l1_pass", "l2_pass", "l3_pass", "l4_pass", "l5_pass",
        "valid_submission", "terminal_state", "endurance_used_s",
        "endurance_budget_s", "n_moves", "n_analyse", "n_action_steps",
        "n_repeated_action_steps", "nodes_visited_unique",
        "min_dist_to_source_h_m", "min_dist_to_source_v_m", "confidence_l1",
        "confidence_l2", "confidence_l3", "confidence_l4", "confidence_l5",
        "raw_output_path", "log_hash", "chain_of_thought_requested",
        "chain_of_thought_stored",
    },
    "task_build_audit.csv": {
        "candidate_event_id", "stage", "stage_order", "retained",
        "exclusion_reason", "audit_scope", "source_artifact",
    },
    "interface_equivalence.csv": {
        "test_id", "observation_equal", "budget_equal", "termination_equal", "passed",
    },
}
PARQUET_COLUMNS = {
    "task_regions.parquet": {
        "task_id", "region_type", "node_id", "region_rule_version",
    },
    "step_log.parquet": {
        "episode_id", "step_index", "sim_time_s", "node_id", "x_m", "y_m", "z_m",
        "action_type", "action_payload_json", "action_valid", "invalid_reason",
        "observation_json", "observation_hash", "state_hash",
        "budget_endurance_remaining_s", "budget_distance_remaining_m",
        "budget_turns_remaining", "visited_nodes_count", "state_record_count",
        "context_tokens_to_model", "hypothesis_source_x_m", "hypothesis_source_y_m",
        "hypothesis_source_z_m", "hypothesis_source_class", "hypothesis_confidence",
        "hypothesis_basis_observation_hash",
        "is_repeated_action", "termination_after_step",
        "evidence_content_hash", "visited_mask_hash", "measurement_history_count",
    },
    "provider_round_log.parquet": {
        "episode_id", "provider_round_index", "observation_json", "observation_hash",
        "messages_hash", "tool_schema_hash", "input_tokens", "output_tokens",
        "evidence_content_hash", "visited_mask_hash", "measurement_history_count",
    },
    "field_grid.parquet": {
        "task_id", "node_id", "x_m", "y_m", "z_m", "concentration",
        "flow_u_m_s", "flow_v_m_s", "flow_w_m_s", "valid_cell",
    },
}
FORMAL_MODELS = {"Gemini", "GPT", "Claude", "Kimi", "DeepSeek", "GLM"}
REPRESENTATION_RESULT_COLUMNS = {
    "readout_id", "model_id", "model_snapshot", "task_id", "run_seed",
    "representation", "snapshot_protocol_version", "snapshot_id",
    "canonical_evidence_hash", "visited_mask_hash", "measurement_history_count",
    "rendered_observation_hash", "valid_submission", "infrastructure_failure",
    "confidence_l2", "confidence_l4", "l2_applicable", "l2_pass",
    "l4_applicable", "l4_pass", "raw_output_path", "log_hash",
    "chain_of_thought_requested", "chain_of_thought_stored",
}
TERMINAL_STATES = {"submitted", "step_limit", "repetition_loop", "invalid_action", "infrastructure"}
PANELS = [
    "Fig1c", "Fig2a", "Fig2c", "Fig3c", "Fig3d", "Fig4a", "Fig4b", "Fig4c",
    "Fig5a", "Fig5b", "Fig5c", "Fig5d", "Fig6a", "Fig6b", "Fig6c", "Fig6d",
]


def _dependencies():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError(
            "v16.1 validation requires pandas and pyarrow; install requirements-v16.1.txt"
        ) from exc
    return pd


def _truthy(series: Any) -> Any:
    return series.astype(str).str.lower().isin({"1", "true", "1.0"})


def _nonempty(series: Any) -> Any:
    return series.notna() & (series.astype(str).str.strip() != "")


def _raw_hashes(path: Path) -> set[str]:
    try:
        return {stable_hash(row) for row in jsonl(path)}
    except (OSError, ValueError, json.JSONDecodeError):
        return set()


def validate_contract(root: Path, profile: str = "collection") -> dict[str, Any]:
    pd = _dependencies()
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    tables: dict[str, Any] = {}
    for name, columns in REQUIRED_COLUMNS.items():
        path = root / name
        if not path.is_file():
            errors.append(f"missing file: {name}")
            continue
        frame = pd.read_csv(path)
        tables[name] = frame
        missing = sorted(columns - set(frame.columns))
        if missing:
            errors.append(f"{name}: missing columns {missing}")
    for name, columns in PARQUET_COLUMNS.items():
        path = root / name
        if not path.is_file():
            errors.append(f"missing file: {name}")
            continue
        try:
            frame = pd.read_parquet(path, engine="pyarrow")
        except (ImportError, ValueError) as exc:
            errors.append(f"{name}: cannot read parquet: {exc}")
            continue
        tables[name] = frame
        missing = sorted(columns - set(frame.columns))
        if missing:
            errors.append(f"{name}: missing columns {missing}")
    thresholds_path = root / "preregistered_thresholds.json"
    figure_manifest_path = root / "figure_manifest.json"
    hashes_path = root / "SHA256SUMS.json"
    for path in (thresholds_path, figure_manifest_path, hashes_path):
        if not path.is_file():
            errors.append(f"missing file: {path.name}")
    if errors:
        report = _report(root, profile, errors, warnings, {}, {})
        return report

    models = tables["model_registry.csv"]
    tasks = tables["task_registry.csv"]
    episodes = tables["episode_results.csv"]
    audit = tables["task_build_audit.csv"]
    equivalence = tables["interface_equivalence.csv"]
    regions = tables["task_regions.parquet"]
    steps = tables["step_log.parquet"]
    rounds = tables["provider_round_log.parquet"]
    field = tables["field_grid.parquet"]
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    representation_results = None
    representation_path = root / "representation_ablation_results.csv"
    if representation_path.is_file():
        representation_results = pd.read_csv(representation_path)
        missing = sorted(REPRESENTATION_RESULT_COLUMNS - set(representation_results.columns))
        if missing:
            errors.append(f"representation_ablation_results.csv: missing columns {missing}")
            representation_results = None

    if len(tasks) != 84:
        errors.append(f"task_registry must contain 84 tasks, found {len(tasks)}")
    if tasks.task_id.duplicated().any():
        errors.append("task_registry task_id is not unique")
    if episodes.episode_id.duplicated().any():
        errors.append("episode_results episode_id is not unique")
    if not set(episodes.model_id).issubset(set(models.model_id)):
        errors.append("episode_results contains unregistered model_id")
    if not set(episodes.task_id).issubset(set(tasks.task_id)):
        errors.append("episode_results contains unknown task_id")
    if not set(steps.episode_id).issubset(set(episodes.episode_id)):
        errors.append("step_log contains unknown episode_id")
    if not set(rounds.episode_id).issubset(set(episodes.episode_id)):
        errors.append("provider_round_log contains unknown episode_id")
    for column in ["l1_pass", "l2_pass", "l3_pass", "l4_pass", "l5_pass"]:
        bad = ~episodes[column].dropna().isin([0, 1, True, False])
        if bad.any():
            errors.append(f"{column} contains values outside 0/1/NA")
    unknown = set(episodes.terminal_state.dropna()) - TERMINAL_STATES
    if unknown:
        errors.append(f"unknown terminal_state values: {sorted(unknown)}")
    if (episodes.endurance_budget_s.dropna() <= 0).any():
        errors.append("non-positive endurance_budget_s")
    if (tasks.source_tol_h_m.dropna() <= 0).any() or (tasks.source_tol_v_m.dropna() <= 0).any():
        errors.append("non-positive source tolerance")
    if not _truthy(equivalence.passed).all():
        errors.append("interface equivalence contains failed tests")
    if not (_truthy(equivalence.observation_equal) & _truthy(equivalence.budget_equal) & _truthy(equivalence.termination_equal)).all():
        errors.append("interface equivalence subchecks are not all equal")

    positive = tasks[_truthy(tasks.anomaly_present)]
    calc = ((positive.peak_to_source_h_m / positive.source_tol_h_m) ** 2 + (positive.peak_to_source_v_m / positive.source_tol_v_m) ** 2) ** 0.5
    if "peak_source_disp_norm" not in tasks.columns:
        errors.append("task_registry missing peak_source_disp_norm")
    elif ((calc - positive.peak_source_disp_norm).abs().dropna() > 1e-6).any():
        errors.append("peak_source_disp_norm is inconsistent with registered formula")
    if set(positive.source_truth_kind.dropna()) != {"source_nodes"}:
        errors.append("positive tasks do not consistently use canonical source_nodes truth")

    region_counts = regions.groupby(["task_id", "region_type"]).size()
    for row in positive.itertuples(index=False):
        for region_type in (
            "contaminated_region", "evidence_region", "peak_node",
            "peak_neighbourhood", "source_node", "source_neighbourhood",
        ):
            if (row.task_id, region_type) not in region_counts:
                errors.append(f"{row.task_id}: missing {region_type} rows")
        actual_source_nodes = int(region_counts.get((row.task_id, "source_node"), 0))
        if actual_source_nodes != int(row.source_node_count):
            errors.append(f"{row.task_id}: source_node_count mismatch")
    field_counts = field.groupby("task_id").size().to_dict()
    for row in tasks.itertuples(index=False):
        if int(field_counts.get(row.task_id, 0)) != int(row.domain_node_count):
            errors.append(f"{row.task_id}: field_grid row count differs from domain_node_count")

    if steps.duplicated(["episode_id", "step_index"]).any():
        errors.append("step_log episode_id + step_index is not unique")
    step_counts = steps.groupby("episode_id").size().to_dict()
    for episode_id, group in steps.groupby("episode_id"):
        indices = sorted(map(int, group.step_index))
        if indices != list(range(len(indices))):
            errors.append(f"{episode_id}: step_index is not contiguous from zero")
        last = group.sort_values("step_index").iloc[-1]
        if not _nonempty(group.sort_values("step_index").tail(1).termination_after_step).all():
            errors.append(f"{episode_id}: final step has no termination_after_step")
    for row in episodes.itertuples(index=False):
        count = int(step_counts.get(row.episode_id, 0))
        if count != int(row.n_action_steps or 0):
            errors.append(f"{row.episode_id}: n_action_steps={row.n_action_steps} but step_log has {count}")
    allowed_actions = {"move", "analyse", "submit", "invalid", "infeasible"}
    unknown_actions = set(steps.action_type.dropna()) - allowed_actions
    if unknown_actions:
        errors.append(f"step_log has unknown action_type values: {sorted(unknown_actions)}")
    for row in steps[_nonempty(steps.observation_json)].itertuples(index=False):
        try:
            observation = json.loads(row.observation_json)
        except (TypeError, json.JSONDecodeError):
            errors.append(f"{row.episode_id} step {row.step_index}: invalid observation_json")
            continue
        if stable_hash(observation) != row.observation_hash:
            errors.append(f"{row.episode_id} step {row.step_index}: observation_hash mismatch")
        forbidden = {"source_nodes", "L2_plume_positions", "exam_truth", "ground_truth"}
        if forbidden & set(_keys_recursive(observation)):
            errors.append(f"{row.episode_id} step {row.step_index}: truth-derived observation key detected")
    valid_analyse = steps[(steps.action_type == "analyse") & _truthy(steps.action_valid)]
    # Null class and an all-null coordinate triple explicitly mean "no current
    # source hypothesis" and are valid.  Confidence itself must always be
    # present, including when the hypothesis is null.
    if not valid_analyse.hypothesis_confidence.notna().all():
        errors.append("valid analyse rows have missing hypothesis_confidence")
    hypothesis_classes = set(valid_analyse.hypothesis_source_class.dropna())
    allowed_hypothesis_classes = {
        "external_point", "external_diffuse", "internal_release",
        "physical_resuspension",
    }
    if not hypothesis_classes <= allowed_hypothesis_classes:
        errors.append(f"unknown hypothesis_source_class values: {sorted(hypothesis_classes - allowed_hypothesis_classes)}")
    coordinates = valid_analyse[[
        "hypothesis_source_x_m", "hypothesis_source_y_m", "hypothesis_source_z_m",
    ]]
    partial_coordinates = coordinates.notna().sum(axis=1).isin([1, 2])
    if partial_coordinates.any():
        errors.append("analyse hypotheses contain partial coordinate triples")
    if not (valid_analyse.hypothesis_basis_observation_hash == valid_analyse.observation_hash).all():
        errors.append("analyse hypothesis basis hash does not match its provider-round observation")
    valid_submissions = episodes[_truthy(episodes.valid_submission)]
    confidence_columns = [f"confidence_l{level}" for level in range(1, 6)]
    if not valid_submissions.empty and not valid_submissions[confidence_columns].notna().all(axis=None):
        errors.append("valid submissions have missing level confidence")
    for column in confidence_columns:
        values = valid_submissions[column].dropna()
        if ((values < 0) | (values > 1)).any():
            errors.append(f"{column} outside 0..1")
    if _truthy(episodes.chain_of_thought_requested).any() or _truthy(episodes.chain_of_thought_stored).any():
        errors.append("chain-of-thought was requested or stored; only structured hypotheses are allowed")

    round_keys = rounds[["episode_id", "provider_round_index"]]
    if round_keys.duplicated().any():
        errors.append("provider_round_log episode_id + provider_round_index is not unique")
    for row in rounds[_nonempty(rounds.observation_json)].itertuples(index=False):
        try:
            observation = json.loads(row.observation_json)
        except (TypeError, json.JSONDecodeError):
            errors.append(f"{row.episode_id} round {row.provider_round_index}: invalid observation_json")
            continue
        if stable_hash(observation) != row.observation_hash:
            errors.append(f"{row.episode_id} round {row.provider_round_index}: observation_hash mismatch")

    raw_cache: dict[str, set[str]] = {}
    for row in episodes[_nonempty(episodes.raw_output_path)].itertuples(index=False):
        path = Path(row.raw_output_path)
        if not path.is_file():
            errors.append(f"raw output path does not exist: {path}")
            continue
        hashes = raw_cache.setdefault(str(path), _raw_hashes(path))
        if row.log_hash not in hashes:
            errors.append(f"{row.episode_id}: log_hash not found in raw JSONL")

    if audit.audit_scope.nunique() != 1 or audit.audit_scope.iloc[0] != "post_registration_release_pipeline":
        warnings.append("task_build_audit scope is not the evidenced post-registration release pipeline")
    if _truthy(audit.retained).all():
        warnings.append("task_build_audit contains no evidenced upstream exclusions; Fig2a must not imply pre-registration attrition")

    registered_keys: dict[tuple[str, str], int] = {}
    for row in models.itertuples(index=False):
        registered_keys[(row.model_id, row.representation)] = int(row.n_registered_repeats)
    task_ids = set(tasks.task_id)
    for (model_id, representation), repeats in registered_keys.items():
        subset = episodes[
            (episodes.model_id == model_id)
            & (episodes.representation == representation)
            & _truthy(episodes.analysis_eligible)
        ]
        keys = set(zip(subset.task_id, subset.run_seed))
        expected_count = len(task_ids) * repeats
        if len(keys) != expected_count:
            message = f"{model_id}/{representation}: {len(keys)} eligible task×repeat cells, expected {expected_count}"
            (errors if profile == "formal" else warnings).append(message)

    pair_rows = []
    round_join = rounds.merge(
        episodes[["episode_id", "model_id", "task_id", "run_seed", "representation"]],
        on="episode_id", how="left",
    )
    for key, group in round_join.groupby(["model_id", "task_id", "run_seed"]):
        representations = sorted(set(group.representation.dropna()))
        if len(representations) < 2:
            continue
        sequences = {}
        for representation, subset in group.groupby("representation"):
            subset = subset.sort_values("provider_round_index")
            sequences[representation] = [
                (row.evidence_content_hash, row.visited_mask_hash, row.measurement_history_count)
                for row in subset.itertuples(index=False)
            ]
        equivalent = len({stable_hash(value) for value in sequences.values()}) == 1
        pair_rows.append({
            "model_id": key[0], "task_id": key[1], "run_seed": key[2],
            "representations": "+".join(representations),
            "n_representations": len(representations),
            "n_provider_rounds_min": min(map(len, sequences.values())),
            "n_provider_rounds_max": max(map(len, sequences.values())),
            "evidence_history_equal": int(equivalent),
        })
    write_csv(
        root / "representation_content_equivalence.csv",
        ["model_id", "task_id", "run_seed", "representations", "n_representations", "n_provider_rounds_min", "n_provider_rounds_max", "evidence_history_equal"],
        pair_rows,
    )
    autonomous_exact_triplets = sum(
        row["representations"] == "flat_list+image_map+text_map" and row["evidence_history_equal"] == 1
        for row in pair_rows
    )

    snapshot_pair_rows = []
    exact_snapshot_pairs = 0
    if representation_results is not None and not representation_results.empty:
        if representation_results.readout_id.duplicated().any():
            errors.append("representation_ablation_results readout_id is not unique")
        if _truthy(representation_results.chain_of_thought_requested).any() or _truthy(representation_results.chain_of_thought_stored).any():
            errors.append("representation readout requested or stored chain-of-thought")
        allowed_representations = {"flat_list", "text_map"}
        unknown_representations = set(representation_results.representation.dropna()) - allowed_representations
        if unknown_representations:
            errors.append(f"representation readout has unknown representations: {sorted(unknown_representations)}")
        for key, group in representation_results.groupby(["model_id", "task_id", "run_seed"]):
            representations = set(group.representation.dropna())
            exact = (
                representations == allowed_representations
                and len(group) == 2
                and group.snapshot_id.nunique(dropna=False) == 1
                and group.canonical_evidence_hash.nunique(dropna=False) == 1
                and group.visited_mask_hash.nunique(dropna=False) == 1
                and group.measurement_history_count.nunique(dropna=False) == 1
                and _truthy(group.valid_submission).all()
                and not _truthy(group.infrastructure_failure).any()
            )
            snapshot_pair_rows.append({
                "model_id": key[0], "task_id": key[1], "run_seed": key[2],
                "representations": "+".join(sorted(representations)),
                "snapshot_id": group.snapshot_id.iloc[0] if len(group) else None,
                "canonical_evidence_equal": int(group.canonical_evidence_hash.nunique(dropna=False) == 1),
                "visited_mask_equal": int(group.visited_mask_hash.nunique(dropna=False) == 1),
                "measurement_history_equal": int(group.measurement_history_count.nunique(dropna=False) == 1),
                "exact_pair": int(exact),
            })
            exact_snapshot_pairs += int(exact)
        snapshot_raw_cache: dict[str, set[str]] = {}
        snapshot_raw_invalid: set[str] = set()
        for row in representation_results[_nonempty(representation_results.raw_output_path)].itertuples(index=False):
            path = Path(row.raw_output_path)
            if not path.is_file():
                errors.append(f"representation raw output path does not exist: {path}")
                continue
            cache_key = str(path.resolve())
            if cache_key not in snapshot_raw_cache:
                raw_log_hashes = set()
                try:
                    for item in jsonl(path):
                        claimed = item.get("log_hash")
                        payload = {key: value for key, value in item.items() if key != "log_hash"}
                        if claimed and stable_hash(payload) == claimed:
                            raw_log_hashes.add(claimed)
                        else:
                            snapshot_raw_invalid.add(cache_key)
                except (OSError, ValueError, json.JSONDecodeError):
                    snapshot_raw_invalid.add(cache_key)
                snapshot_raw_cache[cache_key] = raw_log_hashes
            raw_log_hashes = snapshot_raw_cache[cache_key]
            if row.log_hash not in raw_log_hashes:
                errors.append(f"{row.readout_id}: log_hash not found in representation raw JSONL")
        for path in sorted(snapshot_raw_invalid):
            errors.append(f"representation raw JSONL contains invalid self-hash rows: {path}")
        write_csv(
            root / "representation_snapshot_equivalence.csv",
            [
                "model_id", "task_id", "run_seed", "representations", "snapshot_id",
                "canonical_evidence_equal", "visited_mask_equal",
                "measurement_history_equal", "exact_pair",
            ],
            snapshot_pair_rows,
        )

    if profile == "formal":
        missing_models = sorted(FORMAL_MODELS - set(models.model_id))
        if missing_models:
            errors.append(f"formal registry missing models: {missing_models}")
        primary = models[_truthy(models.primary_condition)]
        if set(primary.model_id) != FORMAL_MODELS:
            errors.append("formal primary condition must contain exactly the six registered main models")
        registered_readout_repeats = int(
            (thresholds.get("representation_readout") or {}).get("registered_repeats", 1)
        )
        expected_pair_keys = {
            (model_id, task_id, run_seed)
            for model_id in FORMAL_MODELS
            for task_id in set(tasks.task_id)
            for run_seed in range(registered_readout_repeats)
        }
        actual_pair_keys = {
            (row["model_id"], row["task_id"], int(row["run_seed"]))
            for row in snapshot_pair_rows
            if row["exact_pair"] == 1
        }
        if representation_results is None:
            errors.append("representation_ablation_results.csv is missing")
        elif set(representation_results.model_id) != FORMAL_MODELS:
            errors.append(
                "frozen-evidence representation readout must contain exactly the six formal models"
            )
        elif actual_pair_keys != expected_pair_keys:
            errors.append(
                f"frozen-evidence representation readout has {len(actual_pair_keys)} exact registered cells, "
                f"expected {len(expected_pair_keys)} (six models × 84 tasks × registered repeats)"
            )

    panel_status = _panel_readiness(
        models, tasks, episodes, steps, rounds, regions, field, audit,
        equivalence, thresholds, exact_snapshot_pairs,
    )
    counts = {
        "models": len(models),
        "tasks": len(tasks),
        "episodes": len(episodes),
        "steps": len(steps),
        "provider_rounds": len(rounds),
        "task_region_rows": len(regions),
        "field_grid_rows": len(field),
        "panels_ready": sum(item["status"] == "READY" for item in panel_status),
        "panels_total": len(panel_status),
        "autonomous_exact_representation_triplets": autonomous_exact_triplets,
        "exact_frozen_evidence_representation_pairs": exact_snapshot_pairs,
    }
    return _report(root, profile, errors, warnings, counts, panel_status)


def _keys_recursive(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys_recursive(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys_recursive(item)


def _panel_readiness(models, tasks, episodes, steps, rounds, regions, field, audit, equivalence, thresholds, exact_snapshot_pairs):
    six_models = FORMAL_MODELS <= set(models.model_id)
    full_primary = False
    if six_models:
        primary = episodes[_truthy(episodes.is_primary_condition) & _truthy(episodes.analysis_eligible)]
        full_primary = all(len(set(primary.loc[primary.model_id == model, "task_id"])) == 84 for model in FORMAL_MODELS)
    step_complete = not steps.empty and steps.observation_hash.notna().all()
    confidence_complete = not episodes.empty and episodes[[f"confidence_l{x}" for x in range(1, 6)]].notna().all(axis=None)
    representation_ready = exact_snapshot_pairs > 0
    valid_submissions = episodes[_truthy(episodes.valid_submission)]
    l5_categories_ready = not valid_submissions.empty and _nonempty(valid_submissions.l5_category).all()
    conditions = {
        "Fig1c": (six_models and full_primary, "six-model primary Cartesian coverage"),
        "Fig2a": (not audit.empty, "post-registration task-build audit; upstream exclusions unavailable"),
        "Fig2c": (len(tasks) == 84, "84-task registry"),
        "Fig3c": (step_complete and not rounds.empty, "step/round state and context tokens"),
        "Fig3d": (_truthy(equivalence.passed).all(), "interface equivalence"),
        "Fig4a": (six_models and full_primary, "six-model level scores"),
        "Fig4b": (step_complete and not episodes.empty, "six behavioral metrics"),
        "Fig4c": (not episodes.empty, "terminal states and diagnostic inputs"),
        "Fig5a": (not episodes.empty and tasks.peak_source_disp_norm.notna().any(), "Dstar and L2/L4 outcomes"),
        "Fig5b": (not episodes.empty, "horizontal/vertical errors"),
        "Fig5c": (representation_ready, f"paired flat_list/text_map terminal readouts from frozen identical evidence; exact pairs={exact_snapshot_pairs}"),
        "Fig5d": (confidence_complete and l5_categories_ready, "episode-level close/abstain × L4 correctness, confidence, and descriptive risk-coverage data"),
        "Fig6a": (step_complete and not field.empty, "field plus replayable trajectories"),
        "Fig6b": (step_complete and not regions.empty, "failure regions and steps"),
        "Fig6c": (step_complete and not regions.empty, "stage regions and steps"),
        "Fig6d": (not episodes.empty and tasks[["stratified", "recirculation", "multi_peak", "complex_hydro"]].notna().all(axis=None), "physical flags plus failures"),
    }
    return [
        {"panel": panel, "status": "READY" if conditions[panel][0] else "BLOCKED", "evidence_or_blocker": conditions[panel][1]}
        for panel in PANELS
    ]


def _report(root: Path, profile: str, errors: list[str], warnings: list[str], counts: dict[str, Any], panels: Any) -> dict[str, Any]:
    panel_rows = panels if isinstance(panels, list) else []
    if panel_rows:
        write_csv(root / "panel_readiness.csv", ["panel", "status", "evidence_or_blocker"], panel_rows)
    report = {
        "contract": "LakeTrace-v16.1",
        "profile": profile,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "counts": counts,
        "panels": panel_rows,
    }
    (root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / f"validation_report_{profile}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "LakeTrace v16.1 data validation",
        f"profile={profile}",
        f"status={report['status']}",
        f"errors={len(errors)}",
        f"warnings={len(warnings)}",
    ]
    lines += [f"ERROR {value}" for value in errors]
    lines += [f"WARNING {value}" for value in warnings]
    (root / "validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / f"validation_report_{profile}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
