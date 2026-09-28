"""Offline outcome metrics; truth is opened only after agent execution/replay."""
from collections import Counter
from pathlib import Path

from laketrace.data.store import DataStore
from laketrace.scoring.scorer import score_submission
from laketrace.util import dump_json, write_csv


def evaluate(episodes, data_root, output, caveats=None):
    store = DataStore(data_root, evaluator=True)
    tracks = {e.get("task_track", "end_to_end") for e in episodes}
    levels = {tuple(e.get("capability_levels", ["L1", "L2", "L3", "L4", "L5"])) for e in episodes}
    protocols = {e.get("protocol_id") for e in episodes}
    modes = {e.get("evidence_mode", "active") for e in episodes}
    if len(tracks) > 1 or len(levels) > 1 or len(protocols) > 1 or len(modes) > 1:
        raise ValueError("different information tracks cannot share a capability summary")
    active_levels = set(next(iter(levels), ()))
    rows = []
    for episode in episodes:
        truth = store.truth(episode["instance_id"])
        finalized = episode["status"] == "completed"
        answer = episode["answer"] if finalized else {}
        answer = dict(answer)
        if truth.get('case_contract') == 'source_event_v2':
            answer['report_time_s'] = episode.get('environment_time_s', max((s['mission_time_s'] for s in episode['sample_ledger']), default=0.))
        score = score_submission(answer, truth, store.protocol["tolerances"][truth["waterbody"]])
        if "task_track" in episode:
            for level in ("L1", "L2", "L3", "L4"):
                if level not in active_levels:
                    score[f"{level}_pass"] = None
                    score[f"{level}_applicable"] = False
            if score["L5_category"] == "correct_true_negative":
                score["L5_category"] = "negative_no_declaration"
        report = episode["report"] if finalized else {}
        present_prob = report.get("source_presence_probability") if "L1" in active_levels else None
        source_prob = report.get("source_confidence")
        proposed = report.get("source_position") is not None
        true_plume = {(v["x"], v["y"], v["z"]) for v in truth["L2_plume_positions"]}
        measured = {s["node_index"]: s for s in episode["sample_ledger"]}
        provenance = report.get("plume_node_id") is None or report["plume_node_id"] in measured
        if episode.get('closure_version') and 'temporal_excess' in truth:
            from laketrace.scoring.source_event import validate_measurement
            provenance = validate_measurement(answer, report, episode['sample_ledger'])
        if episode.get('closure_version') and 'temporal_excess' not in truth and report.get('plume_node_id') is not None:
            node = measured.get(report['plume_node_id'])
            provenance = node is not None and answer.get('peak_position') == {a:node[a] for a in 'xyz'}
        if not provenance:
            score["L2_pass"] = False if score["L2_applicable"] else None
            if episode.get('closure_version'):
                score['L2_reported_position_in_plume'] = score['L2_pass']
        row = {"instance_id": episode["instance_id"], "status": episode["status"],
               "event_cluster": truth["event_cluster_id"], "background": truth["background_id"],
               "domain": truth["domain_id"], "detectable": truth["detectable_source"],
               "mechanism_truth": truth["L3_mechanism"], "mechanism_answer": answer.get("mechanism"),
               "source_present_answer": report.get("source_present"),
               "L2_answered": report.get("plume_node_id") is not None or report.get('plume_sample_id') is not None,
               "L3_answered": answer.get("mechanism") is not None,
               "L4_answered": proposed,
               **score, "L2_measured_provenance": provenance,
               "L2_route_contact": any((s["x"], s["y"], s["z"]) in true_plume for s in measured.values()),
               "declared": answer.get("decision") == "declare",
               "location_correct": score["L4_pass"] is True,
               "source_presence_probability": present_prob,
               "source_confidence": source_prob,
               "presence_brier": (present_prob - int(truth["detectable_source"])) ** 2 if present_prob is not None else None,
               "location_brier": (source_prob - int(score["L4_pass"] is True)) ** 2 if proposed and source_prob is not None else None,
               "legacy_location_brier_including_no_coordinate": (source_prob - int(score["L4_pass"] is True)) ** 2 if source_prob is not None else None,
               "visited_nodes": episode["visited_nodes"], "physical_edges": episode["physical_edges"],
               "movement_actions": episode["movement_actions"], "provider_calls": episode["provider_calls"],
               "input_tokens": episode["usage"]["input_tokens"], "output_tokens": episode["usage"]["output_tokens"],
               "cost_cny_proxy": episode["cost_cny_proxy"]}
        if episode.get('delivery_diagnostics'):
            row.update({'delivery_'+k:v for k,v in episode['delivery_diagnostics'].items()})
        if episode.get('budget_diagnostics'):
            row.update({'budget_'+k:v for k,v in episode['budget_diagnostics'].items()})
        if episode.get('closure_version') and 'temporal_excess' in truth:
            from laketrace.scoring.source_event import sample_hit
            row['L2_route_contact'] = any(sample_hit(truth, s) for s in episode['sample_ledger'])
        if episode.get("independent_levels"):
            row["L1_L4_joint_pass"] = finalized and truth["detectable_source"] and all(
                score[f"L{i}_pass"] is True for i in range(1, 5))
            # Same observed evidence; dynamic cases use the actual peak sample
            # with its timestamp, never a cross-time node average.
            # This is a spatial reference, not a source detector or mechanism model.
            samples = {}
            for sample in episode["sample_ledger"]:
                samples.setdefault(sample["node_index"], []).append(sample["channels"][episode["target_channel"]])
            peak_id = min(samples, key=lambda node: (-sum(samples[node])/len(samples[node]), node)) if samples else None
            peak = {a: measured[peak_id][a] for a in "xyz"} if peak_id is not None else None
            reference_answer={"peak_position": peak, "source_position": peak}
            if 'temporal_excess' in truth and episode['sample_ledger']:
                sample=max(episode['sample_ledger'],key=lambda s:s['channels'][episode['target_channel']])
                reference_answer.update(peak_position={a:sample[a] for a in 'xyz'},
                    source_position={a:sample[a] for a in 'xyz'},
                    plume_measurement={'node_id':sample['node_index'],'time_s':sample['mission_time_s'],**{a:sample[a] for a in 'xyz'}})
            if truth.get("case_contract") == "source_event_v2":
                reference_answer["report_time_s"] = answer["report_time_s"]
            reference = score_submission(reference_answer, truth,
                                         store.protocol["tolerances"][truth["waterbody"]])
            row.update(same_evidence_peak_L2=reference["L2_pass"], same_evidence_peak_L4=reference["L4_pass"],
                       contract_caveats=";".join((caveats or {}).get("cases", {}).get(episode["instance_id"], [])))
        if truth.get("case_contract") in {"source_event_v1", "source_event_v2"}:
            row.update(evidence_stratum=truth["evidence_stratum"], design_stratum=truth["design_stratum"],
                       identifiability_status=truth["identifiability_status"])
            row['L2_answered'] |= answer.get('plume_region_positions') is not None
        rows.append(row)
    def rate(key, selected):
        return sum(r[key] is True for r in selected) / len(selected) if selected else None
    positive = [r for r in rows if r["detectable"]]
    negative = [r for r in rows if not r["detectable"]]
    sensitivity, specificity = (rate("L1_pass", positive), rate("L1_pass", negative)) if "L1" in active_levels else (None, None)
    classes = sorted({r["mechanism_truth"] for r in positive if r["L3_applicable"]})
    recalls = {c: rate("L3_pass", [r for r in positive if r["mechanism_truth"] == c]) for c in classes}
    declared = [r for r in rows if r["declared"]]
    confidence_rows = [r for r in rows if r["L4_answered"] and r["source_confidence"] is not None]
    components = {}
    for level in ("L2", "L3", "L4"):
        eligible = [r for r in rows if r[f"{level}_applicable"]]
        answered = [r for r in eligible if r[f"{level}_answered"]]
        components[level] = {"applicable": len(eligible), "answered": len(answered),
            "unsubmitted": sum(r["status"] != "completed" for r in eligible),
            "completed_but_missing": sum(r["status"] == "completed" and not r[f"{level}_answered"] for r in eligible),
            "answered_correct": sum(r[f"{level}_pass"] is True for r in answered),
            "answered_wrong": sum(r[f"{level}_pass"] is not True for r in answered),
            "accuracy_when_answered": rate(f"{level}_pass", answered)}
    detection_groups = []
    for label, selected in [
        ("unsubmitted", [r for r in positive if r["status"] != "completed"]),
        ("present", [r for r in positive if r["status"] == "completed" and r["source_present_answer"] is True]),
        ("absent", [r for r in positive if r["status"] == "completed" and r["source_present_answer"] is False]),
        ("unknown", [r for r in positive if r["status"] == "completed" and r["source_present_answer"] is None])]:
        detection_groups.append({"presence_decision": label, "positive_cases": len(selected),
            **{f"{level}_answered": sum(r[f"{level}_answered"] for r in selected) for level in ("L2", "L3", "L4")},
            **{f"{level}_correct": sum(r[f"{level}_pass"] is True for r in selected) for level in ("L2", "L3", "L4")}})
    if "L1" not in active_levels:
        detection_groups = [{"presence_decision": "given_by_task_not_evaluated", "positive_cases": len(positive),
            **{f"{level}_answered": sum(r[f"{level}_answered"] for r in positive) for level in ("L2", "L3", "L4")},
            **{f"{level}_correct": sum(r[f"{level}_pass"] is True for r in positive) for level in ("L2", "L3", "L4")}}]
    curve = []
    for threshold in sorted({r["source_confidence"] for r in confidence_rows}, reverse=True):
        chosen = [r for r in confidence_rows if r["source_confidence"] >= threshold]
        curve.append({"threshold": threshold, "count": len(chosen), "coverage": len(chosen) / len(rows),
                      "error_risk": sum(not r["location_correct"] for r in chosen) / len(chosen)})
    summary = {
        "result_status": "development_evaluation_not_unseen_test", "metrics_schema_version": 2, "episodes": len(rows),
        "component_outcomes": components, "positive_outcomes_by_presence_decision": detection_groups,
        "conditional_metric_note": "Accuracy when answered uses model-selected subsets; it cannot replace all-case accuracy or prove an independent component capability.",
        "representation_normalizations": sum(e.get('stability_diagnostics', {}).get('representation_normalizations',0) for e in episodes),
        "execution_outcomes": dict(Counter(
            'submitted' if e['status']=='completed' else
            'protocol_or_runtime_failure' if e.get('termination') in {'report_delivery_failed','provider_contract_violation','request_deadline','unresolved_request','deadline_runtime_unsupported','deadline_timer_already_active','HTTP_400','HTTP_401','HTTP_403','model_identity_mismatch'} else
            'budget_exit' if e.get('termination') in {'cost_limit','provider_call_limit','context_token_limit','request_size_limit'} else
            'other_incomplete' for e in episodes)),
        "completed": sum(r["status"] == "completed" for r in rows), "positive": len(positive), "negative": len(negative),
        "L1_balanced_accuracy": (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None,
        "L1_sensitivity": sensitivity, "L1_specificity": specificity,
        "L2_accuracy_positive_only": rate("L2_pass", positive),
        "L2_accuracy_all_applicable": rate("L2_pass", [r for r in rows if r["L2_applicable"]]),
        "L2_accuracy_legacy_note": "L2_accuracy retains positive-only scope; use the explicitly named metrics.",
        "L2_accuracy": rate("L2_pass", positive), "L2_route_contact": rate("L2_route_contact", positive),
        "L3_accuracy": rate("L3_pass", [r for r in positive if r["L3_applicable"]]), "L3_per_class_recall": recalls,
        "L3_macro_recall": sum(recalls.values()) / len(recalls) if recalls else None,
        "L3_majority_baseline_accuracy": max(Counter(r["mechanism_truth"] for r in positive).values()) / len(positive) if positive and classes else None,
        "L4_accuracy": rate("L4_pass", positive),
        "declaration_coverage": len(declared) / len(rows) if rows else None,
        "declaration_error_risk": sum(not r["location_correct"] for r in declared) / len(declared) if declared else None,
        "declaration_counts": dict(Counter(r["L5_category"] for r in rows)),
        "presence_brier": sum(r["presence_brier"] for r in rows if r["presence_brier"] is not None) / sum(r["presence_brier"] is not None for r in rows) if any(r["presence_brier"] is not None for r in rows) else None,
        "location_brier": sum(r["location_brier"] for r in confidence_rows) / len(confidence_rows) if confidence_rows else None,
        "presence_probability_count": sum(r["source_presence_probability"] is not None for r in rows),
        "location_probability_count": len(confidence_rows),
        "coordinate_proposal_count": sum(r["L4_answered"] for r in rows),
        "coordinate_proposal_coverage": sum(r["L4_answered"] for r in rows) / len(rows) if rows else None,
        "location_calibration_scope": "Explicit coordinate proposals only; no-coordinate reports never earn calibration credit or enter the risk-coverage curve.",
        "always_abstain_baseline": {"coverage": 0, "risk": None},
        "event_clusters": len({r["event_cluster"] for r in rows}), "background_groups": len({r["background"] for r in rows}),
        "input_tokens": sum(r["input_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows),
        "provider_calls": sum(r["provider_calls"] for r in rows), "cost_cny_proxy": sum(r["cost_cny_proxy"] for r in rows),
        "limitations": ["Existing 84 tasks are development data, not independent unseen field events.",
                        "Known-target condition differs from historical hidden-target prompts; differences are not a causal harness effect.",
                        "No physical-validity or field-deployment claim follows from these scores.",
                        "Incomplete episodes remain in denominators; retained drafts are not counted as final submissions."]}
    if any("task_track" in e for e in episodes):
        summary.update(metrics_schema_version=3, task_track=next(iter(tracks)), capability_levels=sorted(active_levels),
                       numerical_fits=sum(e.get("numerical_fits", 0) for e in episodes),
                       tool_calls=sum(e.get("tool_calls", 0) for e in episodes), tool_errors=sum(e.get("tool_errors", 0) for e in episodes),
                       request_bytes=sum(e.get("request_bytes", 0) for e in episodes), tool_result_bytes=sum(e.get("tool_result_bytes", 0) for e in episodes),
                       detection_evaluation="not assessed: existence supplied" if "L1" not in active_levels else "registered-event diagnostic; instantaneous detectability not independently certified")
        if all(e.get("dataset_role") == "analytical_integration_fixture" for e in episodes):
            summary["limitations"] = ["Analytical point-pulse integration fixture with given transport and source existence; not real lakes or an unseen benchmark.",
                "Numerical fitting and LLM tool orchestration are distinct capabilities; compare with the same-evidence solver reference.",
                "No source-detection, mechanism-classification or field-deployment claim; incomplete reports stay in denominators."]
    if episodes and all(e.get("independent_levels") for e in episodes):
        summary.update(metrics_schema_version=4,
            L1_L4_joint_positive_success={"correct": sum(r["L1_L4_joint_pass"] for r in positive),
                "applicable": len(positive), "accuracy": rate("L1_L4_joint_pass", positive),
                "meaning": "All four answers correct in one finalized positive episode; failures/omissions remain in denominator. Report alongside negative specificity, never as a stand-alone overall score."},
            presence_decisions={label: dict(Counter("unsubmitted" if r["status"] != "completed" else
                "present" if r["source_present_answer"] is True else "absent" if r["source_present_answer"] is False else "unknown"
                for r in selected)) for label, selected in (("positive", positive), ("negative", negative))},
            same_evidence_peak_reference={"positive_cases": len(positive),
                "L2_accuracy": rate("same_evidence_peak_L2", positive), "L4_accuracy": rate("same_evidence_peak_L4", positive),
                "note": "Raw target maximum after averaging repeats; same actual measurements including incomplete runs. No L1/L3 or confidence claim. Different model routes imply different references."},
            evaluation_validity={"status": "legacy_development_not_independently_validated",
                "flagged_cases": sum(bool(r["contract_caveats"]) for r in rows),
                "caveats_registered": caveats is not None,
                "note": "All cases retained with original labels/tolerances. Known timing/geometry caveats appear in scores.csv; unflagged does not mean physically validated."})
    if any("compute_calls" in e for e in episodes):
        summary.update(metrics_schema_version=5, evidence_mode=next(iter(modes)),
            compute_calls=sum(e.get("compute_calls", 0) for e in episodes),
            termination_counts=dict(Counter(e.get('termination', 'unknown') for e in episodes)),
            path_length_m=sum(e.get('path_length_m', 0) for e in episodes),
            environment_time_s=sum(e.get('environment_time_s', 0) for e in episodes),
            model_wall_time_s=sum(e.get('wall_time_this_process_s', 0) for e in episodes),
            planned_episodes=None, batch_complete=None)
    if rows and all('evidence_stratum' in r for r in rows):
        summary['evaluation_validity'] = {'status':'controlled_injection_time_consistency_checked',
            'identifiability':'not_certified', 'field_validation':False}
        summary['detection_evaluation'] = 'event presence within declared window; weak sources retain positive labels'
        summary['limitations'] = ['Controlled closed-box transport with proxy multichannel interventions; not real-lake mechanisms.',
            'Grouped development population; numerical consistency does not establish blind identifiability or field capability.',
            'Weak positive events may have empty thresholded regions. Missing reports remain failures.']
        summary['spatial_regions'] = {}
        for key in ('L2_region', 'L4_region'):
            answered = [r[key] for r in rows if r[key]['answered']]
            summary['spatial_regions'][key] = {'answered':len(answered),'total':len(rows),
                'mean_iou_when_answered':sum(v['iou'] for v in answered)/len(answered) if answered else None}
        summary['evidence_strata'] = {key:{'n':len(selected), 'L1_accuracy':rate('L1_pass',selected),
            'joint_positive_accuracy':rate('L1_L4_joint_pass',[r for r in selected if r['detectable']])}
            for key in sorted({r['evidence_stratum'] for r in rows})
            for selected in [[r for r in rows if r['evidence_stratum']==key]]}
    if episodes and all(e.get('closure_version') for e in episodes):
        from .statistics import clustered_summary
        summary['metrics_schema_version'] = 6
        summary['uncertainty'] = clustered_summary(rows)
        summary['provider_errors'] = sum(e.get('provider_errors',0) for e in episodes)
        summary['context_estimator_exceeded'] = sum(e.get('context_estimator_exceeded',0) for e in episodes)
        if 'spatial_regions' in summary:
            for key in ('L2_region','L4_region'):
                eligible = rows if key=='L2_region' else [r for r in rows if r['L4_applicable']]
                summary['spatial_regions'][key].update(applicable=len(eligible),
                    mean_iou_all_applicable=sum((r[key]['iou'] or 0.) for r in eligible)/len(eligible) if eligible else None,
                    missing_policy='zero in all-applicable mean; conditional mean also retained',
                    mean_mass_coverage_all_applicable=sum((r[key].get('mass_coverage') or 0.) for r in eligible)/len(eligible) if eligible and key=='L4_region' else None)
        summary['same_evidence_peak_reference']['note'] = 'Static: node mean maximum. Dynamic: actual highest sample at its acquisition time. Same measurements; spatial reference only.'
    output = Path(output)
    dump_json(output / "summary.json", summary)
    write_csv(output / "scores.csv", list(rows[0]) if rows else [], rows)
    write_csv(output / "risk_coverage.csv", ["threshold", "count", "coverage", "error_risk"], curve)
    write_csv(output / "component_outcomes.csv", ["level", *next(iter(components.values()))],
              [{"level": level, **values} for level, values in components.items()])
    write_csv(output / "positive_detection_breakdown.csv", list(detection_groups[0]), detection_groups)
    grouped = []
    for key in ("domain", "event_cluster", "background"):
        for value in sorted({r[key] for r in rows}):
            group = [r for r in rows if r[key] == value]
            eligible = [r for r in group if r["detectable"]]
            grouped.append({"group_type": key, "group": value, "n": len(group), "positive_n": len(eligible),
                            "L2_accuracy": rate("L2_pass", [r for r in eligible if r["L2_applicable"]]), "L3_accuracy": rate("L3_pass", [r for r in eligible if r["L3_applicable"]]), "L4_accuracy": rate("L4_pass", [r for r in eligible if r["L4_applicable"]])})
    write_csv(output / "grouped_scores.csv", list(grouped[0]) if grouped else [], grouped)
    return summary
