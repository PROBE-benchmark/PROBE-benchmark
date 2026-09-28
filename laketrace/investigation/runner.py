"""One resumable native-tool loop; protocol snapshots bind each experiment."""
from __future__ import annotations

import copy
import json
import time
import importlib.metadata
import platform
import sys
from pathlib import Path

from laketrace.data.store import DataStore
from laketrace.util import append_jsonl, dump_json, jsonl, sha256_file, stable_hash
from .provider import complete, payload, request_upper_cost, set_thinking, usage_cost
from .session import Session, system_prompt

ROOT = Path(__file__).resolve().parents[2]


def load_settings(model_path):
    config = json.loads(Path(model_path).read_text())
    protocol = json.loads((ROOT / config.get("protocol_path", "configs/investigation/protocol.json")).read_text())
    public = json.loads((ROOT / config.get("public_tasks_path", "configs/investigation/public_tasks.json")).read_text())
    if config["provider"] not in {"ollama", "bailian"} or type(config["thinking"]) is not bool:
        raise ValueError("unsupported provider or missing explicit thinking setting")
    if config["provider"] == "bailian" and not config["base_url"].startswith("https://"):
        raise ValueError("remote credentials require HTTPS")
    if config.get("thinking_wire_format") not in (None, "xiaomi"):
        raise ValueError("unsupported thinking wire format")
    for key in ("max_output_tokens", "max_provider_calls", "max_request_bytes", "history_turns"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"invalid {key}")
    if not 0 < config["max_provider_calls"] <= protocol["max_provider_calls"]:
        raise ValueError("model compute allowance exceeds protocol")
    if protocol.get('budget_policy_version'):
        from .context import input_allowance
        if (protocol.get('physical_budget_enabled') or
                type(config.get('max_input_tokens')) is not int or config['max_input_tokens'] < 1 or
                input_allowance({}, config) < config['max_input_tokens'] or
                config['max_provider_calls'] <= protocol['budgeted_finalization']['reserve_calls']):
            raise ValueError('invalid budget policy: input/output capacity, final reserve or endurance')
    if protocol.get('continuity_version'):
        if type(protocol.get('finalization_thinking')) is not bool:
            raise ValueError('continuity requires an explicit finalization thinking policy')
        if config.get('final_tool_choice','auto') not in ('auto','named'):
            raise ValueError('unsupported final tool choice')
        if protocol['finalization_thinking'] and config.get('final_tool_choice')=='named':
            raise ValueError('named final tool requires registered non-thinking finalization')
    if protocol.get('recovery_version') and (type(protocol.get('truncation_recovery_after')) is not int or protocol['truncation_recovery_after'] < 1):
        raise ValueError('recovery requires a positive consecutive-truncation threshold')
    if protocol.get('stability_version'):
        if config.get('runtime_contract_version') != 1 or config.get('output_budget_semantics') not in ('total', 'separate'):
            raise ValueError('stable runtime requires registered provider budget semantics')
        if config.get('request_deadline_s', 0) <= 0:
            raise ValueError('stable runtime requires a whole-request deadline')
        if config['thinking'] and config['output_budget_semantics'] == 'separate' and config.get('thinking_budget', 0) <= 0:
            raise ValueError('separate thinking requires an explicit thinking_budget')
    if config.get("unlimited_provider_calls") and (config["provider"] != "ollama" or not protocol.get("unlimited_provider_calls")):
        raise ValueError("Unlimited calls require an explicit local-only protocol")
    return config, protocol, public


def manifest(config, protocol, public, instances, seed):
    paths = list((ROOT / "laketrace/investigation").glob("*.py"))
    paths += list((ROOT / "laketrace/env").glob("*.py"))
    paths += list((ROOT / "laketrace/scoring").glob("*.py"))
    paths += [ROOT / p for p in ["laketrace/data/store.py", "laketrace/util.py", "scripts/run_investigation.py"]]
    data_root = ROOT / config.get("data_root", "data/laketrace_v2")
    result = {"protocol": protocol, "model_config": config, "public_tasks_sha256": stable_hash(public),
            "data_manifest_sha256": sha256_file(data_root / "MANIFEST.json"),
            "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)},
            "instances": instances, "seed": seed, "prompt_sha256": stable_hash(system_prompt(protocol)),
            "result_status": "development_evaluation", "unseen_test": False}
    if protocol.get('scientific_workbench'):
        result['runtime'] = {'python':sys.version,'platform':platform.platform(),
            'packages':{name:importlib.metadata.version(name) for name in ('numpy','scipy','matplotlib','pandas', *(['jsonschema'] if protocol.get('closure_version') else []))},
            'computation_isolation':'macOS Seatbelt' if sys.platform=='darwin' else 'Linux bubblewrap', 'evidence_mode':protocol.get('evidence_mode','active')}
        registered = json.loads((data_root/'MANIFEST.json').read_text())
        if isinstance(registered.get('files'), dict):
            for relative, digest in registered['files'].items():
                path = (data_root / relative).resolve()
                if not path.is_relative_to(data_root.resolve()) or sha256_file(path) != digest:
                    raise ValueError('dataset content differs from its manifest')
    if protocol.get("evaluation_caveats_path"):
        caveats = json.loads((ROOT / protocol["evaluation_caveats_path"]).read_text())
        if caveats["data_manifest_sha256"] != result["data_manifest_sha256"]:
            raise ValueError("evaluation caveats refer to a different dataset")
        result["evaluation_caveats"] = caveats  # Evaluator-only; never in Session context.
    if protocol.get('closure_version'):
        from .lineage import validate_formal
        result['split_validation'] = validate_formal(config, data_root)
        result['formal_evaluation'] = bool(config.get('formal_evaluation'))
        if result['split_validation']:
            result.update(result_status='registered_heldout_evaluation',unseen_test=True)
    return result


def commit(path, rows, kind, **fields):
    row = {"index": len(rows), "previous": rows[-1]["hash"] if rows else None, "kind": kind, **fields}
    row["hash"] = stable_hash(row)
    append_jsonl(path, row)
    rows.append(row)


def read_trace(path):
    rows = list(jsonl(path)) if path.exists() else []
    previous = None
    for i, row in enumerate(rows):
        unsigned = {k: v for k, v in row.items() if k != "hash"}
        if row["index"] != i or row["previous"] != previous or stable_hash(unsigned) != row["hash"]:
            raise ValueError("trace integrity mismatch")
        previous = row["hash"]
    return rows


def apply_response(session, response):
    group = _apply_response(session, response)
    if session.protocol.get('continuity_version'):
        from .continuity import record
        record(session, group, response)
    return group


def _apply_response(session, response):
    if response.get('error') and session.protocol.get('provider_error_feedback'):
        return [{'role':'user','content':'The preceding provider request failed (' + str(response['error']) + '). No tool from that request executed. Existing measurements and drafts remain available. Continue with a concise, valid tool call; do not assume the failed action happened.'}]
    message = copy.deepcopy(response.get("message", {}))
    calls = message.get("tool_calls") or []
    results = []
    # Truncated actions never execute. A later valid report may still complete.
    if response.get("finish_reason") == "length":
        return [{"role": "user", "content": "Your response reached the output limit; no tool action executed. Use a concise tool call or save the current report."}]
    for call in calls:
        function = call["function"]
        try:
            args = json.loads(function.get("arguments") or "{}")
            result = session.execute(function["name"], args)
        except (ValueError, TypeError, KeyError):
            result = {"ok": False, "error": "invalid tool arguments; use the declared schema"}
        results.append({"role": "tool", "tool_call_id": call["id"], "tool_name": function.get("name"),
                        "content": json.dumps(result, ensure_ascii=False, separators=(",", ":"))})
    if not calls:
        results.append({"role": "user", "content": "Continue with the available investigation tools or finalize your report. Your preceding text is retained."})
    return [message, *results]


def prepare_request(session, config, protocol, context, groups, remaining, cost):
    """Reserve a real final request, keeping read-only analysis until finalization.

    Legacy protocols retain complete assistant/tool groups and thinking settings.
    Continuity protocols pin receipts and use the registered finalization policy.
    Never fills an answer or increases the configured budget.
    """
    local = config["provider"] == "ollama"
    ceiling = config["hard_cost_ceiling_cny"]
    reserve = config.get("submit_reserve_cny", .5)
    final_calls = protocol["budgeted_finalization"]["reserve_calls"]
    session.review_only |= remaining <= protocol["final_report_reserve_calls"] or (
        not local and cost >= ceiling - reserve)
    session.report_only |= remaining <= final_calls

    compact_memory = False
    presentation_level = 0
    def build(retained):
        state = session.state(remaining, round(cost, 6))
        if protocol.get('closure_version'):
            from .evidence import memory
            if protocol.get('spatial_memory_enabled',True):
                state['spatial_memory'] = memory(session, compact=compact_memory)
            state['context_budget'] = {'capacity_tokens':config.get('context_tokens',32768),
                'output_reserve':config['max_output_tokens'], 'margin_tokens':config.get('context_margin_tokens',1024),
                'counting':'registered UTF8 byte/token estimate; actual provider input usage recorded',
                'bytes_per_token':config.get('context_bytes_per_token',1.),
                'history_groups':len(retained), 'compact_grid':compact_memory}
        if protocol.get('budget_policy_version'):
            state['context_budget']['input_admission_tokens'] = config['max_input_tokens']
        if protocol.get('continuity_version'):
            from .continuity import state as continuity_state
            state.update(continuity_state(session, remaining))
            state['context_budget']['assembly'] = 'state_and_receipts; no replayed reasoning transcript'
        from .continuity import action_recovery
        recovering = action_recovery(session)
        if recovering:
            state['action_recovery'] = 'Consecutive output truncations executed no action. This single request disables extra reasoning; choose a concise tool action, save a draft, or finalize using existing evidence. Normal investigation thinking resumes after a non-truncated response. No extra call or scientific answer is supplied.'
        if protocol.get('progress_version'):
            from .progress import execution_state
            state.update(execution_state(session))
            if recovering:
                state['action_recovery'] = 'An earlier deliberation on this observation ledger exhausted its output limit. Extra thinking stays disabled for this ledger; all tools and the full grid remain available. Use calculations or new measurements to resolve a material uncertainty, or submit the supported report. New physical samples restore configured thinking. No answer is generated by the harness.'
        if recovering and protocol.get('progress_version', 0) >= 3:
            state['action_recovery'] = 'The last deliberation exhausted its output limit. This recovery request disables extra thinking. Choose a concise useful tool action or submit; new physical samples OR a successful analyze/fit_source result restore configured thinking for synthesis. Queries and failed calculations do not. Derived results are not new observations or certified conclusions. All tools and exact grid remain available; no answer is supplied.'
        if protocol.get('stability_version'):
            from .stability import state_receipt
            state['execution_receipt'] = state_receipt(session)
            state.pop('recent_measurements', None)  # exact values already in the native grid
            state['progress']['instruction'] = 'Review execution_receipt and grid before choosing an action. Use save_report for your tentative answers and submit_report to finish. No draft is automatically submitted.'
        prompt = system_prompt(protocol)
        if protocol.get('continuity_version'):
            prompt += "\nEach request contains persistent measured state, your notebook/draft and recent tool receipts. Read their results before choosing another action. Full feedback remains retrievable. Persist concise findings, alternatives and a next discriminating question in notes, not long private reasoning. Save tentative answers with report(final=false), and stop when further actions will not resolve material uncertainty.\n"
        if protocol.get('progress_version'):
            prompt += '\nTreat this as an observation-action loop. Review the complete grid and execution_history/computation_memory first. Produce a concise next tool action or submit the report; do not repeat completed descriptive analyses on unchanged evidence. Use inspect for standard retrieval, and analyze for a specific calculation. Failed code is not successful analysis; do not claim fits or checks that were not executed. Finish when evidence supports your answers; available calls are a ceiling, not a work quota.\n'
        if protocol.get("explicit_final_context"):
            phase = "report_only" if session.report_only else "review_only" if session.review_only else "active"
            prompt = f"CURRENT PHASE: {phase}. " + (
                "Your next response must call report. Sampling and SQL queries are closed. Set final=true and supply missing report fields; use explicit nulls for unsupported hypotheses. You can correct a saved report.\n\n" if session.report_only else
                "Sampling is closed. You may query existing measurements, save or finalize a report.\n\n" if session.review_only else
                "You may sample, analyze existing measurements and save or finalize a report.\n\n") + prompt
            if session.report_only and not protocol.get('continuity_version'):
                # Keep the observed results, not an assistant message still planning
                # the now-closed investigation. Notes/drafts preserve model memory.
                state["last_tool_results"] = [m for group in groups[-1:] for m in group if m["role"] == "tool"]
                if protocol.get("preserve_runtime_feedback"):
                    # User-role messages in these groups are generated by the
                    # runner (truncation/no-tool feedback), not model prose.
                    state["runtime_feedback"] = [m["content"] for group in groups[-2:] for m in group if m["role"] == "user"]
                state["last_tool_results_truncated"] = False
                while len(json.dumps(state["last_tool_results"]).encode()) > protocol["query_max_bytes"]:
                    state["last_tool_results"].pop(0)
                    state["last_tool_results_truncated"] = True
        state["compute_budget"] = {
            "initial_provider_calls": None if config.get("unlimited_provider_calls") else config["max_provider_calls"],
            "review_reserve_calls": protocol["final_report_reserve_calls"],
            "final_report_reserve_calls": final_calls,
            "cost_ceiling_cny_proxy": None if local else ceiling,
            "remaining_cost_cny_proxy": None if local else max(0, ceiling - cost),
                "policy": "Review permits analysis/report. Finalization permits report only. Output allowance may shrink to fit the remaining cost; thinking stays enabled."}
        if protocol.get('continuity_version'):
            state['compute_budget']['policy'] = 'Review permits analysis/report. Finalization permits report only, with the registered finalization_thinking setting. Output may shrink only to fit remaining cost.'
        if protocol.get('stability_version'):
            prompt = prompt.replace('report(final=false)', 'save_report').replace('call report', 'call submit_report').replace('Set final=true', 'Use submit_report').replace('Finalize with final=true', 'Finalize with submit_report')
            prompt += '\nDelivery contract: save_report persists your tentative fields; submit_report validates and ends. These tools have no final argument. Use inspect(profile/timeseries/summary/correlation) for supported descriptive calculations before writing custom code. Unknown answers are permitted; a valid submission is not a scientific success.\n'
        if protocol.get('delivery_version'):
            from .stability import delivery_context
            prompt, state = delivery_context(prompt, state, session)
        if config.get("unlimited_provider_calls"):
            state["provider_calls_remaining"] = None
            state["progress"]["calls_until_report_only"] = None
            state["compute_budget"]["call_limit"] = "unlimited; finish explicitly when evidence is sufficient"
        tool_schema = session.tools()
        if protocol.get('cockpit_version'):
            from .cockpit import present
            state, tool_schema = present(state, tool_schema, session, presentation_level)
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
        messages += [m for group in retained for m in group]
        messages.append({"role": "user", "content": json.dumps({"current_state": state}, ensure_ascii=False, **({"separators": (",", ":")} if protocol.get("native_grid_version") else {}))})
        body = payload(config, messages, tool_schema)
        if recovering:
            set_thinking(config, body, False)
            body.pop('thinking_budget', None)
        if protocol.get('continuity_version') and session.report_only:
            from .continuity import final_payload
            body = final_payload(body, config, protocol)
        return body

    def compact(retained):
        nonlocal presentation_level, compact_memory
        retained = [] if protocol.get('continuity_version') else list(retained)
        if session.report_only and protocol.get("explicit_final_context"):
            retained = []
        body = build(retained)
        from .context import fits
        def oversized(body):
            return len(json.dumps(body, ensure_ascii=False).encode()) > config['max_request_bytes'] or (protocol.get('closure_version') and not fits(body,config))
        while retained and oversized(body):
            retained.pop(0)
            body = build(retained)
        # Admit the actual presentation before estimating its cost. Large native
        # grids otherwise trigger finalization using an unsendable full-grid body.
        if protocol.get('closure_version'):
            if protocol.get('cockpit_version'):
                while presentation_level < 2 and oversized(body):
                    presentation_level += 1
                    body = build(retained)
            if oversized(body) and not compact_memory:
                compact_memory = True
                body = build(retained)
        return body, retained

    if protocol.get('stability_version'):
        from .stability import report_failures
        if report_failures(session) >= protocol.get('report_repair_attempts', 1) + 1:
            return None, 'report_delivery_failed', 0
    body, retained = compact(groups[-config["history_turns"]:])
    if not local and not session.report_only:
        # Look ahead before an active request can spend the reporting reserve.
        if cost + request_upper_cost(config, body) > ceiling - reserve:
            session.review_only = True
            body, retained = compact(retained)
        if session.review_only:
            session.report_only = True
            final_body, _ = compact(groups[-1:])
            session.report_only = False
            if cost + request_upper_cost(config, body) + final_calls * request_upper_cost(config, final_body) > ceiling:
                session.report_only = True
    if session.report_only:
        body, retained = compact(groups[-1:])
        if not local:
            while retained and cost + request_upper_cost(config, body) > ceiling:
                retained.pop(0)
                body = build(retained)
            # The cap belongs to this request, not to the model's global config.
            minimum = min(config["max_output_tokens"], protocol["budgeted_finalization"]["minimum_output_tokens"])
            low, high, affordable = minimum, config["max_output_tokens"], None
            while low <= high:
                tokens = (low + high) // 2
                body["max_tokens"] = tokens
                if cost + request_upper_cost(config, body) <= ceiling:
                    affordable, low = tokens, tokens + 1
                else:
                    high = tokens - 1
            if affordable is None:
                return None, "cost_limit", len(retained)
            body["max_tokens"] = affordable
            if protocol.get('stability_version'):
                from .stability import report_failures
                if report_failures(session):
                    body['max_tokens'] = min(affordable, protocol.get('report_repair_output_tokens', 2048))
    if protocol.get('closure_version'):
        from .context import fits
        if protocol.get('cockpit_version'):
            for level in (1, 2):
                if fits(body, config):
                    break
                presentation_level = level
                body, retained = compact(retained)
        if not fits(body, config):
            compact_memory = True
            body, retained = compact(retained)
        if not fits(body, config):
            return None, 'context_token_limit', len(retained)
    if len(json.dumps(body, ensure_ascii=False).encode()) > config["max_request_bytes"]:
        return None, "request_size_limit", len(retained)
    if not local and cost + request_upper_cost(config, body) > ceiling:
        return None, "cost_limit", len(retained)
    return body, None, len(retained)


def run_episode(store, iid, seed, config, protocol, target, directory, provider=complete):
    directory.mkdir(parents=True, exist_ok=True)
    trace_path = directory / "trace.jsonl"
    trace = read_trace(trace_path)
    session = Session(store, iid, seed, protocol, target, workspace=directory / "analysis")
    context = session.context()
    groups = []
    cost = 0.0
    request_count = 0
    usage = {"input_tokens": 0, "output_tokens": 0}
    outcomes = {row["request_index"]: row for row in trace if row["kind"] == "response"}
    for request in [row for row in trace if row["kind"] == "request"]:
        session.env.apply_resource_stage()
        request_count += 1
        response_row = outcomes.get(request["index"])
        if response_row is None:
            cost += request["upper_cost_cny"]
            continue
        result = response_row["response"]
        cost += response_row["cost_cny"]
        session.env.set_provider_call_count(request_count)
        session.report_only = request["report_only"]
        session.review_only = request.get("review_only", False)
        if "message" in result:
            session.replay = True
            group = apply_response(session, result)
            session.replay = False
            if stable_hash(group) != response_row["tool_results_sha256"]:
                raise ValueError("deterministic action/query replay mismatch")
            groups.append(group)
            for key in usage:
                usage[key] += result["usage"][key]
        elif result.get('error') and protocol.get('provider_error_feedback'):
            group=apply_response(session,result)
            if stable_hash(group)!=response_row['tool_results_sha256']:
                raise ValueError('provider feedback replay mismatch')
            groups.append(group)
    started = time.monotonic()
    failures = 0
    unresolved = protocol.get('stability_version') and any(r['kind']=='request' and r['index'] not in outcomes for r in trace)
    stop = "submitted" if session.final else "provider_call_limit"
    if unresolved:
        stop = 'unresolved_request'
    while not unresolved and not session.final and (config.get("unlimited_provider_calls") or request_count < config["max_provider_calls"]):
        remaining = config["max_provider_calls"] if config.get("unlimited_provider_calls") else config["max_provider_calls"] - request_count
        if protocol.get("budgeted_finalization"):
            body, budget_stop, retained_count = prepare_request(session, config, protocol, context, groups, remaining, cost)
            if budget_stop:
                stop = budget_stop
                break
            upper = request_upper_cost(config, body)
        else:
            body, upper, retained_count = legacy_request(session, config, protocol, context, groups, remaining, cost)
            if body is None:
                stop = "cost_limit"
                break
        request_index = len(trace)
        commit(trace_path, trace, "request", payload=body, report_only=session.report_only,
               review_only=session.review_only, upper_cost_cny=upper, history_groups_retained=retained_count)
        response = provider(config, body)
        from .stability import transport_result
        response = transport_result(response, protocol)
        if protocol.get('stability_version'):
            from .stability import contract_violations
            violations = contract_violations(config, body, response)
            if violations:
                response = {**response, 'error': 'provider_contract_violation', 'contract_violations': violations}
                # Retain raw message for evidence, but error feedback prevents execution.
        request_count += 1
        session.env.set_provider_call_count(request_count)
        actual_cost = usage_cost(config, response["usage"]) if "usage" in response else upper if response.get("charged") else 0.0
        cost += actual_cost
        group = []
        if "message" in response:
            if response.get("reported_model") != config["model"]:
                response = {"error": "model_identity_mismatch", "reported_model": response.get("reported_model"), "charged": True}
                stop = "model_identity_mismatch"
                if protocol.get('provider_error_feedback'):
                    group = apply_response(session, response)
                    groups.append(group)
            else:
                group = apply_response(session, response)
                groups.append(group)
                for key in usage:
                    usage[key] += response["usage"][key]
        elif response.get('error') and protocol.get('provider_error_feedback'):
            group=apply_response(session,response)
            groups.append(group)
        commit(trace_path, trace, "response", request_index=request_index, response=response,
               cost_cny=actual_cost, tool_results_sha256=stable_hash(group))
        if response.get("error"):
            failures += 1
            print(json.dumps({"instance": iid, "call": request_count, "provider_error": response["error"]}), flush=True)
            from .stability import FATAL
            if failures >= 3 or response["error"] in {"model_identity_mismatch", "HTTP_401", "HTTP_403", "HTTP_400"} | (FATAL if protocol.get('stability_version') else set()):
                stop = response["error"]
                break
            time.sleep(min(4, failures))
        else:
            failures = 0
            if request_count % 5 == 0 or session.final:
                names = [c["function"]["name"] for c in response["message"].get("tool_calls") or []]
                print(json.dumps({"instance": iid, "call": request_count, "tools": names, "nodes": len(session.env.measurements), "cost_cny": round(cost, 4), "final": session.final}), flush=True)
    episode = {"instance_id": iid, "seed": seed, "protocol_id": protocol["id"], "model": config["model"],
               "condition": protocol["condition"], "status": "completed" if session.final else "incomplete",
               "termination": "submitted" if session.final else stop, "report": session.report,
               "answer": session.canonical_answer(), "sample_ledger": session.env.sample_ledger,
               "sample_ledger_sha256": stable_hash(session.env.sample_ledger), "events": session.env.events,
               "event_sha256": stable_hash(session.env.events), "trace_head": trace[-1]["hash"] if trace else None,
               "provider_calls": request_count, "usage": usage, "cost_cny_proxy": round(cost, 8),
               "wall_time_this_process_s": time.monotonic() - started, "visited_nodes": len(session.env.measurements),
               "total_nodes": len(session.env.graph.positions), "physical_edges": session.env.edge_sequence,
               "movement_actions": session.env.successful_action_count, "path_length_m": session.env.budget.used_m,
               "notes": session.notes, "thinking": config["thinking"], "dataset_role": protocol["dataset_role"]}
    if protocol.get("efficient_tools"):
        tool_messages = [m for group in groups for m in group if m.get("role") == "tool"]
        episode.update(task_track=protocol.get("task_track", "end_to_end"),
                       capability_levels=protocol.get("capability_levels", ["L1", "L2", "L3", "L4", "L5"]),
                       numerical_fits=session.analysis_count, latest_source_fit=session.last_analysis,
                       tool_calls=len(tool_messages),
                       tool_errors=sum(json.loads(m["content"]).get("ok") is False for m in tool_messages),
                       request_bytes=sum(len(json.dumps(r["payload"], ensure_ascii=False).encode()) for r in trace if r["kind"] == "request"),
                       tool_result_bytes=sum(len(m["content"].encode()) for m in tool_messages))
    if protocol.get('stability_version'):
        results = [json.loads(m['content']) for m in tool_messages]
        episode['stability_diagnostics'] = {
            'version': 1,
            'representation_normalizations': sum(len(r.get('normalizations', [])) for r in results),
            'schema_rejections': sum(r.get('error')=='tool_schema' for r in results),
            'unanswered_requests': sum(r['kind']=='request' and r['index'] not in {x['request_index'] for x in trace if x['kind']=='response'} for r in trace),
            'usage_policy': 'provider usage when returned; charged errors estimated; unresolved request actual usage unknown'}
    if protocol.get("independent_levels"):
        episode.update(independent_levels=True, target_channel=target["target_channel"])
    if protocol.get("scientific_workbench"):
        episode.update(compute_calls=session.compute_count, latest_computation=session.last_compute,
                       evidence_mode=protocol.get("evidence_mode", "active"),
                       environment_time_s=session.env.budget.used_s,
                       physical_budget_enabled=protocol.get("physical_budget_enabled", False))
    if protocol.get('closure_version'):
        episode['closure_version'] = protocol['closure_version']
        episode['final_spatial_memory'] = session.state(0, cost).get('spatial_memory')
        episode['provider_errors'] = sum(bool(r['response'].get('error')) for r in trace if r['kind']=='response')
        episode['context_estimator_exceeded'] = sum(r['response'].get('usage',{}).get('input_tokens',0) > config.get('context_tokens',32768)-config['max_output_tokens']-config.get('context_margin_tokens',1024) for r in trace if r['kind']=='response')
    if protocol.get('budget_policy_version'):
        responses = [r['response'] for r in trace if r['kind']=='response']
        requests = [r for r in trace if r['kind']=='request']
        episode['budget_diagnostics'] = {
            'policy_version': protocol['budget_policy_version'],
            'provider_call_limit': None if config.get('unlimited_provider_calls') else config['max_provider_calls'],
            'output_tokens_per_call': config['max_output_tokens'],
            'input_admission_tokens': config['max_input_tokens'],
            'length_truncations': sum(r.get('finish_reason')=='length' for r in responses),
            'provider_call_limit_reached': not config.get('unlimited_provider_calls') and len(responses)>=config['max_provider_calls'],
            'movement_guard_reached': session.env.successful_action_count>=protocol['max_movement_actions'],
            'observation_horizon_stops': sum(e.get('code')=='E_OBSERVATION_HORIZON' for e in session.env.events),
            'raw_samples': len(session.env.sample_ledger),
            'compute_limit_reached': session.compute_count>=protocol['max_compute_calls'],
            'financial_guard_affected': config['provider']!='ollama' and (
                'cost' in stop or any(r['payload'].get('max_tokens',config['max_output_tokens'])<config['max_output_tokens'] or
                    (r.get('review_only') and protocol.get('evidence_mode')!='fixed' and
                     config['max_provider_calls']-i>protocol['final_report_reserve_calls']) for i,r in enumerate(requests))),
            'input_admission_exceeded': sum(r.get('usage',{}).get('input_tokens',0)>config['max_input_tokens'] for r in responses)}
    if protocol.get('continuity_version'):
        episode['continuity_version'] = protocol['continuity_version']
        episode['feedback_receipts_sha256'] = stable_hash(session.feedback_receipts)
        episode['model_notebook'] = session.notebook
        episode['delivery_diagnostics'] = {
            'submitted': session.final, 'draft_saved': bool(session.report),
            'finalization_thinking': protocol['finalization_thinking'],
            'report_phase_tool_violations': sum(o['result'].get('error')=='tool unavailable in the current phase'
                for r in session.feedback_receipts for o in r['outputs']),
            'final_phase_length_truncations': sum(r['response'].get('finish_reason')=='length' and
                next(q for q in trace if q['index']==r['request_index']).get('report_only',False)
                for r in trace if r['kind']=='response')}
    dump_json(directory / "episode.json", episode)
    session.close()
    return episode


def legacy_request(session, config, protocol, context, groups, remaining, cost):
    """Preserve the recorded 1.1/1.2 request policy for historical use."""
    session.report_only = remaining <= protocol["final_report_reserve_calls"] or (
        config["provider"] != "ollama" and cost >= config["hard_cost_ceiling_cny"] - config.get("submit_reserve_cny", 0.5))
    state = session.state(remaining, round(cost, 6))
    state["compute_budget"] = {"initial_provider_calls": config["max_provider_calls"],
        "final_report_reserve_calls": protocol["final_report_reserve_calls"],
        "cost_ceiling_cny_proxy": config["hard_cost_ceiling_cny"] if config["provider"] != "ollama" else None,
        "report_only_at_cost_cny_proxy": config["hard_cost_ceiling_cny"] - config.get("submit_reserve_cny", .5) if config["provider"] != "ollama" else None}
    retained = groups[-config["history_turns"]:]
    messages = [{"role": "system", "content": system_prompt(protocol)}, {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
    messages += [m for group in retained for m in group]
    messages.append({"role": "user", "content": json.dumps({"current_state": state}, ensure_ascii=False, **({"separators": (",", ":")} if protocol.get("native_grid_version") else {}))})
    body = payload(config, messages, session.tools())
    upper = request_upper_cost(config, body)
    if config["provider"] != "ollama" and cost + upper > config["hard_cost_ceiling_cny"]:
        if not session.report_only:
            session.report_only = True
            state["mode"] = "report_only"
        # Use a compact final request, retaining exact measurements via the
        # database until report-only, and the model's own report and notes.
        messages = messages[:2] + [m for group in groups[-1:] for m in group] + [{"role": "user", "content": json.dumps({"current_state": state, "instruction": "Finalize or revise the current report within the remaining cost budget."})}]
        body = payload(config, messages, session.tools())
        upper = request_upper_cost(config, body)
        if cost + upper > config["hard_cost_ceiling_cny"]:
            return None, upper, len(retained)
    return body, upper, len(retained)


def verify_episode(store, episode, protocol, target, directory):
    session = Session(store, episode["instance_id"], episode["seed"], protocol, target,
                      workspace=directory / "analysis", replay=True)
    trace = read_trace(directory / "trace.jsonl")
    request_count = 0
    requests = {}
    for row in trace:
        if row["kind"] == "request":
            session.env.apply_resource_stage()
            request_count += 1
            requests[row["index"]] = (request_count, row)
        elif row["kind"] == "response" and ('message' in row['response'] or (row['response'].get('error') and protocol.get('provider_error_feedback'))):
            count, request = requests[row["request_index"]]
            session.env.set_provider_call_count(count)
            session.report_only = request["report_only"]
            session.review_only = request.get("review_only", False)
            result = apply_response(session, row["response"])
            assert stable_hash(result) == row["tool_results_sha256"], "tool replay mismatch"
    assert stable_hash(session.env.sample_ledger) == episode["sample_ledger_sha256"], "measurement replay mismatch"
    assert stable_hash(session.env.events) == episode["event_sha256"], "event replay mismatch"
    assert session.report == episode["report"], "report replay mismatch"
    if protocol.get('continuity_version'):
        assert stable_hash(session.feedback_receipts) == episode['feedback_receipts_sha256'], 'feedback replay mismatch'
        assert session.notebook == episode['model_notebook'], 'notebook replay mismatch' 
    if "numerical_fits" in episode:
        assert session.analysis_count == episode["numerical_fits"], "numerical fit count mismatch"
        assert session.last_analysis == episode["latest_source_fit"], "numerical result replay mismatch"
    if "compute_calls" in episode:
        assert session.compute_count == episode["compute_calls"], "computation replay count mismatch"
        assert session.last_compute == episode["latest_computation"], "computation replay mismatch"
    if protocol.get('closure_version'):
        assert session.state(0,0).get('spatial_memory') == episode.get('final_spatial_memory'), 'spatial memory replay mismatch'
    session.close()
    return True
