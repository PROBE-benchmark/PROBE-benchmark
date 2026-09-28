"""Deterministic delivery and runtime checks. Never infer scientific answers."""
import copy
from .context import input_allowance
from laketrace.util import stable_hash

FATAL = {'transport_uncertain', 'provider_contract_violation', 'request_deadline', 'unresolved_request',
         'report_delivery_failed', 'deadline_runtime_unsupported', 'deadline_timer_already_active'}


def normalize_report(args, schema):
    """Decode one extra JSON layer only where the declared schema rejects a string.

    Never truncate, clamp, infer a missing value, or interpret free text. An already
    valid string is untouched. Every accepted representation change is recorded.
    """
    import json
    from jsonschema import Draft202012Validator
    out = copy.deepcopy(args)
    changes = []
    if not isinstance(out, dict):
        return out, changes
    def reject_constant(value):
        raise ValueError('non-finite JSON constant')
    def finite_float(value):
        import math
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('non-finite JSON number')
        return number
    for key, value in out.items():
        spec = schema.get('properties', {}).get(key)
        if not spec or not isinstance(value, str) or Draft202012Validator(spec).is_valid(value):
            continue
        text = value.lower() if value.lower() in ('true', 'false', 'null') else value
        try:
            mapped = json.loads(text, parse_constant=reject_constant, parse_float=finite_float)
        except (ValueError, TypeError):
            continue
        if isinstance(mapped, str) or not Draft202012Validator(spec).is_valid(mapped):
            continue
        changes.append({'field': key, 'raw': value, 'canonical': mapped})
        out[key] = mapped
    return out, changes


def report_failures(session):
    count = 0
    for receipt in reversed(session.feedback_receipts):
        reports = [o['result'] for o in receipt['outputs']
                   if o.get('tool') in {'report', 'save_report', 'submit_report'}]
        if not reports:
            continue
        if any(r.get('ok') for r in reports):
            break
        count += 1
    return count


def contract_violations(config, body, response):
    if not config.get('runtime_contract_version') or 'usage' not in response:
        return []
    usage = response['usage']
    errors = []
    if usage['input_tokens'] > input_allowance(body, config):
        errors.append('actual_input_exceeds_registered_allowance')
    cap = body.get('max_tokens', body.get('options', {}).get('num_predict', 0))
    if body.get('enable_thinking') and config.get('output_budget_semantics') == 'separate':
        cap += body.get('thinking_budget', 0)
    if usage['output_tokens'] > cap:
        errors.append('actual_output_exceeds_registered_allowance')
    return errors


def state_receipt(session):
    ledger = session.env.sample_ledger
    return {'ledger_version': len(ledger), 'ledger_sha256': stable_hash(ledger),
            'visited_nodes': len(session.env.measurements),
            'position': session.env._position_snapshot(),
            'report_fields_saved': sorted(session.report),
            'report_failures': report_failures(session),
            'policy': 'Grid and sample history are authoritative. A receipt confirms execution, not scientific correctness. Revisit only for transit or a stated new measurement purpose.'}


def validate_batch_evidence(directory, expected):
    """Require three replay-verified completed smoke cases, not scientific wins."""
    import json
    from pathlib import Path
    from .runner import read_trace
    root = Path(directory)
    registered = json.loads((root/'manifest.json').read_text())
    for key in ('protocol','model_config','source_sha256','data_manifest_sha256','public_tasks_sha256','seed'):
        if registered.get(key) != expected.get(key):
            raise ValueError('provider smoke validation mismatch: '+key)
    validation = json.loads((root/'validation.json').read_text())
    if validation.get('manifest_sha256') != stable_hash(registered):
        raise ValueError('provider smoke manifest receipt mismatch')
    episodes = [json.loads(f.read_text()) for f in root.glob('episodes/*/episode.json')]
    if len({e['instance_id'] for e in episodes}) < 3 or validation.get('replay_verified') != len(episodes):
        raise ValueError('batch requires at least three distinct replay-verified provider smoke cases')
    if any(e.get('termination') != 'submitted' for e in episodes):
        raise ValueError('provider smoke cases must submit; scientific correctness is not a gate')
    traced = 0
    for file in root.glob('episodes/*/trace.jsonl'):
        rows = read_trace(file)
        requests = {r['index']:r for r in rows if r['kind']=='request'}
        responses = {r['request_index']:r for r in rows if r['kind']=='response'}
        if requests.keys() != responses.keys():
            raise ValueError('provider smoke contains unanswered requests')
        for index, response in responses.items():
            value = response['response']
            if value.get('error') or value.get('reported_model') != expected['model_config']['model']:
                raise ValueError('provider smoke contains a provider error or wrong model')
            if contract_violations(expected['model_config'], requests[index]['payload'], value):
                raise ValueError('provider smoke violates token contract')
        traced += 1
    if traced != len(episodes):
        raise ValueError('missing provider smoke trace')
    return {'cases':len(episodes), 'manifest_sha256':stable_hash(registered),
            'scope':'delivery and runtime compatibility only; no scientific pass requirement'}


def delivery_context(prompt, state, session):
    """Expose the current report contract without generating any answer."""
    instruction = ('Use submit_report with every missing_report_fields entry explicitly supplied. '
                   'It merges your supplied fields with saved_report and submits only after validation. '
                   'An empty object is sufficient only for a complete valid saved report. '
                   'Use null only for fields whose schema permits it; probabilities remain numeric.')
    state['instruction'] = instruction if session.report_only else (
        'Investigate using available tools. save_report persists tentative fields; submit_report merges answers and finishes.')
    state['delivery'] = {'missing_fields': sorted(session.required_report() - session.report.keys()),
                         'saved_report': copy.deepcopy(session.report), 'instruction': instruction,
                         'latest_failure': None}
    for receipt in reversed(session.feedback_receipts):
        reports = [o['result'] for o in receipt['outputs']
                   if o.get('tool') in {'save_report', 'submit_report'}]
        if reports:
            result = reports[-1]
            if not result.get('ok'):
                state['delivery']['latest_failure'] = {k: result[k] for k in
                    ('error', 'errors', 'field_errors', 'missing_fields') if k in result}
            break
    # Legacy system text is shared with historical protocols. Rewrite only the new
    # version's delivered prompt; observed feedback and model notes are untouched.
    prompt = prompt.replace('report(final=true)', 'submit_report').replace(
        'report(final=false)', 'save_report').replace('report with final=true', 'submit_report').replace(
        'final=true', 'submit_report').replace('final=false', 'save_report')
    return prompt + '\n' + instruction, state


def stops_run(termination, protocol):
    fatal = FATAL - ({'report_delivery_failed'} if protocol.get('delivery_version') else set())
    return termination in {'HTTP_401', 'HTTP_403', 'HTTP_400', 'model_identity_mismatch'} | fatal


def transport_result(response, protocol):
    """Never execute or retry a response whose remote completion/usage is unknown."""
    if (protocol.get('transport_version') and response.get('error') and
            response.get('charged') and 'usage' not in response):
        return {**response, 'original_error': response['error'],
                'error': 'transport_uncertain', 'usage_unknown': True,
                'retry_allowed': False}
    return response
