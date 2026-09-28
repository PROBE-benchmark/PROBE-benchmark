"""Execution-derived working memory; never generates scientific conclusions."""


def execution_state(session):
    computations = []
    timeline = []
    for receipt in session.feedback_receipts:
        outputs = receipt['outputs']
        timeline.append({'call': receipt['provider_call'],
                         'samples': receipt['sample_count'],
                         'finish': receipt.get('finish_reason'),
                         'tools': [{'name': o['tool'], 'ok': o['result'].get('ok')} for o in outputs]})
        for output in outputs:
            if output['tool'] not in {'analyze', 'fit_source'}:
                continue
            result = output['result']
            text = result.get('stdout', '')
            computations.append({'call': receipt['provider_call'], 'tool': output['tool'],
                                 'ok': result.get('ok'), 'samples': receipt['sample_count'],
                                 'analysis_id': result.get('analysis_id'),
                                 'stdout_excerpt': text[:1600], 'stdout_omitted_chars': max(0, len(text)-1600),
                                 'error': result.get('stderr', result.get('error', ''))[-400:],
                                 'receipt_id': receipt['receipt_id']})
            if session.protocol.get('progress_version', 0) >= 3 and len(text) > 1600:
                computations[-1]['stdout_excerpt'] = text[:800]
                computations[-1]['stdout_tail_excerpt'] = text[-800:]
                computations[-1]['excerpt_order'] = 'head, omitted middle, tail; retrieve full output by receipt'
            if session.protocol.get('progress_version', 0) >= 2 and output['tool'] == 'fit_source':
                item = computations[-1]
                item['fit_result'] = {k: result[k] for k in (
                    'assumptions', 'scope', 'measured_nodes', 'heldout_nodes',
                    'constant_background_heldout_rmse', 'ranking_uses') if k in result}
                candidates = result.get('top_candidates', [])
                item['fit_result']['top_candidates'] = candidates[:2]
                item['fit_result']['omitted_candidates'] = max(0, len(candidates) - 2)
    result = {'execution_history': timeline, 'computation_memory': computations,
            'execution_note': 'These are actual execution records, not verified scientific conclusions. All grid values are unchanged. Read existing results before repeating a calculation. inspect(view=artifact,analysis_id,artifact="stdout.txt") or inspect(view=feedback,receipt_id) retrieves full output. A failed computation may have partial stdout; do not treat it as completed. Never claim a fit, test or validation absent from the actual record.',
            'decision_rule': 'Use the grid and computation results to update L1/L2/L3/L4 separately. Submit report(final=true) once supported, with honest uncertainty. If a material alternative remains, choose a measurement or calculation that discriminates it. Reprinting the same ranking is not new evidence. Save evidence and remaining uncertainty in notes or report(final=false); do not wait for the call limit to submit.'}
    if session.protocol.get('progress_version', 0) >= 2:
        unchanged = 0
        previous = 1 if session.protocol.get('evidence_mode') == 'active' else None
        for receipt in session.feedback_receipts:
            unchanged = unchanged + 1 if receipt['sample_count'] == previous else 0
            previous = receipt['sample_count']
        result['observation_progress'] = {
            'calls_without_new_samples': unchanged,
            'instruction': 'Reading and computing can resolve uncertainty but do not acquire water measurements. Reuse completed results; batch related calculations in one analyze call. Choose a new observation only for a remaining discriminating question, or report supported answers with uncertainty. This counter does not prohibit analysis or force sampling.'}
    return result


def grounded_recovery(session):
    """Do not restart a truncated deliberation on the same observation ledger."""
    samples = len(session.env.sample_ledger)
    for receipt in reversed(session.feedback_receipts):
        if receipt['sample_count'] != samples:
            break
        if receipt.get('finish_reason') == 'length':
            return True
        if session.protocol.get('progress_version', 0) >= 3 and any(
                o['tool'] in {'analyze', 'fit_source'} and o['result'].get('ok') is True
                for o in receipt['outputs']):
            # New derived results warrant synthesis even without moving the sensor.
            # Queries and failed calculations cannot reset truncation recovery.
            return False
    return False
