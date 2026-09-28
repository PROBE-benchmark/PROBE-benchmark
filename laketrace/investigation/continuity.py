"""Deterministic receipts and model-authored memory, never a scientific solver."""
import copy
import json

from laketrace.util import stable_hash


def preview(value, limit=6000):
    """Bound an exact result without pretending that omitted data were summarized."""
    wire=json.dumps(value,ensure_ascii=False,separators=(',',':'))
    if len(wire.encode())<=limit:
        return {'value':copy.deepcopy(value),'truncated':False}
    # Preserve valid tabular structure and column names where possible.
    if isinstance(value,dict) and isinstance(value.get('rows'),list):
        table=copy.deepcopy(value)
        total=len(table['rows'])
        while table['rows'] and len(json.dumps(table,ensure_ascii=False).encode())>limit:
            table['rows'].pop()
        if len(json.dumps(table,ensure_ascii=False).encode())<=limit:
            return {'value':table,'truncated':True,'returned_rows':len(table['rows']),
                    'omitted_rows':total-len(table['rows'])}
    # A labelled text excerpt is preferable to malformed JSON or invented findings.
    return {'text_excerpt':wire.encode()[:limit].decode('utf-8',errors='ignore'),
            'truncated':True,'original_bytes':len(wire.encode())}


def record(session, group, response=None):
    """Called only after execution. Failed and truncated requests remain visible."""
    outputs=[]
    for message in group:
        if message['role']=='tool':
            outputs.append({'tool':message.get('tool_name'),
                            'result':json.loads(message['content'])})
        elif message['role']=='user':
            outputs.append({'tool':None,'result':{'ok':False,'feedback':message['content']}})
    receipt={'receipt_id':len(session.feedback_receipts)+1,
             'provider_call':session.env.provider_call_count,
             'outputs':outputs,'sample_count':len(session.env.sample_ledger)}
    if session.protocol.get('recovery_version'):
        receipt['finish_reason'] = (response or {}).get('finish_reason')
    receipt['sha256']=stable_hash(receipt)
    session.feedback_receipts.append(receipt)


def feedback_context(session):
    result=[]
    for i,receipt in enumerate(session.feedback_receipts[-3:]):
        latest=receipt is session.feedback_receipts[-1]
        cap=6000 if latest else 1200
        outputs=receipt['outputs']
        result.append({k:receipt[k] for k in ('receipt_id','provider_call','sha256','sample_count')} |
            {'outputs':[{'tool':o['tool'],'ok':o['result'].get('ok'),
                         **preview(o['result'],max(256,cap//max(1,len(outputs))))} for o in outputs],
             'retrieve':'inspect(view=feedback, receipt_id, offset, limit); offset/limit count pretty-printed JSON lines'})
    return result


def retrieve(session,args):
    if set(args)-{'view','receipt_id','offset','limit'}:
        raise ValueError('feedback accepts receipt_id, offset, limit only')
    number=args.get('receipt_id')
    if type(number) is not int or not 1<=number<=len(session.feedback_receipts):
        raise ValueError('receipt_id must identify an executed request')
    lines=json.dumps(session.feedback_receipts[number-1],ensure_ascii=False,indent=2).splitlines()
    start,limit=args.get('offset',0),args.get('limit',128)
    return {'ok':True,'receipt_id':number,'content':'\n'.join(lines[start:start+limit]),
            'next_offset':start+limit if start+limit<len(lines) else None,'total_lines':len(lines)}


def state(session,remaining):
    # New requests are reconstructed from authoritative state. No growing CoT transcript.
    return {'recent_tool_feedback':feedback_context(session),
            'model_notebook':[dict(n,text=n['text'][:1000],truncated=len(n['text'])>1000) for n in session.notebook[-6:]],
            'progress':{'samples_acquired':len(session.env.sample_ledger),
                        'calls_until_report_only':max(0,remaining-session.protocol['final_report_reserve_calls']),
                        'draft_saved':bool(session.report),
                        'instruction':'Use the latest receipt before repeating a query. Save your own findings and unresolved alternatives in notes; use report(final=false) for tentative answers. Another action should test a specific remaining uncertainty. You may finish early. No draft is automatically graded or submitted.'}}


def final_payload(body,config,protocol):
    # Registered phase policy, never an unlogged retry or increased budget.
    from .provider import set_thinking
    set_thinking(config, body, protocol['finalization_thinking'])
    if config['provider']!='ollama':
        if not protocol['finalization_thinking']:
            body.pop('thinking_budget', None)
        if config.get('final_tool_choice')=='named':
            body['tool_choice']={'type':'function','function':{'name':'submit_report' if protocol.get('stability_version') else 'report'}}
    return body


def action_recovery(session):
    """One non-thinking action opportunity after each pair of truncated requests."""
    threshold = session.protocol.get('truncation_recovery_after', 2)
    if not session.protocol.get('recovery_version') or session.report_only:
        return False
    if session.protocol.get('progress_version'):
        from .progress import grounded_recovery
        return grounded_recovery(session)
    consecutive = 0
    for receipt in reversed(session.feedback_receipts):
        if receipt.get('finish_reason') != 'length':
            break
        consecutive += 1
    return consecutive > 0 and consecutive % threshold == 0
