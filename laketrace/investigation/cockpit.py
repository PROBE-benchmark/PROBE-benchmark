"""Context presentation only: observed data and scientific decisions stay untouched."""
import copy
from .continuity import preview


def present(state, tools, session, level=0):
    state, tools = copy.deepcopy(state), copy.deepcopy(tools)
    if level:
        # Retain exact receipts externally; show completion/errors before verbose tables.
        receipts = []
        for receipt in session.feedback_receipts[-3:]:
            outputs = []
            for item in receipt['outputs']:
                result = item['result']
                if item.get('tool') in {'goto', 'move', 'survey'}:
                    keys = ('ok','complete','completed_edges','position','stop_reason',
                            'remaining_points','requested_points','handled_points','error')
                    shown = {k:result[k] for k in keys if k in result}
                    shown['full_result'] = 'inspect(view=feedback, receipt_id=%s)' % receipt['receipt_id']
                    outputs.append({'tool':item['tool'], **shown})
                else:
                    outputs.append({'tool':item.get('tool'), 'ok':result.get('ok'),
                                    **preview(result, 2400 if level==1 else 1000)})
            receipts.append({k:receipt[k] for k in ('receipt_id','provider_call','sample_count','sha256')} | {'outputs':outputs})
        state['recent_tool_feedback'] = receipts
        # These are redundant prose, not data or model-authored scientific memory.
        for key in ('execution_note','decision_rule','instruction'):
            state.pop(key, None)
        for tool in tools:
            def strip_descriptions(node):
                if isinstance(node, dict):
                    node.pop('description', None)
                    for value in node.values():strip_descriptions(value)
                elif isinstance(node, list):
                    for value in node:strip_descriptions(value)
            # Keep the tool's own description; remove repetitive property prose only.
            strip_descriptions(tool['function']['parameters'])
    if level >= 2:
        history = state.get('execution_history', [])
        state['execution_history'] = history[-6:]
        state['execution_history_omitted'] = max(0, len(history)-6)
        state['older_execution_retrieval'] = 'inspect(view=feedback, receipt_id, offset, limit)'
    state['context_presentation'] = {'level':level,
        'raw_grid_preserved':True,
        'policy':'Non-grid feedback shortened before selecting any local grid window. All exact receipts and measurements remain retrievable. Truncated excerpts are not complete results.'}
    return state, tools
