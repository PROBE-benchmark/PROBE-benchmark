"""Bounded context assembly; estimates are explicit, never called exact token counts."""
import json
import math


def token_estimate(body, config):
    # Calibrated to the recorded local Qwen JSON/tool workloads. Other models
    # must register their own ratio or tokenizer; actual provider usage is saved.
    chars = len(json.dumps({'messages':body['messages'], 'tools':body.get('tools', [])}, ensure_ascii=False).encode())
    ratio = config.get('context_bytes_per_token', 1.)
    if not isinstance(ratio, (int, float)) or not 1 <= ratio <= 4:
        raise ValueError('context_bytes_per_token must be a registered value in [1,4]')
    return math.ceil(chars/ratio)


def input_allowance(body, config):
    capacity = config.get('context_tokens', 32768)
    output = body.get('max_tokens', body.get('options', {}).get('num_predict', config['max_output_tokens']))
    if config.get('output_budget_semantics') == 'separate' and body.get('enable_thinking', config.get('thinking')):
        output += body.get('thinking_budget', config.get('thinking_budget', 0))
    available = capacity-output-config.get('context_margin_tokens',1024)
    return min(available, config.get("max_input_tokens", available))


def fits(body, config):
    return token_estimate(body,config) <= input_allowance(body,config)
