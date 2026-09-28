"""Native thinking/tool transport; no credentials are written into artifacts."""
from __future__ import annotations

import copy
import http.client
import json
import os
import time
import signal
import threading
import urllib.error
import urllib.request
import re


def set_thinking(config, body, enabled):
    """Apply the registered provider's thinking switch to an outgoing body."""
    if config["provider"] == "ollama":
        body["think"] = enabled
    elif config.get("thinking_wire_format") == "xiaomi":
        body.pop("enable_thinking", None)
        body["thinking"] = {"type": "enabled" if enabled else "disabled"}
    else:
        body["enable_thinking"] = enabled
    return body


def thinking_enabled(config, body):
    if config.get("thinking_wire_format") == "xiaomi":
        return body.get("thinking", {}).get("type") == "enabled"
    return bool(body.get("think" if config["provider"] == "ollama" else "enable_thinking"))


def payload(config, messages, tools):
    if config["provider"] == "ollama":
        normalized = copy.deepcopy(messages)
        for message in normalized:
            for call in message.get("tool_calls", []):
                args = call["function"].get("arguments", {})
                if isinstance(args, str):
                    call["function"]["arguments"] = json.loads(args)
        return {"model": config["model"], "messages": normalized, "tools": tools,
                "think": config["thinking"], "stream": False,
                "options": {"temperature": config["temperature"], "num_predict": config["max_output_tokens"],
                            "num_ctx": config.get("context_tokens", 32768)}, "keep_alive": "30m"}
    body = {"model": config["model"], "messages": messages, "tools": tools, "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": config["temperature"], "max_tokens": config["max_output_tokens"],
            "stream": True, "stream_options": {"include_usage": True}}
    set_thinking(config, body, config["thinking"])
    if config.get('output_budget_semantics') == 'separate' and config['thinking']:
        body['thinking_budget'] = config['thinking_budget']
    if config.get('response_transport') == 'json':
        body['stream'] = False
        body.pop('stream_options', None)
    return body


def request_upper_cost(config, body):
    if config["provider"] == "ollama":
        return 0.0
    output = body['max_tokens']
    if config.get('output_budget_semantics') == 'separate' and thinking_enabled(config, body):
        output += body.get('thinking_budget', 0)
    return (len(json.dumps(body, ensure_ascii=False).encode()) * config["input_cny_per_million"]
            + output * config["output_cny_per_million"]) / 1e6


def usage_cost(config, usage):
    if config["provider"] == "ollama":
        return 0.0
    return (usage["input_tokens"] * config["input_cny_per_million"]
            + usage["output_tokens"] * config["output_cny_per_million"]) / 1e6


def _recover_mimo_xml_calls(message, tools):
    """Recover MiMo's occasional XML tool serialization without inferring data.

    MiMo can return a concatenated/unknown function name in ``tool_calls`` while
    emitting the complete call in its assistant content.  Parse only explicitly
    delimited calls whose function is present in the declared tool set; values are
    JSON-decoded when valid and otherwise retained as strings.
    """
    content = message.get("content") or ""
    allowed = {t.get("function", {}).get("name") for t in tools}
    calls = []
    for block in re.findall(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", content, re.S):
        name, body = block[0].strip(), block[1]
        if name not in allowed:
            continue
        args = {}
        for key, raw in re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.S):
            value = raw.strip()
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                pass
            args[key.strip()] = value
        calls.append({"id": f"mimo_xml_{len(calls)}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    invalid = False
    try:
        from jsonschema import Draft202012Validator
        schemas = {t.get("function", {}).get("name"): t.get("function", {}).get("parameters", {}) for t in tools}
        for call in message.get("tool_calls", []) or []:
            name = call.get("function", {}).get("name")
            args = call.get("function", {}).get("arguments", "{}")
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
            except (TypeError, ValueError):
                parsed = None
            if name not in allowed or not Draft202012Validator(schemas.get(name, {})).is_valid(parsed):
                invalid = True
                break
    except Exception:
        invalid = any(c.get("function", {}).get("name") not in allowed
                      for c in message.get("tool_calls", []) or [])
    if calls and invalid:
        message["tool_calls"] = calls
    return message


class RequestDeadline(BaseException):
    """Escapes network exception handlers so a stream cannot extend its deadline."""


def complete(config, body):
    seconds = config.get('request_deadline_s')
    if not seconds:
        return _complete(config, body)
    if not hasattr(signal, 'setitimer') or threading.current_thread() is not threading.main_thread():
        return {'error': 'deadline_runtime_unsupported', 'charged': False}
    previous = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0]:
        return {'error': 'deadline_timer_already_active', 'charged': False}
    def expire(signum, frame):
        raise RequestDeadline()
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.monotonic()
    try:
        return _complete(config, body)
    except RequestDeadline:
        return {'error': 'request_deadline', 'charged': True, 'usage_unknown': True,
                'elapsed_s': time.monotonic() - started}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _complete(config, body):
    wire = json.dumps(body, ensure_ascii=False).encode()
    if len(wire) > config["max_request_bytes"]:
        return {"error": "request_size_limit", "charged": False}
    headers = {"Content-Type": "application/json"}
    local = config["provider"] == "ollama"
    if not local:
        key = os.environ.get(config["api_key_env"])
        if not key:
            raise RuntimeError("API credential is not available in this process")
        headers["Authorization"] = "Bearer " + key
    url = config["base_url"].rstrip("/") + ("/api/chat" if local else "/chat/completions")
    request = urllib.request.Request(url, data=wire, headers=headers, method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=config["timeout_s"]) as response:
            if local:
                raw = json.loads(response.read())
                message = raw["message"]
                usage = {"input_tokens": raw.get("prompt_eval_count", 0), "output_tokens": raw.get("eval_count", 0)}
                finish = raw.get("done_reason")
                model = raw.get("model")
            elif not body.get('stream', True) or 'application/json' in response.headers.get('Content-Type', ''):
                raw = json.loads(response.read())
                choices = raw.get('choices') or []
                choice = next((c for c in choices if c.get('index', 0) == 0), {})
                finish = choice.get('finish_reason')
                raw_usage = raw.get('usage') or {}
                if not finish or not isinstance(choice.get('message'), dict):
                    return {'error': 'incomplete_json_response', 'charged': True, 'usage_unknown': True,
                            'diagnostic': {'transport': 'json', 'has_finish': bool(finish),
                                           'has_usage': bool(raw_usage), 'has_error': bool(raw.get('error'))},
                            'elapsed_s': time.monotonic() - started}
                if not {'prompt_tokens', 'completion_tokens'} <= raw_usage.keys():
                    return {'error': 'missing_usage', 'charged': True, 'usage_unknown': True,
                            'elapsed_s': time.monotonic() - started}
                message = choice['message']
                model = raw.get('model')
                usage = {'input_tokens': raw_usage['prompt_tokens'], 'output_tokens': raw_usage['completion_tokens'],
                         'reasoning_tokens': (raw_usage.get('completion_tokens_details') or {}).get('reasoning_tokens')}
            else:
                message = {"role": "assistant", "content": "", "reasoning_content": ""}
                chunks = {}
                raw_usage = {}
                finish = model = None
                events, received_bytes, done = 0, 0, False
                for line in response:
                    received_bytes += len(line)
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    events += 1
                    part = json.loads(data)
                    if part.get("usage"):
                        raw_usage = part["usage"]
                    model = part.get("model", model)
                    for choice in part.get("choices", []):
                        if choice.get("index", 0) != 0:
                            continue
                        finish = choice.get("finish_reason") or finish
                        delta = choice.get("delta") or {}
                        for key in ("content", "reasoning_content"):
                            message[key] += delta.get(key) or ""
                        for call in delta.get("tool_calls") or []:
                            idx = call.get("index", 0)
                            current = chunks.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                            if call.get("id"):
                                current["id"] = call["id"]
                            for key in ("name", "arguments"):
                                current["function"][key] += (call.get("function") or {}).get(key) or ""
                if finish is None:
                    return {"error": "incomplete_stream", "charged": True, "usage_unknown": True,
                            "diagnostic": {"transport": "sse", "events": events, "received_bytes": received_bytes,
                                           "done_received": done, "has_usage": bool(raw_usage)},
                            "elapsed_s": time.monotonic() - started}
                if chunks:
                    message["tool_calls"] = [chunks[i] for i in sorted(chunks)]
                if not {"prompt_tokens", "completion_tokens"} <= raw_usage.keys():
                    return {"error": "missing_usage", "charged": True, "elapsed_s": time.monotonic() - started}
                usage = {"input_tokens": raw_usage["prompt_tokens"], "output_tokens": raw_usage["completion_tokens"],
                         "reasoning_tokens": (raw_usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}
        message = _recover_mimo_xml_calls(message, body.get("tools", []))
        message.setdefault("role", "assistant")
        message.setdefault("content", "")
        for index, call in enumerate(message.get("tool_calls") or []):
            call.setdefault("id", f"local_{index}")
            call.setdefault("type", "function")
            if isinstance(call["function"].get("arguments"), dict):
                call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
        return {"message": message, "usage": usage, "finish_reason": finish,
                "reported_model": model, "elapsed_s": time.monotonic() - started,
                "reasoning_chars": len(message.get("thinking") or message.get("reasoning_content") or "")}
    except urllib.error.HTTPError as exc:
        # Never retain an HTTP body that might echo authorization information.
        detail = exc.read().decode(errors="replace")
        for secret in (os.environ.get(config.get("api_key_env", "")),):
            if secret:
                detail = detail.replace(secret, "[REDACTED]")
        return {"error": f"HTTP_{exc.code}", "detail": detail[:500], "charged": exc.code >= 500,
                "elapsed_s": time.monotonic() - started}
    except (OSError, ValueError, KeyError, http.client.HTTPException) as exc:
        return {"error": type(exc).__name__, "charged": True, "usage_unknown": True,
                "diagnostic": {"exception_type": type(exc).__name__,
                    "reason_type": type(getattr(exc, 'reason', None)).__name__,
                    "errno": getattr(getattr(exc, 'reason', exc), 'errno', None)},
                "elapsed_s": time.monotonic() - started}
