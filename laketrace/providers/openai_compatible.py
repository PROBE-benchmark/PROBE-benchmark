from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderResponse:
    tool_calls: list[dict[str, Any]]
    content: str | None
    usage: dict[str, int]
    raw_model: str | None


def parse_chat_completion(body: Any) -> ProviderResponse:
    """Validate the minimal native tool-call response used by the harness."""

    try:
        choices = body["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("choices must be a non-empty array")
        choice = choices[0]["message"]
        raw_calls = choice.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise ValueError("tool_calls must be an array")
        calls = []
        for item in raw_calls:
            function = item.get("function") or {}
            arguments = function.get("arguments") or "{}"
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            if not isinstance(arguments, str):
                raise ValueError("tool-call arguments must be a JSON string or object")
            calls.append({
                "id": item.get("id"),
                "name": function.get("name"),
                "arguments": arguments,
            })
        usage = body.get("usage") or {}
        return ProviderResponse(
            calls,
            choice.get("content"),
            {
                "input_tokens": int(usage.get("prompt_tokens", 0)),
                "output_tokens": int(usage.get("completion_tokens", 0)),
            },
            body.get("model"),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"malformed provider response: {exc}") from exc


class OpenAICompatibleProvider:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.url = config["base_url"].rstrip("/") + "/chat/completions"

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ProviderResponse:
        payload = {
            "model": self.config["model"], "messages": messages, "tools": tools,
            # A provider round has no valid prose-only outcome in this agent
            # protocol: move, analyze, and terminal submission are all tools.
            # Native required mode prevents the provider from spending the
            # action round on an unexecutable assistant message.
            "tool_choice": "required", "temperature": float(self.config.get("temperature", 1.0)),
            "stream": False,
        }
        if self.config.get("max_output_tokens"):
            payload["max_tokens"] = int(self.config["max_output_tokens"])
        extra = self.config.get("extra_body") or {}
        if not isinstance(extra, dict):
            raise ValueError("extra_body must be an object")
        protected = set(payload) & set(extra)
        if protected:
            raise ValueError(f"extra_body may not override standard fields: {sorted(protected)}")
        payload.update(extra)
        headers = {"Content-Type": "application/json"}
        key_env = self.config.get("api_key_env")
        if key_env:
            key = os.environ.get(key_env)
            if not key:
                raise RuntimeError(f"required API key environment variable is unset: {key_env}")
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        retries = int(self.config.get("retries", 3))
        body = None
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=float(self.config.get("timeout_s", 180))) as response:
                    body = json.loads(response.read().decode())
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")
                if attempt >= retries or exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"provider HTTP {exc.code}: {detail[:1000]}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt >= retries:
                    raise RuntimeError(f"provider transport failure: {exc}") from exc
            except json.JSONDecodeError as exc:
                if attempt >= retries:
                    raise RuntimeError(f"provider returned invalid JSON: {exc}") from exc
            time.sleep(min(8.0, 0.5 * (2 ** attempt)))
        assert body is not None
        return parse_chat_completion(body)
