from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from laketrace.providers.openai_compatible import ProviderResponse, parse_chat_completion
from laketrace.util import stable_hash


_AUDIT_HEADERS = (
    "x-request-id", "request-id", "x-dashscope-request-id", "retry-after",
    "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests", "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens",
)


def _safe_headers(headers: Any) -> dict[str, str]:
    if headers is None:
        return {}
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    return {key: lowered[key] for key in _AUDIT_HEADERS if key in lowered}


class AuditedOpenAICompatibleProvider:
    """v18 transport with retry/request evidence and no credential persistence."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.url = str(config["base_url"]).rstrip("/") + "/chat/completions"
        self.last_call_audit: dict[str, Any] | None = None
        self.call_count = 0

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ProviderResponse:
        payload = {
            "model": self.config["model"], "messages": messages, "tools": tools,
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
            credential = os.environ.get(str(key_env))
            if not credential:
                raise RuntimeError(f"required API key environment variable is unset: {key_env}")
            headers["Authorization"] = f"Bearer {credential}"
        encoded = json.dumps(payload).encode()
        request = urllib.request.Request(self.url, data=encoded, headers=headers, method="POST")
        retries = int(self.config.get("retries", 3))
        started = time.monotonic()
        audit = {
            "provider_call_index": self.call_count,
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "endpoint_origin": str(self.config["base_url"]).rstrip("/"),
            "request_payload_hash": stable_hash(payload),
            "request_bytes": len(encoded),
            "credential_env": key_env,
            "credential_persisted": False,
            "attempts": [],
        }
        self.call_count += 1
        self.last_call_audit = audit
        body: Any = None
        for attempt_index in range(retries + 1):
            attempt_started = time.monotonic()
            attempt: dict[str, Any] = {"attempt_index": attempt_index}
            retry_after = None
            try:
                with urllib.request.urlopen(request, timeout=float(self.config.get("timeout_s", 180))) as response:
                    raw = response.read().decode()
                    attempt.update({
                        "status": "ok", "http_status": int(response.status),
                        "response_headers": _safe_headers(response.headers),
                        "response_bytes": len(raw),
                    })
                body = json.loads(raw)
                attempt["elapsed_s"] = time.monotonic() - attempt_started
                audit["attempts"].append(attempt)
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")
                safe = _safe_headers(exc.headers)
                retry_after = safe.get("retry-after")
                attempt.update({
                    "status": "http_error", "http_status": int(exc.code),
                    "response_headers": safe, "error_body_sha256": stable_hash(detail),
                    "error_body_excerpt": detail[:1000],
                })
                retryable = exc.code in {408, 429, 500, 502, 503, 504}
                if attempt_index >= retries or not retryable:
                    attempt["elapsed_s"] = time.monotonic() - attempt_started
                    audit["attempts"].append(attempt)
                    audit.update({"status": "failed", "elapsed_s": time.monotonic() - started})
                    raise RuntimeError(f"provider HTTP {exc.code}: {detail[:1000]}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                attempt.update({"status": "transport_error", "error_type": type(exc).__name__, "error": str(exc)[:1000]})
                if attempt_index >= retries:
                    attempt["elapsed_s"] = time.monotonic() - attempt_started
                    audit["attempts"].append(attempt)
                    audit.update({"status": "failed", "elapsed_s": time.monotonic() - started})
                    raise RuntimeError(f"provider transport failure: {exc}") from exc
            except json.JSONDecodeError as exc:
                attempt.update({"status": "invalid_json", "error": str(exc)})
                if attempt_index >= retries:
                    attempt["elapsed_s"] = time.monotonic() - attempt_started
                    audit["attempts"].append(attempt)
                    audit.update({"status": "failed", "elapsed_s": time.monotonic() - started})
                    raise RuntimeError(f"provider returned invalid JSON: {exc}") from exc
            attempt["elapsed_s"] = time.monotonic() - attempt_started
            audit["attempts"].append(attempt)
            delay = min(8.0, 0.5 * (2 ** attempt_index))
            if retry_after is not None:
                try:
                    delay = min(30.0, max(delay, float(retry_after)))
                except ValueError:
                    pass
            attempt["retry_delay_s"] = delay
            time.sleep(delay)
        assert body is not None
        try:
            parsed = parse_chat_completion(body)
        except RuntimeError:
            audit.update({"status": "malformed_response", "elapsed_s": time.monotonic() - started})
            raise
        audit.update({
            "status": "ok", "elapsed_s": time.monotonic() - started,
            "reported_model": parsed.raw_model, "usage": parsed.usage,
        })
        return parsed
