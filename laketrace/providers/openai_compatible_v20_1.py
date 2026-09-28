"""Fail-visible OpenAI-compatible transport for the v20.1 overlay.

The frozen provider remains untouched.  This successor fixes Python 3.9's
``socket.timeout`` hierarchy mismatch and deliberately performs no hidden
science-provider retry.  Durable retry/resume policy belongs to the contract
and launcher, where it can be logged and audited.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from laketrace.providers.openai_compatible import (
    ProviderResponse,
    parse_chat_completion,
)
from laketrace.util import stable_hash


def build_request_payload(
    config: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the exact provider payload without performing network I/O.

    Keeping this operation public lets the agent enforce the registered byte
    ceiling over the *complete* request before entering the transport layer.
    The provider uses the same function, so the local gate and wire payload
    cannot drift apart.
    """

    payload: dict[str, Any] = {
        "model": config["model"],
        "messages": messages,
        "tools": tools,
        "tool_choice": "required",
        "temperature": float(config.get("temperature", 1.0)),
        "stream": False,
    }
    if config.get("max_output_tokens"):
        payload["max_tokens"] = int(config["max_output_tokens"])
    extra = config.get("extra_body") or {}
    if not isinstance(extra, dict):
        raise ValueError("extra_body must be an object")
    protected = set(payload) & set(extra)
    if protected:
        raise ValueError(
            f"extra_body may not override standard fields: {sorted(protected)}"
        )
    payload.update(extra)
    return payload


def encode_request_payload(payload: dict[str, Any]) -> bytes:
    """Return the canonical UTF-8 bytes used by the HTTP request."""

    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


class RetriableTransportInterruption(RuntimeError):
    """A request whose remote completion/charge status may be unknown."""

    def __init__(
        self,
        message: str,
        *,
        exception_class: str,
        request_sha256: str,
        request_bytes: int,
        elapsed_s: float,
        timeout_s: float,
    ):
        super().__init__(message)
        self.detail = {
            "classification": "retriable_transport_interruption",
            "charge_status": "unknown",
            "exception_class": exception_class,
            "request_sha256": request_sha256,
            "request_bytes": request_bytes,
            "elapsed_s": round(elapsed_s, 6),
            "timeout_s": timeout_s,
        }


class InvalidProviderResponse(RuntimeError):
    """A received response that cannot be committed as a model action."""

    def __init__(
        self,
        message: str,
        *,
        exception_class: str,
        request_sha256: str,
        request_bytes: int,
        response_bytes: int,
        elapsed_s: float,
    ):
        super().__init__(message)
        self.detail = {
            "classification": "invalid_provider_response",
            "charge_status": "unknown",
            "exception_class": exception_class,
            "request_sha256": request_sha256,
            "request_bytes": request_bytes,
            "response_bytes": response_bytes,
            "elapsed_s": round(elapsed_s, 6),
        }


class OpenAICompatibleProviderV20_1:
    """Native required-tool provider with explicit transport classification."""

    RETRIABLE_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})
    TRANSPORT_EXCEPTIONS = (
        socket.timeout,
        TimeoutError,
        urllib.error.URLError,
        http.client.RemoteDisconnected,
        ConnectionResetError,
        BrokenPipeError,
    )

    @staticmethod
    def _is_retriable_grammar_compile_timeout(
        status_code: int, detail: str,
    ) -> bool:
        """Recognize Bailian's transient grammar compiler timeout.

        The service currently returns this infrastructure failure as HTTP 400
        with ``invalid_parameter_error`` even though retrying the identical
        request succeeds.  Keep the exception deliberately narrow so ordinary
        client/schema 400 responses remain terminal.
        """

        if status_code != 400:
            return False
        try:
            body = json.loads(detail)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        error = body.get("error") if isinstance(body, dict) else None
        if not isinstance(error, dict):
            return False
        code = str(error.get("code") or "").lower()
        message = str(error.get("message") or "").lower()
        return (
            code == "invalid_parameter_error"
            and "grammar" in message
            and "compile timed out" in message
        )

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.url = str(config["base_url"]).rstrip("/") + "/chat/completions"
        if int(config.get("retries", 0)) != 0:
            raise ValueError(
                "v20.1 provider retries must be zero; the append-only contract owns resume"
            )

    def _payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return build_request_payload(self.config, messages, tools)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        payload = self._payload(messages, tools)
        encoded = encode_request_payload(payload)
        request_hash = stable_hash(payload)
        ceiling = self.config.get("max_request_utf8_bytes")
        if ceiling is not None and len(encoded) > int(ceiling):
            raise RuntimeError(
                f"provider request exceeds frozen local byte ceiling: {len(encoded)} > {int(ceiling)}"
            )
        headers = {"Content-Type": "application/json"}
        key_env = self.config.get("api_key_env")
        if key_env:
            key = os.environ.get(str(key_env))
            if not key:
                raise RuntimeError(
                    f"required API key environment variable is unset: {key_env}"
                )
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            self.url, data=encoded, headers=headers, method="POST"
        )
        timeout_s = float(self.config.get("timeout_s", 180))
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            # Some HTTP clients/tests expose a status without a response body.
            # Preserve the original provider status in that case instead of
            # letting ``HTTPError.read`` mask it with a secondary exception.
            try:
                detail = exc.read().decode(errors="replace")
            except (AttributeError, KeyError, OSError):
                detail = ""
            grammar_compile_timeout = self._is_retriable_grammar_compile_timeout(
                exc.code, detail
            )
            if exc.code in self.RETRIABLE_HTTP_CODES or grammar_compile_timeout:
                raise RetriableTransportInterruption(
                    f"provider HTTP {exc.code}",
                    exception_class=(
                        "HTTPError:400:grammar_compile_timeout"
                        if grammar_compile_timeout
                        else f"HTTPError:{exc.code}"
                    ),
                    request_sha256=request_hash,
                    request_bytes=len(encoded),
                    elapsed_s=time.monotonic() - started,
                    timeout_s=timeout_s,
                ) from exc
            raise RuntimeError(f"provider HTTP {exc.code}: {detail[:1000]}") from exc
        except self.TRANSPORT_EXCEPTIONS as exc:
            raise RetriableTransportInterruption(
                "provider transport interrupted",
                exception_class=type(exc).__name__,
                request_sha256=request_hash,
                request_bytes=len(encoded),
                elapsed_s=time.monotonic() - started,
                timeout_s=timeout_s,
            ) from exc
        try:
            body = json.loads(raw.decode())
            return parse_chat_completion(body)
        except (
            IndexError,
            KeyError,
            TypeError,
            UnicodeDecodeError,
            ValueError,
            json.JSONDecodeError,
            RuntimeError,
        ) as exc:
            raise InvalidProviderResponse(
                "provider returned an unusable completion response",
                exception_class=type(exc).__name__,
                request_sha256=request_hash,
                request_bytes=len(encoded),
                response_bytes=len(raw),
                elapsed_s=time.monotonic() - started,
            ) from exc


__all__ = [
    "OpenAICompatibleProviderV20_1",
    "InvalidProviderResponse",
    "RetriableTransportInterruption",
    "build_request_payload",
    "encode_request_payload",
]
