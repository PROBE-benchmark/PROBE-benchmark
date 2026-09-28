from __future__ import annotations

import json
from typing import Any

from .openai_compatible import ProviderResponse


class ScriptedProvider:
    """No-model contract adapter used only for clean-clone acceptance tests."""

    def __init__(self, config: dict[str, Any]):
        self.turn = 0
        self.config = config

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ProviderResponse:
        self.turn += 1
        observation = json.loads(messages[-1]["content"].split("\n", 1)[-1])
        if self.turn == 1:
            available = [name for name, item in observation["neighbors"].items() if item["reachable"]]
            if available:
                call = {"id": "script-1", "name": "move", "arguments": json.dumps({"direction": available[0], "steps": 1})}
            else:
                call = {"id": "script-1", "name": "analyze", "arguments": "{}"}
        else:
            call = {"id": "script-2", "name": "submit", "arguments": json.dumps({
                "anomaly": "no_anomaly", "peak_x": None, "peak_y": None, "peak_z": None,
                "mechanism": None, "source_x": None, "source_y": None, "source_z": None,
                "source_candidate_id": None, "decision": "cannot_determine",
            })}
        return ProviderResponse([call], None, {"input_tokens": 0, "output_tokens": 0}, "scripted-contract-adapter")
