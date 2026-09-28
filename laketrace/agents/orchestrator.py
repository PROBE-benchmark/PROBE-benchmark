"""Version-neutral public entry point for LakeTrace agent control loops.

The implementation is currently compatibility-backed by the proven v20.1
loop, but successor protocols import this module rather than injecting directly
into a version-named adapter.  The callable is entirely configured by tool,
validation, presentation, execution, memory, and accounting hooks.
"""

from __future__ import annotations

from typing import Any

from laketrace.agents.v20_1_model import run_model_v20_1


def run_agent_loop(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return run_model_v20_1(*args, **kwargs)


__all__ = ["run_agent_loop"]
