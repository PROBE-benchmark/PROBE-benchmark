"""Exact, bounded v20.2 presentation derived only from measured state."""

from __future__ import annotations

import copy
from typing import Any

from laketrace.agents.v20_1_model import present_observation as present_v20_1
from laketrace.env.v20_2 import EnvironmentV20_2


PRESENTATION_MODE_V20_2 = "hierarchical_measured_map_v2"


def _trim_topology(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    dense = list(result.pop("column_depth_count_rows_base36"))
    sparse = []
    for y_index, text in enumerate(dense):
        payload = str(text).strip("0")
        if payload:
            sparse.append([y_index, len(str(text)) - len(str(text).lstrip("0")), payload])
    result["column_depth_count_sparse_rows"] = sparse
    result["column_depth_count_sparse_schema"] = ["y_index", "x_start", "base36_counts"]
    result["column_depth_count_legend"] = (
        "Unlisted columns are land; base36 digits from x_start give z-layer counts."
    )
    return result


def _drop_redundant_last_time(value: dict[str, Any]) -> dict[str, Any]:
    """Remove a value exactly recoverable from mission time and measurement age."""

    result = copy.deepcopy(value)
    cells = result["cells"]
    schema = list(cells["schema"])
    removed = ["last_sample_time_s_min", "last_sample_time_s_max"]
    indices = [schema.index(name) for name in removed]
    cells["schema"] = [name for name in schema if name not in removed]
    cells["rows"] = [
        [item for index, item in enumerate(row) if index not in indices]
        for row in cells["rows"]
    ]
    result["time_encoding"] = (
        "last_time = mission_time - measurement_age; first_time is explicit"
    )
    result["mode"] = PRESENTATION_MODE_V20_2
    return result


def present_observation_v20_2(
    env: EnvironmentV20_2, last_results: list[dict[str, Any]],
) -> dict[str, Any]:
    value = present_v20_1(env, last_results)
    value["navigation_topology"] = _trim_topology(value["navigation_topology"])
    value["cumulative_measurement_state"] = _drop_redundant_last_time(
        value["cumulative_measurement_state"]
    )
    value["measurement_presentation_mode"] = PRESENTATION_MODE_V20_2
    return value


__all__ = ["PRESENTATION_MODE_V20_2", "present_observation_v20_2"]
