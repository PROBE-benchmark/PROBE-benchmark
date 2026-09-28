"""Bounded, measured-only state for the v20.1 hierarchical harness.

The global state is a phase-separated 8 x 8 horizontal map. Values at
different measured depths inside one horizontal cell are pooled arithmetically;
unmeasured locations are never filled. Native 3-D detail is retained for the
current water column and for the last executed route.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from typing import Any

from laketrace.perception.v20_state import summarize_measurement_ledger
from laketrace.util import stable_hash


MAP_BINS_X = 8
MAP_BINS_Y = 8
BIT_ORDER = "node_index_n;byte=floor(n/8);bit=2**(n%8);lsb0"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _ordered_fields(rows: Sequence[Mapping[str, Any]], *names: str) -> list[str]:
    fields: set[str] = set()
    for row in rows:
        value: Any = row
        for name in names:
            value = _mapping(value).get(name)
        fields.update(_mapping(value))
    return sorted(fields)


def _vector(value: Any, order: Sequence[str]) -> list[Any]:
    mapping = _mapping(value)
    return [mapping.get(field) for field in order]


def _bin(value: float, minimum: float, maximum: float, count: int) -> int:
    if maximum <= minimum:
        return 0
    scaled = int((value - minimum) / (maximum - minimum) * count)
    return max(0, min(count - 1, scaled))


def _bin_edges(minimum: float, maximum: float, count: int) -> list[float]:
    if maximum <= minimum:
        return [round(minimum, 3) for _ in range(count + 1)]
    width = (maximum - minimum) / count
    return [round(minimum + index * width, 3) for index in range(count + 1)]


def _present_range(values: Sequence[Any]) -> tuple[float | None, float | None]:
    present = [float(value) for value in values if value is not None]
    return (min(present), max(present)) if present else (None, None)


def compact_route_measurements(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Encode all public route-arrival readings as lossless schema/rows."""

    copied = [dict(row) for row in rows]
    channel_order = sorted({
        field for row in copied for field in _mapping(row.get("channels"))
    })
    flow_order = sorted({
        field for row in copied for field in _mapping(row.get("flow"))
    })
    schema = [
        "sample_sequence", "edge_sequence", "provider_round",
        "mission_time_s", "node_index", "x", "y", "z", "bed_depth_m",
        "depth_layer", "phase", "visit_index", "channels", "flow",
    ]
    encoded = [
        [
            int(row["sample_sequence"]),
            int(row["edge_sequence"]),
            int(row["provider_round"]),
            row["mission_time_s"],
            int(row["node_index"]),
            row["x"],
            row["y"],
            row["z"],
            row["bed_depth_m"],
            (
                "surface"
                if float(row["z"]) / float(row["bed_depth_m"]) < 1 / 3
                else "middle"
                if float(row["z"]) / float(row["bed_depth_m"]) < 2 / 3
                else "bottom"
            ),
            row["phase"],
            int(row["visit_index"]),
            _vector(row.get("channels"), channel_order),
            _vector(row.get("flow"), flow_order),
        ]
        for row in copied
    ]
    return {
        "encoding": "schema_rows_v1",
        "schema": schema,
        "channel_order": channel_order,
        "flow_order": flow_order,
        "row_count": len(encoded),
        "rows": encoded,
    }


def _compact_current_column(
    rows: Sequence[Mapping[str, Any]],
    raw_ledger: Sequence[Mapping[str, Any]],
    current_position: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain native depth rows with only the registered column facts."""

    channel_order = _ordered_fields(rows, "mean", "channels")
    flow_order = _ordered_fields(rows, "mean", "flow")
    del raw_ledger
    schema = [
        "node_index", "z", "bed_depth_m", "depth_layer", "phase",
        "visit_count", "sample_count", "first_sample_time_s",
        "last_sample_time_s", "measurement_age_s",
        "channel_value_mean", "flow_value_mean",
    ]
    encoded = []
    for row in rows:
        mean = _mapping(row.get("mean"))
        encoded.append([
            int(row["node_index"]),
            row["z"], row.get("bed_depth_m"),
            row.get("depth_layer"), row.get("phase"),
            int(row.get("visit_count", row.get("sample_count", 1))),
            int(row.get("sample_count", row.get("visit_count", 1))),
            row.get("first_sample_time_s"), row.get("last_sample_time_s"),
            row.get("measurement_age_s"),
            _vector(mean.get("channels"), channel_order),
            _vector(mean.get("flow"), flow_order),
        ])
    return {
        "encoding": "schema_rows_v1",
        "column_x": current_position["x"],
        "column_y": current_position["y"],
        "schema": schema,
        "channel_order": channel_order,
        "flow_order": flow_order,
        "rows": encoded,
    }


def _visited_bitset(
    summaries: Sequence[Mapping[str, Any]], total_node_count: int,
) -> dict[str, Any]:
    if isinstance(total_node_count, bool) or not isinstance(total_node_count, int):
        raise ValueError("total_node_count must be an integer")
    if total_node_count <= 0:
        raise ValueError("total_node_count must be positive")
    byte_count = (total_node_count + 7) // 8
    grouped: dict[str, set[int]] = {}
    for row in summaries:
        node_index = int(row["node_index"])
        if not 0 <= node_index < total_node_count:
            raise ValueError(
                f"measured node_index outside total_node_count: {node_index}"
            )
        grouped.setdefault(str(row.get("phase", "")), set()).add(node_index)

    encoded: dict[str, str] = {}
    counts: dict[str, int] = {}
    for phase, indices in sorted(grouped.items()):
        value = bytearray(byte_count)
        for node_index in indices:
            value[node_index // 8] |= 1 << (node_index % 8)
        encoded[phase] = base64.b64encode(bytes(value)).decode("ascii")
        counts[phase] = len(indices)
    return {
        "encoding": "base64_fixed_bitset_v1",
        "bit_order": BIT_ORDER,
        "total_node_count": total_node_count,
        "bytes_per_phase": byte_count,
        "set_bit_count_by_phase": counts,
        "base64_by_phase": encoded,
    }


def decode_visited_bitset(value: Mapping[str, Any]) -> dict[str, list[int]]:
    """Decode and validate an exact v20.1 visited-node bitset."""

    if value.get("encoding") != "base64_fixed_bitset_v1":
        raise ValueError("unknown visited bitset encoding")
    if value.get("bit_order") != BIT_ORDER:
        raise ValueError("visited bitset order mismatch")
    total = int(value.get("total_node_count", 0))
    expected_bytes = (total + 7) // 8
    if total <= 0 or int(value.get("bytes_per_phase", -1)) != expected_bytes:
        raise ValueError("visited bitset length metadata mismatch")
    registered_counts = _mapping(value.get("set_bit_count_by_phase"))
    encoded = _mapping(value.get("base64_by_phase"))
    if set(registered_counts) != set(encoded):
        raise ValueError("visited bitset phase metadata mismatch")

    output: dict[str, list[int]] = {}
    for phase, text in sorted(encoded.items()):
        try:
            raw = base64.b64decode(str(text), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("visited bitset contains invalid base64") from exc
        if len(raw) != expected_bytes:
            raise ValueError("visited bitset byte length mismatch")
        if total % 8 and raw and raw[-1] & ~((1 << (total % 8)) - 1):
            raise ValueError("visited bitset has nonzero padding bits")
        indices = [
            node_index
            for node_index in range(total)
            if raw[node_index // 8] & (1 << (node_index % 8))
        ]
        if len(indices) != int(registered_counts[phase]):
            raise ValueError("visited bitset set-bit count mismatch")
        output[str(phase)] = indices
    return output


def _planar_aggregate_rows(
    summaries: Sequence[Mapping[str, Any]],
    raw_ledger: Sequence[Mapping[str, Any]],
    bounds: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    x_min = float(bounds["x"]["min"])
    x_max = float(bounds["x"]["max"])
    y_min = float(bounds["y"]["min"])
    y_max = float(bounds["y"]["max"])

    summary_groups: dict[tuple[int, int, str], list[Mapping[str, Any]]] = {}
    for row in summaries:
        key = (
            _bin(float(row["x"]), x_min, x_max, MAP_BINS_X),
            _bin(float(row["y"]), y_min, y_max, MAP_BINS_Y),
            str(row.get("phase", "")),
        )
        summary_groups.setdefault(key, []).append(row)

    event_groups: dict[tuple[int, int, str], list[Mapping[str, Any]]] = {}
    for event in raw_ledger:
        key = (
            _bin(float(event["x"]), x_min, x_max, MAP_BINS_X),
            _bin(float(event["y"]), y_min, y_max, MAP_BINS_Y),
            str(event.get("phase", "")),
        )
        event_groups.setdefault(key, []).append(event)

    output: list[dict[str, Any]] = []
    for (x_bin, y_bin, phase), nodes in sorted(summary_groups.items()):
        events = event_groups[(x_bin, y_bin, phase)]
        channel_order = sorted({
            field for event in events for field in _mapping(event.get("channels"))
        })
        flow_order = sorted({
            field for event in events for field in _mapping(event.get("flow"))
        })
        channel_values = {
            field: [
                float(_mapping(event.get("channels"))[field])
                for event in events if field in _mapping(event.get("channels"))
            ]
            for field in channel_order
        }
        flow_values = {
            field: [
                float(_mapping(event.get("flow"))[field])
                for event in events if field in _mapping(event.get("flow"))
            ]
            for field in flow_order
        }
        age_min, age_max = _present_range([
            row.get("measurement_age_s") for row in nodes
        ])
        first_min, first_max = _present_range([
            row.get("first_sample_time_s") for row in nodes
        ])
        last_min, last_max = _present_range([
            row.get("last_sample_time_s") for row in nodes
        ])
        output.append({
            "x_bin": x_bin,
            "y_bin": y_bin,
            "phase": phase,
            "measured_nodes": len(nodes),
            "visit_count": sum(int(row.get("visit_count", 1)) for row in nodes),
            "sample_count": len(events),
            "measurement_age_s_min": age_min,
            "measurement_age_s_max": age_max,
            "first_sample_time_s_min": first_min,
            "first_sample_time_s_max": first_max,
            "last_sample_time_s_min": last_min,
            "last_sample_time_s_max": last_max,
            "observed_z_min_m": round(min(float(row["z"]) for row in nodes), 3),
            "observed_z_max_m": round(max(float(row["z"]) for row in nodes), 3),
            "channel_mean": {
                field: round(sum(values) / len(values), 4)
                for field, values in channel_values.items()
            },
            "channel_max": {
                field: round(max(values), 4)
                for field, values in channel_values.items()
            },
            "flow_mean": {
                field: round(sum(values) / len(values), 5)
                for field, values in flow_values.items()
            },
        })
    return output


def _compact_planar_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    channel_order = sorted({
        field
        for row in rows
        for name in ("channel_mean", "channel_max")
        for field in _mapping(row.get(name))
    })
    flow_order = sorted({
        field for row in rows for field in _mapping(row.get("flow_mean"))
    })
    schema = [
        "x_bin", "y_bin", "phase", "measured_nodes", "visit_count",
        "sample_count", "measurement_age_s_min", "measurement_age_s_max",
        "first_sample_time_s_min", "first_sample_time_s_max",
        "last_sample_time_s_min", "last_sample_time_s_max",
        "observed_z_min_m", "observed_z_max_m",
        "channel_mean", "channel_max", "flow_mean",
    ]
    encoded = [
        [
            int(row["x_bin"]), int(row["y_bin"]), row.get("phase"),
            int(row.get("measured_nodes", 0)),
            int(row.get("visit_count", 0)), int(row.get("sample_count", 0)),
            row.get("measurement_age_s_min"), row.get("measurement_age_s_max"),
            row.get("first_sample_time_s_min"), row.get("first_sample_time_s_max"),
            row.get("last_sample_time_s_min"), row.get("last_sample_time_s_max"),
            row.get("observed_z_min_m"), row.get("observed_z_max_m"),
            _vector(row.get("channel_mean"), channel_order),
            _vector(row.get("channel_max"), channel_order),
            _vector(row.get("flow_mean"), flow_order),
        ]
        for row in rows
    ]
    return {
        "encoding": "schema_rows_v1",
        "schema": schema,
        "channel_order": channel_order,
        "flow_order": flow_order,
        "rows": encoded,
    }


def render_compact_measurements(
    rows: list[dict[str, Any]],
    *,
    bounds: dict[str, dict[str, float]],
    current_position: dict[str, Any],
    raw_ledger: Sequence[Mapping[str, Any]],
    current_mission_time_s: float,
    z_layers: Sequence[float],
    total_node_count: int,
) -> dict[str, Any]:
    """Return a bounded, phase-separated measured-only belief map.

    ``rows`` is retained for adapter compatibility, but the append-only ledger
    is the sole value source. The global map is 2-D; the exact current column
    and separately rendered recent route preserve native vertical detail.
    """

    del rows
    ledger = [dict(event) for event in raw_ledger]
    summaries = summarize_measurement_ledger(
        ledger, current_mission_time_s=current_mission_time_s
    )
    bitset = _visited_bitset(summaries, total_node_count)
    cells = _planar_aggregate_rows(summaries, ledger, bounds)
    current_column = [
        row for row in summaries
        if float(row["x"]) == float(current_position["x"])
        and float(row["y"]) == float(current_position["y"])
    ]
    return {
        "mode": "hierarchical_measured_map_v1",
        "state_kind": "planar_aggregate_plus_native_3d_detail",
        "interpolation": "forbidden",
        "unvisited_cells": "omitted",
        "cross_phase_averaging": False,
        "measured_node_phase_rows": len(summaries),
        "visited_mask_node_count": len(summaries),
        "visited_node_bitset_by_phase": bitset,
        "ledger_event_count": len(ledger),
        "ledger_sha256": stable_hash(ledger),
        "grid_shape": [MAP_BINS_X, MAP_BINS_Y],
        "depth_aggregation": "measured_depths_pooled_within_xy_cell",
        "registered_native_z_layer_count": len({float(value) for value in z_layers}),
        "x_bin_edges_m": _bin_edges(
            float(bounds["x"]["min"]), float(bounds["x"]["max"]), MAP_BINS_X
        ),
        "y_bin_edges_m": _bin_edges(
            float(bounds["y"]["min"]), float(bounds["y"]["max"]), MAP_BINS_Y
        ),
        "cells": _compact_planar_rows(cells),
        "current_column": _compact_current_column(
            current_column, ledger, current_position
        ),
    }


__all__ = [
    "BIT_ORDER",
    "compact_route_measurements",
    "decode_visited_bitset",
    "render_compact_measurements",
]
