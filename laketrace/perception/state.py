from __future__ import annotations

from typing import Any


PRESENTATION_MODES = ("records", "columnar", "sparse_map")
EXACT_NODE_LIMIT = 48
MAP_BINS_X = 8
MAP_BINS_Y = 8
RECENT_DETAIL_ROWS = 12


def _rounded_mapping(value: dict[str, Any], digits: int) -> dict[str, float]:
    return {key: round(float(item), digits) for key, item in value.items()}


def public_measurement(value: dict[str, Any], current_turn: int) -> dict[str, Any]:
    """Render one cumulative, measured-only node without truth or interpolation."""

    count = int(value["visit_count"])
    return {
        "node_index": value["node_index"],
        "x": round(float(value["x"]), 3),
        "y": round(float(value["y"]), 3),
        "z": round(float(value["z"]), 3),
        "phase": value["phase"],
        "measurement_age_turns": current_turn - int(value["turn"]),
        "sample_count": count,
        "channels": _rounded_mapping(
            {key: total / count for key, total in value["channel_sums"].items()}, 4
        ),
        "flow": _rounded_mapping(
            {key: total / count for key, total in value["flow_sums"].items()}, 5
        ),
    }


def _bin(value: float, minimum: float, maximum: float, count: int) -> int:
    if maximum <= minimum:
        return 0
    return min(count - 1, int((value - minimum) / (maximum - minimum) * count))


def _aggregate_rows(
    rows: list[dict[str, Any]], bounds: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            _bin(float(row["x"]), float(bounds["x"]["min"]), float(bounds["x"]["max"]), MAP_BINS_X),
            _bin(float(row["y"]), float(bounds["y"]["min"]), float(bounds["y"]["max"]), MAP_BINS_Y),
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for (x_bin, y_bin), values in sorted(grouped.items()):
        samples = sum(int(row["sample_count"]) for row in values)
        x_width = (float(bounds["x"]["max"]) - float(bounds["x"]["min"])) / MAP_BINS_X
        y_width = (float(bounds["y"]["max"]) - float(bounds["y"]["min"])) / MAP_BINS_Y
        x_center = float(bounds["x"]["min"]) + (x_bin + 0.5) * x_width
        y_center = float(bounds["y"]["min"]) + (y_bin + 0.5) * y_width
        channels = sorted(values[0]["channels"])
        flows = sorted(values[0]["flow"])
        channel_mean = {
            channel: round(
                sum(float(row["channels"][channel]) * int(row["sample_count"]) for row in values) / samples,
                4,
            )
            for channel in channels
        }
        channel_max = {
            channel: round(max(float(row["channels"][channel]) for row in values), 4)
            for channel in channels
        }
        flow_mean = {
            field: round(
                sum(float(row["flow"][field]) * int(row["sample_count"]) for row in values) / samples,
                5,
            )
            for field in flows
        }
        output.append({
            "cell": f"{x_bin},{y_bin}",
            "x_bin": x_bin,
            "y_bin": y_bin,
            "measured_nodes": len(values),
            "sample_count": samples,
            "nearest_measured_node_distance_m": round(min(
                ((float(row["x"]) - x_center) ** 2 + (float(row["y"]) - y_center) ** 2) ** 0.5
                for row in values
            ), 3),
            "measurement_age_turns_min": min(int(row["measurement_age_turns"]) for row in values),
            "measurement_age_turns_max": max(int(row["measurement_age_turns"]) for row in values),
            "phases": sorted({row["phase"] for row in values}),
            "channel_mean": channel_mean,
            "channel_max": channel_max,
            "flow_mean": flow_mean,
        })
    return output


def render_measurements(
    rows: list[dict[str, Any]], mode: str,
    *, bounds: dict[str, dict[str, float]] | None = None,
    current_position: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render measured-only state, aggregating spatially instead of truncating."""

    if mode not in PRESENTATION_MODES:
        raise ValueError(f"presentation_mode must be one of {PRESENTATION_MODES}")
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["node_index"]))
    if len(ordered) > EXACT_NODE_LIMIT:
        if bounds is None or current_position is None:
            raise ValueError("spatial aggregation requires domain bounds and current position")
        values = _aggregate_rows(ordered, bounds)
        current_column = [
            row for row in ordered
            if float(row["x"]) == float(current_position["x"])
            and float(row["y"]) == float(current_position["y"])
        ]
        recent = sorted(
            ordered, key=lambda row: (int(row["measurement_age_turns"]), int(row["node_index"]))
        )[:RECENT_DETAIL_ROWS]
        common = {
            "state_kind": "spatial_aggregate",
            "source_measured_nodes": len(ordered),
            "grid_shape": [MAP_BINS_X, MAP_BINS_Y],
            "unvisited_cells": "omitted",
            "current_column": current_column,
            "recent_detail": recent,
        }
    else:
        values = ordered
        common = {
            "state_kind": "exact_nodes",
            "source_measured_nodes": len(ordered),
        }
    if mode == "records":
        return {"mode": mode, **common, "rows": values}
    if mode == "columnar":
        fields = list(values[0]) if values else []
        return {
            "mode": mode, **common,
            "row_count": len(values),
            "columns": {field: [row[field] for row in values] for field in fields},
        }
    return {
        "mode": mode, **common,
        "nodes": {
            str(row.get("node_index", row.get("cell"))): {
                key: value for key, value in row.items() if key not in {"node_index", "cell"}
            }
            for row in values
        },
    }


def canonical_measurements(value: dict[str, Any]) -> list[dict[str, Any]]:
    """Invert a public presentation so parity can be tested mechanically."""

    mode = value.get("mode")
    if mode == "records":
        rows = [dict(row) for row in value.get("rows") or []]
    elif mode == "columnar":
        columns = value.get("columns") or {}
        lengths = {len(column) for column in columns.values()}
        if len(lengths) > 1:
            raise ValueError("columnar presentation has unequal column lengths")
        count = next(iter(lengths), 0)
        if int(value.get("row_count", count)) != count:
            raise ValueError("columnar presentation row_count mismatch")
        rows = [{field: columns[field][index] for field in columns} for index in range(count)]
    elif mode == "sparse_map":
        rows = [
            ({"node_index": int(node_index)} if value.get("state_kind") == "exact_nodes" else {"cell": node_index}) | dict(row)
            for node_index, row in (value.get("nodes") or {}).items()
        ]
    else:
        raise ValueError(f"unknown measurement presentation mode: {mode!r}")
    if value.get("state_kind") == "spatial_aggregate":
        return sorted(rows, key=lambda row: (int(row["x_bin"]), int(row["y_bin"])))
    return sorted(rows, key=lambda row: int(row["node_index"]))


def derived_summary(
    rows: list[dict[str, Any]], current_position: dict[str, Any],
    directions_to_unvisited_neighbors: list[str], observed_local_gradients: dict[str, Any],
    total_nodes: int,
) -> dict[str, Any]:
    """Arithmetic summaries from the agent's own measured rows only."""

    maxima = {}
    ranges = {}
    if rows:
        for channel in sorted(rows[0]["channels"]):
            best = max(rows, key=lambda row: float(row["channels"][channel]))
            values = [float(row["channels"][channel]) for row in rows]
            dx = float(best["x"]) - float(current_position["x"])
            dy = float(best["y"]) - float(current_position["y"])
            dz = float(best["z"]) - float(current_position["z"])
            maxima[channel] = {
                "value": best["channels"][channel],
                "node_index": best["node_index"],
                "x": best["x"], "y": best["y"], "z": best["z"],
                "relative_dx_m": round(dx, 3),
                "relative_dy_m": round(dy, 3),
                "relative_dz_m": round(dz, 3),
                "distance_m": round((dx * dx + dy * dy + dz * dz) ** 0.5, 3),
            }
            ranges[channel] = {"min": min(values), "max": max(values)}
    return {
        "visited_nodes": len(rows),
        "total_nodes": total_nodes,
        "visited_fraction": round(len(rows) / total_nodes, 6) if total_nodes else 0.0,
        "observed_channel_ranges": ranges,
        "observed_channel_maxima": maxima,
        "observed_local_gradients": observed_local_gradients,
        "directions_to_unvisited_neighbors": sorted(directions_to_unvisited_neighbors),
    }
