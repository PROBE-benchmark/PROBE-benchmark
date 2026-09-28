"""V20 truth-blind three-dimensional mission memory representation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


PRESENTATION_MODES = ("records", "columnar", "sparse_map")
EXACT_NODE_LIMIT = 48
MAP_BINS_X = 8
MAP_BINS_Y = 8
MAP_BINS_Z = 3
RECENT_DETAIL_ROWS = 12
AGGREGATE_SCALAR_FIELDS = (
    "x_bin", "y_bin", "z_bin", "z_center_m", "z_band_m", "phase", "measured_nodes",
    "visit_count", "sample_count", "nearest_measured_node_distance_m",
    "measurement_age_s_min", "measurement_age_s_max",
    "measurement_age_turns_min", "measurement_age_turns_max",
    "first_sample_time_s_min", "first_sample_time_s_max",
    "last_sample_time_s_min", "last_sample_time_s_max",
    "observed_z_min_m", "observed_z_max_m", "bed_depth_m_min",
    "bed_depth_m_max", "depth_layers",
)

__all__ = [
    "EXACT_NODE_LIMIT",
    "MAP_BINS_X",
    "MAP_BINS_Y",
    "MAP_BINS_Z",
    "PRESENTATION_MODES",
    "RECENT_DETAIL_ROWS",
    "canonical_measurements",
    "derived_summary",
    "exact_visited_mask",
    "public_measurement",
    "render_measurements",
    "summarize_measurement_ledger",
]


def _rounded_mapping(value: Mapping[str, Any], digits: int) -> dict[str, float]:
    return {key: round(float(item), digits) for key, item in value.items()}


def public_measurement(value: dict[str, Any], current_turn: int) -> dict[str, Any]:
    """Render one cumulative, measured-only node without truth or interpolation."""

    count = int(value["visit_count"])
    output = {
        "node_index": value["node_index"],
        "x": round(float(value["x"]), 3),
        "y": round(float(value["y"]), 3),
        "z": round(float(value["z"]), 3),
        "bed_depth_m": round(float(value["bed_depth_m"]), 3),
        "depth_layer": str(value["depth_layer"]),
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
    # A future environment can retain physical mission time without changing the
    # legacy public row. Turn age is not silently relabelled as seconds.
    for field in (
        "node_id", "first_sample_time_s", "last_sample_time_s",
        "measurement_age_s",
    ):
        if field in value:
            output[field] = value[field]
    return output


def _bin(value: float, minimum: float, maximum: float, count: int) -> int:
    if maximum <= minimum:
        return 0
    scaled = int((value - minimum) / (maximum - minimum) * count)
    return max(0, min(count - 1, scaled))


def _node_sort_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["node_index"]),
        str(row.get("phase", "")),
        str(row.get("node_id", row["node_index"])),
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _stat_component(
    row: Mapping[str, Any], statistic: str, component: str,
) -> dict[str, Any]:
    value = row.get(statistic)
    if isinstance(value, Mapping):
        nested = value.get(component)
        if isinstance(nested, Mapping):
            return dict(nested)
        # Also accept a flat ``mean``/``std`` channel mapping from early
        # prototypes. Flow remains distinguishable via its explicit legacy key.
        if component == "channels" and all(
            not isinstance(item, Mapping) for item in value.values()
        ):
            return dict(value)
    if statistic in {"mean", "latest"}:
        legacy = row.get("channels" if component == "channels" else "flow")
        if isinstance(legacy, Mapping):
            return dict(legacy)
    return {}


def _row_channels(row: Mapping[str, Any]) -> dict[str, Any]:
    return _stat_component(row, "mean", "channels")


def _row_flow(row: Mapping[str, Any]) -> dict[str, Any]:
    return _stat_component(row, "mean", "flow")


def _sample_count(row: Mapping[str, Any]) -> int:
    return int(row.get("sample_count", row.get("visit_count", 1)))


def _optional_number(row: Mapping[str, Any], *fields: str) -> float | None:
    for field in fields:
        if row.get(field) is not None:
            return float(row[field])
    return None


def _factual_depth_layer(z: float, bed_depth_m: float) -> str:
    ratio = z / bed_depth_m if bed_depth_m > 0 else 0.0
    return "surface" if ratio < 1 / 3 else "middle" if ratio < 2 / 3 else "bottom"


def _population_stats(
    events: Sequence[Mapping[str, Any]], component: str,
) -> tuple[dict[str, float], dict[str, float]]:
    fields = sorted({
        field
        for event in events
        for field in _mapping(event.get(component)).keys()
    })
    means: dict[str, float] = {}
    deviations: dict[str, float] = {}
    for field in fields:
        values = [
            float(_mapping(event.get(component))[field])
            for event in events
            if field in _mapping(event.get(component))
        ]
        mean = sum(values) / len(values)
        means[field] = mean
        deviations[field] = math.sqrt(
            sum((value - mean) ** 2 for value in values) / len(values)
        )
    return means, deviations


def _event_order(item: tuple[int, Mapping[str, Any]]) -> tuple[int, float, int]:
    index, event = item
    sequence = event.get("sample_sequence")
    if sequence is not None:
        return (0, float(sequence), index)
    mission_time = event.get("mission_time_s")
    if mission_time is not None:
        return (1, float(mission_time), index)
    return (2, float(index), index)


def summarize_measurement_ledger(
    raw_ledger: Iterable[Mapping[str, Any]],
    *,
    current_mission_time_s: float | None = None,
) -> list[dict[str, Any]]:
    """Reduce measured ledger events to exact node/phase statistics.

    The reduction is arithmetic over recorded samples only. It neither fills
    unvisited nodes nor interpolates between coordinates or depth layers.
    """

    indexed = [(index, dict(event)) for index, event in enumerate(raw_ledger)]
    if not indexed:
        return []
    required = {
        "node_index", "x", "y", "z", "bed_depth_m", "phase", "channels", "flow",
    }
    for index, event in indexed:
        missing = sorted(required - set(event))
        if missing:
            raise ValueError(
                f"raw_ledger event {index} missing required fields: {missing}"
            )
        if not isinstance(event["channels"], Mapping) or not isinstance(
            event["flow"], Mapping
        ):
            raise ValueError(
                f"raw_ledger event {index} channels and flow must be mappings"
            )

    timed = [
        float(event["mission_time_s"])
        for _, event in indexed
        if event.get("mission_time_s") is not None
    ]
    effective_now = (
        float(current_mission_time_s)
        if current_mission_time_s is not None
        else (max(timed) if timed else None)
    )
    grouped: dict[tuple[int, str], list[tuple[int, dict[str, Any]]]] = {}
    for index, event in indexed:
        key = (int(event["node_index"]), str(event["phase"]))
        grouped.setdefault(key, []).append((index, event))

    output: list[dict[str, Any]] = []
    for _, indexed_events in sorted(grouped.items()):
        ordered_events = [
            event for _, event in sorted(indexed_events, key=_event_order)
        ]
        first = ordered_events[0]
        raw_coordinates = (
            float(first["x"]), float(first["y"]), float(first["z"])
        )
        raw_bed_depth = float(first["bed_depth_m"])
        if any(
            (float(event["x"]), float(event["y"]), float(event["z"]))
            != raw_coordinates
            or float(event["bed_depth_m"]) != raw_bed_depth
            for event in ordered_events[1:]
        ):
            raise ValueError(
                "raw_ledger changes coordinates or bed depth for the same node_index and phase"
            )

        channel_mean, channel_std = _population_stats(ordered_events, "channels")
        flow_mean, flow_std = _population_stats(ordered_events, "flow")
        coordinates = tuple(round(value, 3) for value in raw_coordinates)
        latest = ordered_events[-1]
        times = [
            float(event["mission_time_s"])
            for event in ordered_events
            if event.get("mission_time_s") is not None
        ]
        raw_first_time = min(times) if times else None
        raw_last_time = max(times) if times else None
        first_time = round(raw_first_time, 3) if raw_first_time is not None else None
        last_time = round(raw_last_time, 3) if raw_last_time is not None else None
        visit_indices = {
            str(event["visit_index"])
            for event in ordered_events
            if event.get("visit_index") is not None
        }
        visit_count = len(visit_indices) if visit_indices else len(ordered_events)
        output.append({
            "node_index": int(first["node_index"]),
            "node_id": first.get("node_id", int(first["node_index"])),
            "x": coordinates[0],
            "y": coordinates[1],
            "z": coordinates[2],
            "bed_depth_m": round(raw_bed_depth, 3),
            "depth_layer": _factual_depth_layer(raw_coordinates[2], raw_bed_depth),
            "phase": str(first["phase"]),
            "visit_count": visit_count,
            "sample_count": len(ordered_events),
            "first_sample_time_s": first_time,
            "last_sample_time_s": last_time,
            "measurement_age_s": (
                round(max(0.0, effective_now - raw_last_time), 3)
                if effective_now is not None and raw_last_time is not None
                else None
            ),
            "latest": {
                "channels": {
                    key: round(float(value), 4)
                    for key, value in _mapping(latest["channels"]).items()
                },
                "flow": {
                    key: round(float(value), 5)
                    for key, value in _mapping(latest["flow"]).items()
                },
            },
            "mean": {
                "channels": _rounded_mapping(channel_mean, 4),
                "flow": _rounded_mapping(flow_mean, 5),
            },
            "std": {
                "channels": _rounded_mapping(channel_std, 4),
                "flow": _rounded_mapping(flow_std, 5),
            },
        })
    return sorted(output, key=_node_sort_key)


def _looks_like_raw_ledger(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and any(
        "sample_sequence" in row
        or "mission_time_s" in row
        or "visit_index" in row
        for row in rows
    ) and not all("sample_count" in row for row in rows)


def exact_visited_mask(
    rows: Iterable[Mapping[str, Any]],
    *,
    raw_ledger: Iterable[Mapping[str, Any]] | None = None,
    current_mission_time_s: float | None = None,
) -> list[dict[str, Any]]:
    """Return the global exact visited mask; this structure is never binned."""

    source = (
        summarize_measurement_ledger(
            raw_ledger, current_mission_time_s=current_mission_time_s
        )
        if raw_ledger is not None
        else [dict(row) for row in rows]
    )
    mask = []
    for row in source:
        mask.append({
            "node_id": row.get("node_id", row["node_index"]),
            "node_index": int(row["node_index"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "visit_count": int(
                row.get("visit_count", row.get("sample_count", 1))
            ),
            "first_sample_time_s": _optional_number(
                row,
                "first_sample_time_s",
                "first_mission_time_s",
                "first_visit_time_s",
            ),
            "last_sample_time_s": _optional_number(
                row,
                "last_sample_time_s",
                "last_mission_time_s",
                "last_visit_time_s",
            ),
            "phase": row.get("phase"),
        })
    return sorted(mask, key=_node_sort_key)


def _inclusive_ranges(values: Iterable[int]) -> list[list[int]]:
    """Encode an exact integer set as sorted inclusive ranges.

    Node indices are public, stable identifiers in the v20 observation.  This
    lossless representation lets an actor test whether any advertised neighbor
    has been visited without repeating one verbose coordinate record per node.
    Measurement values and ages remain in the exact/recent rows and 3-D map.
    """

    ordered = sorted(set(int(value) for value in values))
    if not ordered:
        return []
    output: list[list[int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        output.append([start, previous])
        start = previous = value
    output.append([start, previous])
    return output


def compact_visited_mask(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Losslessly encode the exact 3-D visited set, separated by phase."""

    grouped: dict[str, list[int]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("phase", "")), []).append(
            int(row["node_index"])
        )
    return {
        "encoding": "inclusive_node_index_ranges_by_phase",
        "membership_rule": "visited iff node_index is inside an inclusive range for the phase",
        "ranges_by_phase": {
            phase: _inclusive_ranges(indices)
            for phase, indices in sorted(grouped.items())
        },
    }


def _depth_layers(
    rows: Sequence[Mapping[str, Any]],
    bounds: Mapping[str, Any],
    explicit: Sequence[float] | None,
) -> list[float]:
    if explicit is not None:
        layers = sorted({float(value) for value in explicit})
    else:
        z_bounds = _mapping(bounds.get("z"))
        registered = z_bounds.get("layers", z_bounds.get("values"))
        layers = (
            sorted({float(value) for value in registered})
            if isinstance(registered, Sequence)
            and not isinstance(registered, (str, bytes))
            else sorted({float(row["z"]) for row in rows})
        )
    missing = sorted({float(row["z"]) for row in rows} - set(layers))
    if missing:
        raise ValueError(f"z_layers omits measured depths: {missing}")
    return layers


def _range(
    values: Iterable[float | int | None],
) -> tuple[float | None, float | None]:
    present = [float(value) for value in values if value is not None]
    return (min(present), max(present)) if present else (None, None)


def _weighted_mean(
    values: Sequence[Mapping[str, Any]], component: str, field: str,
) -> float:
    mapping = _row_channels if component == "channels" else _row_flow
    observed = [row for row in values if field in mapping(row)]
    samples = sum(_sample_count(row) for row in observed)
    if not samples:
        raise ValueError("cannot aggregate a measured field with zero samples")
    return sum(
        float(mapping(row)[field]) * _sample_count(row) for row in observed
    ) / samples


def _aggregate_rows(
    rows: list[dict[str, Any]],
    bounds: dict[str, dict[str, float]],
    *,
    z_layers: Sequence[float] | None = None,
) -> tuple[list[dict[str, Any]], list[float], int]:
    # Validate the registered native depth axis, but do not render one global
    # map per native layer.  That representation grew as 8 x 8 x up-to-25
    # sparse maps and eventually recreated the very observation log this state
    # is intended to replace.  Three absolute-depth bands keep a bounded 3-D
    # global map; the exact current-column profile and recent rows below retain
    # native vertical resolution where it is actionable.
    layers = _depth_layers(rows, bounds, z_layers)
    z_min = float(bounds["z"]["min"])
    z_max = float(bounds["z"]["max"])
    if z_max <= z_min:
        z_edges = [z_min for _ in range(MAP_BINS_Z + 1)]
    else:
        z_width = (z_max - z_min) / MAP_BINS_Z
        z_edges = [z_min + index * z_width for index in range(MAP_BINS_Z + 1)]
    grouped: dict[tuple[int, int, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            _bin(
                float(row["x"]),
                float(bounds["x"]["min"]),
                float(bounds["x"]["max"]),
                MAP_BINS_X,
            ),
            _bin(
                float(row["y"]),
                float(bounds["y"]["min"]),
                float(bounds["y"]["max"]),
                MAP_BINS_Y,
            ),
            _bin(float(row["z"]), z_min, z_max, MAP_BINS_Z),
            str(row.get("phase", "")),
        )
        grouped.setdefault(key, []).append(row)

    output = []
    x_width = (
        float(bounds["x"]["max"]) - float(bounds["x"]["min"])
    ) / MAP_BINS_X
    y_width = (
        float(bounds["y"]["max"]) - float(bounds["y"]["min"])
    ) / MAP_BINS_Y
    for (x_bin, y_bin, z_bin, phase), values in sorted(grouped.items()):
        samples = sum(_sample_count(row) for row in values)
        visits = sum(
            int(row.get("visit_count", _sample_count(row))) for row in values
        )
        x_center = float(bounds["x"]["min"]) + (x_bin + 0.5) * x_width
        y_center = float(bounds["y"]["min"]) + (y_bin + 0.5) * y_width
        z_low = z_edges[z_bin]
        z_high = z_edges[z_bin + 1]
        z_center = (z_low + z_high) / 2.0
        channels = sorted({
            field for row in values for field in _row_channels(row)
        })
        flows = sorted({field for row in values for field in _row_flow(row)})
        channel_mean = {
            channel: round(_weighted_mean(values, "channels", channel), 4)
            for channel in channels
        }
        channel_max = {
            channel: round(
                max(
                    float(_row_channels(row)[channel])
                    for row in values
                    if channel in _row_channels(row)
                ),
                4,
            )
            for channel in channels
        }
        flow_mean = {
            field: round(_weighted_mean(values, "flow", field), 5)
            for field in flows
        }
        age_s_min, age_s_max = _range(
            row.get("measurement_age_s") for row in values
        )
        age_turns_min, age_turns_max = _range(
            row.get("measurement_age_turns") for row in values
        )
        first_time_min, first_time_max = _range(
            row.get("first_sample_time_s") for row in values
        )
        last_time_min, last_time_max = _range(
            row.get("last_sample_time_s") for row in values
        )
        output.append({
            "cell": json.dumps(
                [x_bin, y_bin, z_bin, phase],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "x_bin": x_bin,
            "y_bin": y_bin,
            "z_bin": z_bin,
            "z_center_m": round(z_center, 3),
            "z_band_m": [round(z_low, 3), round(z_high, 3)],
            "observed_z_min_m": round(min(float(row["z"]) for row in values), 3),
            "observed_z_max_m": round(max(float(row["z"]) for row in values), 3),
            "bed_depth_m_min": round(
                min(float(row["bed_depth_m"]) for row in values), 3
            ) if all(row.get("bed_depth_m") is not None for row in values) else None,
            "bed_depth_m_max": round(
                max(float(row["bed_depth_m"]) for row in values), 3
            ) if all(row.get("bed_depth_m") is not None for row in values) else None,
            "depth_layers": sorted({
                str(row["depth_layer"])
                for row in values if row.get("depth_layer") is not None
            }),
            "phase": phase,
            "measured_nodes": len(values),
            "visit_count": visits,
            "sample_count": samples,
            "nearest_measured_node_distance_m": round(
                min(
                    (
                        (float(row["x"]) - x_center) ** 2
                        + (float(row["y"]) - y_center) ** 2
                        + (float(row["z"]) - z_center) ** 2
                    )
                    ** 0.5
                    for row in values
                ),
                3,
            ),
            "measurement_age_s_min": age_s_min,
            "measurement_age_s_max": age_s_max,
            "measurement_age_turns_min": age_turns_min,
            "measurement_age_turns_max": age_turns_max,
            "first_sample_time_s_min": first_time_min,
            "first_sample_time_s_max": first_time_max,
            "last_sample_time_s_min": last_time_min,
            "last_sample_time_s_max": last_time_max,
            "phases": [phase],
            "channel_mean": channel_mean,
            "channel_max": channel_max,
            "flow_mean": flow_mean,
        })
    return output, [round(value, 3) for value in z_edges], len(layers)


def _recent_sort_key(
    row: Mapping[str, Any],
) -> tuple[float, float, int, str]:
    age_s = _optional_number(row, "measurement_age_s")
    age_turns = _optional_number(row, "measurement_age_turns")
    last_time = _optional_number(
        row, "last_sample_time_s", "last_mission_time_s"
    )
    return (
        age_s
        if age_s is not None
        else (age_turns if age_turns is not None else math.inf),
        -(last_time if last_time is not None else -math.inf),
        int(row["node_index"]),
        str(row.get("phase", "")),
    )


def _ledger_hash(raw_ledger: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        [dict(event) for event in raw_ledger],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _render_sparse(
    values: Sequence[Mapping[str, Any]], state_kind: str,
) -> dict[str, dict[str, Any]]:
    if state_kind == "spatial_aggregate":
        return {
            str(row["cell"]): {
                key: value for key, value in row.items() if key != "cell"
            }
            for row in values
        }
    counts = Counter(int(row["node_index"]) for row in values)
    output = {}
    for row in values:
        node_index = int(row["node_index"])
        if counts[node_index] == 1:
            key = str(node_index)
            payload = {
                field: value for field, value in row.items()
                if field != "node_index"
            }
        else:
            key = json.dumps(
                [node_index, str(row.get("phase", ""))],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            payload = dict(row)
        output[key] = payload
    return output


def _render_aggregate_table(
    values: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Encode repeated aggregate keys once while preserving every cell value."""

    channel_order = sorted({
        field
        for row in values
        for component in ("channel_mean", "channel_max")
        for field in _mapping(row.get(component))
    })
    flow_order = sorted({
        field for row in values for field in _mapping(row.get("flow_mean"))
    })
    schema = [
        *AGGREGATE_SCALAR_FIELDS,
        "channel_mean_by_channel_order",
        "channel_max_by_channel_order",
        "flow_mean_by_flow_order",
    ]
    rows = []
    for row in values:
        rows.append([
            *(row.get(field) for field in AGGREGATE_SCALAR_FIELDS),
            [_mapping(row.get("channel_mean")).get(field) for field in channel_order],
            [_mapping(row.get("channel_max")).get(field) for field in channel_order],
            [_mapping(row.get("flow_mean")).get(field) for field in flow_order],
        ])
    return {
        "encoding": "schema_rows_v1",
        "schema": schema,
        "channel_order": channel_order,
        "flow_order": flow_order,
        "rows": rows,
    }


def _decode_aggregate_table(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    if value.get("encoding") != "schema_rows_v1":
        raise ValueError("unknown aggregate table encoding")
    schema = list(value.get("schema") or [])
    expected = [
        *AGGREGATE_SCALAR_FIELDS,
        "channel_mean_by_channel_order",
        "channel_max_by_channel_order",
        "flow_mean_by_flow_order",
    ]
    if schema != expected:
        raise ValueError("aggregate table schema mismatch")
    channel_order = list(value.get("channel_order") or [])
    flow_order = list(value.get("flow_order") or [])
    output = []
    for encoded in value.get("rows") or []:
        if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
            raise ValueError("aggregate table row is not a sequence")
        if len(encoded) != len(schema):
            raise ValueError("aggregate table row width mismatch")
        row = dict(zip(AGGREGATE_SCALAR_FIELDS, encoded[: len(AGGREGATE_SCALAR_FIELDS)]))
        channel_mean, channel_max, flow_mean = encoded[-3:]
        if len(channel_mean) != len(channel_order) or len(channel_max) != len(channel_order):
            raise ValueError("aggregate channel vector width mismatch")
        if len(flow_mean) != len(flow_order):
            raise ValueError("aggregate flow vector width mismatch")
        row.update({
            "cell": json.dumps(
                [row["x_bin"], row["y_bin"], row["z_bin"], row["phase"]],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "phases": [row["phase"]],
            "channel_mean": dict(zip(channel_order, channel_mean)),
            "channel_max": dict(zip(channel_order, channel_max)),
            "flow_mean": dict(zip(flow_order, flow_mean)),
        })
        output.append(row)
    return output


def render_measurements(
    rows: list[dict[str, Any]],
    mode: str,
    *,
    bounds: dict[str, dict[str, float]] | None = None,
    current_position: dict[str, Any] | None = None,
    raw_ledger: Iterable[Mapping[str, Any]] | None = None,
    current_mission_time_s: float | None = None,
    z_layers: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Render a truth-blind 3-D memory state without truncation or interpolation.

    ``rows`` remains the legacy cumulative-summary interface. ``raw_ledger`` is
    an optional forward-compatible event interface; when supplied, exact
    per-node/phase statistics are derived from it, while the raw events remain
    out of the model-facing payload.
    """

    if mode not in PRESENTATION_MODES:
        raise ValueError(f"presentation_mode must be one of {PRESENTATION_MODES}")
    copied_rows = [dict(row) for row in rows]
    ledger = (
        [dict(event) for event in raw_ledger]
        if raw_ledger is not None
        else None
    )
    if ledger is None and _looks_like_raw_ledger(copied_rows):
        ledger, copied_rows = copied_rows, []
    if ledger is not None:
        ordered = summarize_measurement_ledger(
            ledger, current_mission_time_s=current_mission_time_s
        )
    else:
        ordered = sorted(copied_rows, key=_node_sort_key)

    visited_mask_rows = exact_visited_mask(ordered)
    visited_mask = compact_visited_mask(visited_mask_rows)
    common: dict[str, Any] = {
        "source_measured_nodes": len(ordered),
        "visited_mask_kind": "exact_3d_node_index_ranges",
        "visited_mask_node_count": len(visited_mask_rows),
        "visited_mask": visited_mask,
    }
    if ledger is not None:
        common.update({
            "ledger_event_count": len(ledger),
            "ledger_sha256": _ledger_hash(ledger),
        })

    if len(ordered) > EXACT_NODE_LIMIT:
        if bounds is None or current_position is None:
            raise ValueError(
                "spatial aggregation requires domain bounds and current position"
            )
        values, z_bin_edges, registered_z_layer_count = _aggregate_rows(
            ordered, bounds, z_layers=z_layers
        )
        current_column = [
            row for row in ordered
            if float(row["x"]) == float(current_position["x"])
            and float(row["y"]) == float(current_position["y"])
        ]
        recent = sorted(ordered, key=_recent_sort_key)[:RECENT_DETAIL_ROWS]
        common.update({
            "state_kind": "spatial_aggregate",
            "grid_shape": [MAP_BINS_X, MAP_BINS_Y, MAP_BINS_Z],
            "depth_aggregation": "three_absolute_depth_bands",
            "z_bin_edges_m": z_bin_edges,
            "registered_native_z_layer_count": registered_z_layer_count,
            "unvisited_cells": "omitted",
            "interpolation": "forbidden",
            "current_column": sorted(current_column, key=_node_sort_key),
            "recent_detail": recent,
        })
    else:
        values = ordered
        common.update({
            "state_kind": "exact_nodes",
            "interpolation": "forbidden",
        })

    if mode == "records":
        return {"mode": mode, **common, "rows": values}
    if mode == "columnar":
        fields = list(values[0]) if values else []
        return {
            "mode": mode,
            **common,
            "row_count": len(values),
            "columns": {
                field: [row[field] for row in values] for field in fields
            },
        }
    if common["state_kind"] == "spatial_aggregate":
        return {
            "mode": mode,
            **common,
            "cells": _render_aggregate_table(values),
        }
    return {"mode": mode, **common, "nodes": _render_sparse(values, "exact_nodes")}


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
        rows = [
            {field: columns[field][index] for field in columns}
            for index in range(count)
        ]
    elif mode == "sparse_map":
        if value.get("state_kind") == "spatial_aggregate":
            rows = _decode_aggregate_table(_mapping(value.get("cells")))
        else:
            rows = []
            for sparse_key, payload in (value.get("nodes") or {}).items():
                row = dict(payload)
                if "node_index" not in row:
                    row["node_index"] = int(sparse_key)
                rows.append(row)
    else:
        raise ValueError(f"unknown measurement presentation mode: {mode!r}")
    if value.get("state_kind") == "spatial_aggregate":
        return sorted(
            rows,
            key=lambda row: (
                int(row["x_bin"]),
                int(row["y_bin"]),
                int(row.get("z_bin", 0)),
                str(row.get("phase", "")),
            ),
        )
    return sorted(rows, key=_node_sort_key)


def derived_summary(
    rows: list[dict[str, Any]],
    current_position: dict[str, Any],
    directions_to_unvisited_neighbors: list[str],
    observed_local_gradients: dict[str, Any],
    total_nodes: int,
) -> dict[str, Any]:
    """Arithmetic summaries from the agent's own measured rows only."""

    maxima = {}
    ranges = {}
    if rows:
        channels = sorted({
            field for row in rows for field in _row_channels(row)
        })
        for channel in channels:
            eligible = [row for row in rows if channel in _row_channels(row)]
            best = max(
                eligible, key=lambda row: float(_row_channels(row)[channel])
            )
            values = [
                float(_row_channels(row)[channel]) for row in eligible
            ]
            dx = float(best["x"]) - float(current_position["x"])
            dy = float(best["y"]) - float(current_position["y"])
            dz = float(best["z"]) - float(current_position["z"])
            maxima[channel] = {
                "value": _row_channels(best)[channel],
                "node_index": best["node_index"],
                "x": best["x"],
                "y": best["y"],
                "z": best["z"],
                "relative_dx_m": round(dx, 3),
                "relative_dy_m": round(dy, 3),
                "relative_dz_m": round(dz, 3),
                "distance_m": round(
                    (dx * dx + dy * dy + dz * dz) ** 0.5, 3
                ),
            }
            ranges[channel] = {"min": min(values), "max": max(values)}
    visited_nodes = len({int(row["node_index"]) for row in rows})
    return {
        "visited_nodes": visited_nodes,
        "total_nodes": total_nodes,
        "visited_fraction": (
            round(visited_nodes / total_nodes, 6) if total_nodes else 0.0
        ),
        "observed_channel_ranges": ranges,
        "observed_channel_maxima": maxima,
        "observed_local_gradients": observed_local_gradients,
        "directions_to_unvisited_neighbors": sorted(
            directions_to_unvisited_neighbors
        ),
    }
