from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from bisect import bisect_left
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

BASELINES = (
    "random", "lawnmower", "budgeted_serpentine", "greedy_gradient",
    "upstream_flow_heuristic", "adjoint_proxy", "detect_then_climb",
)
BASELINE_INFORMATION_TIER = {name: "public_observation_only" for name in BASELINES}
DETECTION_SCORE_THRESHOLD = 3.0
BUDGETED_SERPENTINE_STOP_FRACTION = 0.55

# Frozen public sparse-map profile.  These are presentation-contract values,
# not environment internals.  A parity test pins them to the model renderer so
# a renderer change cannot silently give either actor a different memory view.
EXACT_NODE_LIMIT = 48
MAP_BINS_X = 8
MAP_BINS_Y = 8
MAP_BINS_Z = 3
RECENT_DETAIL_ROWS = 12

DIRECTIONS = ("north", "south", "east", "west", "up", "down")
SWEEP_FALLBACK_ORDER = ("east", "west", "north", "south", "down", "up")
PROBE_ORDER = ("east", "west", "north", "south", "down", "up")
OPPOSITE = {
    "north": "south", "south": "north", "east": "west",
    "west": "east", "up": "down", "down": "up",
}
AGGREGATE_SCALAR_FIELDS = (
    "x_bin", "y_bin", "z_bin", "z_center_m", "z_band_m", "phase",
    "measured_nodes", "visit_count", "sample_count",
    "nearest_measured_node_distance_m", "measurement_age_s_min",
    "measurement_age_s_max", "measurement_age_turns_min",
    "measurement_age_turns_max", "first_sample_time_s_min",
    "first_sample_time_s_max", "last_sample_time_s_min",
    "last_sample_time_s_max", "observed_z_min_m", "observed_z_max_m",
    "bed_depth_m_min", "bed_depth_m_max", "depth_layers",
)


class PublicBaselineEnvironment(Protocol):
    """The narrow runtime surface shared by model and classical actors.

    A baseline may obtain information only through ``observation_delta`` and may
    affect the episode only through public actions. ``terminated`` is control
    state used by the adapter, not an input to a policy decision.
    """

    terminated: bool

    def observation_delta(
        self, since_sample_sequence: int = 0, *, include_static: bool = False,
    ) -> dict[str, Any]: ...

    def move(self, direction: str) -> dict[str, Any]: ...

    def submit(self, answer: dict[str, Any]) -> dict[str, Any]: ...


def _rows(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    direct = observation.get("cumulative_measurements")
    if direct is not None:
        # Compatibility for direct controller tests and pre-delta public views.
        return list(direct)
    state = observation.get("cumulative_measurement_state", {})
    registered: list[dict[str, Any]] = []
    if state.get("state_kind") == "exact_nodes":
        for key, payload in state.get("nodes", {}).items():
            row = dict(payload)
            if "node_index" not in row:
                row["node_index"] = int(key)
            registered.append(row)
    elif state.get("state_kind") == "spatial_aggregate":
        by_identity: dict[tuple[int, str], dict[str, Any]] = {}
        for row in [
            *state.get("recent_detail", ()), *state.get("current_column", ()),
        ]:
            by_identity[(int(row["node_index"]), str(row.get("phase", "")))] = dict(row)
        registered = list(by_identity.values())
    output = []
    for row in registered:
        mean = row.get("mean", {})
        latest = row.get("latest", {})
        deviation = row.get("std", {})
        channels = dict(mean.get("channels", row.get("channels", {})))
        flow = dict(mean.get("flow", row.get("flow", {})))
        value = dict(row)
        value.update({
            "channels": channels,
            "flow": flow,
            "channel_stats": {
                channel: {
                    "latest": latest.get("channels", {}).get(channel, number),
                    "mean": number,
                    "std": deviation.get("channels", {}).get(channel, 0.0),
                } for channel, number in channels.items()
            },
            "flow_stats": {
                field: {
                    "latest": latest.get("flow", {}).get(field, number),
                    "mean": number,
                    "std": deviation.get("flow", {}).get(field, 0.0),
                } for field, number in flow.items()
            },
        })
        output.append(value)
    return output


def _node_id(observation: Mapping[str, Any]) -> int:
    return int(observation["position"]["node_index"])


def _current_row(observation: Mapping[str, Any]) -> dict[str, Any]:
    current = _node_id(observation)
    for row in _rows(observation):
        if int(row["node_index"]) == current:
            return row
    raise ValueError("public observation does not contain its current measurement")


def _depth_layer(observation: Mapping[str, Any], row: Mapping[str, Any]) -> str | None:
    explicit = row.get("depth_layer")
    if explicit in {"surface", "middle", "bottom"}:
        return str(explicit)
    bed = row.get("bed_depth_m", row.get("bed_depth"))
    if bed is None and int(row.get("node_index", -1)) == _node_id(observation):
        position = observation.get("position", {})
        bed = position.get(
            "bed_depth_m",
            position.get("bed_depth", observation.get("current_bed_depth_m")),
        )
    if bed is None:
        return None
    bed_value = float(bed)
    ratio = float(row["z"]) / bed_value if bed_value > 0 else 0.0
    return "surface" if ratio < 1 / 3 else "middle" if ratio < 2 / 3 else "bottom"


def _calibration_entry(
    observation: Mapping[str, Any], channel: str, layer: str | None,
) -> tuple[float, float] | None:
    """Read registered instrument calibration from its public observation field.

    The primary v20 shape is ``calibration[channel][depth_layer]``. The small
    compatibility branches accept equivalent row-oriented/public renderings so
    policy code stays decoupled from presentation formatting.
    """

    raw: Any = observation.get("calibration", observation.get("measurement_calibration"))
    if not raw:
        return None
    if isinstance(raw, Mapping) and "detection_limits" in raw:
        raw = raw["detection_limits"]
    if isinstance(raw, Mapping) and "channels" in raw:
        raw = raw["channels"]
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if not isinstance(item, Mapping) or str(item.get("channel")) != channel:
                continue
            item_layer = item.get("depth_layer")
            if layer is not None and item_layer not in {None, layer}:
                continue
            if "background_median" in item and "declared_lod" in item:
                return float(item["background_median"]), float(item["declared_lod"])
        return None
    if not isinstance(raw, Mapping):
        return None
    candidate: Any = raw.get(channel)
    if candidate is None and layer is not None and isinstance(raw.get(layer), Mapping):
        candidate = raw[layer].get(channel)
    if isinstance(candidate, Mapping) and layer is not None and layer in candidate:
        candidate = candidate[layer]
    if isinstance(candidate, Mapping) and "depth_layers" in candidate and layer is not None:
        candidate = candidate["depth_layers"].get(layer)
    if isinstance(candidate, Mapping) and {
        "background_median", "declared_lod",
    } <= set(candidate):
        return float(candidate["background_median"]), float(candidate["declared_lod"])
    return None


def _fallback_channel_scales(
    observation: Mapping[str, Any],
) -> dict[str, tuple[float, float]]:
    rows = _rows(observation)
    channels = observation.get("available_channels") or (
        tuple(rows[0].get("channels", ())) if rows else ()
    )
    output: dict[str, tuple[float, float]] = {}
    for channel in channels:
        values = [float(row["channels"][channel]) for row in rows]
        if not values:
            continue
        center = statistics.median(values)
        deviations = [abs(value - center) for value in values]
        mad = statistics.median(deviations)
        # Registered v20 runs expose calibration. This observation-only
        # fallback exists for older public observations and controller tests.
        scale = max(1.4826 * mad, 1e-12)
        output[str(channel)] = (center, scale)
    return output


def _row_scores(
    observation: Mapping[str, Any], row: Mapping[str, Any],
    fallback: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    output: dict[str, float] = {}
    layer = _depth_layer(observation, row)
    for channel, value in row.get("channels", {}).items():
        calibrated = _calibration_entry(observation, str(channel), layer)
        if calibrated is None and layer is None:
            # Aggregated cells intentionally omit column bathymetry.  Use the
            # most conservative score supported by any registered layer,
            # rather than estimating a layer from hidden historical bed depth.
            candidates = {
                entry for candidate_layer in ("surface", "middle", "bottom")
                if (entry := _calibration_entry(
                    observation, str(channel), candidate_layer,
                )) is not None
            }
            if candidates:
                output[str(channel)] = min(
                    abs(float(value) - center) / max(abs(scale), 1e-12)
                    for center, scale in candidates
                )
                continue
        if calibrated is None and fallback is None:
            fallback = _fallback_channel_scales(observation)
        center, scale = calibrated or (fallback or {}).get(str(channel), (0.0, 1.0))
        output[str(channel)] = abs(float(value) - center) / max(abs(scale), 1e-12)
    return output


def _strongest(
    observation: Mapping[str, Any],
) -> tuple[float, dict[str, Any], dict[str, float]]:
    scored = []
    fallback = _fallback_channel_scales(observation)
    for row in [*_evidence_rows(observation), *_derived_maximum_rows(observation)]:
        channel_scores = _row_scores(observation, row, fallback)
        scored.append((max(channel_scores.values(), default=0.0), row, channel_scores))
    if not scored:
        raise ValueError("public observation contains no cumulative measurements")
    return max(scored, key=lambda item: (
        item[0], int(int(item[1]["node_index"]) >= 0), -int(item[1]["node_index"]),
    ))


def _evidence_rows(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return only evidence represented in the registered sparse presentation."""

    exact = _rows(observation)
    state = observation.get("cumulative_measurement_state", {})
    if state.get("state_kind") != "spatial_aggregate":
        return exact
    bounds = observation.get("domain_bounds", {})
    x_bounds, y_bounds = bounds.get("x", {}), bounds.get("y", {})
    x_width = (
        float(x_bounds.get("max", 0.0)) - float(x_bounds.get("min", 0.0))
    ) / MAP_BINS_X
    y_width = (
        float(y_bounds.get("max", 0.0)) - float(y_bounds.get("min", 0.0))
    ) / MAP_BINS_Y
    table = state.get("cells", {})
    schema = list(table.get("schema", ()))
    channel_order = list(table.get("channel_order", ()))
    flow_order = list(table.get("flow_order", ()))
    aggregate = []
    for index, encoded in enumerate(table.get("rows", ()), start=1):
        if len(encoded) != len(schema):
            continue
        cell = dict(zip(AGGREGATE_SCALAR_FIELDS, encoded))
        cell["channel_max"] = dict(zip(channel_order, encoded[-2]))
        cell["flow_mean"] = dict(zip(flow_order, encoded[-1]))
        aggregate.append({
            "node_index": -index,
            "x": float(x_bounds.get("min", 0.0))
            + (int(cell["x_bin"]) + 0.5) * x_width,
            "y": float(y_bounds.get("min", 0.0))
            + (int(cell["y_bin"]) + 0.5) * y_width,
            "z": float(cell["z_center_m"]),
            "phase": cell.get("phase", observation.get("phase", "F1")),
            "sample_count": int(cell.get("sample_count", 1)),
            "channels": dict(cell.get("channel_max", {})),
            "flow": dict(cell.get("flow_mean", {})),
        })
    return exact + aggregate


def _derived_maximum_rows(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recover exact public maxima coordinates from the registered summary."""

    evidence = _evidence_rows(observation)
    exact = {int(row["node_index"]): row for row in _rows(observation)}
    maxima = observation.get(
        "derived_measurement_summary", {},
    ).get("observed_channel_maxima", {})
    output = []
    for channel, maximum in maxima.items():
        if not isinstance(maximum, Mapping) or "value" not in maximum:
            continue
        node = int(maximum["node_index"])
        template = exact.get(node)
        if template is None and evidence:
            template = min(evidence, key=lambda row: (
                (float(row["x"]) - float(maximum["x"])) ** 2
                + (float(row["y"]) - float(maximum["y"])) ** 2
                + (float(row["z"]) - float(maximum["z"])) ** 2,
                int(row["node_index"]),
            ))
        if template is None:
            continue
        row = dict(template)
        row.update({
            "node_index": node,
            "x": float(maximum["x"]),
            "y": float(maximum["y"]),
            "z": float(maximum["z"]),
            "channels": {str(channel): float(maximum["value"])},
        })
        output.append(row)
    return output


def infer_submission(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Infer an answer solely from the public cumulative observation."""

    public = _evidence_rows(observation)
    if not public:
        return {
            "anomaly": "no_anomaly", "peak_position": None, "mechanism": None,
            "source_position": None, "decision": "cannot_determine",
        }
    strongest, strongest_row, channel_scores = _strongest(observation)
    primary = max(channel_scores, key=channel_scores.get)
    # ``_strongest`` includes the exact public maxima coordinates carried by
    # the model-equivalent derived summary, so an aggregated cell center is
    # never substituted for a measured peak node.
    peak_row = strongest_row
    anomaly = strongest >= DETECTION_SCORE_THRESHOLD
    if primary == "turb":
        layer = _depth_layer(observation, strongest_row)
        mechanism = "physical_resuspension" if layer == "bottom" else "external_diffuse"
    elif primary == "fdom":
        mechanism = "external_point"
    elif primary == "cond":
        mechanism = "internal_release"
    else:
        mechanism = "external_diffuse"
    flow = peak_row["flow"]
    u, v = float(flow["u_m_s"]), float(flow["v_m_s"])
    magnitude = math.hypot(u, v)
    offset = min(30.0, magnitude * 300.0)
    if magnitude > 1e-9:
        sx = float(peak_row["x"]) - offset * u / magnitude
        sy = float(peak_row["y"]) - offset * v / magnitude
    else:
        sx, sy = float(peak_row["x"]), float(peak_row["y"])
    source_row = min(
        public,
        key=lambda row: (
            math.hypot(float(row["x"]) - sx, float(row["y"]) - sy)
            + abs(float(row["z"]) - float(peak_row["z"])),
            int(row["node_index"]),
        ),
    )
    return {
        "anomaly": "anomaly" if anomaly else "no_anomaly",
        "peak_position": {
            key: float(peak_row[key]) for key in ("x", "y", "z")
        } if anomaly else None,
        "mechanism": mechanism if anomaly else None,
        "source_position": {
            key: float(source_row[key]) for key in ("x", "y", "z")
        } if anomaly else None,
        "decision": "declare" if anomaly else "cannot_determine",
    }


class PublicSparseObservationMemory:
    """Incrementally build the model-facing sparse memory from public deltas.

    Exact node summaries are retained inside this presentation adapter only so
    changed cells can be updated. Once the registered 48-node threshold is
    crossed, the policy receives only aggregate cells, the current column,
    recent detail, derived summaries, and the exact compressed visited mask—the
    same semantic memory classes exposed to a sparse-map model.
    """

    STATIC_FIELDS = (
        "domain_bounds", "navigation_topology", "calibration",
        "candidate_catalog_semantics", "candidates",
        "available_channels",
    )

    def __init__(self) -> None:
        self.static: dict[str, Any] = {}
        self.nodes: dict[int, dict[str, Any]] = {}
        self.node_cells: dict[int, tuple[int, int, int, str]] = {}
        self.cell_members: dict[tuple[int, int, int, str], set[int]] = defaultdict(set)
        self.cell_cache: dict[tuple[int, int, int, str], dict[str, Any]] = {}
        self.node_columns: dict[int, tuple[float, float]] = {}
        self.column_members: dict[tuple[float, float], set[int]] = defaultdict(set)
        self.recent: OrderedDict[int, None] = OrderedDict()
        self.visited_ranges: dict[str, list[list[int]]] = {}
        self.registered_native_z_layer_count = 0
        self.z_bin_edges: list[float] = []
        self.ledger_event_count = 0
        self._ledger_hash = hashlib.sha256(b"[")

    @staticmethod
    def _axis_values(topology: Mapping[str, Any], axis: str) -> list[float]:
        rule = topology.get("axes", {}).get(axis, {})
        if "values_m" in rule:
            return [float(value) for value in rule["values_m"]]
        count = int(rule.get("count", 0))
        if count < 1:
            return []
        origin = float(rule.get("origin_m", 0.0))
        spacing = rule.get("nominal_spacing_m")
        return [
            origin if spacing is None else origin + index * float(spacing)
            for index in range(count)
        ]

    def _add_visited(self, phase: str, node: int) -> None:
        ranges = self.visited_ranges.setdefault(phase, [])
        starts = [item[0] for item in ranges]
        index = bisect_left(starts, node)
        if index > 0 and ranges[index - 1][0] <= node <= ranges[index - 1][1]:
            return
        if index < len(ranges) and ranges[index][0] <= node <= ranges[index][1]:
            return
        merge_left = index > 0 and ranges[index - 1][1] + 1 == node
        merge_right = index < len(ranges) and ranges[index][0] - 1 == node
        if merge_left and merge_right:
            ranges[index - 1][1] = ranges[index][1]
            ranges.pop(index)
        elif merge_left:
            ranges[index - 1][1] = node
        elif merge_right:
            ranges[index][0] = node
        else:
            ranges.insert(index, [node, node])

    @staticmethod
    def _bin(value: float, minimum: float, maximum: float, count: int) -> int:
        if maximum <= minimum:
            return 0
        scaled = int((value - minimum) / (maximum - minimum) * count)
        return max(0, min(count - 1, scaled))

    def _cell_key(self, row: Mapping[str, Any]) -> tuple[int, int, int, str]:
        bounds = self.static["domain_bounds"]
        return (
            self._bin(
                float(row["x"]), float(bounds["x"]["min"]),
                float(bounds["x"]["max"]), MAP_BINS_X,
            ),
            self._bin(
                float(row["y"]), float(bounds["y"]["min"]),
                float(bounds["y"]["max"]), MAP_BINS_Y,
            ),
            self._bin(
                float(row["z"]), float(bounds["z"]["min"]),
                float(bounds["z"]["max"]), MAP_BINS_Z,
            ),
            str(row.get("phase", "")),
        )

    @staticmethod
    def _column_key(row: Mapping[str, Any]) -> tuple[float, float]:
        return round(float(row["x"]), 3), round(float(row["y"]), 3)

    def _recompute_cell(self, key: tuple[int, int, int, str]) -> None:
        values = [self.nodes[node] for node in sorted(self.cell_members[key])]
        if not values:
            self.cell_cache.pop(key, None)
            return
        x_bin, y_bin, z_bin, phase = key
        samples = sum(int(row.get("sample_count", 1)) for row in values)
        visits = sum(int(row.get("visit_count", row.get("sample_count", 1))) for row in values)
        channels = sorted({field for row in values for field in row.get("channels", {})})
        flows = sorted({field for row in values for field in row.get("flow", {})})
        channel_mean = {
            channel: round(sum(
                float(row["channels"][channel]) * int(row.get("sample_count", 1))
                for row in values if channel in row.get("channels", {})
            ) / sum(
                int(row.get("sample_count", 1))
                for row in values if channel in row.get("channels", {})
            ), 4)
            for channel in channels
        }
        channel_argmax = {
            channel: max(
                (row for row in values if channel in row.get("channels", {})),
                key=lambda row: (
                    float(row["channels"][channel]), -int(row["node_index"]),
                ),
            )
            for channel in channels
        }
        channel_max = {
            channel: round(float(channel_argmax[channel]["channels"][channel]), 4)
            for channel in channels
        }
        channel_min = {
            channel: round(min(
                float(row["channels"][channel])
                for row in values if channel in row.get("channels", {})
            ), 4)
            for channel in channels
        }
        flow_mean = {
            name: round(sum(
                float(row["flow"][name]) * int(row.get("sample_count", 1))
                for row in values if name in row.get("flow", {})
            ) / sum(
                int(row.get("sample_count", 1))
                for row in values if name in row.get("flow", {})
            ), 5)
            for name in flows
        }
        bounds = self.static["domain_bounds"]
        x_width = (float(bounds["x"]["max"]) - float(bounds["x"]["min"])) / MAP_BINS_X
        y_width = (float(bounds["y"]["max"]) - float(bounds["y"]["min"])) / MAP_BINS_Y
        x_center = float(bounds["x"]["min"]) + (x_bin + 0.5) * x_width
        y_center = float(bounds["y"]["min"]) + (y_bin + 0.5) * y_width
        z_low, z_high = self.z_bin_edges[z_bin:z_bin + 2]
        z_center = (z_low + z_high) / 2.0
        first_times = [float(row["first_sample_time_s"]) for row in values]
        last_times = [float(row["last_sample_time_s"]) for row in values]
        self.cell_cache[key] = {
            "x_bin": x_bin, "y_bin": y_bin, "z_bin": z_bin,
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
                str(row.get("depth_layer") or (
                    "surface" if float(row["z"]) / float(row["bed_depth_m"]) < 1 / 3
                    else "middle" if float(row["z"]) / float(row["bed_depth_m"]) < 2 / 3
                    else "bottom"
                ))
                for row in values if row.get("bed_depth_m") is not None
            }),
            "phase": phase,
            "measured_nodes": len(values), "visit_count": visits,
            "sample_count": samples,
            "nearest_measured_node_distance_m": round(min(
                math.sqrt(
                    (float(row["x"]) - x_center) ** 2
                    + (float(row["y"]) - y_center) ** 2
                    + (float(row["z"]) - z_center) ** 2
                ) for row in values
            ), 3),
            "first_sample_time_s_min": min(first_times),
            "first_sample_time_s_max": max(first_times),
            "last_sample_time_s_min": min(last_times),
            "last_sample_time_s_max": max(last_times),
            "phases": [phase], "channel_mean": channel_mean,
            "channel_max": channel_max, "flow_mean": flow_mean,
            "measurement_age_turns_min": None,
            "measurement_age_turns_max": None,
            "_channel_min": channel_min, "_channel_argmax": channel_argmax,
        }

    @staticmethod
    def _registered_row(row: Mapping[str, Any], now: float) -> dict[str, Any]:
        """Render the same exact-node semantics as the model sparse renderer."""

        channel_stats = row.get("channel_stats", {})
        flow_stats = row.get("flow_stats", {})
        bed_depth = float(row["bed_depth_m"])
        ratio = float(row["z"]) / bed_depth if bed_depth > 0 else 0.0
        depth_layer = row.get("depth_layer") or (
            "surface" if ratio < 1 / 3 else "middle" if ratio < 2 / 3 else "bottom"
        )
        return {
            "node_index": int(row["node_index"]),
            "node_id": row.get("node_id", int(row["node_index"])),
            "x": float(row["x"]), "y": float(row["y"]), "z": float(row["z"]),
            "bed_depth_m": bed_depth,
            "depth_layer": str(depth_layer),
            "phase": str(row.get("phase", "")),
            "visit_count": int(row.get("visit_count", row.get("sample_count", 1))),
            "sample_count": int(row.get("sample_count", 1)),
            "first_sample_time_s": float(row["first_sample_time_s"]),
            "last_sample_time_s": float(row["last_sample_time_s"]),
            "measurement_age_s": round(
                max(0.0, now - float(row["last_sample_time_s"])), 3,
            ),
            "latest": {
                "channels": {
                    key: value.get("latest", row.get("channels", {}).get(key))
                    for key, value in channel_stats.items()
                },
                "flow": {
                    key: value.get("latest", row.get("flow", {}).get(key))
                    for key, value in flow_stats.items()
                },
            },
            "mean": {
                "channels": dict(row.get("channels", {})),
                "flow": dict(row.get("flow", {})),
            },
            "std": {
                "channels": {
                    key: value.get("std", 0.0) for key, value in channel_stats.items()
                },
                "flow": {
                    key: value.get("std", 0.0) for key, value in flow_stats.items()
                },
            },
        }

    def _record_public_samples(self, samples: list[Mapping[str, Any]]) -> None:
        for sample in samples:
            if self.ledger_event_count:
                self._ledger_hash.update(b",")
            self._ledger_hash.update(json.dumps(
                dict(sample), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ).encode("utf-8"))
            self.ledger_event_count += 1

    def _ledger_digest(self) -> str:
        digest = self._ledger_hash.copy()
        digest.update(b"]")
        return digest.hexdigest()

    def ingest(self, delta: Mapping[str, Any]) -> dict[str, Any]:
        for field in self.STATIC_FIELDS:
            if field in delta:
                self.static[field] = delta[field]
        if (
            not self.registered_native_z_layer_count
            and "navigation_topology" in self.static
        ):
            self.registered_native_z_layer_count = len(
                self._axis_values(self.static["navigation_topology"], "z")
            )
        if "domain_bounds" not in self.static:
            raise ValueError("first public observation delta must include static fields")
        if not self.z_bin_edges:
            z_bounds = self.static["domain_bounds"]["z"]
            z_min, z_max = float(z_bounds["min"]), float(z_bounds["max"])
            if z_max <= z_min:
                self.z_bin_edges = [z_min for _ in range(MAP_BINS_Z + 1)]
            else:
                width = (z_max - z_min) / MAP_BINS_Z
                self.z_bin_edges = [
                    z_min + index * width for index in range(MAP_BINS_Z + 1)
                ]

        dirty: set[tuple[int, int, int, str]] = set()
        for summary in delta.get("changed_node_summaries", ()):
            row = dict(summary)
            node = int(row["node_index"])
            previous_key = self.node_cells.get(node)
            key = self._cell_key(row)
            if previous_key is not None and previous_key != key:
                self.cell_members[previous_key].discard(node)
                dirty.add(previous_key)
            is_new = node not in self.nodes
            self.nodes[node] = row
            self.node_cells[node] = key
            self.cell_members[key].add(node)
            previous_column = self.node_columns.get(node)
            column = self._column_key(row)
            if previous_column is not None and previous_column != column:
                self.column_members[previous_column].discard(node)
            self.node_columns[node] = column
            self.column_members[column].add(node)
            dirty.add(key)
            if is_new:
                self._add_visited(str(row.get("phase", "")), node)
        samples = sorted(
            delta.get("new_samples", ()), key=lambda row: int(row["sample_sequence"]),
        )
        self._record_public_samples(samples)
        for sample in samples:
            node = int(sample["node_index"])
            self.recent.pop(node, None)
            self.recent[node] = None
        for key in dirty:
            self._recompute_cell(key)
        return self._policy_observation(delta)

    def _cell_payload(
        self, value: Mapping[str, Any], now: float,
    ) -> dict[str, Any]:
        output = {
            key: item for key, item in value.items() if not key.startswith("_")
        }
        output["measurement_age_s_min"] = round(
            max(0.0, now - float(value["last_sample_time_s_max"])), 3,
        )
        output["measurement_age_s_max"] = round(
            max(0.0, now - float(value["last_sample_time_s_min"])), 3,
        )
        return output

    def _aggregate_table(self, now: float) -> dict[str, Any]:
        values = [
            self._cell_payload(cell, now)
            for _, cell in sorted(self.cell_cache.items())
        ]
        channel_order = sorted({
            channel for row in values
            for component in ("channel_mean", "channel_max")
            for channel in row.get(component, {})
        })
        flow_order = sorted({
            field for row in values for field in row.get("flow_mean", {})
        })
        schema = [
            *AGGREGATE_SCALAR_FIELDS,
            "channel_mean_by_channel_order",
            "channel_max_by_channel_order",
            "flow_mean_by_flow_order",
        ]
        return {
            "encoding": "schema_rows_v1",
            "schema": schema,
            "channel_order": channel_order,
            "flow_order": flow_order,
            "rows": [[
                *(row.get(field) for field in AGGREGATE_SCALAR_FIELDS),
                [row.get("channel_mean", {}).get(field) for field in channel_order],
                [row.get("channel_max", {}).get(field) for field in channel_order],
                [row.get("flow_mean", {}).get(field) for field in flow_order],
            ] for row in values],
        }

    def _derived_summary(
        self, delta: Mapping[str, Any], now: float,
    ) -> dict[str, Any]:
        maxima: dict[str, Any] = {}
        ranges: dict[str, Any] = {}
        channels = sorted({
            channel for cell in self.cell_cache.values()
            for channel in cell["channel_max"]
        })
        position = delta["position"]
        for channel in channels:
            cells = [cell for cell in self.cell_cache.values() if channel in cell["channel_max"]]
            best_cell = max(cells, key=lambda cell: float(cell["channel_max"][channel]))
            best = best_cell["_channel_argmax"][channel]
            dx = float(best["x"]) - float(position["x"])
            dy = float(best["y"]) - float(position["y"])
            dz = float(best["z"]) - float(position["z"])
            maxima[channel] = {
                "value": best["channels"][channel],
                "node_index": best["node_index"],
                "x": best["x"], "y": best["y"], "z": best["z"],
                "relative_dx_m": round(dx, 3),
                "relative_dy_m": round(dy, 3),
                "relative_dz_m": round(dz, 3),
                "distance_m": round(math.sqrt(dx * dx + dy * dy + dz * dz), 3),
            }
            ranges[channel] = {
                "min": min(float(cell["_channel_min"][channel]) for cell in cells),
                "max": max(float(cell["channel_max"][channel]) for cell in cells),
            }
        return {
            "visited_nodes": len(self.nodes),
            "total_nodes": int(delta["n_total_nodes"]),
            "visited_fraction": round(
                len(self.nodes) / int(delta["n_total_nodes"]), 6,
            ),
            "observed_channel_ranges": ranges,
            "observed_channel_maxima": maxima,
            "observed_local_gradients": {},
            "directions_to_unvisited_neighbors": sorted(
                delta.get("directions_to_unvisited_neighbors", ())
            ),
        }

    def _policy_observation(self, delta: Mapping[str, Any]) -> dict[str, Any]:
        now = float(delta["mission_time_s"])
        current = delta["position"]
        if len(self.nodes) <= EXACT_NODE_LIMIT:
            visible_ids = set(self.nodes)
            state_kind = "exact_nodes"
        else:
            recent_ids = list(reversed(self.recent))[:RECENT_DETAIL_ROWS]
            column_ids = self.column_members.get(self._column_key(current), set())
            visible_ids = set(recent_ids) | column_ids
            state_kind = "spatial_aggregate"
        mask = {
            "encoding": "inclusive_node_index_ranges_by_phase",
            "membership_rule": (
                "visited iff node_index is inside an inclusive range for the phase"
            ),
            "ranges_by_phase": {
                phase: [list(item) for item in ranges]
                for phase, ranges in sorted(self.visited_ranges.items())
            },
        }
        common: dict[str, Any] = {
            "mode": "sparse_map", "state_kind": state_kind,
            "source_measured_nodes": len(self.nodes),
            "visited_mask_kind": "exact_3d_node_index_ranges",
            "visited_mask_node_count": len(self.nodes),
            "visited_mask": mask, "interpolation": "forbidden",
            "ledger_event_count": self.ledger_event_count,
            "ledger_sha256": self._ledger_digest(),
        }
        if state_kind == "exact_nodes":
            presentation_nodes = {
                str(row["node_index"]): {
                    key: item for key, item in row.items() if key != "node_index"
                } for row in (
                    self._registered_row(self.nodes[node], now)
                    for node in sorted(visible_ids)
                )
            }
            presentation_fields = {"nodes": presentation_nodes}
        else:
            recent_ids = list(reversed(self.recent))[:RECENT_DETAIL_ROWS]
            common.update({
                "grid_shape": [MAP_BINS_X, MAP_BINS_Y, MAP_BINS_Z],
                "depth_aggregation": "three_absolute_depth_bands",
                "z_bin_edges_m": [round(value, 3) for value in self.z_bin_edges],
                "registered_native_z_layer_count": (
                    self.registered_native_z_layer_count
                ),
                "unvisited_cells": "omitted",
                "current_column": [
                    self._registered_row(self.nodes[node], now)
                    for node in sorted(
                        self.column_members.get(self._column_key(current), set())
                    )
                ],
                "recent_detail": [
                    self._registered_row(self.nodes[node], now) for node in recent_ids
                ],
            })
            presentation_fields = {"cells": self._aggregate_table(now)}
        value = {
            key: item for key, item in delta.items()
            if key not in {
                "new_samples", "changed_node_summaries", "since_sample_sequence",
                "directions_to_unvisited_neighbors",
            }
        }
        value.update(self.static)
        value.update({
            "schema_version": "5.0-baseline-sparse-delta",
            "observed_node_count": len(self.nodes),
            "sample_ledger_count": self.ledger_event_count,
            "sample_ledger_sha256": self._ledger_digest(),
            "measurement_presentation_mode": "sparse_map",
            "cumulative_measurement_state": {**common, **presentation_fields},
            "derived_measurement_summary": self._derived_summary(delta, now),
        })
        return value


@dataclass
class BaselineController:
    """Stateful public-observation to single-cell-action baseline policy."""

    baseline_id: str
    seed: int
    rng: random.Random = field(init=False)
    adjacency: dict[int, dict[str, int]] = field(default_factory=dict, init=False)
    incoming: dict[int, set[tuple[int, str]]] = field(default_factory=dict, init=False)
    reachable: dict[int, set[str]] = field(default_factory=dict, init=False)
    frontier: dict[int, set[str]] = field(default_factory=dict, init=False)
    positions: dict[int, tuple[float, float, float]] = field(default_factory=dict, init=False)
    failed: set[tuple[int, str]] = field(default_factory=set, init=False)
    pending_move: tuple[int, str] | None = field(default=None, init=False)
    x_heading: str = field(default="east", init=False)
    y_heading: str = field(default="north", init=False)
    z_heading: str = field(default="down", init=False)
    gradient_probe: tuple[int, str, int | None] | None = field(default=None, init=False)
    gradient_channel: str | None = field(default=None, init=False)
    detection_target: int | None = field(default=None, init=False)
    climbing: bool = field(default=False, init=False)
    previous_best: float = field(default=-float("inf"), init=False)
    stagnation: int = field(default=0, init=False)
    strongest_score: float = field(default=-float("inf"), init=False)
    strongest_node: int | None = field(default=None, init=False)
    strongest_channel: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.baseline_id not in BASELINES:
            raise ValueError(f"unknown baseline: {self.baseline_id}")
        self.rng = random.Random(self.seed)

    def _remember_edge(self, origin: int, direction: str, target: int) -> None:
        opposite = OPPOSITE[direction]
        self.adjacency.setdefault(origin, {})[direction] = target
        self.adjacency.setdefault(target, {})[opposite] = origin
        self.incoming.setdefault(target, set()).add((origin, direction))
        self.incoming.setdefault(origin, set()).add((target, opposite))

    def _ingest(self, observation: Mapping[str, Any]) -> int:
        current = _node_id(observation)
        if self.pending_move is not None:
            origin, direction = self.pending_move
            if current != origin:
                self._remember_edge(origin, direction, current)
            else:
                self.failed.add((origin, direction))
            self.pending_move = None
        position = observation["position"]
        self.positions[current] = tuple(float(position[axis]) for axis in "xyz")
        neighbors = observation.get("neighbors", {})
        self.reachable[current] = {
            str(direction) for direction, metadata in neighbors.items()
            if isinstance(metadata, Mapping) and bool(metadata.get("reachable"))
        }
        for direction, metadata in neighbors.items():
            if not isinstance(metadata, Mapping):
                continue
            target = metadata.get("target_node_index")
            if metadata.get("reachable") and target is not None:
                self._remember_edge(current, str(direction), int(target))
        unvisited_directions = observation.get("directions_to_unvisited_neighbors")
        if unvisited_directions is None:
            unvisited_directions = observation.get(
                "derived_measurement_summary", {},
            ).get("directions_to_unvisited_neighbors", ())
        self.frontier[current] = {
            str(direction) for direction in unvisited_directions
            if str(direction) in self.reachable[current]
        }
        for origin, direction in self.incoming.get(current, set()):
            self.frontier.setdefault(origin, set()).discard(direction)
        return current

    @staticmethod
    def _used_fraction(observation: Mapping[str, Any]) -> float:
        endurance = observation.get("budget", {}).get("endurance", {})
        limit = float(endurance.get("limit", 0.0) or 0.0)
        if limit > 0:
            return float(endurance.get("used", 0.0)) / limit
        return 1.0 - float(observation.get("endurance_remaining_fraction", 1.0))

    def _legal_directions(self, observation: Mapping[str, Any]) -> list[str]:
        current = _node_id(observation)
        budget = observation.get("budget", {})
        remaining_s = float(
            budget.get("endurance", {}).get(
                "remaining", observation.get("endurance_remaining_s", float("inf")),
            )
        )
        remaining_m = float(budget.get("range", {}).get("remaining", float("inf")))
        output = []
        for direction, metadata in observation.get("neighbors", {}).items():
            if not isinstance(metadata, Mapping) or not metadata.get("reachable"):
                continue
            if (current, str(direction)) in self.failed:
                continue
            if metadata.get("affordable") is False:
                continue
            cost_s = metadata.get("one_step_cost_s")
            distance_m = metadata.get("one_step_distance_m")
            if "affordable" not in metadata:
                # The legacy environment terminates exactly at physical-budget
                # exhaustion, so retain enough budget to submit there. V20
                # advertises affordability and has an explicit submit-only mode.
                if cost_s is not None and float(cost_s) >= remaining_s - 1e-12:
                    continue
                if distance_m is not None and float(distance_m) >= remaining_m - 1e-12:
                    continue
            output.append(str(direction))
        return output

    def _move(self, observation: Mapping[str, Any], direction: str) -> dict[str, Any]:
        current = _node_id(observation)
        self.frontier.setdefault(current, set()).discard(direction)
        self.pending_move = (current, direction)
        return {"action": "move", "direction": direction}

    def _submit(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        answer = infer_submission(observation)
        if self.baseline_id == "detect_then_climb" and answer["anomaly"] == "anomaly":
            current = _current_row(observation)
            position = {key: float(current[key]) for key in ("x", "y", "z")}
            answer["peak_position"] = position
            answer["source_position"] = dict(position)
        return {"action": "submit", "answer": answer}

    def _known_route(self, current: int, targets: set[int]) -> list[str]:
        if current in targets:
            return []
        queue = deque([current])
        previous: dict[int, tuple[int, str]] = {}
        seen = {current}
        found: int | None = None
        while queue:
            node = queue.popleft()
            edges = self.adjacency.get(node, {})
            for direction in DIRECTIONS:
                target = edges.get(direction)
                if target is None or target in seen:
                    continue
                seen.add(target)
                previous[target] = node, direction
                if target in targets:
                    found = target
                    queue.clear()
                    break
                queue.append(target)
        if found is None:
            return []
        route = []
        cursor = found
        while cursor != current:
            parent, direction = previous[cursor]
            route.append(direction)
            cursor = parent
        route.reverse()
        return route

    def _route_to_frontier(self, current: int) -> str | None:
        targets = {node for node, directions in self.frontier.items() if directions}
        targets.discard(current)
        route = self._known_route(current, targets)
        return route[0] if route else None

    def _coverage_direction(self, observation: Mapping[str, Any]) -> str | None:
        current = _node_id(observation)
        legal = set(self._legal_directions(observation))
        unvisited = self.frontier.get(current, set()) & legal
        if self.x_heading in unvisited:
            return self.x_heading
        if self.y_heading in unvisited:
            self.x_heading = OPPOSITE[self.x_heading]
            return self.y_heading
        if self.z_heading in unvisited:
            self.x_heading = OPPOSITE[self.x_heading]
            self.y_heading = OPPOSITE[self.y_heading]
            return self.z_heading
        if OPPOSITE[self.z_heading] in unvisited:
            self.x_heading = OPPOSITE[self.x_heading]
            self.y_heading = OPPOSITE[self.y_heading]
            self.z_heading = OPPOSITE[self.z_heading]
            return self.z_heading
        for direction in SWEEP_FALLBACK_ORDER:
            if direction in unvisited:
                if direction in {"east", "west"}:
                    self.x_heading = direction
                elif direction in {"north", "south"}:
                    self.y_heading = direction
                    self.x_heading = OPPOSITE[self.x_heading]
                return direction
        route_direction = self._route_to_frontier(current)
        return route_direction if route_direction in legal else None

    def _gradient_direction(self, observation: Mapping[str, Any]) -> str | None:
        rows = _rows(observation)
        by_node = {int(row["node_index"]): row for row in rows}
        current = _current_row(observation)
        fallback = _fallback_channel_scales(observation)
        current_layer = _depth_layer(observation, current)
        measured_neighbors = []
        for direction, metadata in observation.get("neighbors", {}).items():
            if not isinstance(metadata, Mapping) or not metadata.get("reachable"):
                continue
            target = metadata.get("target_node_index")
            if target is None or int(target) not in by_node:
                continue
            target_row = by_node[int(target)]
            # Compare like-for-like measured channel deltas.  An absolute
            # anomaly score requires the target's depth layer, which is not in
            # the registered historical-detail rendering.  Scaling the delta
            # by the current layer's public LOD preserves units without
            # inventing or retaining hidden target bathymetry.
            normalized_deltas = []
            for channel, current_value in current.get("channels", {}).items():
                if self.gradient_channel is not None and channel != self.gradient_channel:
                    continue
                if channel not in target_row.get("channels", {}):
                    continue
                calibrated = _calibration_entry(
                    observation, str(channel), current_layer,
                )
                scale = (
                    calibrated
                    or fallback.get(str(channel), (0.0, 1.0))
                )[1]
                normalized_deltas.append((
                    (
                        float(target_row["channels"][channel])
                        - float(current_value)
                    ) / max(abs(scale), 1e-12),
                    str(channel),
                ))
            score, channel = max(
                normalized_deltas, default=(-float("inf"), ""),
            )
            measured_neighbors.append(
                (score, -DIRECTIONS.index(str(direction)), channel, str(direction))
            )
        if measured_neighbors:
            score, _, channel, direction = max(measured_neighbors)
            if score > 1e-9:
                if self.gradient_channel is None:
                    self.gradient_channel = channel
                return direction
            return None

        # Compatibility with the legacy public observation, which exposed a
        # measured-only local-gradient summary instead of target node ids.
        layer = _depth_layer(observation, current)
        candidates = []
        gradients = observation.get("observed_local_gradients", {})
        for channel, summary in gradients.items():
            if self.gradient_channel is not None and channel != self.gradient_channel:
                continue
            if not isinstance(summary, Mapping):
                continue
            direction = summary.get("direction_of_largest_measured_increase")
            delta = summary.get("largest_measured_delta")
            if direction is None or delta is None or float(delta) <= 0:
                continue
            calibrated = _calibration_entry(observation, str(channel), layer)
            scale = (calibrated or fallback.get(str(channel), (0.0, 1.0)))[1]
            normalized = float(delta) / max(abs(scale), 1e-12)
            candidates.append((
                normalized, -DIRECTIONS.index(str(direction)), str(channel), str(direction),
            ))
        if not rows or not candidates:
            return None
        _, _, channel, direction = max(candidates)
        if self.gradient_channel is None:
            self.gradient_channel = channel
        return direction

    def _gradient_direction_or_probe(self, observation: Mapping[str, Any]) -> str | None:
        current = _node_id(observation)
        legal = set(self._legal_directions(observation))
        if self.gradient_probe is not None:
            origin, outbound, target = self.gradient_probe
            if current != origin:
                self.gradient_probe = (origin, outbound, current)
                back = OPPOSITE[outbound]
                return back if back in legal else None
            # Returning to the origin (or a failed outbound move) completes one
            # paid comparison before another direction is considered.
            self.gradient_probe = None
        unvisited = self.frontier.get(current, set()) & legal
        for direction in PROBE_ORDER:
            if direction in unvisited:
                self.gradient_probe = (current, direction, None)
                return direction
        direction = self._gradient_direction(observation)
        return direction if direction in legal else None

    def _flow_direction(self, observation: Mapping[str, Any]) -> str | None:
        current = _current_row(observation)
        u = float(current["flow"]["u_m_s"])
        v = float(current["flow"]["v_m_s"])
        preferred = (
            ("west" if u > 0 else "east") if abs(u) >= abs(v)
            else ("south" if v > 0 else "north")
        )
        horizontal = ["east", "west", "north", "south"]
        if self.baseline_id == "adjoint_proxy" and self.rng.random() < 0.25:
            preferred = self.rng.choice(horizontal)
        legal = self._legal_directions(observation)
        if preferred in legal:
            return preferred
        return self.rng.choice(legal) if legal else None

    def _update_stagnation(self, observation: Mapping[str, Any]) -> None:
        score = max(_row_scores(observation, _current_row(observation)).values(), default=0.0)
        if score > self.previous_best + 1e-9:
            self.previous_best, self.stagnation = score, 0
        else:
            self.stagnation += 1

    def _update_strongest_current(self, observation: Mapping[str, Any]) -> None:
        row = _current_row(observation)
        node = int(row["node_index"])
        scores = _row_scores(observation, row)
        channel = max(
            scores, key=lambda name: (scores[name], str(name)), default=None,
        )
        score = scores[channel] if channel is not None else 0.0
        if (
            score > self.strongest_score + 1e-9
            or (
                abs(score - self.strongest_score) <= 1e-9
                and (self.strongest_node is None or node < self.strongest_node)
            )
        ):
            self.strongest_score = score
            self.strongest_node = node
            self.strongest_channel = channel

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Return exactly one public action for the current public observation."""

        current = self._ingest(observation)
        budget = observation.get("budget", {})
        guard = budget.get("tool_turns", budget.get("provider_round_guard", {}))
        turns_remaining = int(observation.get("turns_remaining", guard.get("remaining", 2)))
        coverage_complete = (
            self.baseline_id in {"lawnmower", "budgeted_serpentine"}
            or (self.baseline_id == "detect_then_climb" and not self.climbing)
        ) and int(observation.get(
            "observed_node_count", len(observation.get("cumulative_measurements", ())),
        )) >= int(
            observation.get("n_total_nodes", 1)
        )
        if (
            observation.get("episode_mode") == "submit_only"
            or turns_remaining <= 1
            or not self._legal_directions(observation)
            or coverage_complete
        ):
            return self._submit(observation)
        if self.baseline_id == "budgeted_serpentine" and (
            self._used_fraction(observation) >= BUDGETED_SERPENTINE_STOP_FRACTION
        ):
            return self._submit(observation)
        if self.baseline_id in {"upstream_flow_heuristic", "adjoint_proxy"}:
            self._update_stagnation(observation)
            if self.stagnation >= 8 or self._used_fraction(observation) >= 0.20:
                return self._submit(observation)

        direction: str | None
        if self.baseline_id == "random":
            legal = self._legal_directions(observation)
            direction = self.rng.choice(legal) if legal else None
        elif self.baseline_id in {"lawnmower", "budgeted_serpentine"}:
            direction = self._coverage_direction(observation)
        elif self.baseline_id == "greedy_gradient":
            direction = self._gradient_direction_or_probe(observation)
        elif self.baseline_id == "detect_then_climb":
            if not self.climbing:
                self._update_strongest_current(observation)
                if self.strongest_score >= DETECTION_SCORE_THRESHOLD:
                    self.climbing = True
                    self.detection_target = self.strongest_node
                    self.gradient_channel = self.strongest_channel
            if self.climbing:
                if self.detection_target is not None and current != self.detection_target:
                    route = self._known_route(current, {self.detection_target})
                    direction = route[0] if route else self._gradient_direction_or_probe(observation)
                else:
                    self.detection_target = None
                    direction = self._gradient_direction_or_probe(observation)
            else:
                direction = self._coverage_direction(observation)
        else:
            direction = self._flow_direction(observation)
        return self._move(observation, direction) if direction is not None else self._submit(observation)


def make_baseline_controller(baseline_id: str, seed: int) -> BaselineController:
    return BaselineController(baseline_id=baseline_id, seed=seed)


def run_baseline(env: PublicBaselineEnvironment, baseline_id: str, seed: int) -> None:
    """Run through public deltas and registered sparse-map mission memory."""

    controller = make_baseline_controller(baseline_id, seed)
    memory = PublicSparseObservationMemory()
    first = True
    sample_sequence = 0
    while not env.terminated:
        delta = env.observation_delta(sample_sequence, include_static=first)
        sample_sequence = int(delta["sample_sequence"])
        action = controller.act(memory.ingest(delta))
        first = False
        if action["action"] == "move":
            result = env.move(str(action["direction"]))
        elif action["action"] == "submit":
            env.submit(dict(action["answer"]))
            break
        else:  # pragma: no cover - controller actions are closed above
            raise ValueError(f"unknown baseline action: {action['action']!r}")
        if result.get("code") == "E_TERMINATED":
            break
