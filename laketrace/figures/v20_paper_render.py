"""Publication-safe SVG rendering for the LakeTrace v20 paper figures.

The renderer has one deliberately narrow data boundary: it consumes the
publication tables ``episode_outcomes``, ``task_geometry`` and ``steps``.  It
does not open run logs or evaluator-only inputs, and it does not recompute
registered analysis quantities such as tertiles or source--peak projection.

Every panel is written as a standalone, accessible SVG.  Figure composites are
then assembled mechanically by nesting those already-written SVG documents;
no values are recalculated during composition.  Rehearsal output is always
watermarked.  Formal mode fails closed when any input is marked synthetic.
"""

from __future__ import annotations

import base64
import csv
import html
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_MODEL_ORDER: tuple[str, ...] = (
    "GPT-5.6 Sol",
    "DeepSeek-V4-Pro-0813",
    "Claude Opus 5",
    "Kimi K3",
    "GLM-5.3",
    "Gemini 3.7 Flash",
)

# Public alias retained for callers that treated the roster as a constant.
MODEL_ORDER = DEFAULT_MODEL_ORDER

PANEL_ORDER: tuple[str, ...] = (
    "Fig1a", "Fig1b", "Fig1c",
    "Fig4a", "Fig4b", "Fig4c", "Fig4d",
    "Fig5a", "Fig5b", "Fig5c",
    "Fig6a", "Fig6b", "Fig6c",
)

SYNTHETIC_WATERMARK = "SIMULATED DATA — LAYOUT REHEARSAL"

# Okabe--Ito-derived, colour-vision-deficiency-safe palette.  Shape, position
# and text always duplicate colour meaning.
INK = "#17212B"
NAVY = "#12304A"
SLATE = "#65727D"
GRID = "#D8E2E6"
PAPER = "#FCFCFA"
PANEL = "#FFFFFF"
SOFT = "#EEF4F6"
BLUE = "#0072B2"
TEAL = "#009E73"
AMBER = "#E69F00"
ORANGE = "#D55E00"
PURPLE = "#8E6C9E"
SKY = "#56B4E9"

LEVEL_COLORS = (BLUE, TEAL, AMBER, ORANGE, PURPLE)

# Fig. 4 is the compact benchmark-results figure.  L5 remains a registered
# score in the publication tables, but is deliberately reserved for the
# safety/calibration analysis rather than duplicated in this four-capability
# overview.  Keeping this as an explicit display contract prevents a visual
# edit from mutating the underlying five-level benchmark record.
FIG4_DISPLAY_LEVELS: tuple[int, ...] = (1, 2, 3, 4)

# Okabe-Ito-derived identity palette.  Colour has exactly one meaning
# throughout result figures: model identity.  Registered marker geometry and
# direct labels provide redundant encoding for greyscale and CVD viewing.
MODEL_COLORS = ("#000000", "#0072B2", "#D55E00", "#CC79A7", "#009E73", "#E69F00")
MODEL_MARKERS = ("circle", "square", "triangle", "diamond", "pentagon", "hexagon")


@dataclass(frozen=True)
class TablePayload:
    """Normalized publication-table payload and non-row metadata."""

    rows: tuple[dict[str, Any], ...]
    metadata: Mapping[str, Any]
    source: str


@dataclass(frozen=True)
class PanelSpec:
    panel_id: str
    width: int
    height: int
    title: str
    description: str


PANEL_SPECS: Mapping[str, PanelSpec] = {
    "Fig1a": PanelSpec("Fig1a", 1000, 540, "Single-edge 3-D survey episode", "Registered geometry and one measured route for a preregistered representative task."),
    "Fig1b": PanelSpec("Fig1b", 620, 540, "Five questions and frozen design", "Frozen v20 capability levels and design counts."),
    "Fig1c": PanelSpec("Fig1c", 1640, 470, "Capability profile across L1–L5", "Task-level estimates with two-way cluster-bootstrap intervals."),
    "Fig4a": PanelSpec("Fig4a", 1640, 390, "Four-level capability profile", "Four aligned dot-and-interval plots (L1–L4) with registered controls and fixed model identity encoding."),
    "Fig4b": PanelSpec("Fig4b", 1640, 440, "Observable behavioural fingerprints", "Task-level distributions for six registered action-ledger measures on native scales; submission is shown as a task-equal mean."),
    "Fig4c": PanelSpec("Fig4c", 820, 400, "Non-submission outcomes", "Mutually exclusive non-submission states on a magnified zero-to-thirty-five-percent scale."),
    "Fig4d": PanelSpec("Fig4d", 800, 400, "Single-run versus repeated reliability", "Single-run pass rate linked to the fraction of tasks passed in all three registered runs."),
    "Fig5a": PanelSpec("Fig5a", 360, 850, "Peak–source separability", "Registered D* geometry and task distribution."),
    "Fig5b": PanelSpec("Fig5b", 730, 850, "L2-to-L4 gap across D*", "Paired capability estimates across the four registered peak-source displacement strata."),
    "Fig5c": PanelSpec("Fig5c", 510, 850, "Submitted position, source to peak", "Registered rho set-distance diagnostic for valid submitted coordinates."),
    "Fig6a": PanelSpec("Fig6a", 1640, 500, "Same-task observable storyboard", "Common registered geometry, measured paths and fixed decision-envelope snapshots."),
    "Fig6b": PanelSpec("Fig6b", 800, 430, "Conditional observable chain", "A: footprint entry; B: source approach after A; C: correct L4 submission after B."),
    "Fig6c": PanelSpec("Fig6c", 800, 430, "First observable failure stage or success", "Exhaustive behaviour categories; no private reasoning state is inferred."),
}

# x, y placements are exact native panel units.  Composite construction never
# rescales a panel, preserving the 20-unit (about 6 pt at 180 mm) minimum type.
FIGURE_LAYOUTS: Mapping[str, tuple[tuple[str, int, int], ...]] = {
    "Fig1": (("Fig1a", 20, 72), ("Fig1b", 1040, 72), ("Fig1c", 20, 632)),
    "Fig4": (("Fig4a", 20, 20), ("Fig4b", 20, 430), ("Fig4c", 20, 890), ("Fig4d", 860, 890)),
    "Fig5": (("Fig5a", 20, 20), ("Fig5b", 400, 20), ("Fig5c", 1150, 20)),
    "Fig6": (("Fig6a", 20, 72), ("Fig6b", 20, 592), ("Fig6c", 840, 592)),
}

FIGURE_TITLES: Mapping[str, tuple[str, str]] = {
    "Fig1": ("LakeTrace v20 task and five-level capability overview", "Three independently rendered panels composed without recomputing data."),
    "Fig4": ("Capability, behaviour and repeated-run reliability", "Four independently rendered panels compare capability, observable action patterns, terminal states and all-three-run reliability in a fixed display order."),
    "Fig5": ("Task geometry separates anomaly-footprint and source outcomes", "Three independently rendered panels consuming publication geometry and outcomes only."),
    "Fig6": ("Observable episode trajectories and failure anatomy", "Three independently rendered panels using measured steps and published behavioural stages."),
}


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _truth_named(path: Path) -> bool:
    return any(re.search(r"(?:^|[-_.])truth(?:$|[-_.])", part.lower()) for part in path.parts)


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y", "pass", "passed", "submitted"}:
            return True
        if token in {"false", "0", "no", "n", "fail", "failed", "unsubmitted", ""}:
            return False
    return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _rows_from_json(value: Any, table_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata: dict[str, Any] = {}
    rows_value = value
    if isinstance(value, Mapping):
        if table_name in value:
            rows_value = value[table_name]
        elif "rows" in value:
            rows_value = value["rows"]
        elif "data" in value and isinstance(value["data"], Sequence):
            rows_value = value["data"]
        else:
            raise ValueError(f"{table_name} JSON object must contain {table_name!r}, 'rows', or 'data'")
        metadata = {str(k): v for k, v in value.items() if k not in {table_name, "rows", "data"}}
        if isinstance(value.get("metadata"), Mapping):
            metadata.update(value["metadata"])
    if not isinstance(rows_value, Sequence) or isinstance(rows_value, (str, bytes, bytearray)):
        raise ValueError(f"{table_name} must be a sequence of row objects")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows_value):
        if not isinstance(row, Mapping):
            raise ValueError(f"{table_name} row {index} is not an object")
        rows.append({str(k): v for k, v in row.items()})
    return rows, metadata


def _load_table(source: Any, table_name: str) -> TablePayload:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if _truth_named(path):
            raise ValueError(f"refusing evaluator/truth-named input for publication renderer: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
            metadata: dict[str, Any] = {}
        elif suffix in {".jsonl", ".ndjson"}:
            rows = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise ValueError(f"{path}:{line_number} is not a row object")
                    rows.append({str(k): v for k, v in row.items()})
            metadata = {}
        elif suffix == ".json":
            rows, metadata = _rows_from_json(json.loads(path.read_text(encoding="utf-8")), table_name)
        else:
            raise ValueError(f"unsupported {table_name} format {suffix!r}; use CSV, JSON, or JSONL")
        return TablePayload(tuple(rows), metadata, str(path))

    if isinstance(source, Mapping):
        rows, metadata = _rows_from_json(source, table_name)
        return TablePayload(tuple(rows), metadata, f"memory:{table_name}")

    if isinstance(source, Iterable) and not isinstance(source, (str, bytes, bytearray)):
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(source):
            if not isinstance(row, Mapping):
                raise ValueError(f"{table_name} row {index} is not an object")
            rows.append({str(k): v for k, v in row.items()})
        return TablePayload(tuple(rows), {}, f"memory:{table_name}")
    raise TypeError(f"unsupported {table_name} source: {type(source).__name__}")


def _synthetic_marker(value: Any) -> bool:
    if _as_bool(value) is True:
        return True
    token = str(value).strip().lower()
    return token in {"synthetic", "simulated", "mock", "wireframe", "rehearsal"}


def _payload_is_synthetic(payload: TablePayload) -> bool:
    for key in ("synthetic", "is_synthetic", "data_kind", "provenance", "mode"):
        if key in payload.metadata and _synthetic_marker(payload.metadata[key]):
            return True
    for row in payload.rows:
        for key in ("synthetic", "is_synthetic", "data_kind", "provenance"):
            if key in row and _synthetic_marker(row[key]):
                return True
    return False


def _metadata_model_order(payloads: Sequence[TablePayload]) -> tuple[str, ...] | None:
    for payload in payloads:
        candidates: list[Any] = [payload.metadata.get("model_order")]
        for container in ("contract", "scenario"):
            nested = payload.metadata.get(container)
            if isinstance(nested, Mapping):
                candidates.append(nested.get("model_order"))
        for candidate in candidates:
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
                order = tuple(str(value) for value in candidate if str(value).strip())
                if order:
                    return order
    return None


def _canonical_model(value: Any, declared: Sequence[str]) -> str:
    raw = str(value or "unregistered model")
    token = _norm(raw)
    for model in declared:
        if token == _norm(model):
            return model
    family_checks = (
        ("gpt", "GPT"), ("deepseek", "DeepSeek"), ("claude", "Claude"),
        ("kimi", "Kimi"), ("glm", "GLM"), ("gemini", "Gemini"),
    )
    for needle, family in family_checks:
        if needle in token:
            match = next((model for model in declared if family.lower() in model.lower()), None)
            if match:
                return match
    return raw


def _model_value(row: Mapping[str, Any]) -> Any:
    # Actor identity is the grouping key.  A provider/model snapshot is
    # provenance and may legitimately be shared (for example the explicit
    # synthetic rehearsal marker), so it must never take precedence over an
    # actor or reader-facing model name.
    return _first(
        row,
        "model_display_name",
        "model_display",
        "actor",
        "model_id",
        "model",
        "system",
        "model_snapshot",
    )


def _control_kind(row: Mapping[str, Any]) -> str | None:
    """Return a registered control kind without treating it as a model."""

    declared = str(_first(row, "control_kind", "control_id", "row_type") or "").lower()
    actor = _norm(_first(row, "actor", "model_id", "model", "system") or "")
    if declared in {"reachable_oracle", "reachable_upper", "oracle", "registered_oracle"} or actor in {"reachableupper", "reachableoracle"}:
        return "reachable_oracle"
    if declared in {"design_random", "random", "registered_random", "3_seed_design_random"} or actor in {"random", "designrandom", "3seeddesignrandom"}:
        return "design_random"
    return None


def _resolve_model_order(
    payloads: Sequence[TablePayload], explicit: Sequence[str] | None,
) -> tuple[str, ...]:
    declared = tuple(explicit or _metadata_model_order(payloads) or DEFAULT_MODEL_ORDER)
    if len({_norm(value) for value in declared}) != len(declared):
        raise ValueError("model_order contains duplicate normalized names")
    observed: list[str] = []
    for payload in payloads:
        for row in payload.rows:
            if _control_kind(row):
                continue
            value = _model_value(row)
            if value is None:
                continue
            model = _canonical_model(value, declared)
            if model not in observed:
                observed.append(model)
    # Declared rows remain visible even if data are absent, making missing
    # coverage obvious. Unknown systems are appended in encounter order, never
    # score order.
    return tuple(declared) + tuple(model for model in observed if model not in declared)


def _task_value(row: Mapping[str, Any]) -> str:
    return str(_first(row, "task_id", "instance_id", "case_id") or "")


def _model_rows(rows: Sequence[Mapping[str, Any]], model: str, order: Sequence[str]) -> list[Mapping[str, Any]]:
    return [row for row in rows if _canonical_model(_model_value(row), order) == model]


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _rate(values: Iterable[Any]) -> float | None:
    clean = [_as_bool(value) for value in values]
    known = [value for value in clean if value is not None]
    return statistics.fmean(1.0 if value else 0.0 for value in known) if known else None


def _level_value(row: Mapping[str, Any], level: int) -> bool | float | None:
    wide = _first(row, f"L{level}_pass", f"l{level}_pass", f"L{level}", f"l{level}")
    if wide is not None:
        boolean = _as_bool(wide)
        return boolean if boolean is not None else _as_float(wide)
    row_level = str(_first(row, "level", "capability_level") or "").upper()
    if row_level == f"L{level}":
        value = _first(row, "pass", "score", "rate", "value")
        boolean = _as_bool(value)
        return boolean if boolean is not None else _as_float(value)
    return None


def _model_level_rates(rows: Sequence[Mapping[str, Any]], order: Sequence[str]) -> dict[str, list[float | None]]:
    output: dict[str, list[float | None]] = {}
    for model in order:
        subset = _model_rows(rows, model, order)
        rates: list[float | None] = []
        for level in range(1, 6):
            if level == 1:
                positives = [
                    _as_bool(_level_value(row, 1))
                    for row in subset
                    if _as_bool(_first(row, "detectable_source", "positive_case")) is True
                ]
                negatives = [
                    _as_bool(_level_value(row, 1))
                    for row in subset
                    if _as_bool(_first(row, "detectable_source", "positive_case")) is False
                ]
                sensitivity_values = [value for value in positives if value is not None]
                specificity_values = [value for value in negatives if value is not None]
                if sensitivity_values and specificity_values:
                    sensitivity = statistics.fmean(1.0 if value else 0.0 for value in sensitivity_values)
                    specificity = statistics.fmean(1.0 if value else 0.0 for value in specificity_values)
                    rates.append((sensitivity + specificity) / 2.0)
                    continue
            values = [_level_value(row, level) for row in subset]
            numeric = [float(value) for value in values if value is not None]
            rates.append(statistics.fmean(numeric) if numeric else None)
        output[model] = rates
    return output


def _task_level_records(
    rows: Sequence[Mapping[str, Any]], level: int,
) -> list[dict[str, Any]]:
    """Collapse stochastic seeds before any uncertainty calculation."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        task = _task_value(row)
        value = _level_value(row, level)
        if task and value is not None:
            grouped[task].append(row)
    records: list[dict[str, Any]] = []
    for task, task_rows in grouped.items():
        values = [float(value) for row in task_rows if (value := _level_value(row, level)) is not None]
        if not values:
            continue
        first = task_rows[0]
        records.append({
            "task": task,
            "value": statistics.fmean(values),
            "event": str(_first(first, "event_cluster_id") or task),
            "background": str(_first(first, "background_id") or "unregistered"),
            "positive": _as_bool(_first(first, "detectable_source", "positive_case")),
            "seed_n": len(values),
        })
    return records


def _cluster_point(records: Sequence[Mapping[str, Any]], *, l1: bool) -> float | None:
    if not records:
        return None
    if not l1:
        return statistics.fmean(float(row["value"]) for row in records)
    positive = [float(row["value"]) for row in records if row.get("positive") is True]
    negative = [float(row["value"]) for row in records if row.get("positive") is False]
    if not positive or not negative:
        return None
    return (statistics.fmean(positive) + statistics.fmean(negative)) / 2


def _cluster_interval(
    rows: Sequence[Mapping[str, Any]], level: int, *,
    seed: int = 20260828, replicates: int = 1200,
) -> tuple[float | None, float | None, float | None, int, int, int, int]:
    """Registered two-way pigeonhole bootstrap after within-task collapse."""

    records = _task_level_records(rows, level)
    point = _cluster_point(records, l1=level == 1)
    if point is None:
        return None, None, None, len(records), sum(int(row["seed_n"]) for row in records), 0, 0
    events = sorted({str(row["event"]) for row in records})
    backgrounds = sorted({str(row["background"]) for row in records})
    rng = random.Random(seed + 1009 * level + 17 * len(rows))
    estimates: list[float] = []
    for _ in range(replicates):
        event_weights = Counter(rng.choice(events) for _ in events)
        background_weights = Counter(rng.choice(backgrounds) for _ in backgrounds)
        weighted: list[dict[str, Any]] = []
        for row in records:
            weight = event_weights[str(row["event"])] * background_weights[str(row["background"])]
            weighted.extend([row] * weight)
        estimate = _cluster_point(weighted, l1=level == 1)
        if estimate is not None and math.isfinite(estimate):
            estimates.append(estimate)
    estimates.sort()
    if not estimates:
        low = high = None
    else:
        low = estimates[int(.025 * (len(estimates) - 1))]
        high = estimates[int(.975 * (len(estimates) - 1))]
    return (
        point, low, high, len(records),
        sum(int(row["seed_n"]) for row in records), len(events), len(backgrounds),
    )


def _draw_vertical_interval(
    parts: list[str], x: float, low: float | None, high: float | None,
    y0: float, height: float, color: str,
) -> None:
    if low is None or high is None:
        return
    y_low, y_high = y0 + height * (1 - low), y0 + height * (1 - high)
    _line(parts, x, y_low, x, y_high, color, 2)
    _line(parts, x - 5, y_low, x + 5, y_low, color, 2)
    _line(parts, x - 5, y_high, x + 5, y_high, color, 2)


def _control_level_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float | None]]:
    output: dict[str, list[float | None]] = {}
    for control in ("reachable_oracle", "design_random"):
        subset = [row for row in rows if _control_kind(row) == control]
        rates: list[float | None] = []
        for level in range(1, 6):
            # The registered design-validation summary is authoritative for
            # controls because its L1 value is balanced accuracy, whereas the
            # per-task score CSV contains ordinary pass indicators.
            summary_values = [
                _as_float(_first(row, "control_rate", "registered_control_rate"))
                for row in subset
                if str(_first(row, "level", "capability_level") or "").upper() == f"L{level}"
                and _first(row, "control_rate", "registered_control_rate") is not None
            ]
            summary_values = [value for value in summary_values if value is not None]
            values = summary_values or [_level_value(row, level) for row in subset]
            numeric = [float(value) for value in values if value is not None]
            rates.append(statistics.fmean(numeric) if numeric else None)
        output[control] = rates
    return output


def _short_model(model: str) -> str:
    replacements = {
        "GPT-5.6 Sol": "GPT-5.6 Sol",
        "DeepSeek-V4-Pro-0813": "DeepSeek-V4-Pro-0813",
        "Claude Opus 5": "Claude Opus 5",
        "Kimi K3": "Kimi K3",
        "GLM-5.3": "GLM-5.3",
        "Gemini 3.7 Flash": "Gemini 3.7 Flash",
    }
    value = replacements.get(model, model)
    return value if model in replacements or len(value) <= 21 else value[:19] + "…"


def _compact_model(model: str) -> str:
    """Compact labels for narrow panels; identity remains complete in SVG metadata."""

    aliases = {
        "GPT-5.6 Sol": "GPT-5.6",
        "DeepSeek-V4-Pro-0813": "DeepSeek V4",
        "Claude Opus 5": "Claude Opus",
        "Kimi K3": "Kimi K3",
        "GLM-5.3": "GLM-5.3",
        "Gemini 3.7 Flash": "Gemini 3.7",
    }
    return aliases.get(model, _short_model(model))


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def _line(parts: list[str], x1: float, y1: float, x2: float, y2: float, color: str = GRID, width: float = 2, dash: str | None = None, opacity: float = 1.0) -> None:
    attrs = f' stroke-dasharray="{dash}"' if dash else ""
    opacity_attr = "" if opacity >= 1 else f' opacity="{opacity:.3f}"'
    parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width:.1f}"{opacity_attr}{attrs}/>' )


def _rect(parts: list[str], x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", radius: float = 5, opacity: float = 1.0) -> None:
    opacity_attr = "" if opacity >= 1 else f' opacity="{opacity:.3f}"'
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0, w):.1f}" height="{max(0, h):.1f}" rx="{radius:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{opacity_attr}/>')


def _circle(parts: list[str], x: float, y: float, radius: float, fill: str, stroke: str = PANEL, opacity: float = 1.0) -> None:
    opacity_attr = "" if opacity >= 1 else f' opacity="{opacity:.3f}"'
    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="2"{opacity_attr}/>')


def _marker(
    parts: list[str], x: float, y: float, size: float, shape: str,
    fill: str, stroke: str = PANEL, opacity: float = 1.0,
    *, model: str | None = None,
) -> None:
    """Draw one accessible, deterministic model marker."""

    model_attr = f' data-model="{_esc(model)}" data-model-color="{fill}"' if model else ""
    opacity_attr = "" if opacity >= 1 else f' opacity="{opacity:.3f}"'
    common = f' fill="{fill}" stroke="{stroke}" stroke-width="2"{opacity_attr}{model_attr}'
    if shape == "circle":
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}"{common}/>')
        return
    if shape == "square":
        parts.append(f'<rect x="{x-size:.1f}" y="{y-size:.1f}" width="{2*size:.1f}" height="{2*size:.1f}" rx="1"{common}/>')
        return
    if shape == "triangle":
        points = ((x, y - 1.18 * size), (x - 1.08 * size, y + .92 * size), (x + 1.08 * size, y + .92 * size))
    elif shape == "diamond":
        points = ((x, y - 1.22 * size), (x - 1.02 * size, y), (x, y + 1.22 * size), (x + 1.02 * size, y))
    else:
        sides = 5 if shape == "pentagon" else 6
        points = tuple(
            (x + size * math.cos(-math.pi / 2 + 2 * math.pi * index / sides),
             y + size * math.sin(-math.pi / 2 + 2 * math.pi * index / sides))
            for index in range(sides)
        )
    coords = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    parts.append(f'<polygon points="{coords}"{common}/>')


def _model_style(model: str, order: Sequence[str]) -> tuple[str, str]:
    index = order.index(model) if model in order else 0
    return MODEL_COLORS[index % len(MODEL_COLORS)], MODEL_MARKERS[index % len(MODEL_MARKERS)]


def _text(parts: list[str], x: float, y: float, value: Any, css: str = "body", anchor: str = "start", fill: str | None = None) -> None:
    # Inline style intentionally wins over class-level neutral defaults.  This
    # keeps registered identity colours and inverse labels intact in SVG/PDF.
    fill_attr = f' style="fill:{fill}"' if fill else ""
    parts.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{css}" text-anchor="{anchor}"{fill_attr}>{_esc(value)}</text>')


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _visual_summary(value: str, width: int) -> str:
    """Fit visible header prose while retaining the full accessible ``desc``."""

    limit = max(24, int((width - 90) / 11.0))
    return value if len(value) <= limit else value[: max(1, limit - 1)].rstrip() + "…"


def _polyline(parts: list[str], points: Sequence[tuple[float, float]], color: str, width: float = 3, opacity: float = 1.0, dash: str | None = None) -> None:
    if len(points) < 2:
        return
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    opacity_attr = "" if opacity >= 1 else f' opacity="{opacity:.3f}"'
    parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width:.1f}" stroke-linecap="round" stroke-linejoin="round"{opacity_attr}{dash_attr}/>')


def _polygon(
    parts: list[str], points: Sequence[tuple[float, float]], fill: str,
    stroke: str, *, opacity: float = 1.0,
) -> None:
    if len(points) < 3:
        return
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    opacity_attr = "" if opacity >= 1 else f' opacity="{opacity:.3f}"'
    parts.append(
        f'<polygon points="{coords}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="2"{opacity_attr}/>'
    )


def _embed_concept_asset(
    parts: list[str], filename: str, x: float, y: float, width: float, height: float,
    *, opacity: float,
) -> bool:
    """Embed a non-data raster as an aria-hidden, self-contained backdrop."""

    path = Path(__file__).resolve().parents[2] / "figures" / "v20_rehearsal" / "assets" / filename
    if not path.is_file():
        return False
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    parts.append(
        f'<image x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'preserveAspectRatio="xMidYMid slice" opacity="{opacity:.3f}" aria-hidden="true" '
        f'data-layer="concept-background" href="data:image/png;base64,{encoded}"/>'
    )
    return True


def _vector_lake_background(
    parts: list[str], x: float, y: float, width: float, height: float,
    *, label: bool = True,
) -> None:
    """Draw a quiet, auditable schematic substrate without generated pixels."""

    parts.append(f'<g data-layer="concept-background" aria-label="schematic waterbody">')
    _rect(parts, x, y, width, height, "#F6F8F7", GRID, 0, 1.0)
    shoreline = (
        (x, y + .15 * height),
        (x + .12 * width, y + .08 * height),
        (x + .24 * width, y + .18 * height),
        (x + .33 * width, y + .11 * height),
        (x + .46 * width, y + .17 * height),
        (x + .62 * width, y + .10 * height),
        (x + .78 * width, y + .15 * height),
        (x + width, y + .07 * height),
        (x + width, y + height),
        (x, y + height),
    )
    _polygon(parts, shoreline, "#EDF7F9", "#AFC7CF")
    for fraction in (.42, .66, .84):
        _line(
            parts, x + .03 * width, y + fraction * height,
            x + .97 * width, y + fraction * height,
            "#DFEAEF", 1.0, "7 7",
        )
    if label:
        _text(parts, x + 12, y + 24, "schematic, registered overlays only", "small", fill=SLATE)
    parts.append("</g>")


def _map_points_bounds(
    points: Sequence[tuple[float, float]], bounds: tuple[float, float, float, float],
    x: float, y: float, width: float, height: float,
) -> list[tuple[float, float]]:
    xmin, xmax, ymin, ymax = bounds
    xspan, yspan = max(xmax - xmin, 1.0), max(ymax - ymin, 1.0)
    return [
        (x + (px - xmin) / xspan * width, y + height - (py - ymin) / yspan * height)
        for px, py in points
    ]


def _bounds(points: Sequence[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    xpad = max((xmax - xmin) * .06, 1.0)
    ypad = max((ymax - ymin) * .06, 1.0)
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def _panel_shell(spec: PanelSpec, synthetic: bool) -> list[str]:
    title_id = f"{spec.panel_id.lower()}-title"
    desc_id = f"{spec.panel_id.lower()}-desc"
    accessible_suffix = f". {SYNTHETIC_WATERMARK}." if synthetic else ""
    journal_result_panel = spec.panel_id.startswith(("Fig4", "Fig5"))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{spec.width}" height="{spec.height}" viewBox="0 0 {spec.width} {spec.height}" role="img" aria-labelledby="{title_id} {desc_id}" data-panel="{spec.panel_id}" data-synthetic="{str(synthetic).lower()}">',
        f'<title id="{title_id}">{_esc(spec.panel_id + ". " + spec.title + accessible_suffix)}</title>',
        f'<desc id="{desc_id}">{_esc(spec.description + accessible_suffix)}</desc>',
        '<style>:root{--min-font-size:20px}text{font-family:Arial,Helvetica,sans-serif;font-size:20px;fill:#17212B}.body{font-size:20px}.small{font-size:20px;fill:#65727D}.label{font-size:22px;font-weight:700}.subtitle{font-size:20px;fill:#65727D}.title{font-size:22px;font-weight:700;fill:#17212B}.panel-letter{font-size:26px;font-weight:700;fill:#000000}.watermark{font-size:34px;font-weight:700;letter-spacing:2px;fill:#D55E00}</style>',
        f'<metadata>{_esc(json.dumps({"panel": spec.panel_id, "synthetic": synthetic, "input_contract": ["episode_outcomes", "task_geometry", "steps"]}, sort_keys=True))}</metadata>',
    ]
    if not journal_result_panel:
        parts.append(f'<rect width="{spec.width}" height="{spec.height}" fill="{PANEL}"/>')
    _text(parts, 8, 36, spec.panel_id[-1], "panel-letter")
    _text(parts, 48, 34, spec.title, "title")
    if not journal_result_panel:
        _line(parts, 8, 52, spec.width - 8, 52, GRID, 1.2)
    return parts


def _finish_panel(parts: list[str], spec: PanelSpec, synthetic: bool) -> str:
    if synthetic:
        parts.append(
            f'<text x="{spec.width / 2:.1f}" y="{spec.height / 2:.1f}" class="watermark" text-anchor="middle" opacity="0.16" transform="rotate(-18 {spec.width / 2:.1f} {spec.height / 2:.1f})">{_esc(SYNTHETIC_WATERMARK)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _no_data(parts: list[str], spec: PanelSpec, message: str) -> None:
    _rect(parts, 44, 130, spec.width - 88, spec.height - 180, SOFT, GRID, 8)
    _text(parts, spec.width / 2, spec.height / 2, message, "label", "middle", SLATE)


def _xy(row: Mapping[str, Any]) -> tuple[float, float] | None:
    x = _as_float(_first(row, "x", "x_m", "position_x", "node_x"))
    y = _as_float(_first(row, "y", "y_m", "position_y", "node_y"))
    return None if x is None or y is None else (x, y)


def _map_points(points: Sequence[tuple[float, float]], x: float, y: float, w: float, h: float) -> list[tuple[float, float]]:
    if not points:
        return []
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    xspan, yspan = max(xmax - xmin, 1.0), max(ymax - ymin, 1.0)
    return [(x + (px - xmin) / xspan * w, y + h - (py - ymin) / yspan * h) for px, py in points]


def _convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return a deterministic display hull without estimating unmeasured values."""

    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    def cross(
        origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float],
    ) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _representative_task(geometry: Sequence[Mapping[str, Any]], panel: str) -> str | None:
    selected = [row for row in geometry if _as_bool(_first(row, "representative", "panel_selected", "storyboard_selected")) is True and str(_first(row, "panel", "panel_id") or panel).lower() in {panel.lower(), "", panel[-2:].lower()}]
    task_ids = sorted({_task_value(row) for row in selected if _task_value(row)})
    return task_ids[0] if task_ids else None


def _render_fig1a(outcomes: Sequence[Mapping[str, Any]], geometry: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], order: Sequence[str], synthetic: bool) -> str:
    del outcomes
    spec = PANEL_SPECS["Fig1a"]
    parts = _panel_shell(spec, synthetic)
    task = _representative_task(geometry, "Fig1a")
    geo = [row for row in geometry if task and _task_value(row) == task]
    route_candidates = [row for row in steps if task and _task_value(row) == task and _xy(row)]
    route_model = next((model for model in order if _model_rows(route_candidates, model, order)), None)
    route = _model_rows(route_candidates, route_model, order) if route_model else route_candidates
    # One episode is one continuous route.  Never concatenate the three seeds
    # or different attempts into an apparently continuous trajectory.
    route_episode_ids = sorted({str(_first(row, "episode_id") or "") for row in route})
    if route_episode_ids:
        route = [row for row in route if str(_first(row, "episode_id") or "") == route_episode_ids[0]]
    route.sort(key=lambda row: (_as_float(_first(row, "step_index", "step", "sequence")) or 0))
    points = [_xy(row) for row in geo + route]
    clean = [point for point in points if point is not None]
    if not task or not clean:
        _no_data(parts, spec, "Representative publication geometry not supplied")
        return _finish_panel(parts, spec, synthetic)
    plot = (42.0, 76.0, 916.0, 366.0)
    _vector_lake_background(parts, *plot)
    parts.append('<!-- table-driven scientific-vector-overlay begins -->')
    all_mapped = _map_points(clean, *plot)
    map_iter = iter(all_mapped)
    geo_mapped = [next(map_iter) for _ in geo if _xy(_) is not None]
    route_mapped = [next(map_iter) for _ in route if _xy(_) is not None]
    footprint_mapped = [
        point for row, point in zip((row for row in geo if _xy(row) is not None), geo_mapped)
        if any(token in str(_first(row, "geometry_role", "point_type", "role") or "").lower() for token in ("footprint", "anomaly"))
    ]
    hull = _convex_hull(footprint_mapped)
    if hull:
        _polygon(parts, hull, TEAL, TEAL, opacity=.13)
        stride = max(1, math.ceil(len(footprint_mapped) / 42))
        for x, y in footprint_mapped[::stride]:
            _circle(parts, x, y, 3.2, TEAL, PANEL, .58)
    if route_mapped:
        _polyline(parts, route_mapped, NAVY, 4)
        marker_stride = max(1, math.ceil(len(route_mapped) / 36))
        marker_indices = set(range(0, len(route_mapped), marker_stride)) | {0, len(route_mapped) - 1}
        depths = [_as_float(_first(row, "z", "z_m", "position_z", "node_z")) for row in route]
        finite_depths = [value for value in depths if value is not None]
        depth_min = min(finite_depths) if finite_depths else 0.0
        depth_span = max((max(finite_depths) - depth_min) if finite_depths else 0.0, 1.0)
        for index, ((x, y), depth) in enumerate(zip(route_mapped, depths)):
            if index in marker_indices:
                radius = 3.8 + 3.0 * (((depth or depth_min) - depth_min) / depth_span)
                if index in {0, len(route_mapped) - 1}:
                    radius = max(radius, 8)
                _circle(parts, x, y, radius, SKY if index == 0 else NAVY)
        _text(parts, route_mapped[0][0] + 12, route_mapped[0][1] - 10, "shore access", "small", fill=BLUE)
    role_colors = {"peak": PURPLE, "source": ORANGE}
    role_positions: dict[str, tuple[float, float]] = {}
    mapped_index = 0
    for row in geo:
        if _xy(row) is None:
            continue
        x, y = geo_mapped[mapped_index]
        mapped_index += 1
        role = str(_first(row, "geometry_role", "point_type", "role") or "geometry").lower()
        match = next(((key, value) for key, value in role_colors.items() if key in role), None)
        if match is None:
            continue
        label, color = match
        role_positions[label] = (x, y)
        _circle(parts, x, y, 10, color)
        if x > plot[0] + plot[2] - 110:
            _text(parts, x - 14, y - 10, label, "small", "end", color)
        else:
            _text(parts, x + 14, y - 10, label, "small", fill=color)
    if "source" in role_positions and "peak" in role_positions:
        sx, sy = role_positions["source"]
        px, py = role_positions["peak"]
        _line(parts, sx, sy, px, py, SLATE, 2.5, "8 5")
        _text(parts, (sx + px) / 2, (sy + py) / 2 - 12, "transport separation", "small", "middle", SLATE)
    route_label = f" · {_short_model(route_model)}" if route_model else ""
    _text(parts, 44, 472, f"Task {task}{route_label} · one seed · adjacent-edge route · arrival measurements", "small")
    if route_mapped and finite_depths:
        if math.isclose(min(finite_depths), max(finite_depths)):
            depth_note = f" · route depth {min(finite_depths):g} m"
        else:
            depth_note = f" · route depth {min(finite_depths):g}–{max(finite_depths):g} m"
        depth_note += "; marker size encodes depth"
    else:
        depth_note = ""
    _text(parts, 44, 506, f"Unknown field (pale) · visited ribbon (navy) · registered footprint (teal){depth_note}.", "small", fill=SLATE)
    return _finish_panel(parts, spec, synthetic)


def _render_fig1b(outcomes: Sequence[Mapping[str, Any]], geometry: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], order: Sequence[str], synthetic: bool) -> str:
    del steps
    spec = PANEL_SPECS["Fig1b"]
    parts = _panel_shell(spec, synthetic)
    science_outcomes = [row for row in outcomes if _control_kind(row) is None]
    tasks = {_task_value(row) for row in geometry or science_outcomes if _task_value(row)}
    repeats = {str(_first(row, "repeat_id", "repeat", "seed")) for row in science_outcomes if _first(row, "repeat_id", "repeat", "seed") is not None}
    task_classes: dict[str, str] = {}
    for row in geometry:
        task = _task_value(row)
        task_class = _first(row, "task_class", "case_type", "positive_control")
        if task and task_class not in (None, ""):
            task_classes.setdefault(task, str(task_class).lower())
    classes = Counter(task_classes.values())
    summaries = [row for row in geometry if str(_first(row, "geometry_role") or "") == "task_summary"]
    waterbodies = {str(_first(row, "waterbody") or "") for row in summaries if _first(row, "waterbody")}
    domains = {str(_first(row, "domain_id") or "") for row in summaries if _first(row, "domain_id")}
    task_status: dict[str, bool] = {}
    for row in science_outcomes:
        task = _task_value(row)
        positive = _as_bool(_first(row, "detectable_source", "positive_case"))
        if task and positive is not None:
            task_status[task] = positive
    positive_n = sum(task_status.values())
    negative_n = len(task_status) - positive_n
    metrics = (
        ("waterbodies", len(waterbodies)), ("domains", len(domains)),
        ("tasks", f"{len(tasks)} ({positive_n}+{negative_n})" if task_status else len(tasks)),
        ("design", f"{len(order)} × {len(repeats)}" if repeats else f"{len(order)} × n/a"),
    )
    for index, (label, value) in enumerate(metrics):
        x = 20 + (index % 2) * 300
        y = 72 + (index // 2) * 55
        _text(parts, x, y + 20, label, "small")
        _text(parts, x + 270, y + 20, value if value else "n/a", "label", "end", NAVY)
        _line(parts, x, y + 30, x + 270, y + 30, GRID, 1)
    level_names = ("detect anomaly", "localize footprint", "identify mechanism", "localize source", "make decision")
    for level, name in enumerate(level_names, 1):
        y = 206 + (level - 1) * 50
        _rect(parts, 22, y, 48, 34, LEVEL_COLORS[level - 1], "none", 3)
        _text(parts, 46, y + 25, f"L{level}", "label", "middle", PANEL)
        _text(parts, 84, y + 25, name, "body")
        if level == 2:
            _circle(parts, 540, y + 17, 8, TEAL)
            _text(parts, 520, y + 24, "footprint", "small", "end", TEAL)
        if level == 4:
            _circle(parts, 540, y + 17, 8, ORANGE)
            _text(parts, 520, y + 24, "source", "small", "end", ORANGE)
    if classes:
        summary = " · ".join(f"{name}: {count}" for name, count in sorted(classes.items())[:3])
        _text(parts, 22, 476, summary, "small")
    _text(parts, 22, 514, "Five levels are scored separately; no total score or rank.", "label", fill=ORANGE)
    return _finish_panel(parts, spec, synthetic)


def _draw_level_profile(parts: list[str], spec: PanelSpec, rates: Mapping[str, Sequence[float | None]], order: Sequence[str], x0: float, y0: float, width: float, height: float) -> None:
    for tick in (0.0, 0.5, 1.0):
        yy = y0 + height * (1 - tick)
        _line(parts, x0, yy, x0 + width, yy, GRID, 1.5)
        _text(parts, x0 - 12, yy + 7, f"{tick:.1f}", "small", "end")
    for level in range(1, 6):
        xx = x0 + (level - 1) * width / 4
        _line(parts, xx, y0, xx, y0 + height, GRID, 1)
        _text(parts, xx, y0 + height + 30, f"L{level}", "label", "middle", LEVEL_COLORS[level - 1])
    for model_index, model in enumerate(order):
        values = rates.get(model, ())
        plotted = [(x0 + index * width / 4, y0 + height * (1 - value)) for index, value in enumerate(values) if value is not None]
        color = MODEL_COLORS[model_index % len(MODEL_COLORS)]
        _polyline(parts, plotted, color, 3, 0.85, None if model_index < 3 else "8 5")
        for x, y in plotted:
            _circle(parts, x, y, 5 + (model_index % 2), color)


def _draw_control_profiles(
    parts: list[str], controls: Mapping[str, Sequence[float | None]],
    x0: float, y0: float, width: float, height: float,
) -> None:
    styles = (
        ("reachable_oracle", INK, "12 5", "reachable oracle · registered control"),
        ("design_random", SLATE, "3 6", "3-seed design random · registered control"),
    )
    for control, color, dash, _ in styles:
        values = controls.get(control, ())
        points = [(x0 + index * width / 4, y0 + height * (1 - value)) for index, value in enumerate(values) if value is not None]
        _polyline(parts, points, color, 3, .95, dash)
        for x, y in points:
            _circle(parts, x, y, 4, PANEL, color)


def _render_fig1c(outcomes: Sequence[Mapping[str, Any]], geometry: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], order: Sequence[str], synthetic: bool) -> str:
    del geometry, steps
    spec = PANEL_SPECS["Fig1c"]
    parts = _panel_shell(spec, synthetic)
    controls = _control_level_rates(outcomes)
    summaries = {
        (model, level): _cluster_interval(
            _model_rows(outcomes, model, order), level,
            seed=20260828 + model_index * 101,
        )
        for model_index, model in enumerate(order)
        for level in range(1, 6)
    }
    if not any(value[0] is not None for value in summaries.values()) and not any(value is not None for values in controls.values() for value in values):
        _no_data(parts, spec, "Published L1–L5 outcomes not supplied")
        return _finish_panel(parts, spec, synthetic)
    x0, y0, width, height = 92.0, 82.0, 1030.0, 300.0
    for tick in (0.0, .5, 1.0):
        yy = y0 + height * (1 - tick)
        _line(parts, x0, yy, x0 + width, yy, GRID, 1)
        _text(parts, x0 - 10, yy + 7, f"{tick:.1f}", "small", "end")
    for level in range(1, 6):
        xx = x0 + (level - 1) * width / 4
        _line(parts, xx, y0, xx, y0 + height, "#E7ECEF", 1)
        n_tasks = max((summaries[(model, level)][3] for model in order), default=0)
        _text(parts, xx, y0 + height + 28, f"L{level}", "label", "middle", LEVEL_COLORS[level - 1])
        _text(parts, xx, y0 + height + 53, f"n={n_tasks} tasks", "small", "middle", SLATE)
    offsets = (-25, -15, -5, 5, 15, 25)
    marker_shapes = ("●", "■", "▲", "◆", "○", "□")
    for model_index, model in enumerate(order):
        color = MODEL_COLORS[model_index % len(MODEL_COLORS)]
        for level in range(1, 6):
            mean, low, high, *_ = summaries[(model, level)]
            if mean is None:
                continue
            x = x0 + (level - 1) * width / 4 + offsets[model_index % len(offsets)]
            _draw_vertical_interval(parts, x, low, high, y0, height, color)
            y = y0 + height * (1 - mean)
            if model_index < 4:
                _circle(parts, x, y, 5.5, color, PANEL)
            else:
                _circle(parts, x, y, 5.5, PANEL, color)
    _draw_control_profiles(parts, controls, x0, y0, width, height)
    _line(parts, 1170, 86, 1210, 86, INK, 3, "12 5")
    _text(parts, 1225, 93, "reachable oracle · registered control", "small")
    _line(parts, 1170, 118, 1210, 118, SLATE, 3, "3 6")
    _text(parts, 1225, 125, "3-seed design random · registered control", "small")
    _text(parts, 1170, 154, "Controls are references, not models.", "small", fill=SLATE)
    for index, model in enumerate(order):
        x = 1170 + (index % 2) * 220
        y = 180 + (index // 2) * 58
        color = MODEL_COLORS[index % len(MODEL_COLORS)]
        _text(parts, x, y + 6, marker_shapes[index], "label", fill=color)
        _text(parts, x + 28, y + 6, _short_model(model), "small")
    _text(parts, 1170, 372, "Mean ± 95% two-way cluster CI", "small", fill=SLATE)
    _text(parts, 1170, 401, "Seeds collapsed within task.", "small", fill=SLATE)
    _text(parts, 1170, 426, "Fixed display order; not ranked.", "small", fill=SLATE)
    _text(parts, 1170, 452, "L1 balanced accuracy; L2–L5 pass rates.", "small", fill=ORANGE)
    return _finish_panel(parts, spec, synthetic)


def _render_fig4a(outcomes: Sequence[Mapping[str, Any]], geometry: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], order: Sequence[str], synthetic: bool) -> str:
    del geometry, steps
    spec = PANEL_SPECS["Fig4a"]
    parts = _panel_shell(spec, synthetic)
    controls = _control_level_rates(outcomes)
    fallback_rates = _model_level_rates(outcomes, order)
    left, col_w, axis_w = 300.0, 320.0, 238.0
    row_y = (132, 176, 220, 264, 308, 352)
    level_names = (
        ("L1", "balanced accuracy"),
        ("L2", "anomaly footprint"),
        ("L3", "mechanism"),
        ("L4", "source localization"),
    )
    for level_index, (code, construct) in enumerate(level_names):
        x = left + level_index * col_w
        _text(parts, x, 74, code, "label", fill=INK)
        _text(parts, x + 38, 74, construct, "small", fill=SLATE)
        _line(parts, x, 110, x + axis_w, 110, INK, 1.2)
        for tick, label in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
            tx = x + tick * axis_w
            _line(parts, tx, 106, tx, 114, INK, 1.0)
            _text(parts, tx, 103, label, "small", "middle", SLATE)
        for control, color, dash in (
            ("reachable_oracle", INK, None),
            ("design_random", SLATE, "5 5"),
        ):
            values = controls.get(control, ())
            value = values[level_index] if level_index < len(values) else None
            if value is not None:
                px = x + max(0.0, min(1.0, value)) * axis_w
                _line(parts, px, 118, px, 365, color, 1.5, dash)
    _line(parts, 1080, 28, 1120, 28, INK, 1.5)
    _text(parts, 1132, 35, "reachable oracle", "small", fill=INK)
    _line(parts, 1315, 28, 1355, 28, SLATE, 1.5, "5 5")
    _text(parts, 1367, 35, "design random", "small", fill=SLATE)
    for model_index, model in enumerate(order):
        y = row_y[model_index]
        color, shape = _model_style(model, order)
        _marker(parts, 24, y, 5.8, shape, color, PANEL, model=model)
        _text(parts, 274, y + 6, _short_model(model), "body", "end", INK)
        subset = _model_rows(outcomes, model, order)
        for level_index, level in enumerate(FIG4_DISPLAY_LEVELS):
            point, low, high, *_ = _cluster_interval(
                subset, level, seed=20260830 + model_index * 100 + level,
            )
            if point is None:
                point = fallback_rates.get(model, (None,) * 5)[level_index]
            x = left + level_index * col_w
            _line(parts, x, y, x + axis_w, y, "#E5E8EA", 1)
            if point is None:
                _text(parts, x + axis_w / 2, y + 6, "NA", "small", "middle", SLATE)
                continue
            if low is not None and high is not None:
                _line(parts, x + low * axis_w, y, x + high * axis_w, y, color, 3)
            _marker(parts, x + point * axis_w, y, 5.6, shape, color, PANEL, model=model)
    return _finish_panel(parts, spec, synthetic)


BEHAVIOUR_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("Endurance", "% used", "endurance_used_fraction"),
    ("Move / analyze", "native log axis", "move_analyze_ratio"),
    ("Submission", "% episodes", "submitted"),
    ("Visited nodes", "unique 3-D nodes", "visited_nodes"),
    ("Nearest source", "metres", "trajectory_source_closest_distance_m"),
    ("Repeated actions", "% actions", "repetition_fraction"),
)


def _behaviour_value(row: Mapping[str, Any], field: str) -> float | None:
    """Return one registered episode-level process measure without clipping."""

    if field == "submitted":
        value = _as_bool(row.get(field))
        return None if value is None else (1.0 if value else 0.0)
    value = _as_float(row.get(field))
    if value is not None:
        return value
    if field == "move_analyze_ratio":
        moves = _as_float(_first(row, "move_calls", "n_move"))
        analyses = _as_float(_first(row, "analyze_calls", "n_analyze"))
        return moves / analyses if moves is not None and analyses and analyses > 0 else None
    if field == "repetition_fraction":
        repeats = _as_float(_first(row, "repeat_diagnostic_count", "repeated_action_steps"))
        attempts = _as_float(_first(row, "action_attempt_n", "total_action_steps"))
        return repeats / attempts if repeats is not None and attempts and attempts > 0 else None
    if field == "coverage_fraction":
        visited = _as_float(_first(row, "visited_nodes", "unique_visited_nodes"))
        total = _as_float(_first(row, "n_total_nodes", "domain_node_count"))
        return visited / total if visited is not None and total and total > 0 else None
    return None


def _task_equal_actor_mean(
    rows: Sequence[Mapping[str, Any]], field: str,
) -> float | None:
    """Aggregate episode → actor×task → actor, giving every task equal weight."""

    task_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _behaviour_value(row, field)
        if value is not None and math.isfinite(value):
            task_values[_task_value(row)].append(value)
    task_means = [statistics.fmean(values) for values in task_values.values() if values]
    return statistics.fmean(task_means) if task_means else None


def _task_equal_actor_values(
    rows: Sequence[Mapping[str, Any]], field: str,
) -> list[float]:
    """Return one seed-collapsed value per task for a process measure."""

    task_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _behaviour_value(row, field)
        if value is not None and math.isfinite(value):
            task_values[_task_value(row)].append(value)
    return [statistics.fmean(values) for _, values in sorted(task_values.items()) if values]


def _behaviour_label(field: str, value: float | None) -> str:
    if value is None:
        return "n/a (0 analyses)" if field == "move_analyze_ratio" else "n/a"
    if field in {"endurance_used_fraction", "submitted", "coverage_fraction", "repetition_fraction"}:
        return f"{100 * value:.1f}%"
    if field == "visited_nodes":
        return f"{value:.0f}"
    if field == "trajectory_source_closest_distance_m":
        return f"{value:.1f} m"
    return f"{value:.1f}×"


def _blend_hex(color: str, fraction: float, background: str = "#FFFFFF") -> str:
    """Return an opaque colour preblended against a known background."""

    fraction = max(0.0, min(1.0, fraction))
    foreground = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
    backdrop = tuple(int(background[index:index + 2], 16) for index in (1, 3, 5))
    channels = tuple(round(fraction * a + (1 - fraction) * b) for a, b in zip(foreground, backdrop))
    return "#%02X%02X%02X" % channels


FIG4B_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("Endurance", "% used", "endurance_used_fraction"),
    ("Move / analyze", "ratio, log scale", "move_analyze_ratio"),
    ("Visited nodes", "unique 3-D nodes", "visited_nodes"),
    ("Nearest source", "metres", "trajectory_source_closest_distance_m"),
)


def _render_fig4b(outcomes: Sequence[Mapping[str, Any]], geometry: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], order: Sequence[str], synthetic: bool) -> str:
    del geometry, steps
    spec = PANEL_SPECS["Fig4b"]
    parts = _panel_shell(spec, synthetic)
    x0, col_w, axis_w = 300.0, 320.0, 238.0
    row_y = (164, 206, 248, 290, 332, 374)
    task_values: dict[tuple[str, str], list[float]] = {}
    metric_pool: dict[str, list[float]] = defaultdict(list)
    for model in order:
        subset = _model_rows(outcomes, model, order)
        for _, _, field in FIG4B_FIELDS:
            values = _task_equal_actor_values(subset, field)
            task_values[(model, field)] = values
            metric_pool[field].extend(values)

    def domain(field: str) -> tuple[float, float, bool]:
        values = metric_pool[field]
        if field == "move_analyze_ratio":
            upper = max(10.0, 10 ** math.ceil(math.log10(max(values, default=1.0))))
            return 1.0, upper, True
        if field in {"endurance_used_fraction", "submitted", "coverage_fraction", "repetition_fraction"}:
            return 0.0, 1.0, False
        if field == "visited_nodes":
            upper = max(10.0, 10.0 * math.ceil(max(values, default=10.0) / 10.0))
            return 0.0, upper, False
        upper = max(25.0, 25.0 * math.ceil(max(values, default=25.0) / 25.0))
        return 0.0, upper, False

    def scale(value: float, lo: float, hi: float, log_axis: bool) -> float:
        if log_axis:
            return 0.0 if value <= 0 else math.log10(max(lo, value) / lo) / math.log10(hi / lo)
        return (value - lo) / (hi - lo) if hi > lo else 0.0

    for metric_index, (title, subtitle, field) in enumerate(FIG4B_FIELDS):
        x = x0 + metric_index * col_w
        lo, hi, log_axis = domain(field)
        _text(parts, x, 75, title, "label", fill=INK)
        _text(parts, x, 98, subtitle, "small", fill=SLATE)
        _line(parts, x, 116, x + axis_w, 116, INK, 1.2)
        if log_axis:
            ticks = [10.0 ** power for power in range(0, int(math.log10(hi)) + 1)]
        else:
            ticks = [lo, (lo + hi) / 2, hi]
        for value in ticks:
            fraction = max(0.0, min(1.0, scale(value, lo, hi, log_axis)))
            tx = x + fraction * axis_w
            _line(parts, tx, 112, tx, 120, INK, 1.0)
            if log_axis:
                label = f"{value:g}"
            elif hi == 1:
                label = f"{100 * value:.0f}"
            else:
                label = f"{value:g}"
            _text(parts, tx, 139, label, "small", "middle", SLATE)
    has_data = any(metric_pool.values())
    for model_index, model in enumerate(order):
        y = row_y[model_index]
        color, shape = _model_style(model, order)
        _line(parts, 22, y, 1560, y, "#ECEEEF", 1)
        _marker(parts, 24, y, 5.8, shape, color, PANEL, model=model)
        _text(parts, 274, y + 6, _compact_model(model), "body", "end", INK)
        for metric_index, (_, _, field) in enumerate(FIG4B_FIELDS):
            x = x0 + metric_index * col_w
            values = task_values[(model, field)]
            lo, hi, log_axis = domain(field)
            if not values:
                _text(parts, x, y + 6, "NA", "small", fill=SLATE)
                continue
            stride = max(1, math.ceil(len(values) / 36))
            pale = _blend_hex(color, .38)
            for value_index, value in enumerate(values[::stride][:36]):
                fraction = max(0.0, min(1.0, scale(value, lo, hi, log_axis)))
                jitter = ((value_index % 7) - 3) * 2.2
                _circle(parts, x + fraction * axis_w, y + jitter, 3.3, pale, "none")
            median = statistics.median(values)
            q1 = _quantile(values, .25)
            q3 = _quantile(values, .75)
            if q1 is not None and q3 is not None:
                _line(
                    parts,
                    x + max(0.0, min(1.0, scale(q1, lo, hi, log_axis))) * axis_w, y,
                    x + max(0.0, min(1.0, scale(q3, lo, hi, log_axis))) * axis_w, y,
                    color, 3,
                )
            _marker(
                parts, x + max(0.0, min(1.0, scale(median, lo, hi, log_axis))) * axis_w,
                y, 5.2, shape, color, PANEL, model=model,
            )
    if not has_data:
        _no_data(parts, spec, "Published behaviour metrics not supplied")
    return _finish_panel(parts, spec, synthetic)


TERMINAL_ORDER = ("submitted", "step_limit", "repetition_loop", "consecutive_invalid", "infrastructure_failure")
TERMINAL_COLORS = ("#58636A", "#939CA1", "#BDC4C7", "#D9DDDF", "#F2F3F3")


def _terminal(row: Mapping[str, Any]) -> str:
    raw = str(_first(row, "terminal_state", "termination_category", "termination_reason") or "").lower()
    submitted = _as_bool(_first(row, "submitted", "submission_valid"))
    if submitted is True or raw in {"submitted", "valid_submission", "success"}:
        return "submitted"
    if any(token in raw for token in ("step", "round_guard", "budget", "endurance", "range")):
        return "step_limit"
    if "repeat" in raw or "loop" in raw:
        return "repetition_loop"
    if "infrastructure" in raw or "transport" in raw or "provider" in raw:
        return "infrastructure_failure"
    return "consecutive_invalid"


def _render_fig4c(outcomes: Sequence[Mapping[str, Any]], geometry: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], order: Sequence[str], synthetic: bool) -> str:
    del geometry, steps
    spec = PANEL_SPECS["Fig4c"]
    parts = _panel_shell(spec, synthetic)
    left, axis_x, axis_w, bar_h = 24.0, 229.0, 510.0, 21.0
    cap = .35
    for tick in (0.0, .10, .20, .30, .35):
        x = axis_x + tick / cap * axis_w
        _line(parts, x, 74, x, 340, "#E2E5E7", 1)
        _line(parts, x, 70, x, 78, INK, 1)
        _text(parts, x, 65, f"{100*tick:.0f}%", "small", "middle", SLATE)
    _line(parts, axis_x, 74, axis_x + axis_w, 74, INK, 1.2)
    any_rows = False
    for model_index, model in enumerate(order):
        subset = _model_rows(outcomes, model, order)
        y = 104 + model_index * 44
        color, shape = _model_style(model, order)
        _marker(parts, left, y, 5.6, shape, color, PANEL, model=model)
        _text(parts, left + 187, y + 6, _compact_model(model), "body", "end", INK)
        x = axis_x
        counts = Counter(_terminal(row) for row in subset)
        total = len(subset)
        any_rows = any_rows or bool(total)
        for category, state_color in zip(TERMINAL_ORDER[1:], TERMINAL_COLORS[1:]):
            fraction = counts[category] / total if total else 0
            width = axis_w * fraction / cap
            if width:
                _rect(parts, x, y - bar_h / 2, min(width, axis_x + axis_w - x), bar_h, state_color, "#7B858A", 0)
            x += width
    if not any_rows:
        _no_data(parts, spec, "Published terminal states not supplied")
    else:
        for index, (category, state_color) in enumerate(zip(TERMINAL_ORDER[1:], TERMINAL_COLORS[1:])):
            x = 24 + (index % 2) * 390
            y = 360 + (index // 2) * 26
            _rect(parts, x, y - 15, 17, 17, state_color, "#7B858A", 0)
            _text(parts, x + 24, y, category.replace("_", " "), "small")
    return _finish_panel(parts, spec, synthetic)


def _three_run_reliability(
    rows: Sequence[Mapping[str, Any]], level: int,
) -> tuple[float | None, int]:
    """Return the fraction of tasks passed in all three registered runs.

    L1 keeps the registered class balance by averaging positive- and
    negative-task reliability.  L2--L4 use their ordinary applicable-task
    denominators.  Tasks lacking three scored repeats are excluded and their
    reduced denominator is exposed in the panel-data export.
    """

    task_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if _level_value(row, level) is not None:
            task_rows[_task_value(row)].append(row)
    records: list[tuple[bool, bool | None]] = []
    for task, subset in sorted(task_rows.items()):
        by_seed: dict[str, bool] = {}
        for row in subset:
            seed = str(_first(row, "seed", "repeat_id", "replicate", "trial") or len(by_seed))
            value = _level_value(row, level)
            if value is not None:
                by_seed[seed] = bool(value)
        if len(by_seed) < 3:
            continue
        passed = all(value for _, value in sorted(by_seed.items())[:3])
        positive = _as_bool(_first(subset[0], "detectable_source", "positive_case"))
        records.append((passed, positive))
    if not records:
        return None, 0
    if level == 1:
        positive_values = [float(passed) for passed, positive in records if positive is True]
        negative_values = [float(passed) for passed, positive in records if positive is False]
        if positive_values and negative_values:
            return (
                (statistics.fmean(positive_values) + statistics.fmean(negative_values)) / 2,
                len(records),
            )
    return statistics.fmean(float(passed) for passed, _ in records), len(records)


def _render_fig4d(outcomes: Sequence[Mapping[str, Any]], geometry: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], order: Sequence[str], synthetic: bool) -> str:
    del geometry, steps
    spec = PANEL_SPECS["Fig4d"]
    parts = _panel_shell(spec, synthetic)
    left, top, plot_h, facet_w = 64.0, 76.0, 216.0, 168.0
    single_rates = _model_level_rates(outcomes, order)
    for tick in (0.0, .5, 1.0):
        y = top + plot_h * (1 - tick)
        _text(parts, left - 10, y + 6, f"{100*tick:.0f}", "small", "end", SLATE)
        _line(parts, left, y, left + len(FIG4_DISPLAY_LEVELS) * facet_w - 20, y, "#E2E5E7", 1)
    reliability: dict[tuple[str, int], tuple[float | None, int]] = {}
    supplied = False
    for level_index, level in enumerate(FIG4_DISPLAY_LEVELS):
        facet_left = left + level_index * facet_w
        x_single, x_repeat = facet_left + 38, facet_left + 108
        _text(parts, facet_left + 73, 62, f"L{level}", "label", "middle", INK)
        _text(parts, x_single, 319, "1 run", "small", "middle", SLATE)
        _text(parts, x_repeat, 319, "3/3", "small", "middle", SLATE)
        for x in (x_single, x_repeat):
            _line(parts, x, top, x, top + plot_h, "#E2E5E7", 1)
        for model_index, model in enumerate(order):
            subset = _model_rows(outcomes, model, order)
            color, shape = _model_style(model, order)
            value, task_n = _three_run_reliability(subset, level)
            reliability[(model, level)] = (value, task_n)
            supplied = supplied or value is not None
            single = single_rates.get(model, (None,) * 5)[level - 1]
            if value is None or single is None:
                continue
            y_single = top + plot_h * (1 - max(0.0, min(1.0, single)))
            y_repeat = top + plot_h * (1 - max(0.0, min(1.0, value)))
            _line(parts, x_single, y_single, x_repeat, y_repeat, color, 2)
            _marker(parts, x_single, y_single, 4.4, shape, PANEL, color, model=model)
            _marker(parts, x_repeat, y_repeat, 4.4, shape, color, PANEL, model=model)
    if not supplied:
        _text(parts, left, 350, "No complete three-run task sets supplied.", "small", fill=ORANGE)
    else:
        for model_index, model in enumerate(order):
            color, shape = _model_style(model, order)
            x = 18 + (model_index % 3) * 255
            y = 354 + (model_index // 3) * 28
            _marker(parts, x, y - 5, 4.5, shape, color, PANEL, model=model)
            _text(parts, x + 15, y, _compact_model(model), "small", fill=INK)
    return _finish_panel(parts, spec, synthetic)


def _geometry_dstar(geometry: Sequence[Mapping[str, Any]]) -> dict[str, tuple[float | None, str | None]]:
    output: dict[str, tuple[float | None, str | None]] = {}
    for row in geometry:
        task = _task_value(row)
        if not task:
            continue
        value = _as_float(_first(row, "dstar", "d_star", "D_star", "Dstar", "registered_d_star"))
        stratum_raw = _first(row, "d_star_stratum", "D_star_stratum", "geometry_stratum")
        if value is not None:
            stratum = (
                "D* < 1" if value < 1 else
                "1 ≤ D* < 2" if value < 2 else
                "2 ≤ D* < 4" if value < 4 else
                "D* ≥ 4"
            )
        else:
            stratum = str(stratum_raw) if stratum_raw not in (None, "") else None
        previous = output.get(task)
        if previous is None or previous == (None, None):
            output[task] = (value, stratum)
    return output


def _render_fig5a(outcomes: Sequence[Mapping[str, Any]], geometry: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], order: Sequence[str], synthetic: bool) -> str:
    del outcomes, steps, order
    spec = PANEL_SPECS["Fig5a"]
    parts = _panel_shell(spec, synthetic)
    values = [(task, value, stratum) for task, (value, stratum) in sorted(_geometry_dstar(geometry).items()) if value is not None]
    if not values:
        _no_data(parts, spec, "Registered D* values not supplied by task_geometry")
        return _finish_panel(parts, spec, synthetic)
    _vector_lake_background(parts, 16, 66, 328, 260, label=False)
    parts.append('<!-- task_geometry-driven scientific-vector-overlay begins -->')
    representative = _representative_task(geometry, "Fig5a") or _representative_task(geometry, "Fig1a")
    geo_rows = [row for row in geometry if representative and _task_value(row) == representative and _xy(row)]
    footprint_rows = [row for row in geo_rows if "footprint" in str(_first(row, "geometry_role", "point_type", "role") or "").lower()]
    role_rows = [row for row in geo_rows if any(token in str(_first(row, "geometry_role", "point_type", "role") or "").lower() for token in ("source", "peak"))]
    all_points = [point for row in footprint_rows + role_rows if (point := _xy(row)) is not None]
    mapped_all = _map_points(all_points, 40, 94, 275, 185) if all_points else []
    footprint_mapped = mapped_all[:len(footprint_rows)]
    role_mapped = mapped_all[len(footprint_rows):]
    hull = _convex_hull(footprint_mapped)
    if hull:
        _polygon(parts, hull, _blend_hex(TEAL, .20), TEAL)
    role_positions: dict[str, tuple[float, float]] = {}
    for row, (mx, my) in zip(role_rows, role_mapped):
        role = str(_first(row, "geometry_role", "point_type", "role") or "geometry").lower()
        label, color = ("source", ORANGE) if "source" in role else ("peak", PURPLE)
        role_positions[label] = (mx, my)
        _circle(parts, mx, my, 10, color)
        offset = -12 if mx > 260 else 12
        anchor = "end" if offset < 0 else "start"
        _text(parts, mx + offset, my - 10, label, "small", anchor, color)
    if "source" in role_positions and "peak" in role_positions:
        sx, sy = role_positions["source"]
        px, py = role_positions["peak"]
        _line(parts, sx, sy, px, py, INK, 2, "6 4")
        _text(parts, (sx + px) / 2, (sy + py) / 2 - 10, "D*", "label", "middle", INK)
    x0, y0, width, height = 38.0, 398.0, 285.0, 324.0
    vmax = max(4.0, max(value for _, value, _ in values))
    transformed_max = math.log2(1 + vmax)
    ticks = [tick for tick in (0, .5, 1, 2, 4, 8, 16, 32) if tick <= vmax]
    for tick in ticks:
        x = x0 + math.log2(1 + tick) / transformed_max * width
        _line(parts, x, y0, x, y0 + height, "#E2E5E7", 1)
        _line(parts, x, y0 + height, x, y0 + height + 8, INK, 1)
        _text(parts, x, y0 + height + 31, tick, "small", "middle")
    boundary = x0 + math.log2(2) / transformed_max * width
    _rect(parts, x0, y0, boundary - x0, height, "#F7ECE8", "none", 0)
    _line(parts, boundary, y0 - 8, boundary, y0 + height, ORANGE, 2, "6 5")
    _text(parts, x0, y0 - 17, "D* < 1", "small", fill=ORANGE)
    _text(parts, boundary + 10, y0 - 17, "D* ≥ 1", "small", fill=INK)
    for index, (_, value, _) in enumerate(values):
        x = x0 + min(math.log2(1 + value) / transformed_max, 1) * width
        y = y0 + 14 + (index % 18) * 16.5
        if value < 1:
            _circle(parts, x, y, 4.3, PANEL, ORANGE)
        else:
            _circle(parts, x, y, 4.3, INK, PANEL)
    _line(parts, x0, y0 + height, x0 + width, y0 + height, INK, 1.2)
    _text(parts, x0 + width / 2, 797, "registered D* (log display)", "small", "middle", SLATE)
    return _finish_panel(parts, spec, synthetic)


def _stratum_for(row: Mapping[str, Any], geometry_map: Mapping[str, tuple[float | None, str | None]]) -> str | None:
    mapped = geometry_map.get(_task_value(row), (None, None))[1]
    if mapped is not None:
        return mapped
    supplied = _first(row, "d_star_stratum", "D_star_stratum", "geometry_stratum")
    if supplied not in (None, ""):
        return str(supplied)
    return None


def _render_fig5b(outcomes: Sequence[Mapping[str, Any]], geometry: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], order: Sequence[str], synthetic: bool) -> str:
    del steps
    spec = PANEL_SPECS["Fig5b"]
    parts = _panel_shell(spec, synthetic)
    geometry_map = _geometry_dstar(geometry)
    strata = ("D* < 1", "1 ≤ D* < 2", "2 ≤ D* < 4", "D* ≥ 4")
    any_data = False
    left, top, plot_h, facet_w = 66.0, 116.0, 516.0, 160.0
    for tick in (0.0, .5, 1.0):
        y = top + plot_h * (1 - tick)
        _text(parts, left - 10, y + 6, f"{tick:.1f}", "small", "end", SLATE)
        _line(parts, left, y, left + 4 * facet_w - 18, y, "#E2E5E7", 1)
    for stratum_index, stratum in enumerate(strata):
        task_ids = {
            task for task, (_, mapped_stratum) in geometry_map.items()
            if mapped_stratum == stratum
        }
        facet_left = left + stratum_index * facet_w
        x_l2, x_l4 = facet_left + 38, facet_left + 102
        short_label = ("D* < 1", "1 ≤ D* < 2", "2 ≤ D* < 4", "D* ≥ 4")[stratum_index]
        _text(parts, facet_left + 70, 72, short_label, "label", "middle", INK)
        _text(parts, facet_left + 70, 96, f"n={len(task_ids)} tasks", "small", "middle", SLATE)
        for x in (x_l2, x_l4):
            _line(parts, x, top, x, top + plot_h, "#E2E5E7", 1)
        _text(parts, x_l2, 662, "L2", "small", "middle", SLATE)
        _text(parts, x_l4, 662, "L4", "small", "middle", SLATE)
        for model_index, model in enumerate(order):
            subset = [row for row in _model_rows(outcomes, model, order) if _stratum_for(row, geometry_map) == stratum]
            l2, l2_low, l2_high, *_ = _cluster_interval(subset, 2, seed=20260828 + stratum_index * 200 + model_index)
            l4, l4_low, l4_high, *_ = _cluster_interval(subset, 4, seed=20260829 + stratum_index * 200 + model_index)
            if l2 is None or l4 is None:
                continue
            any_data = True
            color, shape = _model_style(model, order)
            y_l2 = top + plot_h * (1 - l2)
            y_l4 = top + plot_h * (1 - l4)
            _line(parts, x_l2, y_l2, x_l4, y_l4, color, 2)
            for x, mean, low, high in (
                (x_l2, l2, l2_low, l2_high),
                (x_l4, l4, l4_low, l4_high),
            ):
                if low is not None and high is not None:
                    y_low = top + plot_h * (1 - low)
                    y_high = top + plot_h * (1 - high)
                    _line(parts, x, y_high, x, y_low, color, 2)
                    _line(parts, x - 4, y_high, x + 4, y_high, color, 1)
                    _line(parts, x - 4, y_low, x + 4, y_low, color, 1)
                fill = PANEL if x == x_l2 else color
                stroke = color if x == x_l2 else PANEL
                _marker(parts, x, top + plot_h * (1 - mean), 4.8, shape, fill, stroke, model=model)
    if not any_data:
        _no_data(parts, spec, "Paired L2/L4 outcomes or D* strata not supplied")
    else:
        for model_index, model in enumerate(order):
            color, shape = _model_style(model, order)
            x = 30 + (model_index % 3) * 230
            y = 725 + (model_index // 3) * 31
            _marker(parts, x, y - 5, 4.7, shape, color, PANEL, model=model)
            _text(parts, x + 16, y, _compact_model(model), "small", fill=INK)
    return _finish_panel(parts, spec, synthetic)


def _projection_value(row: Mapping[str, Any]) -> float | None:
    # v20 publishes rho as a set-distance diagnostic: 0 is source-relative
    # and 1 is peak-relative.  Older projection aliases remain read-only
    # fallbacks for compatibility with preregistration fixtures.
    return _as_float(_first(row, "rho", "source_peak_projection", "projection_pi", "pi", "submitted_axis_projection"))


def _render_fig5c(outcomes: Sequence[Mapping[str, Any]], geometry: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], order: Sequence[str], synthetic: bool) -> str:
    del geometry, steps
    spec = PANEL_SPECS["Fig5c"]
    parts = _panel_shell(spec, synthetic)
    x0, width = 135.0, 340.0
    _line(parts, x0, 682, x0 + width, 682, INK, 1.2)
    for tick, label in ((0.0, "source"), (0.5, "equidistant"), (1.0, "peak")):
        x = x0 + tick * width
        _line(parts, x, 82, x, 690, "#DDE2E4" if tick == .5 else SLATE, 1.5, "6 5" if tick == .5 else None)
        _line(parts, x, 682, x, 690, INK, 1)
        anchor = "start" if tick == 0.0 else "end" if tick == 1.0 else "middle"
        _text(parts, x, 714, label, "small", anchor)
    _text(parts, x0 + width / 2, 750, "relative set distance, ρ", "small", "middle", SLATE)
    any_data = False
    for model_index, model in enumerate(order):
        model_subset = _model_rows(outcomes, model, order)
        has_case_labels = any(
            _as_bool(_first(row, "detectable_source", "positive_case")) is not None
            for row in model_subset
        )
        subset = [
            row for row in model_subset
            if not has_case_labels or _as_bool(_first(row, "detectable_source", "positive_case")) is True
        ]
        values = [_projection_value(row) for row in subset]
        values = [value for value in values if value is not None]
        y = 130 + model_index * 88
        color, shape = _model_style(model, order)
        _marker(parts, 18, y, 5.0, shape, color, PANEL, model=model)
        _text(parts, x0 - 14, y + 6, _compact_model(model), "small", "end", INK)
        _line(parts, x0, y, x0 + width, y, "#ECEEEF", 1)
        if values:
            any_data = True
        stride = max(1, math.ceil(len(values) / 40))
        pale = _blend_hex(color, .36)
        for value_index, value in enumerate(values[::stride][:40]):
            clipped = max(0.0, min(1.0, value))
            jitter = ((value_index % 7) - 3) * 2.5
            _circle(parts, x0 + clipped * width, y + jitter, 3.3, pale, "none")
        if values:
            ordered = sorted(values)
            median = statistics.median(ordered)
            q1 = ordered[int(.25 * (len(ordered) - 1))]
            q3 = ordered[int(.75 * (len(ordered) - 1))]
            _line(parts, x0 + q1 * width, y, x0 + q3 * width, y, color, 3)
            _marker(parts, x0 + median * width, y, 5.3, shape, color, PANEL, model=model)
    if not any_data:
        _no_data(parts, spec, "Published rho values not supplied")
    return _finish_panel(parts, spec, synthetic)


def _storyboard_task(outcomes: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]) -> str | None:
    rows = list(outcomes) + list(steps)
    selected = sorted({_task_value(row) for row in rows if _task_value(row) and _as_bool(_first(row, "storyboard_selected", "case_selected")) is True})
    return selected[0] if selected else None


def _render_fig6a(outcomes: Sequence[Mapping[str, Any]], geometry: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], order: Sequence[str], synthetic: bool) -> str:
    spec = PANEL_SPECS["Fig6a"]
    parts = _panel_shell(spec, synthetic)
    task = _storyboard_task(outcomes, steps)
    selected_outcomes = [
        row for row in outcomes
        if task and _task_value(row) == task
        and _as_bool(_first(row, "storyboard_selected", "case_selected")) is True
    ]
    selected_episode_ids = {
        str(_first(row, "episode_id")) for row in selected_outcomes
        if _first(row, "episode_id") not in (None, "")
    }
    task_steps = [
        row for row in steps
        if task and _task_value(row) == task and _xy(row)
        and (
            _as_bool(_first(row, "storyboard_selected", "case_selected")) is True
            or str(_first(row, "episode_id")) in selected_episode_ids
        )
    ]
    actor_slots: dict[str, float] = {}
    for row in selected_outcomes + task_steps:
        actor = _canonical_model(_model_value(row), order)
        slot = _as_float(_first(row, "storyboard_actor_slot", "actor_slot"))
        if slot is not None:
            actor_slots[actor] = min(slot, actor_slots.get(actor, slot))
    models = sorted(
        { _canonical_model(_model_value(row), order) for row in task_steps },
        key=lambda model: (actor_slots.get(model, float("inf")), order.index(model) if model in order else len(order), model),
    )[:2]
    if not task or not models:
        _no_data(parts, spec, "Upstream-selected storyboard case not supplied")
        return _finish_panel(parts, spec, synthetic)
    task_geo = [row for row in geometry if _task_value(row) == task and _xy(row)]
    footprint_geo = [row for row in task_geo if "footprint" in str(_first(row, "geometry_role", "role") or "").lower()]
    landmark_geo = [
        row for row in task_geo
        if any(token in str(_first(row, "geometry_role", "role") or "").lower() for token in ("source", "peak"))
    ]
    common_points = [point for row in footprint_geo + landmark_geo + task_steps if (point := _xy(row)) is not None]
    common_bounds = _bounds(common_points)
    for lane, model in enumerate(models):
        x, y, w, h = 16 + lane * 810, 68, 792, 390
        model_color = MODEL_COLORS[order.index(model) % len(MODEL_COLORS)] if model in order else MODEL_COLORS[lane]
        _line(parts, x, y + 32, x + w, y + 32, GRID, 1)
        _text(parts, x, y + 24, _short_model(model), "label", fill=model_color)
        rows = _model_rows(task_steps, model, order)
        episode_ids = {
            str(_first(row, "episode_id")) for row in selected_outcomes
            if _canonical_model(_model_value(row), order) == model and _first(row, "episode_id")
        }
        if episode_ids:
            rows = [row for row in rows if str(_first(row, "episode_id")) in episode_ids]
        rows.sort(key=lambda row: (_as_float(_first(row, "step_index", "step", "sequence")) or 0))
        raw_points = [_xy(row) for row in rows]
        clean = [point for point in raw_points if point is not None]
        map_x, map_y, map_w, map_h = x, y + 48, 360, 300
        _vector_lake_background(parts, map_x, map_y, map_w, map_h)
        footprint_points = [point for row in footprint_geo if (point := _xy(row)) is not None]
        footprint_mapped = _map_points_bounds(footprint_points, common_bounds, map_x + 18, map_y + 34, map_w - 36, map_h - 54)
        hull = _convex_hull(footprint_mapped)
        if hull:
            _polygon(parts, hull, TEAL, TEAL, opacity=.16)
        for row in landmark_geo:
            point = _xy(row)
            if point is None:
                continue
            mx, my = _map_points_bounds([point], common_bounds, map_x + 18, map_y + 34, map_w - 36, map_h - 54)[0]
            role = str(_first(row, "geometry_role", "role") or "").lower()
            label, color = ("source", ORANGE) if "source" in role else ("peak", PURPLE)
            _circle(parts, mx, my, 7, color)
            _text(parts, mx + 9, my - 7, label, "small", fill=color)
        mapped = _map_points_bounds(clean, common_bounds, map_x + 18, map_y + 34, map_w - 36, map_h - 54)
        _polyline(parts, mapped, model_color, 4)
        for index, (px, py) in enumerate(mapped):
            if index in {0, len(mapped) // 2, len(mapped) - 1}:
                _circle(parts, px, py, 7, SKY if index == 0 else model_color)
        snapshot_indices = sorted({0, max(0, len(rows) // 2), max(0, len(rows) - 1)})
        snapshot_labels = ("start", "turn", "end")
        for card_index, row_index in enumerate(snapshot_indices[:3]):
            row = rows[row_index]
            cy = y + 52 + card_index * 100
            _rect(parts, x + 382, cy, 398, 86, PANEL, GRID, 2)
            step_index = int(_as_float(_first(row, "step_index", "step")) or 0)
            intent = str(_first(row, "intent") or "not reported")
            evidence = str(_first(row, "evidence_node_ids_csv") or "")
            evidence_n = len([value for value in evidence.split(",") if value.strip()])
            confidence = _as_float(_first(row, "confidence"))
            budget = _as_float(_first(row, "endurance_used_fraction"))
            _text(parts, x + 394, cy + 23, f"{snapshot_labels[card_index]} · step {step_index}", "label", fill=model_color)
            _text(parts, x + 394, cy + 49, f"intent: {intent} · measured evidence: {evidence_n} node(s)", "small")
            confidence_label = "n/a" if confidence is None else f"{confidence:.2f}"
            budget_label = "n/a" if budget is None else f"{100*budget:.1f}%"
            _text(parts, x + 394, cy + 74, f"confidence {confidence_label} · endurance used {budget_label}", "small", fill=SLATE)
        _text(parts, x, y + h - 4, f"{len(mapped)} measured arrivals · reader-only source/peak/footprint overlay", "small", fill=SLATE)
    _text(parts, 16, 488, f"Task {task} · deterministic upstream selection · envelopes are external records, not chain-of-thought.", "small")
    return _finish_panel(parts, spec, synthetic)


STAGE_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("A · entered footprint", ("checkpoint_A_footprint_entry", "entered_anomaly_footprint", "entered_footprint")),
    ("B | A · source approach", ("checkpoint_B_source_approach_after_A", "source_approach_after_footprint")),
    ("C | B · correct submit", ("checkpoint_C_correct_L4_submission", "correct_l4_submit", "L4_pass", "l4_pass")),
)


def _render_fig6b(outcomes: Sequence[Mapping[str, Any]], geometry: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], order: Sequence[str], synthetic: bool) -> str:
    del geometry, steps
    spec = PANEL_SPECS["Fig6b"]
    parts = _panel_shell(spec, synthetic)
    stage_x = (260.0, 455.0, 650.0)
    axis_w = 130.0
    stage_headings = (
        ("A", "entered footprint"),
        ("B | A", "source approach"),
        ("C | B", "correct L4 submit"),
    )
    for x, (code, label) in zip(stage_x, stage_headings):
        _text(parts, x + axis_w / 2, 76, code, "label", "middle", INK)
        _text(parts, x + axis_w / 2, 99, label, "small", "middle", SLATE)
        _line(parts, x, 116, x + axis_w, 116, INK, 1.2)
        _text(parts, x, 114, "0", "small", "start", SLATE)
        _text(parts, x + axis_w, 114, "1", "small", "end", SLATE)
    any_data = False
    for model_index, model in enumerate(order):
        model_subset = _model_rows(outcomes, model, order)
        has_case_labels = any(
            _as_bool(_first(row, "detectable_source", "positive_case")) is not None
            for row in model_subset
        )
        subset = [
            row for row in model_subset
            if not has_case_labels or _as_bool(_first(row, "detectable_source", "positive_case")) is True
        ]
        a_rows = [row for row in subset if _as_bool(_first(row, *STAGE_FIELDS[0][1])) is True]
        b_rows = [row for row in a_rows if _as_bool(_first(row, *STAGE_FIELDS[1][1])) is True]
        c_rows = [row for row in b_rows if _as_bool(_first(row, *STAGE_FIELDS[2][1])) is True]
        numerators = (len(a_rows), len(b_rows), len(c_rows))
        denominators = (len(subset), len(a_rows), len(b_rows))
        y = 145 + model_index * 42
        _text(parts, 225, y + 6, _short_model(model), "small", "end")
        color = MODEL_COLORS[model_index % len(MODEL_COLORS)]
        for x, numerator, denominator in zip(stage_x, numerators, denominators):
            if denominator <= 0:
                _text(parts, x + axis_w / 2, y + 6, "n/a", "small", "middle", SLATE)
                continue
            any_data = True
            value = numerator / denominator
            _line(parts, x, y, x + axis_w, y, "#E5EAED", 6)
            _circle(parts, x + value * axis_w, y, 6, color, PANEL)
            _text(parts, x + axis_w / 2, y + 24, f"{numerator}/{denominator}", "small", "middle", SLATE)
    if not any_data:
        _no_data(parts, spec, "Published observable-stage flags not supplied")
    else:
        labelled_positive = [row for row in outcomes if _as_bool(_first(row, "detectable_source", "positive_case")) is True]
        population = labelled_positive or list(outcomes)
        task_n = len({_task_value(row) for row in population if _task_value(row)})
        _text(parts, 20, 392, f"Conditional seed-row proportions · {task_n} positive tasks × 3 seeds.", "small")
        _text(parts, 20, 416, "Denominators shrink left→right; counts are numerator/denominator.", "small")
    return _finish_panel(parts, spec, synthetic)


FAILURE_ORDER = ("sampling_coverage", "source_approach", "conclusion_submission", "success", "unclassified")
FAILURE_COLORS = (BLUE, AMBER, ORANGE, TEAL, SLATE)


def _failure_stage(row: Mapping[str, Any]) -> str | None:
    value = _first(row, "first_behavioral_stage", "first_failure_stage", "first_error_stage", "failure_stage")
    if value in (None, ""):
        return None
    token = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "footprint": "sampling_coverage",
        "never_entered_footprint": "sampling_coverage",
        "l4_tolerance": "source_approach",
        "never_entered_l4_tolerance": "source_approach",
        "wrong_l4": "conclusion_submission",
        "incorrect_l4_submit": "conclusion_submission",
        "unsubmitted": "conclusion_submission",
        "no_submit": "conclusion_submission",
        "none": "success",
    }
    return aliases.get(token, token if token in FAILURE_ORDER else "unclassified")


def _render_fig6c(outcomes: Sequence[Mapping[str, Any]], geometry: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], order: Sequence[str], synthetic: bool) -> str:
    del geometry, steps
    spec = PANEL_SPECS["Fig6c"]
    parts = _panel_shell(spec, synthetic)
    left, top, bar_w, bar_h = 20.0, 88.0, 500.0, 32.0
    supplied = False
    for model_index, model in enumerate(order):
        model_subset = _model_rows(outcomes, model, order)
        has_case_labels = any(
            _as_bool(_first(row, "detectable_source", "positive_case")) is not None
            for row in model_subset
        )
        subset = [
            row for row in model_subset
            if not has_case_labels or _as_bool(_first(row, "detectable_source", "positive_case")) is True
        ]
        values = [_failure_stage(row) for row in subset]
        values = [value for value in values if value is not None]
        y = top + model_index * 43
        _text(parts, left + 190, y + 23, _short_model(model), "body", "end")
        x = left + 205
        counts = Counter(values)
        supplied = supplied or bool(values)
        for category, color in zip(FAILURE_ORDER, FAILURE_COLORS):
            width = bar_w * counts[category] / len(values) if values else 0
            if width:
                _rect(parts, x, y, width, bar_h, color, "none", 0)
                if width > 45:
                    _text(parts, x + width / 2, y + 23, counts[category], "small", "middle", PANEL if category != "unclassified" else INK)
            x += width
        _text(parts, 780, y + 23, f"n={len(values)}", "small", "end", SLATE)
    if not supplied:
        _no_data(parts, spec, "Published first-failure categories not supplied")
    else:
        labels = ("sampling coverage", "source approach", "conclusion / submit", "success", "unclassified")
        for index, (label, color) in enumerate(zip(labels, FAILURE_COLORS)):
            x = 18 + (index % 3) * 250
            y = 370 + (index // 3) * 28
            _rect(parts, x, y - 15, 16, 16, color, "none", 2)
            _text(parts, x + 24, y, label, "small")
    return _finish_panel(parts, spec, synthetic)


PANEL_RENDERERS = {
    "Fig1a": _render_fig1a,
    "Fig1b": _render_fig1b,
    "Fig1c": _render_fig1c,
    "Fig4a": _render_fig4a,
    "Fig4b": _render_fig4b,
    "Fig4c": _render_fig4c,
    "Fig4d": _render_fig4d,
    "Fig5a": _render_fig5a,
    "Fig5b": _render_fig5b,
    "Fig5c": _render_fig5c,
    "Fig6a": _render_fig6a,
    "Fig6b": _render_fig6b,
    "Fig6c": _render_fig6c,
}


def _write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def _export_fig4_panel_data(
    outcomes: Sequence[Mapping[str, Any]], order: Sequence[str], root: Path,
    *, synthetic: bool, sources: Sequence[str],
) -> list[Path]:
    """Publish the exact reviewed tables consumed by the redesigned Fig. 4."""

    data_root = root / "fig4_panel_data"
    data_root.mkdir(parents=True, exist_ok=True)
    styles = {model: _model_style(model, order) for model in order}

    rows_a: list[dict[str, Any]] = []
    controls = _control_level_rates(outcomes)
    for control in ("reachable_oracle", "design_random"):
        values = controls.get(control, ())
        for level in FIG4_DISPLAY_LEVELS:
            rows_a.append({
                "row_type": "registered_control", "actor": control, "level": f"L{level}",
                "estimate": values[level - 1] if level <= len(values) else None,
                "ci_low": None, "ci_high": None, "task_n": None,
                "color": INK if control == "reachable_oracle" else SLATE,
                "marker": "diamond" if control == "reachable_oracle" else "cross",
            })
    for model_index, model in enumerate(order):
        subset = _model_rows(outcomes, model, order)
        color, marker = styles[model]
        fallback_rates = _model_level_rates(outcomes, order).get(model, (None,) * 5)
        for level in FIG4_DISPLAY_LEVELS:
            point, low, high, task_n, seed_n, event_n, background_n = _cluster_interval(
                subset, level, seed=20260830 + model_index * 100 + level,
            )
            if point is None:
                point = fallback_rates[level - 1]
            rows_a.append({
                "row_type": "model", "actor": model, "level": f"L{level}",
                "estimate": point, "ci_low": low, "ci_high": high,
                "task_n": task_n, "seed_n": seed_n, "event_cluster_n": event_n,
                "background_n": background_n, "color": color, "marker": marker,
            })

    rows_b: list[dict[str, Any]] = []
    for model in order:
        subset = _model_rows(outcomes, model, order)
        color, marker = styles[model]
        for title, unit, field in BEHAVIOUR_FIELDS:
            values = _task_equal_actor_values(subset, field)
            point = statistics.median(values) if values else None
            low = _quantile(values, .25)
            high = _quantile(values, .75)
            interval_kind = "median_iqr"
            if values and field == "submitted":
                point = statistics.fmean(values)
                rng = random.Random(20260831 + order.index(model))
                draws = [
                    statistics.fmean(rng.choice(values) for _ in values)
                    for _ in range(1200)
                ]
                low = _quantile(draws, .025)
                high = _quantile(draws, .975)
                interval_kind = "task_equal_mean_bootstrap_95ci"
            rows_b.append({
                "actor": model, "metric": title, "field": field, "unit": unit,
                "task_n": len(values), "estimate": point,
                "interval_low": low, "interval_high": high,
                "interval_kind": interval_kind,
                "minimum": min(values) if values else None, "maximum": max(values) if values else None,
                "color": color, "marker": marker,
            })

    rows_c: list[dict[str, Any]] = []
    for model in order:
        subset = _model_rows(outcomes, model, order)
        counts = Counter(_terminal(row) for row in subset)
        total = len(subset)
        color, marker = styles[model]
        for category in TERMINAL_ORDER:
            rows_c.append({
                "actor": model, "terminal_state": category, "count": counts[category],
                "denominator": total, "fraction": counts[category] / total if total else None,
                "model_color": color, "model_marker": marker,
            })

    rows_d: list[dict[str, Any]] = []
    all_single_rates = _model_level_rates(outcomes, order)
    for model in order:
        subset = _model_rows(outcomes, model, order)
        color, marker = styles[model]
        for level in FIG4_DISPLAY_LEVELS:
            value, task_n = _three_run_reliability(subset, level)
            rows_d.append({
                "actor": model, "level": f"L{level}",
                "single_run_pass": all_single_rates.get(model, (None,) * 5)[level - 1],
                "pass3": value,
                "task_n": task_n,
                "denominator": "balanced positive/negative tasks" if level == 1 else "applicable tasks",
                "color": color, "marker": marker,
            })

    paths = [data_root / name for name in ("Fig4a.csv", "Fig4b.csv", "Fig4c.csv", "Fig4d.csv")]
    for path, rows in zip(paths, (rows_a, rows_b, rows_c, rows_d)):
        _write_csv_rows(path, rows)
    provenance = data_root / "Fig4_provenance.json"
    provenance.write_text(json.dumps({
        "schema_version": "laketrace-v20-fig4-provenance-v1",
        "synthetic": synthetic,
        "sources": list(sources),
        "model_order": list(order),
        "model_palette": {
            model: {"color": styles[model][0], "marker": styles[model][1]} for model in order
        },
        "statistics": {
            "Fig4a": "seed collapse within task; two-way event-cluster × background bootstrap",
            "Fig4b": "four plotted seed-collapsed task distributions use median and interpolated IQR; all six registered measures remain in Source Data",
            "Fig4c": "mutually exclusive non-submission terminal-state fractions on a magnified 0-35% scale",
            "Fig4d": "single-run pass rate linked to all-three-run pass3 reliability for displayed levels L1-L4; L1 is class-balanced and L2-L4 use applicable-task denominators",
        },
        "display_contract": {
            "levels": [f"L{level}" for level in FIG4_DISPLAY_LEVELS],
            "excluded_from_artwork": ["L5"],
            "exclusion_scope": "Fig. 4 artwork only; L5 remains in episode_outcomes and the registered benchmark score schema",
        },
        "color_policy": "Okabe-Ito colour plus marker shape identifies models; terminal states use neutral tone only",
    }, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return paths + [provenance]


def _compose_figure(figure: str, panel_paths: Mapping[str, Path], output: Path, synthetic: bool) -> Path:
    placements = FIGURE_LAYOUTS[figure]
    width = 1680
    height = max(y + PANEL_SPECS[panel].height for panel, _, y in placements) + 30
    title, description = FIGURE_TITLES[figure]
    accessible_suffix = f". {SYNTHETIC_WATERMARK}." if synthetic else ""
    title_id, desc_id = f"{figure.lower()}-title", f"{figure.lower()}-desc"
    is_journal_result_figure = figure in {"Fig4", "Fig5"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="{180 * height / width:.2f}mm" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{title_id} {desc_id}" data-figure="{figure}" data-synthetic="{str(synthetic).lower()}">',
        f'<title id="{title_id}">{_esc(figure + ". " + title + accessible_suffix)}</title>',
        f'<desc id="{desc_id}">{_esc(description + accessible_suffix)}</desc>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:20px}.figure-title{font-size:30px;font-weight:700;fill:#12304A}.watermark{font-size:38px;font-weight:700;letter-spacing:2px;fill:#D55E00}</style>',
        f'<metadata>{_esc(json.dumps({"figure": figure, "composition": "mechanical_nested_svg", "panel_order": [panel for panel, _, _ in placements], "synthetic": synthetic}, sort_keys=True))}</metadata>',
    ]
    if not is_journal_result_figure:
        parts.extend([
            f'<rect width="{width}" height="{height}" fill="{PAPER}"/>',
            f'<text x="20" y="40" class="figure-title">{_esc(title)}</text>',
            f'<line x1="20" y1="53" x2="1660" y2="53" stroke="{GRID}" stroke-width="1.2"/>',
        ])
    for panel, x, y in placements:
        path = panel_paths[panel]
        panel_svg = path.read_text(encoding="utf-8").strip()
        if not panel_svg.startswith("<svg ") or not panel_svg.endswith("</svg>"):
            raise ValueError(f"invalid standalone panel SVG: {path}")
        if "<title " not in panel_svg or "<desc " not in panel_svg:
            raise ValueError(f"panel lacks accessible title/description: {path}")
        if synthetic and SYNTHETIC_WATERMARK not in panel_svg:
            raise ValueError(f"synthetic panel lacks rehearsal watermark: {path}")
        if synthetic:
            # Standalone panels retain their own warning; the composite carries
            # one figure-level warning rather than stacking four diagonal labels.
            panel_svg = re.sub(
                r'<text [^>]*class="watermark"[^>]*>.*?</text>', "", panel_svg,
            )
        nested = panel_svg.replace(
            "<svg ",
            f'<svg x="{x}" y="{y}" data-composite-panel="{panel}" ',
            1,
        )
        parts.append(nested)
    if synthetic:
        parts.append(f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" class="watermark" text-anchor="middle" opacity="0.13" transform="rotate(-18 {width / 2:.1f} {height / 2:.1f})">{_esc(SYNTHETIC_WATERMARK)}</text>')
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return output


def build_v20_paper_figures(
    episode_outcomes: Any,
    task_geometry: Any,
    steps: Any,
    output: str | Path,
    *,
    mode: str = "formal",
    model_order: Sequence[str] | None = None,
) -> list[Path]:
    """Build all v20 paper panels and mechanically composed figures.

    Sources may be in-memory row sequences or CSV/JSON/JSONL files.  A JSON
    object can carry ``metadata``, ``contract`` or ``scenario`` alongside
    ``rows``; ``contract.model_order`` and ``scenario.model_order`` take
    precedence over :data:`DEFAULT_MODEL_ORDER` unless ``model_order`` is
    passed explicitly.

    ``mode='rehearsal'`` always watermarks every panel and composite.  Formal
    mode refuses any row or metadata marked synthetic/simulated/mock.
    """

    if mode not in {"formal", "rehearsal"}:
        raise ValueError("mode must be 'formal' or 'rehearsal'")
    payloads = (
        _load_table(episode_outcomes, "episode_outcomes"),
        _load_table(task_geometry, "task_geometry"),
        _load_table(steps, "steps"),
    )
    detected_synthetic = any(_payload_is_synthetic(payload) for payload in payloads)
    if mode == "formal" and detected_synthetic:
        raise ValueError("formal paper figures refuse synthetic, simulated, mock, or rehearsal data")
    synthetic = mode == "rehearsal" or detected_synthetic
    order = _resolve_model_order(payloads, model_order)

    root = Path(output)
    panel_root = root / "panels"
    panel_root.mkdir(parents=True, exist_ok=True)
    outcomes_rows, geometry_rows, step_rows = (payload.rows for payload in payloads)

    _export_fig4_panel_data(
        outcomes_rows, order, root, synthetic=synthetic,
        sources=[payload.source for payload in payloads],
    )
    artifacts: list[Path] = []
    panel_paths: dict[str, Path] = {}
    for panel_id in PANEL_ORDER:
        svg = PANEL_RENDERERS[panel_id](outcomes_rows, geometry_rows, step_rows, order, synthetic)
        path = panel_root / f"{panel_id}.svg"
        path.write_text(svg, encoding="utf-8")
        panel_paths[panel_id] = path
        artifacts.append(path)
    for figure in ("Fig1", "Fig4", "Fig5", "Fig6"):
        artifacts.append(_compose_figure(figure, panel_paths, root / f"{figure}.svg", synthetic))
    return artifacts


# Compatibility name for a future generator that may import the shorter API.
build_paper_figures = build_v20_paper_figures


def make_rehearsal_tables() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return a deterministic, conspicuously synthetic layout-only bundle."""

    outcomes: list[dict[str, Any]] = []
    geometry: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for task_index in range(12):
        task = f"REHEARSAL-{task_index + 1:02d}"
        d_star = 0.35 + task_index * 0.16
        geometry.append({
            "task_id": task,
            "task_class": "control" if task_index % 6 == 0 else "positive",
            "d_star": round(d_star, 3),
            "d_star_stratum": "D* < 1" if d_star < 1 else "D* ≥ 1",
            "synthetic": True,
        })
    scene_task = "REHEARSAL-04"
    for index, (role, x, y) in enumerate((("access", 0, 0), ("footprint", 4, 3), ("peak", 7, 5), ("source", 10, 7))):
        geometry.append({"task_id": scene_task, "geometry_role": role, "x": x, "y": y, "representative": True, "panel": "Fig1a", "synthetic": True, "geometry_index": index})
    tertiles = ("low", "middle", "high", "low", "middle", "high")
    for model_index, model in enumerate(DEFAULT_MODEL_ORDER):
        for task_index in range(12):
            task = f"REHEARSAL-{task_index + 1:02d}"
            d_star = 0.35 + task_index * 0.16
            for repeat in range(3):
                base = (model_index * 7 + task_index * 3 + repeat) % 11
                l1 = base not in {0, 1}
                l2 = base not in {0, 1, 2}
                l3 = base not in {0, 1, 2, 3}
                l4 = base not in {0, 1, 2, 3, 4}
                l5 = base not in {0, 1, 2, 3, 4, 5}
                submitted = base != 0
                outcomes.append({
                    "model_id": model,
                    "task_id": task,
                    "repeat_id": repeat,
                    "L1_pass": l1,
                    "L2_pass": l2,
                    "L3_pass": l3,
                    "L4_pass": l4,
                    "L5_pass": l5,
                    "visited_fraction": round(0.08 + 0.025 * ((base + 2) % 8), 3),
                    "endurance_used_fraction": round(0.10 + 0.035 * ((base + 4) % 8), 3),
                    "range_used_fraction": round(0.06 + 0.03 * ((base + 5) % 8), 3),
                    "analyze_fraction": round(0.12 + 0.04 * ((base + 1) % 8), 3),
                    "revisit_fraction": round(0.04 + 0.025 * ((base + 6) % 8), 3),
                    "submitted": submitted,
                    "terminal_state": "submitted" if submitted else "budget_exhausted",
                    "breadth_tertile": tertiles[model_index],
                    "endurance_tertile": tertiles[(model_index + 1) % 6],
                    "analysis_tertile": tertiles[(model_index + 2) % 6],
                    "revisit_tertile": tertiles[(model_index + 3) % 6],
                    "d_star_stratum": "D* < 1" if d_star < 1 else "D* ≥ 1",
                    "source_peak_projection": round(-0.08 + ((base + repeat) % 13) / 11, 3) if submitted and task_index else None,
                    "entered_anomaly_footprint": l2 or l3 or l4,
                    "entered_l4_tolerance": l4 or (base == 5),
                    "correct_l4_submit": l4,
                    "first_failure_stage": "none" if l4 else "never_entered_footprint" if not l2 else "never_entered_l4_tolerance" if base < 5 else "incorrect_l4_submit" if submitted else "no_submit",
                    "storyboard_selected": task == scene_task,
                    "synthetic": True,
                })
        if model_index < 2:
            for step_index in range(9):
                steps.append({
                    "model_id": model,
                    "task_id": scene_task,
                    "step_index": step_index,
                    "x": step_index + model_index * 0.3,
                    "y": (step_index * (0.72 if model_index == 0 else 0.48)) + math.sin(step_index / 2 + model_index),
                    "z": 0.5 + (step_index % 3) * 0.5,
                    "storyboard_selected": True,
                    "synthetic": True,
                })
    metadata = {"synthetic": True, "data_kind": "rehearsal", "scenario": {"model_order": list(DEFAULT_MODEL_ORDER)}}
    return (
        {"rows": outcomes, "metadata": metadata},
        {"rows": geometry, "metadata": metadata},
        {"rows": steps, "metadata": metadata},
    )


__all__ = [
    "DEFAULT_MODEL_ORDER",
    "FIGURE_LAYOUTS",
    "MODEL_ORDER",
    "PANEL_ORDER",
    "SYNTHETIC_WATERMARK",
    "build_paper_figures",
    "build_v20_paper_figures",
    "make_rehearsal_tables",
]
