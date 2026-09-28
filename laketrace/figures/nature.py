from __future__ import annotations

import html
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable

from laketrace.util import read_csv, write_csv


REQUIRED = {
    "actor", "instance_id", "science_eligible", "waterbody", "difficulty_tier",
    "source_geometry_type", "peak_to_source_m", "detectable_source", "answer_anomaly",
    "L1_pass", "L2_pass", "L3_pass", "L4_pass", "L5_pass", "L5_category",
    "endurance_used_fraction", "visited_nodes", "n_total_nodes",
    "action_parse_success_rate", "replay_verified", "first_error_stage",
    "L2_peak_distance_m", "L4_source_distance_m", "model_config_hash", "presentation_mode",
    "trajectory_source_closest_distance_m", "trajectory_entered_L4_tolerance",
}

W = 1800
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
LEVEL_COLORS = {1: BLUE, 2: TEAL, 3: AMBER, 4: ORANGE, 5: PURPLE}
STAGE_COLORS = {"sampling_position": "#6B7280", "observation_interpretation": BLUE, "hypothesis_update": AMBER, "conclusion_submission": INK, "L1": "#6B7280", "L2": BLUE, "L3": AMBER, "L4": ORANGE, "L5": PURPLE, "none": "#C9D3D8"}
ACTOR_COLORS = ["#12304A", "#2F6C8E", "#5B7F95", "#4F8C7A", "#876D52", "#6C647D", "#1F2937", "#87909A"]
FONT = "Arial, Helvetica, sans-serif"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _float(value: str | None) -> float | None:
    if value in {None, "", "None", "nan", "NaN"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: str | None) -> bool | None:
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def _fmt(value: float | None, digits: int = 2, percent: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value * 100:.{digits}f}%" if percent else f"{value:.{digits}f}"


def _short_actor(actor: str, limit: int = 17) -> str:
    return actor if len(actor) <= limit else actor[: limit - 1] + "…"


def _actor_palette(actors: list[str]) -> dict[str, str]:
    return {actor: ACTOR_COLORS[index % len(ACTOR_COLORS)] for index, actor in enumerate(actors)}


def _svg_base(height: int, title: str, desc: str) -> list[str]:
    height_mm = height * 180 / W
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="{height_mm:.2f}mm" viewBox="0 0 {W} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{esc(title)}</title>',
        f'<desc id="desc">{esc(desc)}</desc>',
        f'<style>text{{font-family:{FONT};fill:{INK}}}.title{{font-size:30px;font-weight:700}}.panel{{font-size:22px;font-weight:700}}.small{{font-size:15px}}.tiny{{font-size:13px}}.muted{{fill:{SLATE}}}.axis{{stroke:{INK};stroke-width:1.5}}.grid{{stroke:{GRID};stroke-width:1}} </style>',
        f'<rect width="{W}" height="{height}" fill="{PAPER}"/>',
        '<defs>',
        f'<marker id="arrow-navy" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{NAVY}"/></marker>',
        '</defs>',
    ]


def _text(parts: list[str], x: float, y: float, value: object, size: int = 18, fill: str = INK, weight: str = "400", anchor: str = "start", opacity: float = 1.0, cls: str | None = None) -> None:
    class_attr = f' class="{cls}"' if cls else ""
    parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}px" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}" opacity="{opacity}"{class_attr}>{esc(value)}</text>')


def _rect(parts: list[str], x: float, y: float, w: float, h: float, fill: str = PANEL, stroke: str = GRID, sw: float = 1.2, rx: float = 8, opacity: float = 1.0, dash: str | None = None) -> None:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}" opacity="{opacity}"{extra}/>')


def _line(parts: list[str], x1: float, y1: float, x2: float, y2: float, stroke: str = INK, sw: float = 1.5, dash: str | None = None, opacity: float = 1.0) -> None:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw:.1f}" opacity="{opacity}"{extra}/>')


def _circle(parts: list[str], x: float, y: float, r: float, fill: str, stroke: str = "none", sw: float = 1.0, opacity: float = 1.0) -> None:
    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}" opacity="{opacity}"/>')


def _path(parts: list[str], d: str, stroke: str, sw: float = 2.0, fill: str = "none", opacity: float = 1.0, dash: str | None = None) -> None:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}" opacity="{opacity}"{extra}/>')


def _header(parts: list[str], number: str, title: str, subtitle: str, height: int) -> None:
    _text(parts, 60, 48, number, 20, NAVY, "700")
    _text(parts, 178, 48, title, 28, INK, "700")
    _text(parts, 60, 77, subtitle, 15, SLATE)
    _line(parts, 60, 92, 1740, 92, INK, 1.5)


def _panel(parts: list[str], x: float, y: float, w: float, h: float, label: str, title: str, subtitle: str = "", fill: str = PANEL) -> tuple[float, float, float, float]:
    _rect(parts, x, y, w, h, fill, GRID, 1.4, 10)
    _text(parts, x + 22, y + 37, label, 20, NAVY, "700")
    _text(parts, x + 66, y + 37, title, 19, INK, "700")
    if subtitle:
        _text(parts, x + 22, y + 64, subtitle, 13, SLATE)
    return x + 62, y + 92, w - 86, h - 126


def _footer(parts: list[str], height: int, note: str) -> None:
    _line(parts, 60, height - 38, 1740, height - 38, GRID, 1)
    _text(parts, 60, height - 15, note, 13, SLATE, "700")


def _plot_frame(parts: list[str], x: float, y: float, w: float, h: float, y_ticks: Iterable[float], y_min: float, y_max: float, ylabel: str, xlabel: str | None = None) -> None:
    for tick in y_ticks:
        yy = y + h - (tick - y_min) / (y_max - y_min) * h
        _line(parts, x, yy, x + w, yy, GRID, 1)
        _text(parts, x - 12, yy + 5, _fmt(tick, 0, percent=True), 12, SLATE, anchor="end")
    _line(parts, x, y, x, y + h, INK, 1.6)
    _line(parts, x, y + h, x + w, y + h, INK, 1.6)
    # Keep the axis title above the plotting area. A horizontal title beside the
    # y-axis collides with percentage tick labels in compact multi-panel figures.
    _text(parts, x, y - 14, ylabel, 13, SLATE, "700")
    if xlabel:
        _text(parts, x + w / 2, y + h + 47, xlabel, 14, SLATE, "400", "middle")


def _xmap(value: float, low: float, high: float, x: float, w: float) -> float:
    return x + (value - low) / (high - low) * w


def _ymap(value: float, low: float, high: float, y: float, h: float) -> float:
    return y + h - (value - low) / (high - low) * h


def _applicable(rows: list[dict[str, str]], field: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get(field) in {"True", "False"}]


def _cluster_rate(rows: list[dict[str, str]], field: str, rng: random.Random, replicates: int = 1000) -> tuple[float | None, float | None, float | None, int, int, int]:
    subset = _applicable(rows, field)
    if not subset:
        return None, None, None, 0, 0, 0
    values = [row[field] == "True" for row in subset]
    events = sorted({row.get("event_cluster_id") or row["instance_id"] for row in subset})
    backgrounds = sorted({row.get("background_id") or "unregistered" for row in subset})
    estimates: list[float] = []
    for _ in range(replicates):
        event_weights = Counter(rng.choice(events) for _ in events)
        background_weights = Counter(rng.choice(backgrounds) for _ in backgrounds)
        numerator = denominator = 0
        for row, value in zip(subset, values):
            weight = event_weights[row.get("event_cluster_id") or row["instance_id"]] * background_weights[row.get("background_id") or "unregistered"]
            numerator += weight * value
            denominator += weight
        if denominator:
            estimates.append(numerator / denominator)
    estimates.sort()
    return statistics.mean(estimates), estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))], len(subset), len(events), len(backgrounds)


def _l1_metrics(rows: list[dict[str, str]]) -> dict[str, float | None]:
    positives = [row for row in rows if row.get("detectable_source") == "True"]
    negatives = [row for row in rows if row.get("detectable_source") == "False"]
    sensitivity = sum(row.get("answer_anomaly") == "anomaly" for row in positives) / len(positives) if positives else None
    specificity = sum(row.get("answer_anomaly") == "no_anomaly" for row in negatives) / len(negatives) if negatives else None
    balanced_accuracy = (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None
    return {
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def _cluster_l1_balanced(rows: list[dict[str, str]], rng: random.Random, replicates: int = 1000) -> tuple[float | None, float | None, float | None, int, int, int]:
    subset = [row for row in rows if row.get("detectable_source") in {"True", "False"}]
    if not subset or not any(row["detectable_source"] == "True" for row in subset) or not any(row["detectable_source"] == "False" for row in subset):
        return None, None, None, 0, 0, 0
    events = sorted({row.get("event_cluster_id") or row["instance_id"] for row in subset})
    backgrounds = sorted({row.get("background_id") or "unregistered" for row in subset})
    estimates: list[float] = []
    for _ in range(replicates):
        event_weights = Counter(rng.choice(events) for _ in events)
        background_weights = Counter(rng.choice(backgrounds) for _ in backgrounds)
        positive_correct = positive_total = negative_correct = negative_total = 0
        for row in subset:
            weight = event_weights[row.get("event_cluster_id") or row["instance_id"]] * background_weights[row.get("background_id") or "unregistered"]
            if row["detectable_source"] == "True":
                positive_total += weight
                positive_correct += weight * (row.get("answer_anomaly") == "anomaly")
            else:
                negative_total += weight
                negative_correct += weight * (row.get("answer_anomaly") == "no_anomaly")
        if positive_total and negative_total:
            estimates.append(((positive_correct / positive_total) + (negative_correct / negative_total)) / 2)
    if not estimates:
        return None, None, None, len(subset), len(events), len(backgrounds)
    estimates.sort()
    return statistics.mean(estimates), estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))], len(subset), len(events), len(backgrounds)


def _pooled_cluster_rates(rows: list[dict[str, str]], field: str) -> tuple[float | None, float | None, float | None, int, int, int]:
    # Keep the figure deterministic while using the same two-way bootstrap
    # definition as `laketrace analyze`.
    return _cluster_rate(rows, field, random.Random(20260826))


def _draw_interval(parts: list[str], x: float, y: float, low: float | None, high: float | None, plot_y: float, plot_h: float, color: str, width: float = 2.2) -> None:
    if low is None or high is None:
        return
    _line(parts, x, _ymap(low, 0, 1, plot_y, plot_h), x, _ymap(high, 0, 1, plot_y, plot_h), color, width)
    _line(parts, x - 5, _ymap(low, 0, 1, plot_y, plot_h), x + 5, _ymap(low, 0, 1, plot_y, plot_h), color, width)
    _line(parts, x - 5, _ymap(high, 0, 1, plot_y, plot_h), x + 5, _ymap(high, 0, 1, plot_y, plot_h), color, width)


def build_fig1(rows: list[dict[str, str]], output: Path) -> None:
    actors = sorted({row["actor"] for row in rows})
    actor_colors = _actor_palette(actors)
    height = 1220
    parts = _svg_base(height, "Fig. 1 LakeTrace task and performance overview", "Data-derived result panels for five separately scored capability levels and the paired L2/L4 contrast.")
    _header(parts, "FIG. 1", "LakeTrace turns lake source tracing into a scored survey task", "Data panels b–c; the conceptual survey panel a is delivered separately as Fig1a_experiment_schematic.svg.", height)
    px, py, pw, ph = _panel(parts, 60, 118, 1090, 700, "b", "Performance at each capability level", "L1 is balanced accuracy; L2–L5 are pass rates. Points are two-way bootstrap means; whiskers are 95% intervals.")
    plot_x, plot_y, plot_w, plot_h = px + 20, py + 12, pw - 40, ph - 64
    _plot_frame(parts, plot_x, plot_y, plot_w, plot_h, [0, .25, .5, .75, 1], 0, 1, "score", "capability level")
    level_x = {level: plot_x + (level - 1) * plot_w / 4 for level in range(1, 6)}
    for level in range(1, 6):
        _line(parts, level_x[level], plot_y + plot_h, level_x[level], plot_y + plot_h + 8, LEVEL_COLORS[level], 2)
        _text(parts, level_x[level], plot_y + plot_h + 28, f"L{level}", 15, LEVEL_COLORS[level], "700", "middle")
    rng = random.Random(20260826)
    for actor_index, actor in enumerate(actors):
        offset = (actor_index - (len(actors) - 1) / 2) * 8
        previous = None
        for level in range(1, 6):
            actor_rows = [row for row in rows if row["actor"] == actor]
            if level == 1:
                mean, low, high, _, _, _ = _cluster_l1_balanced(actor_rows, rng)
            else:
                mean, low, high, _, _, _ = _cluster_rate(actor_rows, f"L{level}_pass", rng)
            if mean is None:
                continue
            x = level_x[level] + offset
            y = _ymap(mean, 0, 1, plot_y, plot_h)
            if previous is not None:
                _line(parts, previous[0], previous[1], x, y, actor_colors[actor], 1.3, opacity=.48)
            _draw_interval(parts, x, y, low, high, plot_y, plot_h, actor_colors[actor], 1.4)
            _circle(parts, x, y, 5, LEVEL_COLORS[level], actor_colors[actor], 1.3)
            previous = (x, y)
    legend_y = 755
    _text(parts, plot_x, legend_y, "systems", 13, SLATE, "700")
    for i, actor in enumerate(actors):
        x = plot_x + 92 + (i % 3) * 290
        y = legend_y + (i // 3) * 24
        _line(parts, x, y - 5, x + 26, y - 5, actor_colors[actor], 2)
        _text(parts, x + 36, y, _short_actor(actor), 13, INK)
    px2, py2, pw2, ph2 = _panel(parts, 1180, 118, 560, 700, "c", "Peak versus source", "Same rows; green = L2 peak, orange = L4 source.")
    plot_x, plot_y, plot_w, plot_h = px2 + 35, py2 + 12, pw2 - 55, ph2 - 74
    _plot_frame(parts, plot_x, plot_y, plot_w, plot_h, [0, .25, .5, .75, 1], 0, 1, "pass rate")
    if actors:
        for i, actor in enumerate(actors):
            x = plot_x + (i + .5) * plot_w / len(actors)
            l2 = _cluster_rate([row for row in rows if row["actor"] == actor], "L2_pass", random.Random(20260826 + i))[0]
            l4 = _cluster_rate([row for row in rows if row["actor"] == actor], "L4_pass", random.Random(20260826 + i))[0]
            if l2 is not None and l4 is not None:
                y2, y4 = _ymap(l2, 0, 1, plot_y, plot_h), _ymap(l4, 0, 1, plot_y, plot_h)
                _line(parts, x, y2, x, y4, actor_colors[actor], 3)
                _circle(parts, x, y2, 7, TEAL, actor_colors[actor], 1.5)
                _circle(parts, x, y4, 7, ORANGE, actor_colors[actor], 1.5)
                _text(parts, x, min(y2, y4) - 13, _fmt(l2 - l4, 0, percent=True), 12, actor_colors[actor], "700", "middle")
            _text(parts, x, plot_y + plot_h + 32, _short_actor(actor, 10), 12, INK, "400", "middle")
    _circle(parts, px2 + 28, py2 + ph2 - 50, 6, TEAL); _text(parts, px2 + 42, py2 + ph2 - 45, "L2", 13, TEAL, "700")
    _circle(parts, px2 + 85, py2 + ph2 - 50, 6, ORANGE); _text(parts, px2 + 99, py2 + ph2 - 45, "L4", 13, ORANGE, "700")
    px3, py3, pw3, ph3 = _panel(parts, 60, 850, 1680, 280, "scope", "Result-table coverage", "These counts describe the supplied CSV and are not a combined benchmark score.", SOFT)
    unique = {row["instance_id"]: row for row in rows}
    metrics = [("scored rows", len(rows)), ("unique instances", len(unique)), ("waterbodies", len({row["waterbody"] for row in unique.values()})), ("event clusters", len({row.get("event_cluster_id") or row["instance_id"] for row in rows})), ("backgrounds", len({row.get("background_id") or "unregistered" for row in rows})), ("L2 applicable", len(_applicable(rows, "L2_pass"))), ("L4 applicable", len(_applicable(rows, "L4_pass")))]
    for i, (label, value) in enumerate(metrics):
        x = px3 + (i % 4) * 400; y = py3 + (i // 4) * 92
        _rect(parts, x, y, 350, 68, PANEL, "#D5E0E5", 1.0, 7)
        _text(parts, x + 16, y + 25, label, 13, SLATE, "700")
        _text(parts, x + 16, y + 53, value, 23, NAVY, "700")
    _footer(parts, height, "CSV-derived panels only · no aggregate score · CI: two-way pigeonhole cluster bootstrap · image2 schematic is separate")
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_fig2(rows: list[dict[str, str]], output: Path) -> None:
    height = 900
    parts = _svg_base(height, "Fig. 2 LakeTrace task-set coverage companion", "CSV-derived inventory companion; conceptual task construction is delivered separately as Fig2_task_construction_schematic.svg.")
    _header(parts, "FIG. 2", "Task-set coverage and source-geometry inventory", "A data companion to the image2 construction schematic; no values are used to define truth in this panel.", height)
    unique = {row["instance_id"]: row for row in rows}
    water = Counter(row["waterbody"] for row in unique.values())
    geometry = Counter(row.get("source_geometry_type") or "negative/control" for row in unique.values())
    phases = Counter(row.get("phase") or "unregistered" for row in unique.values())
    px, py, pw, ph = _panel(parts, 60, 118, 820, 640, "a", "Instances by waterbody", "One row per unique instance; comparator rows are collapsed.")
    plot_x, plot_y, plot_w, plot_h = px + 22, py + 24, pw - 42, ph - 75
    max_count = max(water.values(), default=1)
    _line(parts, plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, INK, 1.5)
    for i, (name, count) in enumerate(sorted(water.items())):
        slot = plot_w / max(1, len(water)); bw = slot * .55; x = plot_x + i * slot + (slot - bw) / 2; bh = count / max_count * (plot_h - 24); y = plot_y + plot_h - bh
        _rect(parts, x, y, bw, bh, [BLUE, TEAL, AMBER][i % 3], "none", 0, 5)
        _text(parts, x + bw / 2, y - 10, count, 18, NAVY, "700", "middle")
        _text(parts, x + bw / 2, plot_y + plot_h + 30, name, 14, INK, "400", "middle")
    _text(parts, plot_x, plot_y - 4, "unique instances", 13, SLATE, "700")
    px2, py2, pw2, ph2 = _panel(parts, 920, 118, 820, 640, "b", "Registered source geometry", "Blank geometry rows are explicit negatives or below-detection controls.")
    y = py2 + 26
    total = sum(geometry.values()) or 1
    for i, (name, count) in enumerate(sorted(geometry.items())):
        yy = y + i * 78
        _text(parts, px2, yy + 20, name, 15, INK, "700")
        _rect(parts, px2 + 230, yy, pw2 - 280, 28, "#EAF0F2", "none", 0, 5)
        _rect(parts, px2 + 230, yy, (pw2 - 280) * count / total, 28, [AMBER, ORANGE, PURPLE, SLATE][i % 4], "none", 0, 5)
        _text(parts, px2 + pw2 - 30, yy + 21, f"{count} · {_fmt(count / total, 1, percent=True)}", 14, NAVY, "700", "end")
    _rect(parts, 60, 790, 1680, 70, SOFT, "#C9D9E0", 1.2, 8)
    _text(parts, 84, 820, "phase binding", 14, SLATE, "700")
    _text(parts, 230, 820, " · ".join(f"{phase_name}: {count}" for phase_name, count in sorted(phases.items())), 16, NAVY, "700")
    _footer(parts, height, "Data companion only · construction semantics and truth withholding are shown in Fig2_task_construction_schematic.svg")
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_fig3(rows: list[dict[str, str]], output: Path) -> None:
    height = 900
    actors = sorted({row["actor"] for row in rows})
    actor_colors = _actor_palette(actors)
    parts = _svg_base(height, "Fig. 3 driver and interface validation", "CSV-derived interface validation; conceptual driver/dataflow is delivered separately as Fig3_driver_dataflow_schematic.svg.")
    _header(parts, "FIG. 3", "All systems use the same observation–action interface", "The data panel validates parsing and replay after the conceptual S–L–A interface is shown.", height)
    px, py, pw, ph = _panel(parts, 60, 118, 820, 640, "d", "Parser success", "Mean action-parse success by actor; dots show episodes.")
    plot_x, plot_y, plot_w, plot_h = px + 22, py + 22, pw - 42, ph - 80
    _plot_frame(parts, plot_x, plot_y, plot_w, plot_h, [0, .25, .5, .75, 1], 0, 1, "fraction")
    for i, actor in enumerate(actors):
        x = plot_x + (i + .5) * plot_w / max(1, len(actors)); vals = [_float(r.get("action_parse_success_rate")) for r in rows if r["actor"] == actor]; val = statistics.mean(v for v in vals if v is not None) if any(v is not None for v in vals) else None
        if val is not None:
            _circle(parts, x, _ymap(val, 0, 1, plot_y, plot_h), 8, TEAL, actor_colors[actor], 1.5)
            _text(parts, x, _ymap(val, 0, 1, plot_y, plot_h) - 15, _fmt(val, 1, percent=True), 12, actor_colors[actor], "700", "middle")
        _text(parts, x, plot_y + plot_h + 28, _short_actor(actor, 10), 12, INK, "400", "middle")
    px2, py2, pw2, ph2 = _panel(parts, 920, 118, 820, 640, "e", "Replay and invalid-action audit", "Each point is a fraction of episodes; bottom rows are event counts.")
    y0 = py2 + 30
    for i, actor in enumerate(actors):
        yy = y0 + i * 58; subset = [r for r in rows if r["actor"] == actor]; replay = sum(r.get("replay_verified") == "True" for r in subset) / len(subset) if subset else 0; invalid = sum(int(_float(r.get("invalid_calls")) or 0) for r in subset); infeasible = sum(int(_float(r.get("infeasible_calls")) or 0) for r in subset)
        _text(parts, px2, yy + 4, _short_actor(actor), 14, INK, "700")
        _rect(parts, px2 + 190, yy - 13, 420, 20, "#EAF0F2", "none", 0, 4)
        _rect(parts, px2 + 190, yy - 13, 420 * replay, 20, TEAL, "none", 0, 4)
        _text(parts, px2 + 630, yy + 4, f"replay {_fmt(replay, 1, percent=True)}", 13, TEAL, "700")
        _text(parts, px2 + 190, yy + 27, f"invalid {invalid} · infeasible {infeasible}", 12, SLATE)
    _rect(parts, 60, 790, 1680, 70, SOFT, "#C9D9E0", 1.2, 8)
    _text(parts, 84, 820, "contract", 14, SLATE, "700")
    _text(parts, 190, 820, "move · analyze · submit  →  append-only JSONL  →  replay  →  offline score", 16, NAVY, "700")
    _footer(parts, height, "CSV-derived validation only · exact interface diagram and run/score boundary are in Fig3_driver_dataflow_schematic.svg")
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_fig4(rows: list[dict[str, str]], output: Path) -> None:
    height = 1160
    actors = sorted({row["actor"] for row in rows})
    actor_colors = _actor_palette(actors)
    parts = _svg_base(height, "Fig. 4 system by capability level", "System-by-level matrix and decompositions derived from the result CSV.")
    _header(parts, "FIG. 4", "Performance differs sharply across the five levels", "Rows are systems; columns are independently scored L1–L5. No aggregate score is used.", height)
    px, py, pw, ph = _panel(parts, 60, 118, 840, 490, "a", "System × capability-level matrix", "L1 is balanced accuracy; L2–L5 are pass rates. Values are two-way bootstrap means.")
    left, top = px + 90, py + 24; cell_w = (pw - 120) / 5; cell_h = 43
    for level in range(1, 6):
        _text(parts, left + (level - .5) * cell_w, top - 15, f"L{level}", 14, LEVEL_COLORS[level], "700", "middle")
    rng = random.Random(20260826)
    for row_index, actor in enumerate(actors):
        yy = top + row_index * cell_h
        _text(parts, left - 15, yy + 26, _short_actor(actor), 13, INK, "700", "end")
        for level in range(1, 6):
            actor_rows = [r for r in rows if r["actor"] == actor]
            if level == 1:
                mean, low, high, n, event_n, bg_n = _cluster_l1_balanced(actor_rows, rng)
            else:
                mean, low, high, n, event_n, bg_n = _cluster_rate(actor_rows, f"L{level}_pass", rng)
            xx = left + (level - 1) * cell_w
            fill = "#EFF4F5" if mean is None else f"rgb({int(250 - 95 * mean)},{int(252 - 55 * mean)},{int(252 - 20 * mean)})"
            _rect(parts, xx, yy, cell_w - 5, cell_h - 5, fill, LEVEL_COLORS[level], 1.5, 4)
            _text(parts, xx + (cell_w - 5) / 2, yy + 25, _fmt(mean, 1, percent=True) if mean is not None else "n/a", 13, INK, "700", "middle")
    _text(parts, left, top + len(actors) * cell_h + 18, "n = applicable episodes; intervals use event-cluster × background bootstrap", 12, SLATE)

    px2, py2, pw2, ph2 = _panel(parts, 940, 118, 800, 490, "b", "Waterbody decomposition", "L4 source-location pass rate, resolved by physical waterbody.")
    plot_x, plot_y, plot_w, plot_h = px2 + 45, py2 + 20, pw2 - 70, ph2 - 82
    _plot_frame(parts, plot_x, plot_y, plot_w, plot_h, [0, .5, 1], 0, 1, "L4 pass rate")
    waterbodies = sorted({r["waterbody"] for r in rows}); actor_slot = plot_w / max(1, len(waterbodies));
    for wi, waterbody in enumerate(waterbodies):
        xx = plot_x + (wi + .5) * actor_slot
        _text(parts, xx, plot_y + plot_h + 28, waterbody, 13, INK, "700", "middle")
        for ai, actor in enumerate(actors):
            subset = [r for r in rows if r["actor"] == actor and r["waterbody"] == waterbody]
            mean = _cluster_rate(subset, "L4_pass", random.Random(20260826 + wi * 10 + ai))[0]
            if mean is None: continue
            dx = (ai - (len(actors) - 1) / 2) * min(10, actor_slot / max(1, len(actors) + 2)); _circle(parts, xx + dx, _ymap(mean, 0, 1, plot_y, plot_h), 5, ORANGE, actor_colors[actor], 1.1, .75)
    _text(parts, plot_x, plot_y - 6, "each dot = actor", 12, SLATE)

    px3, py3, pw3, ph3 = _panel(parts, 60, 640, 840, 440, "c", "Displacement-resolved performance", "Pooled descriptive decomposition; the task-set range is shown by the occupied bins.")
    plot_x, plot_y, plot_w, plot_h = px3 + 45, py3 + 18, pw3 - 70, ph3 - 86
    _plot_frame(parts, plot_x, plot_y, plot_w, plot_h, [0, .5, 1], 0, 1, "pass rate", "peak-to-source displacement (m)")
    bins = [(0, 10, "0–10"), (10, 25, "10–25"), (25, 50, "25–50"), (50, 100, "50–100"), (100, float("inf"), ">100")]
    for bi, (low, high, label) in enumerate(bins):
        subset = [r for r in rows if (v := _float(r.get("peak_to_source_m"))) is not None and low <= v < high]
        xx = plot_x + (bi + .5) * plot_w / len(bins)
        _text(parts, xx, plot_y + plot_h + 27, label, 12, INK, "400", "middle")
        for field, color, dx in (("L2_pass", TEAL, -6), ("L4_pass", ORANGE, 6)):
            mean = _pooled_cluster_rates(subset, field)[0]
            if mean is not None: _circle(parts, xx + dx, _ymap(mean, 0, 1, plot_y, plot_h), 5, color, color, 1)
    _circle(parts, px3 + 35, py3 + ph3 - 36, 5, TEAL); _text(parts, px3 + 48, py3 + ph3 - 31, "L2", 12, TEAL, "700")
    _circle(parts, px3 + 80, py3 + ph3 - 36, 5, ORANGE); _text(parts, px3 + 93, py3 + ph3 - 31, "L4", 12, ORANGE, "700")

    px4, py4, pw4, ph4 = _panel(parts, 940, 640, 800, 440, "d", "Budget and first-error composition", "Endurance fraction and observable first scored failure stage.")
    base_x, base_y, bar_w, bar_h = px4 + 42, py4 + 26, pw4 - 90, 185
    max_end = max((_float(r.get("endurance_used_fraction")) or 0 for r in rows), default=1) or 1
    for i, actor in enumerate(actors):
        subset = [r for r in rows if r["actor"] == actor]; value = statistics.mean(_float(r.get("endurance_used_fraction")) or 0 for r in subset); yy = base_y + i * (bar_h / max(1, len(actors)))
        _text(parts, base_x, yy + 12, _short_actor(actor, 12), 12, INK, "700")
        _rect(parts, base_x + 112, yy, bar_w - 210, 16, "#EAF0F2", "none", 0, 3)
        _rect(parts, base_x + 112, yy, (bar_w - 210) * value / max_end, 16, NAVY, "none", 0, 3)
        _text(parts, base_x + bar_w - 80, yy + 13, _fmt(value, 1, percent=True), 12, NAVY, "700", "end")
    _text(parts, base_x, base_y - 12, "mean endurance used", 12, SLATE, "700")
    stack_y = py4 + 275; stages = ["L1", "L2", "L3", "L4", "L5", "none"]
    for i, actor in enumerate(actors):
        subset = [r for r in rows if r["actor"] == actor]; counts = Counter(r.get("first_error_stage") or "none" for r in subset); xx = base_x + i * (bar_w / max(1, len(actors))); yy = stack_y + 62
        running = 0
        for stage in stages:
            fraction = counts[stage] / len(subset) if subset else 0; hh = 92 * fraction
            if hh: _rect(parts, xx, yy - running - hh, max(18, bar_w / max(1, len(actors)) - 8), hh, STAGE_COLORS.get(stage, SLATE), "none", 0, 2)
            running += hh
        _text(parts, xx + max(18, bar_w / max(1, len(actors)) - 8) / 2, stack_y + 118, _short_actor(actor, 8), 10, INK, "400", "middle")
    _text(parts, base_x, stack_y + 18, "first_error_stage", 12, SLATE, "700")
    _footer(parts, height, "All values from benchmark_results.csv · no combined score · rank uncertainty and prompt-variant panels require registered additional CSV fields")
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_fig5(rows: list[dict[str, str]], output: Path) -> None:
    height = 1160
    actors = sorted({row["actor"] for row in rows})
    actor_colors = _actor_palette(actors)
    parts = _svg_base(height, "Fig. 5 peak versus source displacement", "Displacement response and error components derived from result CSV fields.")
    _header(parts, "FIG. 5", "Peak and source localization are evaluated separately", "The displacement analysis is restricted to the range occupied by the current task set; no extrapolation or model-capability claim is implied.", height)
    px, py, pw, ph = _panel(parts, 60, 118, 1090, 500, "a", "L2 and L4 versus peak-to-source displacement", "Bold curves are roster-pooled; thin curves show individual comparator systems.")
    plot_x, plot_y, plot_w, plot_h = px + 48, py + 18, pw - 70, ph - 86
    vals = [_float(r.get("peak_to_source_m")) for r in rows]; vals = [v for v in vals if v is not None]; xmax = max(vals, default=100); xmax = max(10, math.ceil(xmax / 10) * 10)
    _plot_frame(parts, plot_x, plot_y, plot_w, plot_h, [0, .5, 1], 0, 1, "pass rate", "peak-to-source displacement (m)")
    bins = [(0, 10), (10, 25), (25, 50), (50, 100), (100, xmax + .001)]
    occupied = []
    for low, high in bins:
        subset = [r for r in rows if (v := _float(r.get("peak_to_source_m"))) is not None and low <= v < high]
        if subset: occupied.append((low, high, subset))
    for low, high, _ in occupied:
        xx0, xx1 = _xmap(low, 0, xmax, plot_x, plot_w), _xmap(min(high, xmax), 0, xmax, plot_x, plot_w)
        _rect(parts, xx0, plot_y, max(1, xx1 - xx0), plot_h, "#EDF5F3", "none", 0, 0, .75)
    for field, color in (("L2_pass", TEAL), ("L4_pass", ORANGE)):
        pooled_points = []
        for low, high, subset in occupied:
            mean, _, _, _, _, _ = _pooled_cluster_rates(subset, field); mid = (low + min(high, xmax)) / 2
            if mean is not None: pooled_points.append((_xmap(mid, 0, xmax, plot_x, plot_w), _ymap(mean, 0, 1, plot_y, plot_h)))
        if pooled_points:
            d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pooled_points); _path(parts, d, color, 4)
            for x, y in pooled_points: _circle(parts, x, y, 7, color, PANEL, 1.5)
    for actor in actors:
        for field, color, opacity in (("L2_pass", TEAL, .35), ("L4_pass", ORANGE, .35)):
            points = []
            for low, high, subset_all in occupied:
                subset = [r for r in subset_all if r["actor"] == actor]; mean = _pooled_cluster_rates(subset, field)[0]; mid = (low + min(high, xmax)) / 2
                if mean is not None: points.append((_xmap(mid, 0, xmax, plot_x, plot_w), _ymap(mean, 0, 1, plot_y, plot_h)))
            if len(points) > 1: _path(parts, "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points), actor_colors[actor], 1.2, opacity=opacity)
    _circle(parts, px + 26, py + ph - 36, 6, TEAL); _text(parts, px + 40, py + ph - 31, "L2 peak", 12, TEAL, "700")
    _circle(parts, px + 103, py + ph - 36, 6, ORANGE); _text(parts, px + 117, py + ph - 31, "L4 source", 12, ORANGE, "700")
    _text(parts, px + 210, py + ph - 31, "shaded = occupied task-set range", 12, SLATE)

    px2, py2, pw2, ph2 = _panel(parts, 1180, 118, 560, 500, "b", "Localization error components", "Rows with an applicable coordinate answer; metres.")
    plot_x, plot_y, plot_w, plot_h = px2 + 62, py2 + 18, pw2 - 86, ph2 - 82
    vals_x = [_float(r.get("L4_horizontal_error_m")) for r in rows]; vals_y = [_float(r.get("L4_vertical_error_m")) for r in rows]; vals_x = [v for v in vals_x if v is not None]; vals_y = [v for v in vals_y if v is not None]; xmax2 = max(vals_x, default=1); ymax2 = max(vals_y, default=1); xmax2 = max(1, math.ceil(xmax2 / 10) * 10); ymax2 = max(1, math.ceil(ymax2))
    _line(parts, plot_x, plot_y, plot_x, plot_y + plot_h, INK, 1.5); _line(parts, plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, INK, 1.5)
    _text(parts, plot_x + plot_w / 2, plot_y + plot_h + 42, "horizontal error (m)", 13, SLATE, "400", "middle"); _text(parts, plot_x - 44, plot_y + plot_h / 2, "vertical error (m)", 13, SLATE, "400", "middle")
    for r in rows:
        x, y = _float(r.get("L4_horizontal_error_m")), _float(r.get("L4_vertical_error_m"));
        if x is not None and y is not None: _circle(parts, _xmap(x, 0, xmax2, plot_x, plot_w), _ymap(y, 0, ymax2, plot_y, plot_h), 3.5, actor_colors.get(r["actor"], SLATE), "none", 0, .36)
    _text(parts, plot_x, plot_y - 6, "L4 source answer", 12, SLATE, "700")

    px3, py3, pw3, ph3 = _panel(parts, 60, 670, 820, 410, "c", "Success versus endurance used", "Pooled descriptive response; fractions are measured from the episode ledger.")
    plot_x, plot_y, plot_w, plot_h = px3 + 48, py3 + 18, pw3 - 70, ph3 - 82
    _plot_frame(parts, plot_x, plot_y, plot_w, plot_h, [0, .5, 1], 0, 1, "pass rate", "endurance used fraction")
    e_bins = [(0, .005), (.005, .01), (.01, .02), (.02, .05), (.05, .1), (.1, 1.0)]
    for i, (low, high) in enumerate(e_bins):
        subset = [r for r in rows if (v := _float(r.get("endurance_used_fraction"))) is not None and low <= v < high]; xx = plot_x + (i + .5) * plot_w / len(e_bins); _text(parts, xx, plot_y + plot_h + 25, f"{low:.3g}–{high:.3g}", 11, INK, "400", "middle")
        for field, color, dx in (("L2_pass", TEAL, -5), ("L4_pass", ORANGE, 5)):
            mean = _pooled_cluster_rates(subset, field)[0]
            if mean is not None: _circle(parts, xx + dx, _ymap(mean, 0, 1, plot_y, plot_h), 5, color, color, 1)
    px4, py4, pw4, ph4 = _panel(parts, 920, 670, 820, 410, "d", "Terminal decision categories", "L5 categories are reported directly; confidence calibration is not present in this CSV schema.")
    category_order = ["correct_declaration", "correct_uncertainty", "correct_true_negative", "false_declaration", "false_declaration_true_negative", "overconservative", "unsubmitted"]
    category_colors = [TEAL, PURPLE, BLUE, ORANGE, AMBER, SLATE, "#C9D3D8"]
    for i, actor in enumerate(actors):
        subset = [r for r in rows if r["actor"] == actor]; xx = px4 + 55 + i * (pw4 - 90) / max(1, len(actors)); running = 0
        for category in category_order:
            frac = sum(r.get("L5_category") == category for r in subset) / len(subset) if subset else 0; hh = (ph4 - 150) * frac
            if hh: _rect(parts, xx, py4 + ph4 - 58 - running - hh, 42, hh, category_colors[category_order.index(category)], "none", 0, 2)
            running += hh
        _text(parts, xx + 21, py4 + ph4 - 25, _short_actor(actor, 8), 10, INK, "400", "middle")
    _text(parts, px4 + 22, py4 + 40, "stacked fractions", 12, SLATE, "700")
    _footer(parts, height, "CSV-derived response analysis · displacement and endurance ranges are descriptive · calibration and per-step case study require additional registered fields")
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_fig6(rows: list[dict[str, str]], output: Path) -> None:
    height = 1020
    actors = sorted({row["actor"] for row in rows})
    actor_colors = _actor_palette(actors)
    parts = _svg_base(height, "Fig. 6 population failure anatomy", "Population-level failure anatomy from result CSV; step-level case study is withheld because it is not represented in the result CSV schema.")
    _header(parts, "FIG. 6", "Failure anatomy from observable scored outcomes", "Population panels are data-derived; the step-level case study and hydrodynamic mechanism are delivered separately when their inputs exist.", height)
    px, py, pw, ph = _panel(parts, 60, 118, 840, 650, "d", "First scored failure stage", "Stage labels are the frozen observable-behaviour categories.")
    base_x, base_y, bar_w, bar_h = px + 30, py + 38, pw - 80, ph - 110
    stages = ["L1", "L2", "L3", "L4", "L5", "none"]
    for i, actor in enumerate(actors):
        subset = [r for r in rows if r["actor"] == actor]; xx = base_x + (i + .5) * bar_w / max(1, len(actors)); running = 0
        for stage in stages:
            frac = sum((r.get("first_error_stage") or "none") == stage for r in subset) / len(subset) if subset else 0; hh = bar_h * frac
            if hh: _rect(parts, xx - 22, base_y + bar_h - running - hh, 44, hh, STAGE_COLORS[stage], "none", 0, 2)
            running += hh
        _text(parts, xx, base_y + bar_h + 28, _short_actor(actor, 10), 11, INK, "400", "middle")
    _text(parts, base_x, base_y - 12, "episode fraction", 13, SLATE, "700")
    for i, stage in enumerate(stages):
        xx = px + 22 + (i % 3) * 235; yy = py + ph - 64 + (i // 3) * 22; _rect(parts, xx, yy - 11, 12, 12, STAGE_COLORS[stage], "none", 0, 2); _text(parts, xx + 20, yy, stage, 12, INK)
    px2, py2, pw2, ph2 = _panel(parts, 940, 118, 800, 650, "e", "Trajectory closest approach to source", "Minimum 3-D distance from any actually visited node to canonical source_nodes; this is not answer error.")
    plot_x, plot_y, plot_w, plot_h = px2 + 62, py2 + 30, pw2 - 95, ph2 - 105
    vals = [_float(r.get("trajectory_source_closest_distance_m")) for r in rows]; vals = [v for v in vals if v is not None]; xmax = max(vals, default=1); xmax = max(1, math.ceil(xmax / 10) * 10)
    _line(parts, plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, INK, 1.5); _line(parts, plot_x, plot_y, plot_x, plot_y + plot_h, INK, 1.5)
    _text(parts, plot_x + plot_w / 2, plot_y + plot_h + 42, "minimum visited-node distance to source (m)", 13, SLATE, "400", "middle")
    for i, actor in enumerate(actors):
        subset = [_float(r.get("trajectory_source_closest_distance_m")) for r in rows if r["actor"] == actor and _float(r.get("trajectory_source_closest_distance_m")) is not None]; subset.sort(); yy = plot_y + 26 + i * (plot_h - 48) / max(1, len(actors));
        if subset:
            q1, med, q3 = subset[len(subset)//4], subset[len(subset)//2], subset[(3*len(subset))//4]; _line(parts, _xmap(q1, 0, xmax, plot_x, plot_w), yy, _xmap(q3, 0, xmax, plot_x, plot_w), yy, actor_colors[actor], 5); _line(parts, _xmap(q1, 0, xmax, plot_x, plot_w), yy - 8, _xmap(q1, 0, xmax, plot_x, plot_w), yy + 8, actor_colors[actor], 2); _line(parts, _xmap(q3, 0, xmax, plot_x, plot_w), yy - 8, _xmap(q3, 0, xmax, plot_x, plot_w), yy + 8, actor_colors[actor], 2); _circle(parts, _xmap(med, 0, xmax, plot_x, plot_w), yy, 6, ORANGE, PANEL, 1.2)
        _text(parts, plot_x - 12, yy + 5, _short_actor(actor, 10), 11, INK, "700", "end")
    _rect(parts, 60, 800, 1680, 100, SOFT, "#C9D9E0", 1.2, 8)
    _text(parts, 84, 833, "interpretation", 14, ORANGE, "700")
    _text(parts, 230, 833, "Fig. 6e separates physical approach from recognition: a near-source track with a failed L4 answer is a decision failure, not a reachability failure.", 15, NAVY, "700")
    _text(parts, 230, 860, "Fig. 6a–c still requires registered step-level case-study fields. Fig. 6f remains the separate image2 mechanism schematic.", 14, SLATE)
    _footer(parts, height, "Population data only · no model self-report used · closest approach computed offline from logged visited nodes and canonical source_nodes")
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _table_rows(rows: list[dict[str, str]], actors: list[str]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for actor in actors:
        subset = [row for row in rows if row["actor"] == actor]
        unique_instances = {row["instance_id"] for row in subset}
        ambiguous_instances = {row["instance_id"] for row in subset if row.get("candidate_ambiguous") == "True"}
        presentation_modes = sorted({row.get("presentation_mode") or "records" for row in subset})
        item: dict[str, str] = {
            "actor": actor,
            "science_eligible": sorted({row["science_eligible"] for row in subset})[0],
            "episodes": str(len(subset)),
            "model_config_hash": next((row.get("model_config_hash", "") for row in subset if row.get("model_config_hash")), ""),
            "presentation_mode": ";".join(presentation_modes),
            "information_condition": "adjacent-node actions; no neighbour values; complete cumulative own measurements",
            "candidate_identity_limit": f"{len(ambiguous_instances)}/{len(unique_instances)} tasks spatially ambiguous; identity is secondary",
        }
        l1 = _l1_metrics(subset)
        item["L1_balanced_accuracy"] = _fmt(l1["balanced_accuracy"], 12) if l1["balanced_accuracy"] is not None else ""
        item["L1_sensitivity"] = _fmt(l1["sensitivity"], 12) if l1["sensitivity"] is not None else ""
        item["L1_specificity"] = _fmt(l1["specificity"], 12) if l1["specificity"] is not None else ""
        for level in range(2, 6):
            applicable = _applicable(subset, f"L{level}_pass")
            item[f"L{level}_pass_rate"] = _fmt(sum(row[f"L{level}_pass"] == "True" for row in applicable) / len(applicable), 12) if applicable else ""
        for prefix, field in (("L2_peak_median_distance_m", "L2_peak_distance_m"), ("L4_source_median_distance_m", "L4_source_distance_m"), ("trajectory_source_closest_median_distance_m", "trajectory_source_closest_distance_m")):
            values = sorted(v for v in (_float(row.get(field)) for row in subset) if v is not None); item[prefix] = _fmt(values[len(values)//2], 6) if values else ""
        output.append(item)
    return output


def build_figures(results_csv: str | Path, output: str | Path) -> list[Path]:
    source = Path(results_csv)
    if source.name != "benchmark_results.csv" or source.suffix.lower() != ".csv":
        raise ValueError("figures accepts only a file named benchmark_results.csv")
    rows = read_csv(source)
    if not rows:
        raise ValueError("benchmark_results.csv is empty")
    missing = REQUIRED - set(rows[0])
    if missing:
        raise ValueError(f"benchmark_results.csv missing required columns: {sorted(missing)}")
    target = Path(output); target.mkdir(parents=True, exist_ok=True)
    actors = sorted({row["actor"] for row in rows})
    outputs: list[Path] = []
    builders = [("Fig1_exam_overview_scores.svg", build_fig1), ("Fig2_task_construction.svg", build_fig2), ("Fig3_driver_interface_validation.svg", build_fig3), ("Fig4_system_level_matrix.svg", build_fig4), ("Fig5_displacement_dose_response.svg", build_fig5), ("Fig6_failure_anatomy.svg", build_fig6)]
    for filename, builder in builders:
        path = target / filename; builder(rows, path); outputs.append(path)
    table_rows = _table_rows(rows, actors)
    table = target / "Table1_system_roster_scores.csv"; write_csv(table, list(table_rows[0]), table_rows); outputs.append(table)
    return outputs
