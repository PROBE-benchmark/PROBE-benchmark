"""V20-semantic wrapper around the frozen legacy Nature SVG renderer."""

from __future__ import annotations

from pathlib import Path

from laketrace.figures.nature import build_figures as build_legacy_figures
from laketrace.util import read_csv


L2_TARGET = "registered_F1_all_anomalous_footprint_components"


def _replace_registered_text(path: Path, replacements: tuple[tuple[str, str], ...]) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        count = text.count(old)
        if count != 1:
            raise ValueError(
                f"v20 figure semantic patch expected one {old!r} in {path.name}; found {count}"
            )
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def build_v20_figures(results_csv: str | Path, output: str | Path) -> list[Path]:
    """Build CSV-only figures while naming the frozen v20 L2 construct exactly.

    The base renderer belongs to the byte-frozen v4 layer and cannot be edited
    without invalidating historical releases.  This explicit wrapper first
    validates the score target and then changes only registered SVG prose; all
    plotted values and geometry still come from the shared deterministic
    renderer.
    """

    source = Path(results_csv)
    rows = read_csv(source)
    if not rows:
        raise ValueError("benchmark_results.csv is empty")
    targets = {row.get("L2_scoring_target") for row in rows}
    if targets != {L2_TARGET}:
        raise ValueError(
            "v20 figures require one frozen L2 anomaly-footprint target; "
            f"observed {sorted(str(value) for value in targets)}"
        )
    artifacts = build_legacy_figures(source, output)
    root = Path(output)
    _replace_registered_text(
        root / "Fig1_exam_overview_scores.svg",
        (
            ("Peak versus source", "Anomaly footprint versus source"),
            (
                "Same rows; green = L2 peak, orange = L4 source.",
                "Same rows; green = L2 anomaly footprint, orange = L4 source.",
            ),
        ),
    )
    _replace_registered_text(
        root / "Fig5_displacement_dose_response.svg",
        (
            (
                "Fig. 5 peak versus source displacement",
                "Fig. 5 anomaly-footprint and source outcomes",
            ),
            (
                "Displacement response and error components derived from result CSV fields.",
                "Anomaly-footprint and source outcomes versus the peak-to-source diagnostic.",
            ),
            (
                "Peak and source localization are evaluated separately",
                "Anomaly-footprint and source localization are evaluated separately",
            ),
            (
                "The displacement analysis is restricted to the range occupied by the current task set; no extrapolation or model-capability claim is implied.",
                "Peak-to-source displacement is a registered task diagnostic, not the L2 answer target; no extrapolation is implied.",
            ),
            (
                "L2 and L4 versus peak-to-source displacement",
                "L2 anomaly footprint and L4 source versus peak-to-source displacement",
            ),
            (">L2 peak</text>", ">L2 region</text>"),
        ),
    )
    return artifacts


__all__ = ["L2_TARGET", "build_v20_figures"]
