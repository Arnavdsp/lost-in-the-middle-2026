"""Figures. Matplotlib only, colorblind-safe, and never a curve without error bars.

THE ONE RULE
------------
:class:`PositionCurve` and :class:`USISeries` **require** confidence bounds in
their constructors and raise :class:`MissingConfidenceInterval` without them. A
bare accuracy line is the thing that makes a reviewer stop reading, so the plotting
layer refuses to draw one. If you have no CI, you have no chart -- go back to
:mod:`litm2026.stats`.

PALETTE
-------
Three categorical hues, in fixed order, from a validated colorblind-safe set:

    slot 1  blue   #2a78d6
    slot 2  orange #eb6834
    slot 3  aqua   #1baf7a

Validated all-pairs on a light surface: worst CVD (deuteranopia) Delta E 9.2,
worst normal-vision Delta E 24.0, both clear of the >=8 / >=15 floors. Three slots is
also the project's model cap, which is not a coincidence -- past three series this
palette stops being all-pairs safe and the honest move is to facet, not to invent
a fourth hue. Aqua sits below 3:1 contrast on the light surface, so every series
is **direct-labelled** as well as legended: identity never rests on colour alone.

Closed-book and oracle baselines are drawn as neutral grey dashed rules with
direct labels. They are references, not series, and they do not consume a hue.

THUMBNAIL LEGIBILITY
--------------------
These charts get shared as images in Slack and on GitHub cards. Base font 13pt at
200 dpi, 2pt lines, 6pt markers, recessive grid, and no chartjunk, so the shape
survives being scaled to 400px wide.
"""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # figures are written to files; never needs a display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

__all__ = [
    "SERIES_COLORS",
    "MissingConfidenceInterval",
    "PositionCurve",
    "USISeries",
    "plot_accuracy_by_position",
    "plot_2023_vs_2026",
    "plot_usi_vs_context_length",
    "plot_not_yet_run_placeholder",
    "curves_from_summary_csv",
    "main",
]

#: Categorical hues in fixed order. Never cycled, never extended past three.
SERIES_COLORS: Tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e6e5e1"
REFERENCE_LINE = "#52514e"
HISTORICAL_2023 = "#8a8983"


class MissingConfidenceInterval(ValueError):
    """Raised when a plot is asked to draw a series with no confidence bounds."""


def _apply_style() -> None:
    """Set the shared rcParams. Called by every public plotting function."""
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "text.color": INK_PRIMARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.frameon": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
        }
    )


def _validate_ci(
    label: str,
    values: Sequence[float],
    ci_low: Optional[Sequence[float]],
    ci_high: Optional[Sequence[float]],
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if ci_low is None or ci_high is None:
        raise MissingConfidenceInterval(
            f"Series {label!r} has no confidence interval. Every chart in this repo shows CIs; "
            "compute them with litm2026.stats.bootstrap_ci (or u_shape_severity_ci) first."
        )
    if not (len(values) == len(ci_low) == len(ci_high)):
        raise ValueError(
            f"Series {label!r}: values, ci_low and ci_high must be the same length "
            f"({len(values)}, {len(ci_low)}, {len(ci_high)})"
        )
    return tuple(float(v) for v in ci_low), tuple(float(v) for v in ci_high)


@dataclass(frozen=True)
class PositionCurve:
    """Accuracy as a function of gold position, for one model.

    Attributes:
        label: Series name for the legend and the direct label.
        positions: Gold document/key positions, ascending.
        accuracy: Accuracy at each position, in [0, 1].
        ci_low: Lower confidence bounds. **Required.**
        ci_high: Upper confidence bounds. **Required.**

    Raises:
        MissingConfidenceInterval: If either bound is omitted.
        ValueError: On length mismatches or an empty series.
    """

    label: str
    positions: Tuple[int, ...]
    accuracy: Tuple[float, ...]
    ci_low: Tuple[float, ...]
    ci_high: Tuple[float, ...]

    def __init__(
        self,
        label: str,
        positions: Sequence[int],
        accuracy: Sequence[float],
        ci_low: Optional[Sequence[float]] = None,
        ci_high: Optional[Sequence[float]] = None,
    ) -> None:
        if len(positions) == 0:
            raise ValueError(f"Series {label!r} is empty")
        if len(positions) != len(accuracy):
            raise ValueError(f"Series {label!r}: positions and accuracy differ in length")
        low, high = _validate_ci(label, accuracy, ci_low, ci_high)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "positions", tuple(int(p) for p in positions))
        object.__setattr__(self, "accuracy", tuple(float(a) for a in accuracy))
        object.__setattr__(self, "ci_low", low)
        object.__setattr__(self, "ci_high", high)

    def yerr(self) -> List[List[float]]:
        """Asymmetric error bar offsets, as matplotlib's ``yerr`` wants them."""
        lower = [max(0.0, a - lo) for a, lo in zip(self.accuracy, self.ci_low)]
        upper = [max(0.0, hi - a) for a, hi in zip(self.accuracy, self.ci_high)]
        return [lower, upper]


@dataclass(frozen=True)
class USISeries:
    """U-Shape Severity Index versus total context length, for one model.

    Attributes:
        label: Series name.
        context_lengths: X values -- total input length in tokens (or the best
            available proxy, which must be stated in the caption).
        usi: The index at each length.
        ci_low: Lower bootstrap bounds. **Required.**
        ci_high: Upper bootstrap bounds. **Required.**
    """

    label: str
    context_lengths: Tuple[float, ...]
    usi: Tuple[float, ...]
    ci_low: Tuple[float, ...]
    ci_high: Tuple[float, ...]

    def __init__(
        self,
        label: str,
        context_lengths: Sequence[float],
        usi: Sequence[float],
        ci_low: Optional[Sequence[float]] = None,
        ci_high: Optional[Sequence[float]] = None,
    ) -> None:
        if len(context_lengths) == 0:
            raise ValueError(f"Series {label!r} is empty")
        if len(context_lengths) != len(usi):
            raise ValueError(f"Series {label!r}: context_lengths and usi differ in length")
        low, high = _validate_ci(label, usi, ci_low, ci_high)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "context_lengths", tuple(float(x) for x in context_lengths))
        object.__setattr__(self, "usi", tuple(float(u) for u in usi))
        object.__setattr__(self, "ci_low", low)
        object.__setattr__(self, "ci_high", high)


def _finish(
    fig: Figure,
    ax: Any,
    *,
    title: str,
    subtitle: Optional[str],
    caption: Optional[str],
    outpath: Optional[pathlib.Path],
    watermark: Optional[str] = None,
) -> Figure:
    """Apply shared chrome (titles, caption, watermark) and optionally save."""
    ax.set_title(title, loc="left", pad=18 if subtitle else 10, fontweight="bold")
    if subtitle:
        ax.text(
            0.0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            fontsize=11.5,
            color=INK_SECONDARY,
            va="bottom",
        )
    if caption:
        fig.text(0.012, 0.012, caption, fontsize=9.5, color=INK_MUTED, va="bottom", ha="left")
    if watermark:
        fig.text(
            0.5,
            0.5,
            watermark,
            fontsize=30,
            color="#d03b3b",
            alpha=0.22,
            ha="center",
            va="center",
            rotation=22,
            fontweight="bold",
            zorder=100,
        )
    fig.tight_layout(rect=(0, 0.05 if caption else 0.0, 1, 1))
    if outpath is not None:
        outpath = pathlib.Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight")
    return fig


def plot_accuracy_by_position(
    curves: Sequence[PositionCurve],
    *,
    title: str = "Accuracy by position of the gold document",
    subtitle: Optional[str] = None,
    xlabel: str = "Position of the gold document in the input",
    ylabel: str = "Best-subspan EM",
    closedbook: Optional[Mapping[str, float]] = None,
    oracle: Optional[Mapping[str, float]] = None,
    caption: Optional[str] = None,
    outpath: Optional[pathlib.Path] = None,
    watermark: Optional[str] = None,
) -> Figure:
    """The paper's core figure: accuracy versus gold position, with error bars.

    Args:
        curves: One :class:`PositionCurve` per model (at most three).
        title: Chart title.
        subtitle: One line under the title -- put n, seed and model versions here.
        xlabel: X axis label.
        ylabel: Y axis label.
        closedbook: Optional ``{label: accuracy}`` floor references, drawn as
            dashed grey rules. The paper's most striking claim is that middle
            positions can fall *below* the closed-book floor, so this reference is
            what makes the chart interpretable rather than decorative.
        oracle: Optional ``{label: accuracy}`` ceiling references.
        caption: Small print under the figure (data source, metric, run id).
        outpath: If given, save the PNG here.
        watermark: Diagonal overlay text, used to mark synthetic/demo figures.

    Returns:
        The matplotlib :class:`~matplotlib.figure.Figure`.

    Raises:
        ValueError: If more than three curves are supplied.
    """
    if len(curves) > len(SERIES_COLORS):
        raise ValueError(
            f"{len(curves)} series exceeds the {len(SERIES_COLORS)}-slot colorblind-safe palette. "
            "Facet into small multiples instead of inventing a fourth hue."
        )
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    for i, curve in enumerate(curves):
        color = SERIES_COLORS[i]
        ax.errorbar(
            curve.positions,
            curve.accuracy,
            yerr=curve.yerr(),
            label=curve.label,
            color=color,
            marker="o",
            capsize=3.5,
            elinewidth=1.4,
            zorder=3 + i,
        )
        # Direct label: identity must not rest on colour alone (relief rule).
        ax.annotate(
            curve.label,
            xy=(curve.positions[-1], curve.accuracy[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=11,
            va="center",
            fontweight="bold",
        )

    for label, value in (closedbook or {}).items():
        ax.axhline(value, color=REFERENCE_LINE, linestyle=(0, (6, 4)), linewidth=1.4, zorder=2)
        ax.annotate(
            f"closed-book · {label}",
            xy=(0.005, value),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=10,
            color=REFERENCE_LINE,
        )
    for label, value in (oracle or {}).items():
        ax.axhline(value, color=REFERENCE_LINE, linestyle=(0, (2, 3)), linewidth=1.4, zorder=2)
        ax.annotate(
            f"oracle · {label}",
            xy=(0.005, value),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=10,
            color=REFERENCE_LINE,
        )

    all_positions = sorted({p for c in curves for p in c.positions})
    ax.set_xticks(all_positions)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if len(curves) >= 2:
        ax.legend(loc="lower left", ncols=min(3, len(curves)))
    return _finish(
        fig, ax, title=title, subtitle=subtitle, caption=caption, outpath=outpath, watermark=watermark
    )


def plot_2023_vs_2026(
    curves_2026: Sequence[PositionCurve],
    curves_2023: Sequence[PositionCurve],
    *,
    title: str = "2023 models vs 2026 models, same data and same metric",
    subtitle: Optional[str] = None,
    caption: Optional[str] = None,
    outpath: Optional[pathlib.Path] = None,
    watermark: Optional[str] = None,
) -> Figure:
    """Overlay this replication's curves on the original paper's published curves.

    The 2023 curves are drawn in neutral grey with open markers and a dashed line,
    so the eye reads them as historical reference rather than as a fourth and fifth
    model. Their provenance belongs in ``caption``: state plainly which are read
    from the paper's released ``EXPERIMENTS.md`` and which (if any) are digitized
    from a figure and therefore approximate.

    Args:
        curves_2026: This replication's curves.
        curves_2023: Curves from ``data/original_results.csv``.
        title: Chart title.
        subtitle: One line under the title.
        caption: Provenance line. Do not omit it.
        outpath: If given, save the PNG here.
        watermark: Diagonal overlay text for synthetic/demo figures.

    Returns:
        The matplotlib figure.
    """
    if len(curves_2026) > len(SERIES_COLORS):
        raise ValueError(f"{len(curves_2026)} 2026 series exceeds the {len(SERIES_COLORS)}-slot palette")
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    for curve in curves_2023:
        ax.errorbar(
            curve.positions,
            curve.accuracy,
            yerr=curve.yerr(),
            label=f"{curve.label} (2023)",
            color=HISTORICAL_2023,
            marker="o",
            markerfacecolor=SURFACE,
            linestyle=(0, (5, 3)),
            linewidth=1.8,
            capsize=3.0,
            elinewidth=1.2,
            zorder=2,
        )
    for i, curve in enumerate(curves_2026):
        color = SERIES_COLORS[i]
        ax.errorbar(
            curve.positions,
            curve.accuracy,
            yerr=curve.yerr(),
            label=f"{curve.label} (2026)",
            color=color,
            marker="o",
            capsize=3.5,
            elinewidth=1.4,
            zorder=4 + i,
        )
        ax.annotate(
            curve.label,
            xy=(curve.positions[-1], curve.accuracy[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=11,
            va="center",
            fontweight="bold",
        )

    all_positions = sorted({p for c in list(curves_2026) + list(curves_2023) for p in c.positions})
    ax.set_xticks(all_positions)
    ax.set_xlabel("Position of the gold document in the input")
    ax.set_ylabel("Best-subspan EM")
    ax.set_ylim(0.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", ncols=2)
    return _finish(
        fig, ax, title=title, subtitle=subtitle, caption=caption, outpath=outpath, watermark=watermark
    )


def plot_usi_vs_context_length(
    series: Sequence[USISeries],
    *,
    title: str = "Does the U-shape survive longer contexts?",
    subtitle: Optional[str] = "U-Shape Severity Index vs total context length (higher = worse middle)",
    caption: Optional[str] = None,
    outpath: Optional[pathlib.Path] = None,
    log_x: bool = True,
    watermark: Optional[str] = None,
) -> Figure:
    """THE HEADLINE CHART: U-shape severity (y) against total context length (x).

    One line per model, error bars from :func:`litm2026.stats.u_shape_severity_ci`.
    A horizontal rule at 0 marks "no middle penalty", which is the null hypothesis
    of the 2026 extension: if long-context training removed the effect, the lines
    sit on that rule at every length.

    Args:
        series: One :class:`USISeries` per model (at most three).
        title: Chart title.
        subtitle: One line under the title.
        caption: Small print -- say how context length was measured.
        outpath: If given, save the PNG here.
        log_x: Log-scale the x axis. Context lengths span ~2K to ~100K+ tokens, so
            this is on by default.
        watermark: Diagonal overlay text for synthetic/demo figures.

    Returns:
        The matplotlib figure.
    """
    if len(series) > len(SERIES_COLORS):
        raise ValueError(f"{len(series)} series exceeds the {len(SERIES_COLORS)}-slot palette")
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0.0, color=REFERENCE_LINE, linestyle=(0, (6, 4)), linewidth=1.4, zorder=2)
    ax.annotate(
        "no middle penalty",
        xy=(0.005, 0.0),
        xycoords=("axes fraction", "data"),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=10,
        color=REFERENCE_LINE,
    )

    for i, item in enumerate(series):
        color = SERIES_COLORS[i]
        lower = [max(0.0, u - lo) for u, lo in zip(item.usi, item.ci_low)]
        upper = [max(0.0, hi - u) for u, hi in zip(item.usi, item.ci_high)]
        ax.errorbar(
            item.context_lengths,
            item.usi,
            yerr=[lower, upper],
            label=item.label,
            color=color,
            marker="o",
            capsize=3.5,
            elinewidth=1.4,
            zorder=3 + i,
        )
        ax.annotate(
            item.label,
            xy=(item.context_lengths[-1], item.usi[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=11,
            va="center",
            fontweight="bold",
        )

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Total context length (tokens)")
    ax.set_ylabel("U-Shape Severity Index")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if len(series) >= 2:
        ax.legend(loc="upper left", ncols=min(3, len(series)))
    return _finish(
        fig, ax, title=title, subtitle=subtitle, caption=caption, outpath=outpath, watermark=watermark
    )


def plot_not_yet_run_placeholder(
    outpath: pathlib.Path,
    *,
    headline: str = "NO EXPERIMENT HAS BEEN RUN YET",
    detail: str = (
        "This repository is complete and tested, but no model has been called.\n"
        "There are no results, and no numbers have been invented to stand in for them.\n\n"
        "Run:  python -m litm2026.runner --config experiments/configs/pilot.yaml --dry-run\n"
        "then: python -m litm2026.runner --config experiments/configs/pilot.yaml"
    ),
) -> pathlib.Path:
    """Render the README's above-the-fold placeholder image.

    Deliberately contains **no axes and no data**. An empty chart frame would
    suggest a measurement that does not exist; a text card cannot be mistaken for
    one.

    Args:
        outpath: Where to write the PNG.
        headline: Large red line.
        detail: Body text.

    Returns:
        The path written.
    """
    _apply_style()
    fig = plt.figure(figsize=(8.0, 4.0))
    fig.text(0.5, 0.78, headline, ha="center", va="center", fontsize=19, fontweight="bold", color="#d03b3b")
    fig.text(0.5, 0.42, detail, ha="center", va="center", fontsize=11.5, color=INK_SECONDARY, linespacing=1.6)
    fig.text(
        0.5,
        0.07,
        "lost-in-the-middle-2026 · replication of Liu et al. (TACL 2024), arXiv:2307.03172",
        ha="center",
        va="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    outpath = pathlib.Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath


def curves_from_summary_csv(
    path: pathlib.Path,
    *,
    experiment: str = "qa_positional",
    num_documents: Optional[int] = None,
    num_keys: Optional[int] = None,
    labels: Optional[Mapping[str, str]] = None,
) -> Tuple[List[PositionCurve], Dict[str, float], Dict[str, float]]:
    """Build position curves and baseline references from ``results/summary.csv``.

    Args:
        path: Path to a summary CSV produced by :func:`litm2026.stats.write_summary_csv`.
        experiment: Which arm to turn into curves.
        num_documents: Filter to one document count (QA).
        num_keys: Filter to one key count (KV).
        labels: Optional ``model_key -> display label`` overrides.

    Returns:
        ``(curves, closedbook, oracle)`` -- the curves plus ``{label: accuracy}``
        maps for the two baseline arms, empty when those arms were not run.

    Raises:
        FileNotFoundError: If the summary does not exist.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No summary at {path}. Run `python -m litm2026.stats --run-dir ...` first.")
    with open(path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_model: Dict[str, List[Dict[str, Any]]] = {}
    closedbook: Dict[str, float] = {}
    oracle: Dict[str, float] = {}
    for row in rows:
        label = (labels or {}).get(row["model_key"], row["model_key"])
        if row["experiment"] == "qa_closedbook":
            closedbook[label] = float(row["accuracy"])
        elif row["experiment"] == "qa_oracle":
            oracle[label] = float(row["accuracy"])
        elif row["experiment"] == experiment:
            if num_documents is not None and row.get("num_documents") != str(num_documents):
                continue
            if num_keys is not None and row.get("num_keys") != str(num_keys):
                continue
            by_model.setdefault(label, []).append(row)

    curves: List[PositionCurve] = []
    for label, model_rows in sorted(by_model.items()):
        model_rows.sort(key=lambda r: int(r["gold_index"]))
        curves.append(
            PositionCurve(
                label=label,
                positions=[int(r["gold_index"]) for r in model_rows],
                accuracy=[float(r["accuracy"]) for r in model_rows],
                ci_low=[float(r["ci_low"]) for r in model_rows],
                ci_high=[float(r["ci_high"]) for r in model_rows],
            )
        )
    return curves, closedbook, oracle


def _smoke(outdir: pathlib.Path) -> List[pathlib.Path]:
    """Render every chart from obviously synthetic inputs, watermarked as such.

    Exists so the plotting code is exercised end to end without any model call.
    Every figure carries a red diagonal "SYNTHETIC DEMO DATA -- NOT A RESULT"
    watermark and a caption saying the same thing, so a stray PNG cannot be
    mistaken for a finding.
    """
    outdir = pathlib.Path(outdir)
    mark = "SYNTHETIC DEMO DATA - NOT A RESULT"
    caption = "Synthetic numbers, generated to exercise the plotting code. Not a measurement of any model."
    positions = [0, 4, 9, 14, 19]
    written: List[pathlib.Path] = []

    curves = [
        PositionCurve(
            "demo-model-a",
            positions,
            [0.70, 0.60, 0.55, 0.58, 0.68],
            ci_low=[0.63, 0.53, 0.48, 0.51, 0.61],
            ci_high=[0.77, 0.67, 0.62, 0.65, 0.75],
        ),
        PositionCurve(
            "demo-model-b",
            positions,
            [0.55, 0.50, 0.47, 0.49, 0.54],
            ci_low=[0.48, 0.43, 0.40, 0.42, 0.47],
            ci_high=[0.62, 0.57, 0.54, 0.56, 0.61],
        ),
    ]
    written.append(outdir / "smoke_accuracy_by_position.png")
    fig = plot_accuracy_by_position(
        curves,
        subtitle="synthetic inputs, n=demo",
        closedbook={"demo-model-a": 0.32},
        oracle={"demo-model-a": 0.84},
        caption=caption,
        outpath=written[-1],
        watermark=mark,
    )
    plt.close(fig)

    written.append(outdir / "smoke_2023_vs_2026.png")
    fig = plot_2023_vs_2026(
        curves[:1],
        [
            PositionCurve(
                "demo-2023",
                positions,
                [0.68, 0.57, 0.55, 0.53, 0.56],
                ci_low=[0.66, 0.55, 0.53, 0.51, 0.54],
                ci_high=[0.70, 0.59, 0.57, 0.55, 0.58],
            )
        ],
        caption=caption,
        outpath=written[-1],
        watermark=mark,
    )
    plt.close(fig)

    written.append(outdir / "smoke_usi_vs_context_length.png")
    fig = plot_usi_vs_context_length(
        [
            USISeries(
                "demo-model-a",
                [2000, 4000, 12000, 40000],
                [0.05, 0.12, 0.20, 0.31],
                ci_low=[0.01, 0.07, 0.14, 0.23],
                ci_high=[0.10, 0.18, 0.27, 0.39],
            ),
            USISeries(
                "demo-model-b",
                [2000, 4000, 12000, 40000],
                [0.02, 0.03, 0.05, 0.09],
                ci_low=[-0.02, -0.01, 0.01, 0.03],
                ci_high=[0.06, 0.08, 0.10, 0.15],
            ),
        ],
        caption=caption,
        outpath=written[-1],
        watermark=mark,
    )
    plt.close(fig)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: build figures from a summary CSV, or the placeholder, or a smoke render."""
    import argparse

    parser = argparse.ArgumentParser(description="Render the repository's figures.")
    parser.add_argument("--summary", type=pathlib.Path, help="results/summary.csv from litm2026.stats")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("results/figures"))
    parser.add_argument("--num-documents", type=int, default=20, help="Document count to plot for qa_positional")
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Write the 'no experiment has been run' card used above the fold in the README.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Render every chart from synthetic, watermarked inputs. Never a result.",
    )
    args = parser.parse_args(argv)

    if args.placeholder:
        path = plot_not_yet_run_placeholder(args.outdir / "headline_not_yet_run.png")
        print(f"Wrote {path}")
        return 0
    if args.smoke:
        for path in _smoke(args.outdir):
            print(f"Wrote {path}")
        return 0
    if not args.summary:
        parser.error("pass --summary, --placeholder or --smoke")

    curves, closedbook, oracle = curves_from_summary_csv(
        args.summary, experiment="qa_positional", num_documents=args.num_documents
    )
    if not curves:
        print(f"No qa_positional rows with num_documents={args.num_documents} in {args.summary}")
        return 1
    out = args.outdir / f"accuracy_by_position_{args.num_documents}docs.png"
    fig = plot_accuracy_by_position(
        curves,
        subtitle=f"{args.num_documents} total documents",
        closedbook=closedbook,
        oracle=oracle,
        caption=f"Source: {args.summary}. Bars are 95% percentile bootstrap CIs.",
        outpath=out,
    )
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
