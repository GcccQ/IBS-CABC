from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .reporting import (  # noqa: E402
    EXPECTED_ALGORITHMS,
    EXPECTED_JOB_COUNTS,
    load_ibscabc_results,
    validate_complete_four_algorithm_results,
)


GRID = np.linspace(0.0, 1.0, 101)
COLORS = {
    "IBS_CABC": "#1F4E79",
    "Standard ABC": "#D97706",
    "ALNS": "#748238",
    "TS": "#A64D79",
}
LINE_STYLES = {
    "IBS_CABC": "-",
    "Standard ABC": "--",
    "ALNS": "-.",
    "TS": ":",
}


class ConvergenceDataError(ValueError):
    """Raised when formal histories cannot support an honest convergence plot."""


def _identity(row: Mapping[str, object]) -> str:
    digest = str(row.get("instance_sha256", "")).strip().lower()
    if digest:
        return digest
    return str(Path(str(row["instance_path"])).resolve()).lower()


def _canonical_label(value: object) -> str:
    compact = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "ibscabc": "IBS_CABC",
        "ibs_cabc": "IBS_CABC",
        "standard_abc": "Standard ABC",
        "alns": "ALNS",
        "ts": "TS",
        "tabu_search": "TS",
    }.get(compact, str(value).strip())


def _load_combined(
    baseline_results: list[dict[str, object]],
    ibscabc_result_paths: Path | Iterable[Path],
) -> list[dict[str, object]]:
    paths = (
        [ibscabc_result_paths]
        if isinstance(ibscabc_result_paths, Path)
        else list(ibscabc_result_paths)
    )
    ibscabc = [row for path in paths for row in load_ibscabc_results(path)]
    combined = ibscabc + list(baseline_results)
    validate_complete_four_algorithm_results(combined)
    return combined


def _incumbent_curve(row: Mapping[str, object]) -> np.ndarray:
    history = np.asarray(list(row.get("history", [])), dtype=float)
    evaluations = np.asarray(list(row.get("history_evaluations", [])), dtype=float)
    if history.size == 0 or evaluations.size == 0:
        raise ConvergenceDataError(
            f"missing convergence history for {row.get('algorithm_label')}/"
            f"J{row.get('jobs')}/{row.get('instance')}"
        )
    if history.size != evaluations.size:
        raise ConvergenceDataError(
            f"history length mismatch for {row.get('algorithm_label')}/"
            f"J{row.get('jobs')}/{row.get('instance')}"
        )
    if np.any(np.diff(evaluations) < 0):
        raise ConvergenceDataError(f"history evaluations are not ordered: {row.get('task_id')}")
    if np.any(np.diff(history) > 1e-9):
        raise ConvergenceDataError(f"incumbent history is not non-increasing: {row.get('task_id')}")

    total_evaluations = max(float(row.get("evaluations", 0)), float(evaluations[-1]))
    if total_evaluations <= 0:
        raise ConvergenceDataError(f"invalid evaluation count: {row.get('task_id')}")
    final_objective = float(row["total_completion_time"])
    if final_objective > history[-1] + 1e-9:
        raise ConvergenceDataError(
            f"final objective is worse than recorded incumbent: {row.get('task_id')}"
        )
    if final_objective < history[-1] - 1e-9:
        history = np.append(history, final_objective)
        evaluations = np.append(evaluations, total_evaluations)

    progress = np.clip(evaluations / total_evaluations, 0.0, 1.0)
    progress = np.insert(progress, 0, 0.0)
    history = np.insert(history, 0, history[0])
    indices = np.searchsorted(progress, GRID, side="right") - 1
    indices = np.clip(indices, 0, history.size - 1)
    curve = history[indices]
    curve[-1] = final_objective
    return curve


def _mean_ci(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = curves.mean(axis=0)
    if curves.shape[0] <= 1:
        zeros = np.zeros_like(mean)
        return mean, zeros, zeros
    standard_deviation = curves.std(axis=0, ddof=1)
    ci95 = 1.96 * standard_deviation / math.sqrt(curves.shape[0])
    return mean, standard_deviation, ci95


def _auc(values: np.ndarray) -> float:
    widths = np.diff(GRID)
    return float(np.sum((values[:-1] + values[1:]) * 0.5 * widths))


def build_convergence_data(
    baseline_results: list[dict[str, object]],
    ibscabc_result_paths: Path | Iterable[Path],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Build paired mean-gap curves and convergence diagnostics."""
    combined = _load_combined(baseline_results, ibscabc_result_paths)
    grouped: dict[int, dict[str, dict[str, Mapping[str, object]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in combined:
        jobs = int(row["jobs"])
        grouped[jobs][_identity(row)][_canonical_label(row["algorithm_label"])] = row

    curve_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for jobs in EXPECTED_JOB_COUNTS:
        instances = grouped[jobs]
        final_values = {
            identity: {
                algorithm: float(rows[algorithm]["total_completion_time"])
                for algorithm in EXPECTED_ALGORITHMS
            }
            for identity, rows in instances.items()
        }
        references = {
            identity: min(values.values()) for identity, values in final_values.items()
        }
        for algorithm in EXPECTED_ALGORITHMS:
            relative_curves = []
            ranks = []
            wins = 0
            for identity, rows in sorted(instances.items()):
                objective_curve = _incumbent_curve(rows[algorithm])
                reference = references[identity]
                relative_curves.append(100.0 * (objective_curve - reference) / reference)
                target = final_values[identity][algorithm]
                best = min(final_values[identity].values())
                wins += int(abs(target - best) <= 1e-9)
                ranks.append(
                    1
                    + sum(
                        value < target - 1e-9
                        for value in final_values[identity].values()
                    )
                )
            matrix = np.vstack(relative_curves)
            mean, standard_deviation, ci95 = _mean_ci(matrix)
            for index, progress in enumerate(GRID):
                curve_rows.append(
                    {
                        "jobs": jobs,
                        "algorithm": algorithm,
                        "progress_percent": float(progress * 100.0),
                        "mean_relative_gap_percent": float(mean[index]),
                        "sample_std_percent": float(standard_deviation[index]),
                        "ci95_percent": float(ci95[index]),
                        "instance_count": matrix.shape[0],
                    }
                )
            diagnostics.append(
                {
                    "jobs": jobs,
                    "algorithm": algorithm,
                    "instance_count": matrix.shape[0],
                    "initial_mean_gap_percent": float(mean[0]),
                    "final_mean_gap_percent": float(mean[-1]),
                    "normalized_gap_auc": _auc(mean),
                    "final_win_or_tie_count": wins,
                    "final_mean_rank": float(np.mean(ranks)),
                }
            )

    for jobs in EXPECTED_JOB_COUNTS:
        members = [row for row in diagnostics if int(row["jobs"]) == jobs]
        best_auc = min(float(row["normalized_gap_auc"]) for row in members)
        best_final = min(float(row["final_mean_gap_percent"]) for row in members)
        for row in members:
            row["best_auc_or_tied"] = (
                float(row["normalized_gap_auc"]) <= best_auc + 1e-9
            )
            row["best_final_gap_or_tied"] = (
                float(row["final_mean_gap_percent"]) <= best_final + 1e-9
            )

    ibs_rows = [row for row in diagnostics if row["algorithm"] == "IBS_CABC"]
    ibs_best_auc_all = all(bool(row["best_auc_or_tied"]) for row in ibs_rows)
    ibs_best_final_all = all(bool(row["best_final_gap_or_tied"]) for row in ibs_rows)
    if ibs_best_auc_all and ibs_best_final_all:
        status = "IBS_CABC_BEST_OR_TIED_ON_ALL_SCALES"
    elif any(bool(row["best_auc_or_tied"]) for row in ibs_rows):
        status = "MIXED"
    else:
        status = "IBS_CABC_NOT_LEADING"
    audit = {
        "status": status,
        "comparison_is_descriptive_not_forced": True,
        "x_axis": "normalized evaluation progress within each equal-time run (0-100%)",
        "y_axis": "mean relative incumbent gap to paired-instance best final objective (%)",
        "lower_is_better": True,
        "confidence_interval": "normal-approximation 95% CI across 20 paired instances",
        "primary_convergence_metric": "normalized_gap_auc",
        "final_quality_metric": "final_mean_gap_percent",
        "IBS_CABC_best_auc_all_scales": ibs_best_auc_all,
        "IBS_CABC_best_final_gap_all_scales": ibs_best_final_all,
        "scale_count": len(EXPECTED_JOB_COUNTS),
        "algorithms": list(EXPECTED_ALGORITHMS),
    }
    return curve_rows, diagnostics, audit


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _series(
    curve_rows: list[dict[str, object]], jobs: int, algorithm: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [
        row
        for row in curve_rows
        if int(row["jobs"]) == jobs and row["algorithm"] == algorithm
    ]
    return (
        np.asarray([row["progress_percent"] for row in rows], dtype=float),
        np.asarray([row["mean_relative_gap_percent"] for row in rows], dtype=float),
        np.asarray([row["ci95_percent"] for row in rows], dtype=float),
    )


def _draw_axis(
    axis: plt.Axes,
    curve_rows: list[dict[str, object]],
    jobs: int,
) -> None:
    for algorithm in EXPECTED_ALGORITHMS:
        x, mean, ci95 = _series(curve_rows, jobs, algorithm)
        color = COLORS[algorithm]
        axis.plot(
            x,
            mean,
            label=algorithm,
            color=color,
            linestyle=LINE_STYLES[algorithm],
            linewidth=2.6 if algorithm == "IBS_CABC" else 2.0,
        )
        axis.fill_between(
            x,
            np.maximum(0.0, mean - ci95),
            mean + ci95,
            color=color,
            alpha=0.10,
            linewidth=0,
        )
    axis.set_title(f"J{jobs} (20 paired instances)", loc="left", fontweight="semibold")
    axis.set_xlim(0.0, 100.0)
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#75808C")
    axis.tick_params(colors="#3F4852")


def _save_figures(figures: Path, curve_rows: list[dict[str, object]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#20262D",
            "axes.titlecolor": "#20262D",
            "text.color": "#20262D",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=False)
    for axis, jobs in zip(axes.flat, EXPECTED_JOB_COUNTS):
        _draw_axis(axis, curve_rows, jobs)
        axis.set_xlabel("Normalized evaluation progress (%)")
        axis.set_ylabel("Mean relative incumbent gap (%)")
    legend_axis = axes.flat[-1]
    legend_axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_axis.legend(handles, labels, loc="center", frameon=False, fontsize=11)
    legend_axis.text(
        0.5,
        0.22,
        "Lower is better\nShading: 95% CI across paired instances",
        ha="center",
        va="center",
        color="#56616D",
        transform=legend_axis.transAxes,
    )
    figure.suptitle(
        "Four-algorithm convergence under equal time budgets",
        fontsize=17,
        fontweight="semibold",
        x=0.06,
        ha="left",
    )
    figure.text(
        0.06,
        0.935,
        "Gap is measured against the best final objective on each paired instance; "
        "curves use normalized evaluation progress.",
        fontsize=10.5,
        color="#56616D",
    )
    figure.tight_layout(rect=(0.04, 0.03, 0.99, 0.91))
    figure.savefig(figures / "convergence_four_algorithms.png", dpi=300, bbox_inches="tight")
    figure.savefig(figures / "convergence_four_algorithms.pdf", bbox_inches="tight")
    plt.close(figure)

    for jobs in EXPECTED_JOB_COUNTS:
        figure, axis = plt.subplots(figsize=(7.6, 4.8))
        _draw_axis(axis, curve_rows, jobs)
        axis.set_xlabel("Normalized evaluation progress (%)")
        axis.set_ylabel("Mean relative incumbent gap (%)")
        axis.legend(frameon=False, ncol=2, loc="upper right")
        figure.tight_layout()
        figure.savefig(figures / f"convergence_J{jobs}.png", dpi=300, bbox_inches="tight")
        plt.close(figure)


def write_convergence_reports(
    output_dir: Path,
    baseline_results: list[dict[str, object]],
    ibscabc_result_paths: Path | Iterable[Path],
) -> dict[str, object]:
    """Write convergence data, diagnostics, and publication-ready figures."""
    curve_rows, diagnostics, audit = build_convergence_data(
        baseline_results, ibscabc_result_paths
    )
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    _write_csv(tables / "convergence_curves.csv", curve_rows)
    _write_csv(tables / "convergence_diagnostics.csv", diagnostics)
    (tables / "convergence_diagnostics.json").write_text(
        json.dumps(
            {"audit": audit, "algorithm_scale_diagnostics": diagnostics},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _save_figures(figures, curve_rows)
    return audit
