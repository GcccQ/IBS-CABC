from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


GRID = np.linspace(0.0, 1.0, 101)
JOB_COUNTS = (25, 100, 250, 500, 1000)
ALGORITHMS = ("IBS-CABC", "Standard ABC", "ALNS", "TS", "Kraul-GA")
COLORS = {
    "IBS-CABC": "#C83737",
    "Standard ABC": "#D98E04",
    "ALNS": "#2F78B7",
    "TS": "#4F7F3A",
    "Kraul-GA": "#7A5AA6",
}
LINE_STYLES = {
    "IBS-CABC": "-",
    "Standard ABC": (0, (5, 2)),
    "ALNS": "-.",
    "TS": (0, (2, 1.5)),
    "Kraul-GA": (0, (1, 1.2)),
}
MARKERS = {
    "IBS-CABC": "s",
    "Standard ABC": "v",
    "ALNS": "^",
    "TS": "D",
    "Kraul-GA": "o",
}


class ConvergenceMergeError(ValueError):
    """Raised when the formal runs cannot support a paired convergence figure."""


@dataclass(frozen=True)
class RunHistory:
    algorithm: str
    instance: str
    instance_sha256: str
    jobs: int
    seed: int
    machines_stage1: int
    machines_stage2: int
    nominal_budget_seconds: float
    evaluations: int
    history: tuple[float, ...]
    history_evaluations: tuple[float, ...]
    final_incumbent: float
    reported_objective: float
    source_path: str


def _results_path(value: str | Path) -> Path:
    path = Path(value).resolve()
    if path.is_dir():
        path = path / "raw" / "results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"results file not found: {path}")
    return path


def _read_jsonl(value: str | Path) -> tuple[Path, list[dict[str, object]]]:
    path = _results_path(value)
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ConvergenceMergeError(f"non-object JSON at {path}:{line_number}")
            if row.get("status") not in (None, "success"):
                raise ConvergenceMergeError(
                    f"non-success record at {path}:{line_number}: {row.get('status')}"
                )
            rows.append(row)
    if not rows:
        raise ConvergenceMergeError(f"no result records in {path}")
    return path, rows


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConvergenceMergeError(f"missing or invalid {name}")
    return value


def _canonical_algorithm(value: object) -> str:
    compact = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    labels = {
        "ibscabc": "IBS-CABC",
        "ibs_cabc": "IBS-CABC",
        "standard_abc": "Standard ABC",
        "alns": "ALNS",
        "ts": "TS",
        "tabu_search": "TS",
        "kraul_ga": "Kraul-GA",
    }
    if compact not in labels:
        raise ConvergenceMergeError(f"unsupported convergence algorithm: {value}")
    return labels[compact]


def _as_float_tuple(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConvergenceMergeError(f"missing or invalid {name}")
    return tuple(float(item) for item in value)


def _nominal_budget(row: Mapping[str, object], source_kind: str) -> float:
    if source_kind != "ibscabc":
        return float(row["time_limit_seconds"])
    diagnostics = _mapping(row.get("diagnostics"), name="IBS-CABC diagnostics")
    config = _mapping(diagnostics.get("config"), name="IBS-CABC diagnostics.config")
    return float(config["time_limit_seconds"])


def _normalise_record(
    row: Mapping[str, object],
    *,
    source_kind: str,
    source_path: Path,
) -> RunHistory:
    if source_kind == "kraul_ga":
        diagnostics = _mapping(row.get("algorithm_diagnostics"), name="Kraul-GA diagnostics")
        history = _as_float_tuple(
            diagnostics.get("incumbent_history"), name="Kraul-GA incumbent_history"
        )
        history_evaluations = _as_float_tuple(
            diagnostics.get("history_evaluations"), name="Kraul-GA history_evaluations"
        )
        reported_objective = float(row["objective"])
        final_incumbent = float(diagnostics["best_seen_during_search"])
        algorithm = "Kraul-GA"
    else:
        history = _as_float_tuple(row.get("history"), name=f"{source_kind} history")
        history_evaluations = _as_float_tuple(
            row.get("history_evaluations"), name=f"{source_kind} history_evaluations"
        )
        if source_kind == "ibscabc":
            reported_objective = float(row.get("objective", row["total_completion_time"]))
            algorithm = "IBS-CABC"
        else:
            reported_objective = float(row["total_completion_time"])
            algorithm = _canonical_algorithm(row.get("algorithm_label", row["algorithm"]))
        final_incumbent = reported_objective

    run = RunHistory(
        algorithm=algorithm,
        instance=str(row["instance"]),
        instance_sha256=str(row["instance_sha256"]).strip().lower(),
        jobs=int(row["jobs"]),
        seed=int(row["seed"]),
        machines_stage1=int(row["machines_stage1"]),
        machines_stage2=int(row["machines_stage2"]),
        nominal_budget_seconds=_nominal_budget(row, source_kind),
        evaluations=int(row["evaluations"]),
        history=history,
        history_evaluations=history_evaluations,
        final_incumbent=final_incumbent,
        reported_objective=reported_objective,
        source_path=str(source_path),
    )
    _validate_run(run)
    return run


def _validate_run(run: RunHistory) -> None:
    if not run.instance_sha256:
        raise ConvergenceMergeError(f"missing instance hash for {run.algorithm}/{run.instance}")
    if run.jobs not in JOB_COUNTS:
        raise ConvergenceMergeError(f"unexpected job count J{run.jobs}: {run.instance}")
    if len(run.history) != len(run.history_evaluations) or not run.history:
        raise ConvergenceMergeError(
            f"history length mismatch for {run.algorithm}/J{run.jobs}/{run.instance}"
        )
    history = np.asarray(run.history, dtype=float)
    evaluations = np.asarray(run.history_evaluations, dtype=float)
    if not np.all(np.isfinite(history)) or not np.all(np.isfinite(evaluations)):
        raise ConvergenceMergeError(f"non-finite history for {run.algorithm}/{run.instance}")
    if np.any(np.diff(history) > 1e-9):
        raise ConvergenceMergeError(
            f"incumbent history worsens for {run.algorithm}/{run.instance}"
        )
    if np.any(np.diff(evaluations) < 0.0) or evaluations[0] < 0.0:
        raise ConvergenceMergeError(
            f"history evaluations are unordered for {run.algorithm}/{run.instance}"
        )
    if run.evaluations <= 0 or evaluations[-1] > run.evaluations + 1e-9:
        raise ConvergenceMergeError(
            f"invalid total evaluations for {run.algorithm}/{run.instance}"
        )
    if run.final_incumbent > history[-1] + 1e-9:
        raise ConvergenceMergeError(
            f"final incumbent is worse than history for {run.algorithm}/{run.instance}"
        )
    if run.nominal_budget_seconds <= 0.0:
        raise ConvergenceMergeError(f"invalid budget for {run.algorithm}/{run.instance}")


def load_runs(
    *,
    baseline_experiment: str | Path,
    ibscabc_experiments: Sequence[str | Path],
    kraul_experiment: str | Path,
) -> list[RunHistory]:
    baseline_path, baseline_rows = _read_jsonl(baseline_experiment)
    kraul_path, kraul_rows = _read_jsonl(kraul_experiment)
    runs = [
        _normalise_record(row, source_kind="baseline", source_path=baseline_path)
        for row in baseline_rows
    ]
    for experiment in ibscabc_experiments:
        ibs_path, ibs_rows = _read_jsonl(experiment)
        runs.extend(
            _normalise_record(row, source_kind="ibscabc", source_path=ibs_path)
            for row in ibs_rows
        )
    runs.extend(
        _normalise_record(row, source_kind="kraul_ga", source_path=kraul_path)
        for row in kraul_rows
    )
    return runs


def validate_pairing(runs: Sequence[RunHistory]) -> dict[str, object]:
    duplicates: set[tuple[str, str, int]] = set()
    seen: set[tuple[str, str, int]] = set()
    grouped: dict[str, dict[int, list[RunHistory]]] = defaultdict(lambda: defaultdict(list))
    by_instance: dict[str, list[RunHistory]] = defaultdict(list)
    for run in runs:
        key = (run.algorithm, run.instance_sha256, run.seed)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
        grouped[run.algorithm][run.jobs].append(run)
        by_instance[run.instance_sha256].append(run)
    if duplicates:
        raise ConvergenceMergeError(f"duplicate algorithm-instance-seed records: {len(duplicates)}")

    expected_hashes: dict[int, set[str]] = {}
    for jobs in JOB_COUNTS:
        reference = {run.instance_sha256 for run in grouped[ALGORITHMS[0]][jobs]}
        if len(reference) != 20:
            raise ConvergenceMergeError(
                f"{ALGORITHMS[0]} has {len(reference)} unique instances for J{jobs}, expected 20"
            )
        expected_hashes[jobs] = reference
        for algorithm in ALGORITHMS:
            members = grouped[algorithm][jobs]
            hashes = {run.instance_sha256 for run in members}
            if len(members) != 20 or hashes != reference:
                raise ConvergenceMergeError(
                    f"paired instance mismatch for {algorithm}/J{jobs}: "
                    f"runs={len(members)}, unique={len(hashes)}"
                )

    seed_sets = {algorithm: sorted({run.seed for run in runs if run.algorithm == algorithm}) for algorithm in ALGORITHMS}
    if len({tuple(values) for values in seed_sets.values()}) != 1:
        raise ConvergenceMergeError(f"seed-set mismatch: {seed_sets}")

    for digest, members in by_instance.items():
        if {run.algorithm for run in members} != set(ALGORITHMS):
            raise ConvergenceMergeError(f"algorithm coverage mismatch for instance {digest}")
        machines = {(run.machines_stage1, run.machines_stage2) for run in members}
        if len(machines) != 1:
            raise ConvergenceMergeError(f"machine-setting mismatch for instance {digest}")
        budgets = {round(run.nominal_budget_seconds, 9) for run in members}
        if len(budgets) != 1:
            raise ConvergenceMergeError(f"nominal-budget mismatch for instance {digest}")

    budget_by_jobs = {
        str(jobs): sorted(
            {run.nominal_budget_seconds for run in runs if run.jobs == jobs}
        )[0]
        for jobs in JOB_COUNTS
    }
    terminal_population_worse = sum(
        run.algorithm == "Kraul-GA"
        and run.reported_objective > run.final_incumbent + 1e-9
        for run in runs
    )
    return {
        "status": "PASS",
        "algorithms": list(ALGORITHMS),
        "run_count": len(runs),
        "unique_instances": sum(len(values) for values in expected_hashes.values()),
        "instances_per_scale": {str(jobs): len(expected_hashes[jobs]) for jobs in JOB_COUNTS},
        "seed_sets": seed_sets,
        "same_instance_sha256_sets": True,
        "same_machine_settings_by_instance": True,
        "same_nominal_budget_by_instance": True,
        "nominal_budget_seconds_by_jobs": budget_by_jobs,
        "x_axis": "normalised evaluation progress within each run (0-100%)",
        "x_axis_warning": (
            "Evaluation progress is algorithm-relative and is not elapsed wall-clock time; "
            "per-evaluation costs may differ."
        ),
        "y_axis": (
            "mean best-so-far relative gap to the best end-of-run incumbent "
            "on each paired instance (%)"
        ),
        "lower_is_better": True,
        "confidence_interval": "normal-approximation 95% CI across paired instances",
        "kraul_terminal_population_worse_than_best_seen_runs": terminal_population_worse,
        "ffcbspt_omitted_reason": "deterministic constructive rule has no iterative history",
        "source_paths": sorted({run.source_path for run in runs}),
    }


def _incumbent_curve(run: RunHistory, grid: np.ndarray = GRID) -> np.ndarray:
    history = np.asarray(run.history, dtype=float)
    evaluations = np.asarray(run.history_evaluations, dtype=float)
    progress = np.clip(evaluations / float(run.evaluations), 0.0, 1.0)
    x = np.concatenate(([0.0], progress, [1.0]))
    y = np.concatenate(([history[0]], history, [run.final_incumbent]))
    indices = np.searchsorted(x, grid, side="right") - 1
    curve = y[np.clip(indices, 0, y.size - 1)]
    curve[-1] = run.final_incumbent
    if np.any(np.diff(curve) > 1e-9):
        raise ConvergenceMergeError(f"resampled curve worsens for {run.algorithm}/{run.instance}")
    return curve


def _mean_ci(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    if matrix.shape[0] <= 1:
        zeros = np.zeros_like(mean)
        return mean, zeros, zeros
    sample_std = matrix.std(axis=0, ddof=1)
    ci95 = 1.96 * sample_std / math.sqrt(matrix.shape[0])
    return mean, sample_std, ci95


def _auc(values: np.ndarray) -> float:
    return float(np.trapezoid(values, GRID))


def build_convergence_data(
    runs: Sequence[RunHistory],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    lookup: dict[int, dict[str, dict[str, RunHistory]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for run in runs:
        lookup[run.jobs][run.instance_sha256][run.algorithm] = run
    references = {
        digest: min(member.final_incumbent for member in algorithms.values())
        for instances in lookup.values()
        for digest, algorithms in instances.items()
    }

    scopes: list[tuple[str, list[tuple[int, str]]]] = []
    for jobs in JOB_COUNTS:
        scopes.append((f"J{jobs}", [(jobs, digest) for digest in sorted(lookup[jobs])]))
    scopes.append(
        (
            "Overall",
            [
                (jobs, digest)
                for jobs in JOB_COUNTS
                for digest in sorted(lookup[jobs])
            ],
        )
    )

    curve_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for scope, identities in scopes:
        for algorithm in ALGORITHMS:
            relative_curves: list[np.ndarray] = []
            endpoints: list[float] = []
            ranks: list[int] = []
            wins = 0
            for jobs, digest in identities:
                run = lookup[jobs][digest][algorithm]
                reference = references[digest]
                curve = 100.0 * (_incumbent_curve(run) - reference) / reference
                relative_curves.append(np.maximum(0.0, curve))
                endpoints.append(run.final_incumbent)
                paired = [lookup[jobs][digest][name].final_incumbent for name in ALGORITHMS]
                best = min(paired)
                wins += int(abs(run.final_incumbent - best) <= 1e-9)
                ranks.append(1 + sum(value < run.final_incumbent - 1e-9 for value in paired))
            matrix = np.vstack(relative_curves)
            mean, sample_std, ci95 = _mean_ci(matrix)
            for index, progress in enumerate(GRID):
                curve_rows.append(
                    {
                        "scope": scope,
                        "jobs": int(scope[1:]) if scope.startswith("J") else "",
                        "algorithm": algorithm,
                        "progress_percent": float(progress * 100.0),
                        "mean_relative_gap_percent": float(mean[index]),
                        "sample_std_percent": float(sample_std[index]),
                        "ci95_percent": float(ci95[index]),
                        "paired_instance_count": int(matrix.shape[0]),
                    }
                )
            diagnostics.append(
                {
                    "scope": scope,
                    "jobs": int(scope[1:]) if scope.startswith("J") else "",
                    "algorithm": algorithm,
                    "paired_instance_count": int(matrix.shape[0]),
                    "initial_mean_gap_percent": float(mean[0]),
                    "final_mean_gap_percent": float(mean[-1]),
                    "normalised_gap_auc": _auc(mean),
                    "final_win_or_tie_count": wins,
                    "final_mean_rank": float(np.mean(ranks)),
                    "mean_final_incumbent": float(np.mean(endpoints)),
                }
            )

    for scope, _ in scopes:
        members = [row for row in diagnostics if row["scope"] == scope]
        best_auc = min(float(row["normalised_gap_auc"]) for row in members)
        best_final = min(float(row["final_mean_gap_percent"]) for row in members)
        for row in members:
            row["best_auc_or_tied"] = float(row["normalised_gap_auc"]) <= best_auc + 1e-9
            row["best_final_gap_or_tied"] = (
                float(row["final_mean_gap_percent"]) <= best_final + 1e-9
            )
    return curve_rows, diagnostics


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _series(
    rows: Sequence[Mapping[str, object]], scope: str, algorithm: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    members = [row for row in rows if row["scope"] == scope and row["algorithm"] == algorithm]
    return (
        np.asarray([row["progress_percent"] for row in members], dtype=float),
        np.asarray([row["mean_relative_gap_percent"] for row in members], dtype=float),
        np.asarray([row["ci95_percent"] for row in members], dtype=float),
    )


def _draw_axis(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, object]],
    scope: str,
) -> None:
    maximum = 0.0
    for algorithm in ALGORITHMS:
        x, mean, ci95 = _series(rows, scope, algorithm)
        maximum = max(maximum, float(np.max(mean + ci95)))
        axis.plot(
            x,
            mean,
            label=algorithm,
            color=COLORS[algorithm],
            linestyle=LINE_STYLES[algorithm],
            linewidth=2.7 if algorithm == "IBS-CABC" else 1.9,
            marker=MARKERS[algorithm],
            markevery=20,
            markersize=4.0 if algorithm == "IBS-CABC" else 3.5,
            markerfacecolor="white",
            markeredgewidth=0.9,
        )
        axis.fill_between(
            x,
            np.maximum(0.0, mean - ci95),
            mean + ci95,
            color=COLORS[algorithm],
            alpha=0.055,
            linewidth=0.0,
        )
    count = 100 if scope == "Overall" else 20
    title = f"{scope} (n={count})"
    axis.set_title(title, loc="left", fontsize=11.5, fontweight="semibold")
    axis.set_xlim(0.0, 100.0)
    axis.set_xticks(np.arange(0.0, 101.0, 20.0))
    axis.set_ylim(0.0, max(0.5, maximum * 1.08))
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#75808C")
    axis.tick_params(colors="#3F4852", labelsize=9)


def _plot_comparison(rows: Sequence[Mapping[str, object]], path: Path) -> None:
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
    scopes = [f"J{jobs}" for jobs in JOB_COUNTS] + ["Overall"]
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.0))
    for axis, scope in zip(axes.flat, scopes):
        _draw_axis(axis, rows, scope)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=5,
        frameon=False,
        columnspacing=1.8,
        handlelength=3.2,
    )
    figure.suptitle(
        "Five-algorithm convergence on fixed-scale instances",
        fontsize=17,
        fontweight="semibold",
        x=0.055,
        y=0.985,
        ha="left",
    )
    figure.text(
        0.055,
        0.947,
        "Mean best-so-far gap to the best end-of-run incumbent on each paired instance; "
        "x-axis uses within-run normalised evaluation progress.",
        fontsize=10.4,
        color="#56616D",
        ha="left",
    )
    figure.supxlabel("Normalised evaluation progress (%)", y=0.045, fontsize=11)
    figure.supylabel("Mean relative incumbent gap (%)", x=0.018, fontsize=11)
    figure.text(
        0.5,
        0.012,
        "Lower is better · Shading: 95% CI across paired instances · "
        "Same 100 instances and seed 101 · "
        "Overall weights all instances equally · FFCBSPT omitted because it has no iterative history",
        ha="center",
        va="bottom",
        fontsize=9.1,
        color="#56616D",
    )
    figure.tight_layout(rect=(0.035, 0.075, 0.995, 0.885), h_pad=2.0, w_pad=1.4)
    figure.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(figure)


def _summary_markdown(audit: Mapping[str, object]) -> str:
    budgets = audit["nominal_budget_seconds_by_jobs"]
    return "\n".join(
        [
            "# 五种迭代算法固定规模收敛图说明",
            "",
            "- 图中包含 IBS-CABC、Standard ABC、ALNS、TS 和 Kraul-GA；FFCBSPT 为一次性确定性构造规则，不具备逐代收敛历史，因此未画入。",
            "- 横轴为每次运行内部的归一化评价进度（0%—100%），不是墙钟时间；不同算法单次评价成本可能不同，因此该图用于比较搜索过程形态，不应解释为绝对求解速度。",
            "- 纵轴为相对于同一实例五种算法运行结束时最佳历史最优值的平均相对差距，数值越低越好。",
            "- 每个规模包含20个完全配对实例，Overall面板包含100个实例；阴影为配对实例间均值的近似95%置信区间。",
            f"- 名义时间预算（秒）：{json.dumps(budgets, ensure_ascii=False)}。",
            "",
            "建议图题：固定规模实例上五种迭代算法的收敛过程。",
            "",
            "建议图注：横轴为各算法运行内部的归一化评价进度，纵轴为相对于同一实例五种算法运行结束时最佳历史最优值的平均相对差距；数值越低越好，阴影表示95%置信区间。",
            "",
        ]
    )


def build_five_algorithm_convergence(
    *,
    baseline_experiment: str | Path,
    ibscabc_experiments: Sequence[str | Path],
    kraul_experiment: str | Path,
    output_dir: str | Path,
) -> Path:
    runs = load_runs(
        baseline_experiment=baseline_experiment,
        ibscabc_experiments=ibscabc_experiments,
        kraul_experiment=kraul_experiment,
    )
    audit = validate_pairing(runs)
    curves, diagnostics = build_convergence_data(runs)
    output = Path(output_dir).resolve()
    tables = output / "tables"
    figures = output / "figures"
    report = output / "report"
    for directory in (tables, figures, report):
        directory.mkdir(parents=True, exist_ok=True)
    _write_csv(tables / "fixed_scale_convergence_five_algorithms.csv", curves)
    _write_csv(tables / "fixed_scale_convergence_diagnostics.csv", diagnostics)
    (report / "fixed_scale_convergence_integrity.json").write_text(
        json.dumps(
            {"audit": audit, "algorithm_scope_diagnostics": diagnostics},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (report / "fixed_scale_convergence_note_zh.md").write_text(
        _summary_markdown(audit), encoding="utf-8"
    )
    _plot_comparison(curves, figures / "fixed_scale_convergence_five_algorithms.png")
    _plot_comparison(curves, figures / "fixed_scale_convergence_five_algorithms.pdf")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Kraul-GA with the formal fixed-scale convergence histories"
    )
    parser.add_argument("--baseline-experiment", required=True)
    parser.add_argument("--ibscabc-experiment", action="append", required=True)
    parser.add_argument("--kraul-experiment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = build_five_algorithm_convergence(
        baseline_experiment=args.baseline_experiment,
        ibscabc_experiments=args.ibscabc_experiment,
        kraul_experiment=args.kraul_experiment,
        output_dir=args.output,
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
