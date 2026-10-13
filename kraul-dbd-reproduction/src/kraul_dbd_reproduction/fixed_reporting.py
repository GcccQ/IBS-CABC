from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


INSTANCE_NUMERIC_FIELDS = (
    "objective",
    "release_sum",
    "makespan",
    "average_time_in_system",
    "average_waiting_stage1",
    "average_waiting_stage2",
    "utilization_stage1",
    "utilization_stage2",
    "runtime_seconds",
    "process_cpu_seconds",
    "evaluations",
    "generations",
    "restart_count",
)


# Kraul (2025), Table 5.  The printed J250 TIS entry is retained for audit,
# but validation uses the value implied by sum(TIS) = sum(Cj2) - sum(rj).
PAPER_TABLE5_KRAUL_GA: Mapping[int, Mapping[str, float]] = {
    100: {
        "objective_mean": 129_202.0,
        "objective_sample_std": 5_754.0,
        "time_in_system_printed_mean": 481.0,
        "time_in_system_printed_sample_std": 56.0,
        "makespan_mean": 1_615.0,
        "makespan_sample_std": 74.0,
        "utilization_stage1_percent": 19.45,
        "utilization_stage2_percent": 29.14,
    },
    250: {
        "objective_mean": 297_876.0,
        "objective_sample_std": 14_445.0,
        "time_in_system_printed_mean": 481.0,
        "time_in_system_printed_sample_std": 56.0,
        "makespan_mean": 1_664.0,
        "makespan_sample_std": 118.0,
        "utilization_stage1_percent": 36.35,
        "utilization_stage2_percent": 51.95,
    },
    500: {
        "objective_mean": 638_982.0,
        "objective_sample_std": 59_192.0,
        "time_in_system_printed_mean": 468.0,
        "time_in_system_printed_sample_std": 119.0,
        "makespan_mean": 1_929.0,
        "makespan_sample_std": 247.0,
        "utilization_stage1_percent": 65.84,
        "utilization_stage2_percent": 82.73,
    },
    1000: {
        "objective_mean": 1_724_170.0,
        "objective_sample_std": 249_863.0,
        "time_in_system_printed_mean": 918.0,
        "time_in_system_printed_sample_std": 251.0,
        "makespan_mean": 2_993.0,
        "makespan_sample_std": 487.0,
        "utilization_stage1_percent": 83.41,
        "utilization_stage2_percent": 89.70,
    },
}

PAPER_TABLE5_TOLERANCES: Mapping[str, float] = {
    "objective_relative_percent": 1.0,
    "time_in_system_minutes": 5.0,
    "makespan_minutes": 20.0,
    "utilization_percentage_points": 2.0,
}


def _with_derived_metrics(record: Mapping[str, object]) -> dict[str, object]:
    """Add fixed-scale metrics that older raw records did not store explicitly."""

    output = dict(record)
    if output.get("makespan") is None:
        schedule = output.get("schedule")
        if not isinstance(schedule, Mapping):
            raise ValueError(f"missing decoded schedule for {output.get('task_id', '<unknown>')}")
        completion_stage2 = schedule.get("completion_stage2")
        if not isinstance(completion_stage2, list) or not completion_stage2:
            raise ValueError(
                f"missing stage-2 completion vector for {output.get('task_id', '<unknown>')}"
            )
        if output.get("jobs") is not None and len(completion_stage2) != int(output["jobs"]):
            raise ValueError(
                f"stage-2 completion vector has the wrong length for "
                f"{output.get('task_id', '<unknown>')}"
            )
        makespan = max(float(value) for value in completion_stage2)
        if not np.isfinite(makespan):
            raise ValueError(f"non-finite makespan for {output.get('task_id', '<unknown>')}")
        output["makespan"] = makespan
    return output


def _read_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"result line {line_number} is not an object")
            if value.get("status") == "success":
                records.append(_with_derived_metrics(value))
    return records


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sample_std(values: Sequence[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _run_rows(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    fields = (
        "task_id",
        "instance_index",
        "instance",
        "relative_instance_path",
        "instance_sha256",
        "dataset_group",
        "jobs",
        "families",
        "machines_stage1",
        "machines_stage2",
        "capacity_stage1",
        "capacity_stage2",
        "algorithm",
        "algorithm_label",
        "profile",
        "seed",
        "protocol_name",
        "time_limit_seconds",
        "objective",
        "release_sum",
        "makespan",
        "average_time_in_system",
        "average_waiting_stage1",
        "average_waiting_stage2",
        "utilization_stage1",
        "utilization_stage2",
        "runtime_seconds",
        "process_cpu_seconds",
        "evaluations",
        "generations",
        "restart_count",
        "feasible",
    )
    rows: list[dict[str, object]] = []
    for record in sorted(
        records,
        key=lambda item: (int(item["jobs"]), str(item["instance"]), int(item["seed"])),
    ):
        row = {field: record.get(field) for field in fields}
        diagnostics = record.get("algorithm_diagnostics")
        if isinstance(diagnostics, Mapping):
            row.update(
                {
                    "last_generation_best": diagnostics.get("last_generation_best"),
                    "final_population_best": diagnostics.get("final_population_best"),
                    "best_seen_during_search": diagnostics.get("best_seen_during_search"),
                    "termination_reason": diagnostics.get("termination_reason"),
                    "time_limit_overrun_seconds": diagnostics.get("time_limit_overrun_seconds"),
                }
            )
        rows.append(row)
    return rows


def _instance_rows(
    records: Sequence[Mapping[str, object]],
    experiment_name: str,
) -> list[dict[str, object]]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        groups[str(record["instance_sha256"])].append(record)
    rows: list[dict[str, object]] = []
    for members in groups.values():
        reference = members[0]
        row: dict[str, object] = {
            "experiment": experiment_name,
            "algorithm": "kraul_ga",
            "schedule_algorithm": "Kraul-GA",
            "protocol": reference["protocol_name"],
            "instance": reference["instance"],
            "instance_path": reference["relative_instance_path"],
            "instance_sha256": reference["instance_sha256"],
            "dataset_group": reference["dataset_group"],
            "planning_mode": "SINGLE",
            "scenario": "base",
            "day_length_minutes": reference["day_length_minutes"],
            "jobs": reference["jobs"],
            "families": reference["families"],
            "machines_stage1": reference["machines_stage1"],
            "machines_stage2": reference["machines_stage2"],
            "feasible": all(bool(item.get("feasible")) for item in members),
            "seed_runs": len(members),
            "seeds": ",".join(str(item["seed"]) for item in sorted(members, key=lambda x: int(x["seed"]))),
        }
        for field in INSTANCE_NUMERIC_FIELDS:
            values = [float(item[field]) for item in members]
            row[field] = fmean(values)
            row[f"{field}_seed_std"] = _sample_std(values)
        rows.append(row)
    return sorted(rows, key=lambda item: (int(item["jobs"]), str(item["instance"])))


def _scale_rows(instance_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in instance_rows:
        groups[int(row["jobs"])].append(row)
    rows: list[dict[str, object]] = []
    for jobs, members in sorted(groups.items()):
        row: dict[str, object] = {"scale": f"J{jobs}", "jobs": jobs, "instances": len(members)}
        for field in INSTANCE_NUMERIC_FIELDS:
            values = [float(item[field]) for item in members]
            row[f"{field}_mean"] = fmean(values)
            row[f"{field}_sample_std"] = _sample_std(values)
        rows.append(row)
    return rows


def _paper_table5_validation(
    scale_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    by_jobs = {int(row["jobs"]): row for row in scale_rows}
    required = set(PAPER_TABLE5_KRAUL_GA)
    missing = sorted(required - set(by_jobs))
    if missing:
        return (
            {
                "status": "NOT_APPLICABLE",
                "reason": "the report does not contain all four Table 5 scales",
                "missing_scales": [f"J{jobs}" for jobs in missing],
                "tolerances": dict(PAPER_TABLE5_TOLERANCES),
            },
            [],
        )

    comparison_rows: list[dict[str, object]] = []
    failed_checks: list[str] = []
    for jobs in sorted(required):
        observed = by_jobs[jobs]
        paper = PAPER_TABLE5_KRAUL_GA[jobs]
        instances = int(observed["instances"])
        objective_observed = float(observed["objective_mean"])
        objective_paper = float(paper["objective_mean"])
        objective_difference_percent = 100.0 * (
            objective_observed - objective_paper
        ) / objective_paper
        release_mean = float(observed["release_sum_mean"])
        tis_implied = (objective_paper - release_mean) / jobs
        tis_observed = float(observed["average_time_in_system_mean"])
        tis_difference = tis_observed - tis_implied
        makespan_observed = float(observed["makespan_mean"])
        makespan_difference = makespan_observed - float(paper["makespan_mean"])
        utilization_stage1_observed = 100.0 * float(observed["utilization_stage1_mean"])
        utilization_stage2_observed = 100.0 * float(observed["utilization_stage2_mean"])
        utilization_stage1_difference = (
            utilization_stage1_observed - float(paper["utilization_stage1_percent"])
        )
        utilization_stage2_difference = (
            utilization_stage2_observed - float(paper["utilization_stage2_percent"])
        )
        checks = {
            "instance_count": instances == 20,
            "objective": abs(objective_difference_percent)
            <= PAPER_TABLE5_TOLERANCES["objective_relative_percent"],
            "time_in_system": abs(tis_difference)
            <= PAPER_TABLE5_TOLERANCES["time_in_system_minutes"],
            "makespan": abs(makespan_difference)
            <= PAPER_TABLE5_TOLERANCES["makespan_minutes"],
            "utilization_stage1": abs(utilization_stage1_difference)
            <= PAPER_TABLE5_TOLERANCES["utilization_percentage_points"],
            "utilization_stage2": abs(utilization_stage2_difference)
            <= PAPER_TABLE5_TOLERANCES["utilization_percentage_points"],
        }
        for name, passed in checks.items():
            if not passed:
                failed_checks.append(f"J{jobs}:{name}")
        comparison_rows.append(
            {
                "scale": f"J{jobs}",
                "instances": instances,
                "objective_observed": objective_observed,
                "objective_paper": objective_paper,
                "objective_difference_percent": objective_difference_percent,
                "objective_sample_std_observed": observed["objective_sample_std"],
                "objective_sample_std_paper": paper["objective_sample_std"],
                "time_in_system_observed": tis_observed,
                "time_in_system_paper_printed": paper["time_in_system_printed_mean"],
                "time_in_system_implied_by_paper_objective": tis_implied,
                "time_in_system_difference_from_implied": tis_difference,
                "time_in_system_sample_std_observed": observed[
                    "average_time_in_system_sample_std"
                ],
                "time_in_system_sample_std_paper_printed": paper[
                    "time_in_system_printed_sample_std"
                ],
                "makespan_observed": makespan_observed,
                "makespan_paper": paper["makespan_mean"],
                "makespan_difference_minutes": makespan_difference,
                "makespan_sample_std_observed": observed["makespan_sample_std"],
                "makespan_sample_std_paper": paper["makespan_sample_std"],
                "utilization_stage1_observed_percent": utilization_stage1_observed,
                "utilization_stage1_paper_percent": paper[
                    "utilization_stage1_percent"
                ],
                "utilization_stage1_difference_percentage_points": (
                    utilization_stage1_difference
                ),
                "utilization_stage2_observed_percent": utilization_stage2_observed,
                "utilization_stage2_paper_percent": paper[
                    "utilization_stage2_percent"
                ],
                "utilization_stage2_difference_percentage_points": (
                    utilization_stage2_difference
                ),
                "all_checks_pass": all(checks.values()),
            }
        )

    status = "PASS" if not failed_checks else "FAIL"
    return (
        {
            "status": status,
            "reference": "Kraul (2025), Table 5, Genetic rows",
            "tolerances": dict(PAPER_TABLE5_TOLERANCES),
            "failed_checks": failed_checks,
            "j250_note": (
                "The printed J250 TIS value 481 (56) is inconsistent with the published "
                "objective and the common release-time sum. The comparison therefore uses "
                "the TIS implied by sum(TIS)=sum(Cj2)-sum(rj)."
            ),
            "comparisons": comparison_rows,
        },
        comparison_rows,
    )


def _history(record: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    diagnostics = record.get("algorithm_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError(f"missing GA diagnostics for {record['task_id']}")
    generation_best = np.asarray(diagnostics.get("history", []), dtype=np.float64)
    incumbent = np.asarray(diagnostics.get("incumbent_history", []), dtype=np.float64)
    elapsed = np.asarray(diagnostics.get("history_elapsed_seconds", []), dtype=np.float64)
    evaluations = np.asarray(diagnostics.get("history_evaluations", []), dtype=np.int64)
    lengths = {len(generation_best), len(incumbent), len(elapsed), len(evaluations)}
    if len(lengths) != 1 or not len(generation_best):
        raise ValueError(f"incomplete convergence history for {record['task_id']}")
    if not (
        np.all(np.isfinite(generation_best))
        and np.all(np.isfinite(incumbent))
        and np.all(np.isfinite(elapsed))
    ):
        raise ValueError(f"non-finite convergence history for {record['task_id']}")
    if np.any(np.diff(elapsed) < 0) or np.any(np.diff(evaluations) < 0):
        raise ValueError(f"non-monotone convergence coordinates for {record['task_id']}")
    if np.any(np.diff(incumbent) > 1e-9):
        raise ValueError(f"incumbent objective worsened for {record['task_id']}")
    return generation_best, incumbent, elapsed, evaluations


def _convergence_rows(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in sorted(
        records,
        key=lambda item: (int(item["jobs"]), str(item["instance"]), int(item["seed"])),
    ):
        generation_best, incumbent, elapsed, evaluations = _history(record)
        initial = float(incumbent[0])
        limit = float(record["time_limit_seconds"])
        for index in range(len(generation_best)):
            actual_fraction = 100.0 * float(elapsed[index]) / limit
            rows.append(
                {
                    "task_id": record["task_id"],
                    "instance": record["instance"],
                    "dataset_group": record["dataset_group"],
                    "jobs": record["jobs"],
                    "seed": record["seed"],
                    "phase": "generation",
                    "generation": index + 1,
                    "evaluations": int(evaluations[index]),
                    "elapsed_seconds": float(elapsed[index]),
                    "nominal_time_limit_seconds": limit,
                    "actual_budget_fraction_percent": actual_fraction,
                    "plotted_budget_fraction_percent": min(100.0, actual_fraction),
                    "generation_best_objective": float(generation_best[index]),
                    "best_so_far_objective": float(incumbent[index]),
                    "relative_improvement_percent": 100.0
                    * (initial - float(incumbent[index]))
                    / initial,
                    "returned_objective": record["objective"],
                }
            )
        final_elapsed = float(record["runtime_seconds"])
        final_fraction = 100.0 * final_elapsed / limit
        diagnostics = record["algorithm_diagnostics"]
        final_population_best = float(diagnostics["final_population_best"])
        best_seen = float(diagnostics["best_seen_during_search"])
        rows.append(
            {
                "task_id": record["task_id"],
                "instance": record["instance"],
                "dataset_group": record["dataset_group"],
                "jobs": record["jobs"],
                "seed": record["seed"],
                "phase": "final_population",
                "generation": len(generation_best) + 1,
                "evaluations": record["evaluations"],
                "elapsed_seconds": final_elapsed,
                "nominal_time_limit_seconds": limit,
                "actual_budget_fraction_percent": final_fraction,
                "plotted_budget_fraction_percent": 100.0,
                "generation_best_objective": final_population_best,
                "best_so_far_objective": best_seen,
                "relative_improvement_percent": 100.0 * (initial - best_seen) / initial,
                "returned_objective": record["objective"],
            }
        )
    return rows


def _step_series(record: Mapping[str, object], grid: np.ndarray) -> np.ndarray:
    _, incumbent, elapsed, _ = _history(record)
    initial = float(incumbent[0])
    fractions = np.minimum(100.0, 100.0 * elapsed / float(record["time_limit_seconds"]))
    improvements = 100.0 * (initial - incumbent) / initial
    diagnostics = record["algorithm_diagnostics"]
    final_best_seen = float(diagnostics["best_seen_during_search"])
    final_improvement = 100.0 * (initial - final_best_seen) / initial
    x = np.concatenate(([0.0], fractions, [100.0]))
    y = np.concatenate(([0.0], improvements, [final_improvement]))
    indices = np.searchsorted(x, grid, side="right") - 1
    return y[np.maximum(indices, 0)]


def _aggregate_convergence(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        groups[int(record["jobs"])].append(record)
    grid = np.linspace(0.0, 100.0, 101)
    rows: list[dict[str, object]] = []
    for jobs, members in sorted(groups.items()):
        matrix = np.vstack([_step_series(record, grid) for record in members])
        for column, fraction in enumerate(grid):
            values = matrix[:, column]
            rows.append(
                {
                    "scale": f"J{jobs}",
                    "jobs": jobs,
                    "runs": len(members),
                    "budget_fraction_percent": float(fraction),
                    "mean_relative_improvement_percent": float(np.mean(values)),
                    "median_relative_improvement_percent": float(np.median(values)),
                    "sample_std_relative_improvement_percent": float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0,
                    "q25_relative_improvement_percent": float(np.quantile(values, 0.25)),
                    "q75_relative_improvement_percent": float(np.quantile(values, 0.75)),
                }
            )
    return rows


def _plot_convergence(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    groups: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[int(row["jobs"])].append(row)
    colors = {
        25: "#4C78A8",
        100: "#E3BA22",
        250: "#F28E2B",
        500: "#797D1D",
        1000: "#D45087",
    }
    line_styles = {25: "-", 100: "--", 250: "-.", 500: ":", 1000: (0, (5, 2))}
    markers = {25: "o", 100: "s", 250: "^", 500: "D", 1000: "P"}
    figure, axis = plt.subplots(figsize=(8.6, 5.0))
    for jobs, members in sorted(groups.items()):
        members = sorted(members, key=lambda item: float(item["budget_fraction_percent"]))
        axis.plot(
            [float(item["budget_fraction_percent"]) for item in members],
            [float(item["mean_relative_improvement_percent"]) for item in members],
            label=f"J{jobs}",
            color=colors.get(jobs),
            linestyle=line_styles.get(jobs, "-"),
            linewidth=1.8,
            marker=markers.get(jobs, "o"),
            markevery=10,
            markersize=3.5,
        )
    axis.set_xlim(0, 100)
    axis.set_xticks(np.arange(0, 101, 20))
    maximum = max(float(item["mean_relative_improvement_percent"]) for item in rows)
    axis.set_ylim(0, max(1.0, maximum * 1.10))
    axis.set_xlabel("Nominal time budget consumed (%)")
    axis.set_ylabel("Mean incumbent improvement from initial best (%)")
    axis.set_title("Kraul-GA convergence by instance scale", loc="left", pad=28)
    run_counts = sorted({int(item["runs"]) for item in rows})
    count_text = (
        f"{run_counts[0]} instance run{'s' if run_counts[0] != 1 else ''} per scale"
        if len(run_counts) == 1
        else "instance-run count shown in the source table"
    )
    axis.text(
        0.0,
        1.015,
        f"Mean best-so-far trajectory; {count_text}; final population included at 100%",
        transform=axis.transAxes,
        fontsize=9,
        color="#555555",
        ha="left",
        va="bottom",
    )
    axis.grid(color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, ncol=5)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _summary_markdown(
    scale_rows: Sequence[Mapping[str, object]],
    paper_validation: Mapping[str, object],
) -> str:
    lines = [
        "# Kraul-GA 固定规模实验汇总",
        "",
        "| 规模 | 实例数 | 目标均值 | 目标样本标准差 | Cmax均值 | 平均运行时间/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in scale_rows:
        lines.append(
            f"| {row['scale']} | {row['instances']} | "
            f"{float(row['objective_mean']):.2f} | "
            f"{float(row['objective_sample_std']):.2f} | "
            f"{float(row['makespan_mean']):.2f} | "
            f"{float(row['runtime_seconds_mean']):.2f} |"
        )
    lines.extend(
        [
            "",
            "收敛图横轴为名义时间预算消耗比例，纵轴为相对于各次运行首次完整种群评价后最优值的平均累计改善率。",
            "不同规模先在各实例内标准化，再按规模求均值，避免绝对目标量级差异主导曲线。",
            "原作者在时间循环后执行的最终种群评价显示在100%端点；原始文件保留其实际用时比例以及代数、评价次数、当代最优值和历史最优值。",
        ]
    )
    if paper_validation.get("status") != "NOT_APPLICABLE":
        lines.extend(
            [
                "",
                f"与原论文表5中 Kraul-GA（Genetic）行的自动核验结果：{paper_validation['status']}。",
                "J250 行印刷的 TIS=481 (56) 与该行目标值及释放时刻总和不满足恒等式，故自动核验采用论文目标值反推的 TIS。",
            ]
        )
    return "\n".join(lines) + "\n"


def build_fixed_report(experiment: str | Path) -> Path:
    root = Path(experiment).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("fixed-scale report requires a complete experiment")
    records = _read_records(root / "raw" / "results.jsonl")
    expected = int(manifest["task_count_total"])
    if len(records) != expected:
        raise ValueError(f"found {len(records)} successful records, expected {expected}")
    task_ids = [str(record["task_id"]) for record in records]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate successful task IDs")
    if not all(record.get("algorithm") == "kraul_ga" for record in records):
        raise ValueError("fixed-scale results contain a non-Kraul algorithm")
    if not all(bool(record.get("feasible")) for record in records):
        raise ValueError("fixed-scale results contain an infeasible schedule")

    experiment_spec = manifest.get("experiment")
    audit = manifest.get("audit")
    if not isinstance(experiment_spec, Mapping) or not isinstance(audit, Mapping):
        raise ValueError("manifest is missing the fixed-scale experiment contract")
    expected_sequence = audit.get("sequence")
    if not isinstance(expected_sequence, list):
        raise ValueError("manifest audit sequence is missing")
    expected_hashes = {str(item["instance_sha256"]) for item in expected_sequence}
    actual_hashes = {str(record["instance_sha256"]) for record in records}
    if actual_hashes != expected_hashes:
        raise ValueError("result instance hashes differ from the audited split")
    expected_seeds = {int(seed) for seed in experiment_spec["seeds"]}
    records_by_hash: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        records_by_hash[str(record["instance_sha256"])].append(record)
        diagnostics = record.get("algorithm_diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise ValueError(f"missing diagnostics for {record['task_id']}")
        if abs(float(record["objective"]) - float(diagnostics["final_population_best"])) > 1e-8:
            raise ValueError(f"returned objective differs from final population for {record['task_id']}")
        if float(diagnostics["best_seen_during_search"]) > float(record["objective"]) + 1e-8:
            raise ValueError(f"best-seen objective is inconsistent for {record['task_id']}")
    for instance_hash, members in records_by_hash.items():
        seeds = {int(item["seed"]) for item in members}
        if seeds != expected_seeds:
            raise ValueError(f"seed set mismatch for instance hash {instance_hash}")

    run_rows = _run_rows(records)
    instance_rows = _instance_rows(records, str(experiment_spec["name"]))
    scale_rows = _scale_rows(instance_rows)
    expected_counts = {int(key): int(value) for key, value in experiment_spec["expected_by_jobs"].items()}
    actual_counts = {int(row["jobs"]): int(row["instances"]) for row in scale_rows}
    if actual_counts != expected_counts:
        raise ValueError(f"reported scale counts {actual_counts} differ from {expected_counts}")
    convergence_rows = _convergence_rows(records)
    aggregate_convergence = _aggregate_convergence(records)
    paper_validation, paper_comparison_rows = _paper_table5_validation(scale_rows)
    tables = root / "tables"
    figures = root / "figures"
    report = root / "report"
    for directory in (tables, figures, report):
        directory.mkdir(parents=True, exist_ok=True)
    _write_csv(tables / "run_level_kraul_ga.csv", run_rows)
    _write_csv(tables / "instance_level_kraul_ga.csv", instance_rows)
    _write_csv(tables / "scale_summary_kraul_ga.csv", scale_rows)
    _write_csv(tables / "paper_table5_kraul_ga_comparison.csv", paper_comparison_rows)
    _write_csv(tables / "ga_convergence_raw.csv", convergence_rows)
    _write_csv(tables / "ga_convergence_by_scale.csv", aggregate_convergence)
    _plot_convergence(aggregate_convergence, figures / "kraul_ga_convergence_fixed_scale.png")
    _plot_convergence(aggregate_convergence, figures / "kraul_ga_convergence_fixed_scale.pdf")
    (report / "fixed_scale_summary_zh.md").write_text(
        _summary_markdown(scale_rows, paper_validation),
        encoding="utf-8",
    )
    (report / "paper_table5_kraul_ga_validation.json").write_text(
        json.dumps(paper_validation, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    is_formal = experiment_spec.get("contract") == "formal_fixed_scale"
    integrity_status = (
        "PASS"
        if not is_formal or paper_validation.get("status") == "PASS"
        else "FAIL"
    )
    integrity = {
        "status": integrity_status,
        "successful_runs": len(records),
        "unique_instances": len(instance_rows),
        "scales": {str(row["jobs"]): int(row["instances"]) for row in scale_rows},
        "convergence_points_raw": len(convergence_rows),
        "convergence_points_aggregated": len(aggregate_convergence),
        "all_feasible": True,
        "paired_statistical_unit": "instance",
        "paper_table5_kraul_ga_validation": paper_validation.get("status"),
    }
    (report / "integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return root
