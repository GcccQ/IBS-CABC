from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import rankdata, wilcoxon

_SUMMARY_METRICS = {
    "gap_lb_percent": "paper_gap_lb_percent",
    "time_in_system": "average_time_in_system",
    "total_completion_time": "objective",
    "c_max": "makespan",
    "utilization_stage1_percent": "utilization_stage1",
    "utilization_stage2_percent": "utilization_stage2",
    # The paper calls this CPU time, while the released notebook measures
    # elapsed time with time.time(). Preserve the published column label and
    # also export actual process CPU time as a diagnostic.
    "cpu_time_seconds": "runtime_seconds",
    "process_cpu_time_seconds": "process_cpu_runtime_seconds",
    "wall_time_seconds": "wall_runtime_seconds",
}

# External reference values transcribed from Table 5 of Kraul et al. These are
# never used by the solver or for parameter selection; they are appended only
# after aggregation so the proposed algorithm can be audited against the
# published GA result on the same 20-instance groups.
_PUBLISHED_TABLE5_GA = {
    100: {"gap_percent": 13.25, "objective": 129_202.0},
    250: {"gap_percent": 7.85, "objective": 297_876.0},
    500: {"gap_percent": 12.35, "objective": 638_982.0},
    1000: {"gap_percent": 34.15, "objective": 1_724_170.0},
}
_PUBLISHED_TABLE3_GA_GAP_PERCENT = 9.74
_PUBLISHED_TABLE4_GA = {"gap_percent": 12.70, "objective": 310_390.0}


def _load_records(path: Path) -> list[dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                latest[str(record["task_id"])] = record
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"invalid result JSON at line {line_number}") from error
    return list(latest.values())


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        if not fields:
            handle.write("status\nno_matching_records\n")
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    materialized = list(values)
    if not materialized:
        return 0.0, 0.0
    return (
        fmean(materialized),
        stdev(materialized) if len(materialized) > 1 else 0.0,
    )


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Return a deterministic paired-bootstrap percentile interval."""

    sample = np.asarray(values, dtype=float)
    if sample.size == 0:
        return 0.0, 0.0
    if sample.size == 1 or np.all(sample == sample[0]):
        value = float(sample[0])
        return value, value
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    for index in range(resamples):
        means[index] = float(
            np.mean(generator.choice(sample, size=sample.size, replace=True))
        )
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def _bootstrap_paired_objective_gain_ci(
    full_objectives: Sequence[float],
    no_cp_objectives: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap the same ratio-of-means gain reported in the main table."""

    full = np.asarray(full_objectives, dtype=float)
    no_cp = np.asarray(no_cp_objectives, dtype=float)
    if full.size == 0 or full.size != no_cp.size:
        raise ValueError("paired objective vectors must have equal positive length")
    if full.size == 1:
        gain = 100.0 * (float(no_cp[0]) - float(full[0])) / max(
            1.0, float(no_cp[0])
        )
        return gain, gain
    generator = np.random.default_rng(seed)
    gains = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled = generator.integers(0, full.size, size=full.size)
        full_mean = float(np.mean(full[sampled]))
        no_cp_mean = float(np.mean(no_cp[sampled]))
        gains[index] = 100.0 * (no_cp_mean - full_mean) / max(1.0, no_cp_mean)
    lower, upper = np.quantile(gains, [0.025, 0.975])
    return float(lower), float(upper)


def _wilcoxon_p_and_rank_biserial(
    paired_gains: Sequence[float],
) -> tuple[float, float]:
    """Two-sided signed-rank p-value and paired rank-biserial effect size."""

    nonzero = np.asarray(
        [value for value in paired_gains if abs(float(value)) > 1e-12],
        dtype=float,
    )
    if nonzero.size == 0:
        return 1.0, 0.0
    p_value = float(
        wilcoxon(
            nonzero,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        ).pvalue
    )
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(np.sum(ranks[nonzero > 0]))
    negative = float(np.sum(ranks[nonzero < 0]))
    denominator = positive + negative
    effect = (positive - negative) / denominator if denominator else 0.0
    return p_value, float(effect)


def _paper_value(mean: float, standard_deviation: float, decimals: int = 2) -> str:
    return f"{mean:.{decimals}f} ({standard_deviation:.{decimals}f})"


def _instance_level_means(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str, str, str],
        list[Mapping[str, object]],
    ] = defaultdict(list)
    for record in records:
        if not bool(record.get("feasible", False)):
            continue
        grouped[
            (
                str(record.get("algorithm")),
                str(record.get("scenario", "base")),
                str(record.get("protocol")),
                str(record.get("instance_path")),
            )
        ].append(record)
    rows: list[dict[str, object]] = []
    numeric = set(_SUMMARY_METRICS.values()) | {
        "lower_bound",
        "gap_lb_percent",
        "evaluations",
        "iterations",
        "waiting_stage1",
        "waiting_stage2",
    }
    for (algorithm, scenario, protocol, instance_path), members in grouped.items():
        representative = {
            key: value
            for key, value in members[0].items()
            if key
            not in {
                "task_id",
                "seed",
                "history",
                "history_evaluations",
                "diagnostics",
                "source_files",
                "metric_moments",
            }
        }
        representative |= {
            "algorithm": algorithm,
            "scenario": scenario,
            "protocol": protocol,
            "instance_path": instance_path,
            "seed_runs": len(members),
        }
        for field in numeric:
            values = [
                float(member[field])
                for member in members
                if member.get(field) is not None
            ]
            if values:
                representative[field] = fmean(values)
                representative[f"{field}_seed_std"] = (
                    stdev(values) if len(values) > 1 else 0.0
                )
        rows.append(representative)
    return rows


def _summary_row(
    members: Sequence[Mapping[str, object]],
    *,
    algorithm: str = "IBS-CABC",
    group_fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "algorithm": algorithm,
        "instance_count": len(members),
        "seed_runs_per_instance_min": min(
            (int(member.get("seed_runs", 1)) for member in members),
            default=0,
        ),
        "seed_runs_per_instance_max": max(
            (int(member.get("seed_runs", 1)) for member in members),
            default=0,
        ),
        "delta_ga_percent": "",
        "delta_ga_status": "requires_external_GA_results",
    }
    if group_fields:
        row |= dict(group_fields)
    for output_name, source_name in _SUMMARY_METRICS.items():
        scale = (
            100.0
            if source_name in {"utilization_stage1", "utilization_stage2"}
            else 1.0
        )
        mean, standard_deviation = _mean_std(
            float(member[source_name]) * scale
            for member in members
            if member.get(source_name) is not None
        )
        row[f"{output_name}_mean"] = mean
        row[f"{output_name}_std"] = standard_deviation
        row[f"{output_name}_paper"] = _paper_value(mean, standard_deviation)
    return row


def _environment_rows(
    manifest: Mapping[str, object],
    experiment: Mapping[str, object],
) -> list[dict[str, object]]:
    hardware = dict(manifest.get("hardware", {}))
    software = dict(manifest.get("software", {}))
    algorithms = [dict(item) for item in experiment.get("algorithms", [])]
    parameters = dict(algorithms[0].get("parameters", {})) if algorithms else {}
    protocols = [dict(item) for item in experiment.get("protocols", [])]
    protocol = protocols[0] if protocols else {}
    if str(protocol.get("mode")) == "official_time":
        stopping_rule = (
            "paper formula, "
            f"tau={float(protocol.get('tau', 1.0)):g}, cap=1800 s; "
            "elapsed-time clock matching the author's released time.time() "
            "implementation"
        )
    else:
        stopping_rule = str(protocol)
    memory = hardware.get("memory_bytes")
    rows: list[dict[str, object]] = [
        {"item": "CPU", "value": hardware.get("cpu_model", "unknown")},
        {
            "item": "Logical CPUs",
            "value": hardware.get("logical_cpu_count", "unknown"),
        },
        {
            "item": "Memory (GiB)",
            "value": (
                round(float(memory) / 1024**3, 2) if memory is not None else "unknown"
            ),
        },
        {"item": "Operating system", "value": manifest.get("platform", "unknown")},
        {"item": "Python", "value": str(manifest.get("python", "")).split()[0]},
        {
            "item": "External independent-run workers",
            "value": manifest.get("workers", 1),
        },
        {
            "item": "Threads per IBS-CABC run",
            "value": 1,
        },
        {
            "item": "Assignment repair workers",
            "value": parameters.get("assignment_repair_workers", 1),
        },
        {
            "item": "Stopping rule",
            "value": stopping_rule,
        },
        {
            "item": "Primary objective",
            "value": "sum of stage-2 completion times",
        },
        {
            "item": "Utilization denominator",
            "value": "stage-specific dynamic active horizon",
        },
    ]
    rows.extend(
        {"item": f"Package {name}", "value": version}
        for name, version in sorted(software.items())
    )
    return rows


def _e1_parameter_report(
    instance_rows: Sequence[Mapping[str, object]],
    experiment: Mapping[str, object],
    tables: Path,
    figures: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_instance: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in instance_rows:
        by_instance[str(row["instance_path"])].append(row)
    rpi_rows: list[dict[str, object]] = []
    for members in by_instance.values():
        minimum = min(float(member["objective"]) for member in members)
        for member in members:
            objective = float(member["objective"])
            rpi_rows.append(
                {
                    "algorithm": member["algorithm"],
                    "instance": member["instance"],
                    "instance_path": member["instance_path"],
                    "mean_objective_over_seeds": objective,
                    "minimum_configuration_objective": minimum,
                    "rpi_percent": 100.0 * (objective - minimum) / max(1.0, minimum),
                    "seed_runs": member["seed_runs"],
                }
            )
    specifications = {
        str(item.get("label", item.get("name"))): dict(item)
        for item in experiment.get("algorithms", [])
    }
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rpi_rows:
        grouped[str(row["algorithm"])].append(row)
    calibration_parameters = [
        str(value)
        for value in experiment.get(
            "calibration_parameters",
            [
                "colony_size",
                "archive_size",
                "abandonment_limit",
                "operator_attempts",
                "intensification_fraction",
            ],
        )
    ]
    summaries: list[dict[str, object]] = []
    for label, members in sorted(grouped.items()):
        rpi_mean, rpi_std = _mean_std(
            float(member["rpi_percent"]) for member in members
        )
        objectives = [
            float(row["objective"])
            for row in instance_rows
            if str(row["algorithm"]) == label
        ]
        runtimes = [
            float(row["runtime_seconds"])
            for row in instance_rows
            if str(row["algorithm"]) == label
        ]
        parameters = dict(specifications[label].get("parameters", {}))
        summary: dict[str, object] = {
            "configuration": label,
            "instance_count": len(members),
            "rpi_percent_mean": rpi_mean,
            "rpi_percent_std": rpi_std,
            "rpi_percent_paper": _paper_value(rpi_mean, rpi_std),
            "objective_mean": fmean(objectives),
            "cpu_time_seconds_mean": fmean(runtimes),
        }
        summary.update(
            {
                parameter: parameters.get(parameter)
                for parameter in calibration_parameters
            }
        )
        summaries.append(summary)
    selected = min(
        summaries,
        key=lambda row: (
            float(row["rpi_percent_mean"]),
            float(row["objective_mean"]),
            float(row["cpu_time_seconds_mean"]),
        ),
    )
    selected_specification = specifications[str(selected["configuration"])]
    all_parameters = dict(selected_specification.get("parameters", {}))
    selected_parameters = {
        parameter: all_parameters[parameter]
        for parameter in calibration_parameters
        if parameter in all_parameters
    }
    selected_payload = {
        "algorithm": "ibscabc",
        "source_experiment": experiment.get("name"),
        "selection_rule": (
            "minimum mean RPI; ties by mean objective then mean elapsed time"
        ),
        "selected_configuration": selected["configuration"],
        "parameters": selected_parameters,
    }
    _write_csv(tables / "e1_run_level_rpi.csv", rpi_rows)
    _write_csv(tables / "table2_parameter_calibration.csv", summaries)
    _write_csv(tables / "table2_selected_parameters.csv", [selected])
    factor_rows = []
    for parameter in calibration_parameters:
        levels = sorted(
            {
                row[parameter]
                for row in summaries
                if row.get(parameter) is not None
            }
        )
        factor_rows.append(
            {
                "parameter": parameter,
                "candidate_levels": " / ".join(str(level) for level in levels),
                "selected_level": selected.get(parameter, ""),
            }
        )
    factor_rows.append(
        {
            "parameter": "stopping_time",
            "candidate_levels": "same paper CPU-time formula as Original GA",
            "selected_level": "fixed",
        }
    )
    _write_csv(
        tables / "table2_factor_levels_and_selected.csv",
        factor_rows,
    )
    (tables / "selected_parameters.yaml").write_text(
        yaml.safe_dump(selected_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (tables / "selected_parameters.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    labels = [str(row["configuration"]) for row in summaries]
    means = [float(row["rpi_percent_mean"]) for row in summaries]
    errors = [float(row["rpi_percent_std"]) for row in summaries]
    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, means, yerr=errors, capsize=3)
    plt.ylabel("Mean RPI (%)")
    plt.xlabel("IBS-CABC parameter configuration")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "e1_parameter_rpi.png", dpi=180)
    plt.close()
    return summaries, selected_payload


def _e5_report(
    instance_rows: Sequence[Mapping[str, object]],
    tables: Path,
    figures: Path,
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in instance_rows:
        grouped[
            (
                int(row["machines_stage1"]),
                int(row["machines_stage2"]),
            )
        ].append(row)
    rows = [
        _summary_row(
            members,
            group_fields={"machines_stage1": m1, "machines_stage2": m2},
        )
        for (m1, m2), members in sorted(grouped.items())
    ]
    _write_csv(tables / "table6_machine_grid_long.csv", rows)
    m1_levels = sorted({int(row["machines_stage1"]) for row in rows})
    m2_levels = sorted({int(row["machines_stage2"]) for row in rows})
    lookup = {
        (int(row["machines_stage1"]), int(row["machines_stage2"])): row
        for row in rows
    }
    matrix_rows = []
    for m1 in m1_levels:
        matrix_row: dict[str, object] = {"m1\\m2": m1}
        for m2 in m2_levels:
            item = lookup.get((m1, m2))
            matrix_row[str(m2)] = (
                item["gap_lb_percent_paper"] if item is not None else ""
            )
        matrix_rows.append(matrix_row)
    _write_csv(tables / "table6_gap_lb_matrix.csv", matrix_rows)
    _write_csv(
        tables / "table7_parameter_by_machine_status.csv",
        [
            {
                "status": "NOT_APPLICABLE",
                "reason": (
                    "IBS-CABC has no single machine-dependent batch threshold; "
                    "its fixed seed portfolio is part of the algorithm."
                ),
            }
        ],
    )
    if len(rows) == len(m1_levels) * len(m2_levels):
        matrix = np.asarray(
            [
                [
                    float(lookup[(m1, m2)]["gap_lb_percent_mean"])
                    for m2 in m2_levels
                ]
                for m1 in m1_levels
            ]
        )
        plt.figure(figsize=(8, 7))
        image = plt.imshow(matrix, aspect="auto", cmap="viridis_r")
        plt.xticks(range(len(m2_levels)), m2_levels)
        plt.yticks(range(len(m1_levels)), m1_levels)
        plt.xlabel("Stage-2 machines (m2)")
        plt.ylabel("Stage-1 machines (m1)")
        plt.colorbar(image, label="Mean Gap_LB (%)")
        plt.tight_layout()
        plt.savefig(figures / "table6_gap_lb_heatmap.png", dpi=180)
        plt.close()
    return rows


def _cp_sat_ablation_report(
    records: Sequence[Mapping[str, object]],
    experiment: Mapping[str, object],
    tables: Path,
    figures: Path,
    aggregate: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build the paired full-versus-no-CP-SAT ablation evidence package."""

    ablation = dict(experiment.get("ablation", {}))
    full_label = str(ablation.get("full_label", "IBS-CABC"))
    no_cp_label = str(
        ablation.get("no_cp_label", "IBS-CABC-w/o-CP-SAT")
    )
    marginal_threshold = float(
        ablation.get("marginal_threshold_percent", 0.30)
    )
    bootstrap_resamples = int(ablation.get("bootstrap_resamples", 10_000))
    bootstrap_seed = int(ablation.get("bootstrap_seed", 20_260_825))
    if marginal_threshold < 0 or bootstrap_resamples <= 0:
        raise ValueError("invalid CP-SAT ablation reporting parameters")

    specifications = {
        str(item.get("label", item.get("name"))): dict(item)
        for item in experiment.get("algorithms", [])
    }
    if set(specifications) != {full_label, no_cp_label}:
        raise ValueError(
            "CP-SAT ablation requires exactly the configured full and no-CP labels"
        )
    full_parameters = dict(specifications[full_label].get("parameters", {}))
    no_cp_parameters = dict(specifications[no_cp_label].get("parameters", {}))
    if full_parameters.get("assignment_repair_enabled") is not True:
        raise ValueError("the full ablation variant must enable assignment repair")
    if no_cp_parameters.get("assignment_repair_enabled") is not False:
        raise ValueError("the no-CP ablation variant must disable assignment repair")
    aligned_full = dict(full_parameters)
    aligned_no_cp = dict(no_cp_parameters)
    aligned_full.pop("assignment_repair_enabled", None)
    aligned_no_cp.pop("assignment_repair_enabled", None)
    if aligned_full != aligned_no_cp:
        differing = sorted(
            key
            for key in set(aligned_full) | set(aligned_no_cp)
            if aligned_full.get(key) != aligned_no_cp.get(key)
        )
        raise ValueError(
            "CP-SAT ablation variants differ outside the intervention: "
            + ", ".join(differing)
        )

    per_model_limit = float(
        full_parameters.get("assignment_repair_seconds_per_family", 10.0)
    )
    per_round_limit = float(
        full_parameters.get("mathematical_repair_seconds", 45.0)
    )

    paired: dict[
        tuple[str, str, str, int],
        dict[str, Mapping[str, object]],
    ] = defaultdict(dict)
    relevant_keys: set[tuple[str, str, str, int]] = set()
    for record in records:
        label = str(record.get("algorithm"))
        if label not in {full_label, no_cp_label}:
            continue
        key = (
            str(record.get("instance_path")),
            str(record.get("protocol")),
            str(record.get("scenario", "base")),
            int(record.get("seed", 0)),
        )
        relevant_keys.add(key)
        if not bool(record.get("feasible", False)):
            continue
        if label in paired[key]:
            raise ValueError(f"duplicate CP-SAT ablation record for {key} / {label}")
        paired[key][label] = record

    run_rows: list[dict[str, object]] = []
    missing_pairs = 0
    for key in sorted(relevant_keys):
        members = paired.get(key, {})
        if set(members) != {full_label, no_cp_label}:
            missing_pairs += 1
            continue
        full = members[full_label]
        no_cp = members[no_cp_label]
        if int(full.get("jobs", -1)) != int(no_cp.get("jobs", -2)):
            raise ValueError(f"paired records disagree on job count for {key}")
        full_diagnostics = full.get("diagnostics", {})
        no_cp_diagnostics = no_cp.get("diagnostics", {})
        if not isinstance(full_diagnostics, Mapping):
            full_diagnostics = {}
        if not isinstance(no_cp_diagnostics, Mapping):
            no_cp_diagnostics = {}
        full_objective = float(full["objective"])
        no_cp_objective = float(no_cp["objective"])
        module_runtime = float(
            full_diagnostics.get("assignment_repair_runtime_seconds", 0.0)
        )
        solver_runtime = float(
            full_diagnostics.get(
                "assignment_repair_solver_runtime_seconds", 0.0
            )
        )
        full_runtime = float(full.get("runtime_seconds", 0.0))
        run_rows.append(
            {
                "instance": full.get("instance"),
                "instance_path": key[0],
                "protocol": key[1],
                "scenario": key[2],
                "seed": key[3],
                "jobs": int(full["jobs"]),
                "full_objective": full_objective,
                "no_cp_objective": no_cp_objective,
                "paired_gain_percent": (
                    100.0
                    * (no_cp_objective - full_objective)
                    / max(1.0, no_cp_objective)
                ),
                "full_runtime_seconds": full_runtime,
                "no_cp_runtime_seconds": float(no_cp.get("runtime_seconds", 0.0)),
                "cp_sat_model_calls": int(
                    full_diagnostics.get("assignment_repair_model_calls", 0)
                ),
                "cp_sat_repair_rounds": int(
                    full_diagnostics.get("assignment_repair_rounds", 0)
                ),
                "cp_sat_feasible_models": int(
                    full_diagnostics.get("assignment_repair_feasible_models", 0)
                ),
                "cp_sat_optimal_models": int(
                    full_diagnostics.get("assignment_repair_optimal_models", 0)
                ),
                "cp_sat_accepted_iterations": int(
                    full_diagnostics.get(
                        "assignment_repair_accepted_iterations", 0
                    )
                ),
                "cp_sat_direct_objective_gain": float(
                    full_diagnostics.get("assignment_repair_objective_gain", 0.0)
                ),
                "cp_sat_solver_runtime_seconds": solver_runtime,
                "cp_sat_solver_runtime_share_percent": (
                    100.0 * solver_runtime / full_runtime
                    if full_runtime > 0
                    else 0.0
                ),
                "cp_sat_max_solver_call_runtime_seconds": float(
                    full_diagnostics.get(
                        "assignment_repair_max_solver_call_runtime_seconds",
                        0.0,
                    )
                ),
                "cp_sat_module_runtime_seconds": module_runtime,
                "cp_sat_module_runtime_share_percent": (
                    100.0 * module_runtime / full_runtime
                    if full_runtime > 0
                    else 0.0
                ),
                "cp_sat_applicable": bool(
                    full_diagnostics.get("assignment_repair_applicable", False)
                ),
                "no_cp_sat_model_calls": int(
                    no_cp_diagnostics.get("assignment_repair_model_calls", 0)
                ),
            }
        )
    _write_csv(aggregate / "cp_sat_ablation_paired_runs.csv", run_rows)

    by_instance: dict[
        tuple[str, str, str, int],
        list[Mapping[str, object]],
    ] = defaultdict(list)
    for row in run_rows:
        by_instance[
            (
                str(row["instance_path"]),
                str(row["protocol"]),
                str(row["scenario"]),
                int(row["jobs"]),
            )
        ].append(row)
    instance_rows: list[dict[str, object]] = []
    for (instance_path, protocol, scenario, jobs), members in sorted(
        by_instance.items()
    ):
        full_mean = fmean(float(member["full_objective"]) for member in members)
        no_cp_mean = fmean(
            float(member["no_cp_objective"]) for member in members
        )
        instance_rows.append(
            {
                "instance": members[0]["instance"],
                "instance_path": instance_path,
                "protocol": protocol,
                "scenario": scenario,
                "jobs": jobs,
                "seed_runs": len(members),
                "full_objective": full_mean,
                "no_cp_objective": no_cp_mean,
                "paired_gain_percent": (
                    100.0 * (no_cp_mean - full_mean) / max(1.0, no_cp_mean)
                ),
                "cp_sat_model_calls": fmean(
                    float(member["cp_sat_model_calls"]) for member in members
                ),
                "cp_sat_repair_rounds": fmean(
                    float(member["cp_sat_repair_rounds"]) for member in members
                ),
                "cp_sat_feasible_models": fmean(
                    float(member["cp_sat_feasible_models"]) for member in members
                ),
                "cp_sat_optimal_models": fmean(
                    float(member["cp_sat_optimal_models"]) for member in members
                ),
                "cp_sat_accepted_iterations": fmean(
                    float(member["cp_sat_accepted_iterations"])
                    for member in members
                ),
                "cp_sat_direct_objective_gain": fmean(
                    float(member["cp_sat_direct_objective_gain"])
                    for member in members
                ),
                "cp_sat_solver_runtime_seconds": fmean(
                    float(member["cp_sat_solver_runtime_seconds"])
                    for member in members
                ),
                "cp_sat_solver_runtime_share_percent": fmean(
                    float(member["cp_sat_solver_runtime_share_percent"])
                    for member in members
                ),
                "cp_sat_max_solver_call_runtime_seconds": max(
                    float(member["cp_sat_max_solver_call_runtime_seconds"])
                    for member in members
                ),
                "cp_sat_module_runtime_seconds": fmean(
                    float(member["cp_sat_module_runtime_seconds"])
                    for member in members
                ),
                "cp_sat_module_runtime_share_percent": fmean(
                    float(member["cp_sat_module_runtime_share_percent"])
                    for member in members
                ),
                "cp_sat_applicable": all(
                    bool(member["cp_sat_applicable"]) for member in members
                ),
                "no_cp_sat_model_calls_nonzero": any(
                    int(member["no_cp_sat_model_calls"]) != 0
                    for member in members
                ),
            }
        )
    _write_csv(aggregate / "cp_sat_ablation_paired_instances.csv", instance_rows)

    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in instance_rows:
        grouped[int(row["jobs"])].append(row)
    summary_rows: list[dict[str, object]] = []
    for group_index, (jobs, members) in enumerate(sorted(grouped.items())):
        full_values = [float(member["full_objective"]) for member in members]
        no_cp_values = [float(member["no_cp_objective"]) for member in members]
        paired_gains = [float(member["paired_gain_percent"]) for member in members]
        full_mean, full_std = _mean_std(full_values)
        no_cp_mean, no_cp_std = _mean_std(no_cp_values)
        difference = 100.0 * (no_cp_mean - full_mean) / max(1.0, no_cp_mean)
        ci_low, ci_high = _bootstrap_paired_objective_gain_ci(
            full_values,
            no_cp_values,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + group_index,
        )
        wilcoxon_p, rank_biserial = _wilcoxon_p_and_rank_biserial(
            paired_gains
        )
        wins = sum(value > 1e-12 for value in paired_gains)
        ties = sum(abs(value) <= 1e-12 for value in paired_gains)
        losses = sum(value < -1e-12 for value in paired_gains)
        calls_mean, calls_std = _mean_std(
            float(member["cp_sat_model_calls"]) for member in members
        )
        solver_runtime_mean, solver_runtime_std = _mean_std(
            float(member["cp_sat_solver_runtime_seconds"])
            for member in members
        )
        solver_share_mean, solver_share_std = _mean_std(
            float(member["cp_sat_solver_runtime_share_percent"])
            for member in members
        )
        module_runtime_mean, module_runtime_std = _mean_std(
            float(member["cp_sat_module_runtime_seconds"])
            for member in members
        )
        module_share_mean, module_share_std = _mean_std(
            float(member["cp_sat_module_runtime_share_percent"])
            for member in members
        )
        direct_gain_mean, direct_gain_std = _mean_std(
            float(member["cp_sat_direct_objective_gain"])
            for member in members
        )
        if abs(difference) < marginal_threshold:
            interpretation = "marginal_below_threshold"
        elif difference > 0:
            interpretation = "positive"
        else:
            interpretation = "adverse"
        summary_rows.append(
            {
                "jobs": jobs,
                "instance_count": len(members),
                "seed_runs_per_instance_min": min(
                    int(member["seed_runs"]) for member in members
                ),
                "seed_runs_per_instance_max": max(
                    int(member["seed_runs"]) for member in members
                ),
                "full_objective_mean": full_mean,
                "full_objective_std": full_std,
                "no_cp_objective_mean": no_cp_mean,
                "no_cp_objective_std": no_cp_std,
                "difference_percent": difference,
                "bootstrap_95_ci_low_percent": ci_low,
                "bootstrap_95_ci_high_percent": ci_high,
                "paired_median_gain_percent": float(np.median(paired_gains)),
                "wilcoxon_p": wilcoxon_p,
                "rank_biserial": rank_biserial,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "cp_sat_model_calls_mean": calls_mean,
                "cp_sat_model_calls_std": calls_std,
                "cp_sat_repair_rounds_mean": fmean(
                    float(member["cp_sat_repair_rounds"])
                    for member in members
                ),
                "cp_sat_per_model_time_limit_seconds": per_model_limit,
                "cp_sat_per_round_total_time_limit_seconds": per_round_limit,
                "cp_sat_solver_runtime_seconds_mean": solver_runtime_mean,
                "cp_sat_solver_runtime_seconds_std": solver_runtime_std,
                "cp_sat_solver_runtime_share_percent_mean": solver_share_mean,
                "cp_sat_solver_runtime_share_percent_std": solver_share_std,
                "cp_sat_max_actual_solver_call_seconds": max(
                    float(member["cp_sat_max_solver_call_runtime_seconds"])
                    for member in members
                ),
                "cp_sat_module_runtime_seconds_mean": module_runtime_mean,
                "cp_sat_module_runtime_seconds_std": module_runtime_std,
                "cp_sat_module_runtime_share_percent_mean": module_share_mean,
                "cp_sat_module_runtime_share_percent_std": module_share_std,
                "cp_sat_direct_objective_gain_mean": direct_gain_mean,
                "cp_sat_direct_objective_gain_std": direct_gain_std,
                "cp_sat_accepted_iterations_mean": fmean(
                    float(member["cp_sat_accepted_iterations"])
                    for member in members
                ),
                "cp_sat_improved_instance_percent": (
                    100.0
                    * sum(
                        float(member["cp_sat_accepted_iterations"]) > 0
                        for member in members
                    )
                    / len(members)
                ),
                "cp_sat_applicable_instance_percent": (
                    100.0
                    * sum(bool(member["cp_sat_applicable"]) for member in members)
                    / len(members)
                ),
                "interpretation": interpretation,
                "marginal_threshold_percent": marginal_threshold,
            }
        )
    _write_csv(tables / "table8_1_cp_sat_ablation.csv", summary_rows)
    _write_csv(
        tables / "table8_1_cp_sat_ablation_zh.csv",
        [
            {
                "规模": f"J{int(row['jobs'])}",
                "实例数": int(row["instance_count"]),
                "完整版均值（标准差）": _paper_value(
                    float(row["full_objective_mean"]),
                    float(row["full_objective_std"]),
                ),
                "移除CP-SAT均值（标准差）": _paper_value(
                    float(row["no_cp_objective_mean"]),
                    float(row["no_cp_objective_std"]),
                ),
                "差异/%": f"{float(row['difference_percent']):.2f}",
                "配对Bootstrap 95% CI/%": (
                    f"[{float(row['bootstrap_95_ci_low_percent']):.2f}, "
                    f"{float(row['bootstrap_95_ci_high_percent']):.2f}]"
                ),
                "单实例平均CP-SAT模型调用次数": (
                    f"{float(row['cp_sat_model_calls_mean']):.2f}"
                ),
                "单模型时间上限/s": (
                    f"{float(row['cp_sat_per_model_time_limit_seconds']):.2f}"
                ),
                "单轮修复总时间上限/s": (
                    f"{float(row['cp_sat_per_round_total_time_limit_seconds']):.2f}"
                ),
                "CP-SAT求解时间占比/%": (
                    f"{float(row['cp_sat_solver_runtime_share_percent_mean']):.2f}"
                ),
                "修复模块总时间占比/%": (
                    f"{float(row['cp_sat_module_runtime_share_percent_mean']):.2f}"
                ),
                "W/T/L": f"{row['wins']}/{row['ties']}/{row['losses']}",
            }
            for row in summary_rows
        ],
    )

    if summary_rows:
        x = np.arange(len(summary_rows))
        gains = np.asarray(
            [float(row["difference_percent"]) for row in summary_rows]
        )
        lower = gains - np.asarray(
            [float(row["bootstrap_95_ci_low_percent"]) for row in summary_rows]
        )
        upper = np.asarray(
            [float(row["bootstrap_95_ci_high_percent"]) for row in summary_rows]
        ) - gains
        plt.figure(figsize=(7.5, 4.5))
        plt.axhspan(
            -marginal_threshold,
            marginal_threshold,
            color="0.92",
            label=f"marginal band (±{marginal_threshold:g}%)",
        )
        plt.axhline(0.0, color="black", linewidth=0.9)
        plt.errorbar(
            x,
            gains,
            yerr=np.vstack([lower, upper]),
            fmt="o-",
            capsize=4,
            color="#c73b36",
            label="Full IBS-CABC vs w/o CP-SAT",
        )
        plt.xticks(x, [f"J{int(row['jobs'])}" for row in summary_rows])
        plt.ylabel("Objective improvement from CP-SAT (%)")
        plt.xlabel("Instance scale")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "cp_sat_ablation_paired_gain.png", dpi=180)
        plt.close()

    metadata = {
        "full_label": full_label,
        "no_cp_label": no_cp_label,
        "pair_keys_observed": len(relevant_keys),
        "complete_run_pairs": len(run_rows),
        "complete_instance_pairs": len(instance_rows),
        "missing_run_pairs": missing_pairs,
        "full_nonapplicable_instances": sum(
            not bool(row["cp_sat_applicable"]) for row in instance_rows
        ),
        "no_cp_nonzero_call_instances": sum(
            bool(row["no_cp_sat_model_calls_nonzero"]) for row in instance_rows
        ),
        "per_model_time_limit_seconds": per_model_limit,
        "per_round_total_time_limit_seconds": per_round_limit,
        "marginal_threshold_percent": marginal_threshold,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "difference_definition": (
            "100 * (mean objective without CP-SAT - mean objective full) / "
            "mean objective without CP-SAT; positive favors full IBS-CABC"
        ),
        "call_definition": "one OR-Tools CP-SAT Solve invocation",
        "solver_runtime_share_definition": (
            "mean across paired instances of 100 * wall time measured strictly "
            "inside OR-Tools Solve calls / complete full-run wall time"
        ),
        "module_runtime_share_definition": (
            "mean across paired instances of 100 * complete repair-module wall "
            "time (model construction, Solve, reconstruction, decoding) / "
            "complete full-run wall time"
        ),
    }
    return summary_rows, metadata


def _main_table_report(
    template: str,
    instance_rows: Sequence[Mapping[str, object]],
    tables: Path,
    figures: Path,
) -> list[dict[str, object]]:
    if template in {"e2", "e3"}:
        rows = [_summary_row(instance_rows)]
        filename = (
            "table3_small25_ibscabc.csv"
            if template == "e2"
            else "table4_real365_ibscabc.csv"
        )
        if template == "e2":
            rows[0]["best_known_hits"] = ""
            rows[0]["best_known_hits_status"] = "requires_external_reference"
            observed_gap = float(rows[0]["gap_lb_percent_mean"])
            rows[0] |= {
                "published_ga_gap_percent": _PUBLISHED_TABLE3_GA_GAP_PERCENT,
                "relative_gap_reduction_percent": (
                    100.0
                    * (_PUBLISHED_TABLE3_GA_GAP_PERCENT - observed_gap)
                    / _PUBLISHED_TABLE3_GA_GAP_PERCENT
                ),
            }
        else:
            proposed = float(rows[0]["total_completion_time_mean"])
            published = float(_PUBLISHED_TABLE4_GA["objective"])
            improvement = 100.0 * (published - proposed) / published
            formal = {
                str(member.get("protocol")) for member in instance_rows
            } == {"paper_time_tau1"}
            rows[0] |= {
                "published_ga_gap_percent": _PUBLISHED_TABLE4_GA["gap_percent"],
                "published_ga_objective_mean": published,
                "four_percent_target_objective": 0.96 * published,
                "improvement_over_published_ga_percent": improvement,
                "observed_meets_four_percent_target": improvement >= 4.0,
                "comparison_evidence_status": (
                    "formal_equal_budget" if formal else "development_screening_only"
                ),
                "formal_four_percent_gate": (
                    "PASS"
                    if formal and improvement >= 4.0
                    else ("FAIL" if formal else "PENDING_EQUAL_BUDGET")
                ),
            }
        _write_csv(tables / filename, rows)
        return rows
    if template == "e4":
        grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
        for row in instance_rows:
            grouped[int(row["jobs"])].append(row)
        rows = [
            _summary_row(members, group_fields={"jobs": jobs})
            for jobs, members in sorted(grouped.items())
        ]
        for row in rows:
            reference = _PUBLISHED_TABLE5_GA.get(int(row["jobs"]))
            if reference is None:
                continue
            proposed = float(row["total_completion_time_mean"])
            published = float(reference["objective"])
            improvement = 100.0 * (published - proposed) / published
            members = grouped[int(row["jobs"])]
            formal = {
                str(member.get("protocol")) for member in members
            } == {"paper_time_tau1"}
            row |= {
                "published_ga_gap_percent": reference["gap_percent"],
                "published_ga_objective_mean": published,
                "four_percent_target_objective": 0.96 * published,
                "improvement_over_published_ga_percent": improvement,
                "observed_meets_four_percent_target": improvement >= 4.0,
                "comparison_evidence_status": (
                    "formal_equal_budget" if formal else "development_screening_only"
                ),
                "formal_four_percent_gate": (
                    "PASS"
                    if formal and improvement >= 4.0
                    else ("FAIL" if formal else "PENDING_EQUAL_BUDGET")
                ),
            }
        _write_csv(tables / "table5_fixed_scale_ibscabc.csv", rows)
        plt.figure(figsize=(8, 4.5))
        jobs = [int(row["jobs"]) for row in rows]
        gaps = [float(row["gap_lb_percent_mean"]) for row in rows]
        plt.plot(jobs, gaps, marker="o")
        plt.xscale("log")
        plt.xticks(jobs, jobs)
        plt.xlabel("Number of jobs")
        plt.ylabel("Mean Gap_LB (%)")
        plt.tight_layout()
        plt.savefig(figures / "table5_scalability_gap.png", dpi=180)
        plt.close()
        return rows
    if template == "e5":
        return _e5_report(instance_rows, tables, figures)
    raise ValueError(f"unknown paper experiment report template: {template}")


def _precheck_report(
    records: Sequence[Mapping[str, object]],
    instance_rows: Sequence[Mapping[str, object]],
    tables: Path,
    figures: Path,
) -> list[dict[str, object]]:
    """Write an exploratory, mixed-scale health check for the proposed solver."""

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in instance_rows:
        grouped[str(row.get("dataset_group", "unknown"))].append(row)
    summaries = [
        _summary_row(members, group_fields={"dataset_group": group})
        for group, members in sorted(grouped.items())
    ]
    _write_csv(tables / "local_precheck_summary.csv", summaries)

    diagnostic_rows: list[dict[str, object]] = []
    for record in records:
        if not bool(record.get("feasible", False)):
            diagnostic_rows.append(
                {
                    "instance": Path(str(record.get("instance_path", ""))).stem,
                    "feasible": False,
                    "error": record.get("error", ""),
                }
            )
            continue
        diagnostics = record.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            diagnostics = {}
        history = record.get("history", [])
        initial = (
            float(history[0])
            if isinstance(history, list) and history
            else float(record["objective"])
        )
        final = float(record["objective"])
        role_calls = diagnostics.get("role_calls", {})
        role_improvements = diagnostics.get("role_improvements", {})
        routing = diagnostics.get("routing_stage_counts", {})
        diagnostic_rows.append(
            {
                "dataset_group": record.get("dataset_group"),
                "instance": record.get("instance"),
                "jobs": record.get("jobs"),
                "feasible": True,
                "initial_objective": initial,
                "final_objective": final,
                "initial_to_final_gain_percent": (
                    100.0 * (initial - final) / initial if initial else 0.0
                ),
                "lower_bound_setup_inclusive": record.get("lower_bound"),
                "paper_gap_lb_percent": record.get("paper_gap_lb_percent"),
                "elapsed_time_seconds": record.get("runtime_seconds"),
                "process_cpu_time_seconds": record.get(
                    "process_cpu_runtime_seconds"
                ),
                "evaluations": record.get("evaluations"),
                "iterations": record.get("iterations"),
                "intensification_objective_gain": diagnostics.get(
                    "intensification_objective_gain", 0
                ),
                "assignment_repair_objective_gain": diagnostics.get(
                    "assignment_repair_objective_gain", 0
                ),
                "assignment_repair_rounds": diagnostics.get(
                    "assignment_repair_rounds", 0
                ),
                "adaptive_seed_evaluations": diagnostics.get(
                    "adaptive_seed_evaluations", 0
                ),
                "adaptive_seed_improvements": diagnostics.get(
                    "adaptive_seed_improvements", 0
                ),
                "adaptive_seed_best_gain": diagnostics.get(
                    "adaptive_seed_best_gain", 0
                ),
                "adaptive_seed_injected": diagnostics.get(
                    "adaptive_seed_injected", False
                ),
                "formation_calls": (
                    role_calls.get("formation", 0)
                    if isinstance(role_calls, Mapping)
                    else 0
                ),
                "sequencing_calls": (
                    role_calls.get("sequencing", 0)
                    if isinstance(role_calls, Mapping)
                    else 0
                ),
                "formation_improvements": (
                    role_improvements.get("formation", 0)
                    if isinstance(role_improvements, Mapping)
                    else 0
                ),
                "sequencing_improvements": (
                    role_improvements.get("sequencing", 0)
                    if isinstance(role_improvements, Mapping)
                    else 0
                ),
                "routed_stage1": (
                    routing.get("stage1", 0)
                    if isinstance(routing, Mapping)
                    else 0
                ),
                "routed_stage2": (
                    routing.get("stage2", 0)
                    if isinstance(routing, Mapping)
                    else 0
                ),
                "decode_calls": diagnostics.get("decode_calls", 0),
                "decode_cache_hits": diagnostics.get("decode_cache_hits", 0),
                "termination_reason": diagnostics.get("termination_reason", ""),
            }
        )
    _write_csv(tables / "local_precheck_module_diagnostics.csv", diagnostic_rows)

    feasible_rows = [row for row in diagnostic_rows if bool(row.get("feasible"))]
    if feasible_rows:
        labels = [str(row["instance"]) for row in feasible_rows]
        gains = [
            float(row["initial_to_final_gain_percent"]) for row in feasible_rows
        ]
        plt.figure(figsize=(10, 4.8))
        plt.bar(labels, gains)
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.ylabel("Initial-to-final objective gain (%)")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(figures / "local_precheck_initial_to_final_gain.png", dpi=180)
        plt.close()
    return summaries


def build_paper_experiment_report(
    experiment_directory: str | Path,
) -> Path:
    root = Path(experiment_directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    experiment = dict(dict(manifest.get("config", {})).get("experiment", {}))
    template = str(experiment.get("report_template", "")).lower()
    if template not in {
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "precheck",
        "cp_sat_ablation",
    }:
        raise ValueError(f"unsupported report_template: {template!r}")
    records = _load_records(root / "raw" / "results.jsonl")
    feasible = [record for record in records if bool(record.get("feasible", False))]
    instance_rows = _instance_level_means(records)

    report = root / "report"
    tables = root / "tables"
    figures = root / "figures"
    aggregate = root / "aggregate"
    for directory in (report, tables, figures, aggregate):
        directory.mkdir(exist_ok=True)
    _write_csv(
        tables / "all_runs_ibscabc.csv",
        [
            {
                key: value
                for key, value in record.items()
                if key
                not in {
                    "diagnostics",
                    "history",
                    "history_evaluations",
                    "source_files",
                    "metric_moments",
                }
            }
            for record in records
        ],
    )
    _write_csv(aggregate / "instance_level_ibscabc.csv", instance_rows)
    _write_csv(
        tables / "table1_environment_ibscabc.csv",
        _environment_rows(manifest, experiment),
    )

    selected: dict[str, object] | None = None
    cp_sat_metadata: dict[str, object] | None = None
    if template == "e1":
        rows, selected = _e1_parameter_report(
            instance_rows,
            experiment,
            tables,
            figures,
        )
    elif template == "precheck":
        rows = _precheck_report(records, instance_rows, tables, figures)
    elif template == "cp_sat_ablation":
        rows, cp_sat_metadata = _cp_sat_ablation_report(
            records,
            experiment,
            tables,
            figures,
            aggregate,
        )
    else:
        rows = _main_table_report(template, instance_rows, tables, figures)

    expected_records = int(manifest.get("task_count_total", -1))
    execution_complete = (
        len(records) == expected_records
        and len(feasible) == expected_records
        and int(manifest.get("task_count_pending", 0)) == 0
        and int(manifest.get("task_count_failed", 0)) == 0
    )
    contract = dict(experiment.get("evidence_contract", {}))
    unique_instances = len(
        {str(record.get("instance_path")) for record in feasible}
    )
    unique_scenarios = len(
        {str(record.get("scenario", "base")) for record in feasible}
    )
    unique_labels = len({str(record.get("algorithm")) for record in feasible})
    failures: list[str] = []
    observed = {
        "instances": unique_instances,
        "scenarios": unique_scenarios,
        "algorithm_labels": unique_labels,
    }
    for field, value in observed.items():
        if field in contract and int(contract[field]) != value:
            failures.append(f"{field}_expected_{contract[field]}_observed_{value}")
    solvers = {str(record.get("solver")) for record in feasible}
    if solvers - {"ibscabc"}:
        failures.append("non_ibscabc_solver_present")
    if cp_sat_metadata is not None:
        if int(cp_sat_metadata["missing_run_pairs"]) != 0:
            failures.append(
                "cp_sat_ablation_missing_run_pairs_"
                f"{cp_sat_metadata['missing_run_pairs']}"
            )
        if int(cp_sat_metadata["full_nonapplicable_instances"]) != 0:
            failures.append(
                "cp_sat_full_variant_nonapplicable_instances_"
                f"{cp_sat_metadata['full_nonapplicable_instances']}"
            )
        if int(cp_sat_metadata["no_cp_nonzero_call_instances"]) != 0:
            failures.append(
                "cp_sat_disabled_variant_has_nonzero_calls_"
                f"{cp_sat_metadata['no_cp_nonzero_call_instances']}"
            )
    status = "PASS" if execution_complete and not failures else "INCOMPLETE"
    protocol_names = {str(record.get("protocol")) for record in feasible}
    evidence_status = (
        "exploratory_only_not_for_paper_comparison"
        if template == "precheck"
        else (
            "formal_experiment"
            if protocol_names == {"paper_time_tau1"}
            else "development_screening_only"
        )
    )
    summary = {
        "status": status,
        "report_template": template,
        "records": len(records),
        "expected_records": expected_records,
        "execution_complete": execution_complete,
        "contract_failures": failures,
        "aggregation": (
            "mean over seeds within each instance, then mean/std across instances"
        ),
        "time_budget_clock": (
            "elapsed/wall-clock seconds, matching the author's released "
            "time.time() implementation"
        ),
        "process_cpu_time_definition": "process CPU seconds (diagnostic only)",
        "rows": rows,
        "selected_parameters": selected,
        "cp_sat_ablation": cp_sat_metadata,
        "evidence_status": evidence_status,
    }
    (report / "performance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# IBS-CABC {template.upper()} 实验回填说明",
        "",
        f"- 状态：**{status}**",
        f"- 完成记录：{len(records)} / {expected_records}",
        f"- 独立实例：{unique_instances}",
        f"- 参数/机器场景：{unique_scenarios}",
        "- 本目录只包含 IBS-CABC 及其受控变体，不混入外部算法结果。",
        "- 汇总顺序：先对同一实例的随机种子取均值，再在实例层计算均值和标准差。",
        "- 停止计时使用经过时间，严格对应作者公开代码中的 `time.time()`；进程 CPU 秒另存为诊断。",
    ]
    if template == "precheck":
        lines.extend(
            [
                "- 该配置仅用于本地代码与趋势预检，使用论文时限的 5%，不得回填正式论文表格。",
                "- `local_precheck_module_diagnostics.csv` 可检查初始到最终改进、BFC/BSC 调用与改进、瓶颈路由及精确邻域收益。",
            ]
        )
    if template == "cp_sat_ablation" and cp_sat_metadata is not None:
        lines.extend(
            [
                "- 消融干预：两个版本除 `assignment_repair_enabled` 外参数完全一致，并共享实例、种子、目标函数、可行性检查与总墙钟预算。",
                "- 差异定义：100×（移除 CP-SAT 的目标均值－完整版目标均值）/移除 CP-SAT 的目标均值；正值表示 CP-SAT 带来改善。",
                "- 调用次数按 OR-Tools CP-SAT 的单次 `Solve` 计数；单模型上限与单轮修复总上限分别报告。",
                "- CP-SAT 求解时间占比仅累计 OR-Tools `Solve` 调用；修复模块总占比另含建模、结构重建与解码，二者均以完整版完整墙钟时间为分母。",
            ]
        )
        for row in rows:
            difference = float(row["difference_percent"])
            threshold = float(row["marginal_threshold_percent"])
            if abs(difference) < threshold:
                conclusion = (
                    "低于预设 0.30% 判据，表明该模块在此规模上的边际贡献有限"
                )
            elif difference > 0:
                conclusion = "超过预设 0.30% 判据，表明该模块在此规模上具有正向增量作用"
            else:
                conclusion = "为负，表明该模块在此规模与预算下未产生净收益"
            lines.append(
                f"- J{int(row['jobs'])}：差异为 {difference:.2f}%，{conclusion}。"
            )
    if failures:
        lines.extend(["", "## 未满足", "", "、".join(failures)])
    output = report / f"{template}_summary_zh.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return output
