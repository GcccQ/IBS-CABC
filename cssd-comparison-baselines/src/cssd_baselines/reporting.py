from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Iterable, Mapping


EXPECTED_ALGORITHMS = ("IBS_CABC", "Standard ABC", "ALNS", "TS")
EXPECTED_JOB_COUNTS = (25, 100, 250, 500, 1000)
EXPECTED_INSTANCES_PER_GROUP = 20
EXPECTED_TOTAL_RESULTS = (
    len(EXPECTED_ALGORITHMS)
    * len(EXPECTED_JOB_COUNTS)
    * EXPECTED_INSTANCES_PER_GROUP
)

INSTANCE_FIELDS = (
    "task_id",
    "algorithm",
    "algorithm_label",
    "instance",
    "instance_path",
    "instance_sha256",
    "jobs",
    "seed",
    "time_limit_seconds",
    "evaluations",
    "runtime_seconds",
    "process_cpu_runtime_seconds",
    "total_completion_time",
    "average_time_in_system",
    "c_max",
    "utilization_stage1_percent",
    "utilization_stage2_percent",
    "feasible",
)

REQUIRED_METRICS = (
    "total_completion_time",
    "average_time_in_system",
    "c_max",
    "utilization_stage1_percent",
    "utilization_stage2_percent",
)


class ExperimentIntegrityError(ValueError):
    """Raised when a result set cannot support the formal four-algorithm table."""


def _mean_std(values: list[float]) -> tuple[float, float]:
    return fmean(values), stdev(values) if len(values) > 1 else 0.0


def _paper(mean: float, std: float) -> str:
    return f"{mean:,.2f} ({std:,.2f})"


def _canonical_label(value: object) -> str:
    compact = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ibscabc": "IBS_CABC",
        "ibs_cabc": "IBS_CABC",
        "standard_abc": "Standard ABC",
        "alns": "ALNS",
        "ts": "TS",
        "tabu_search": "TS",
    }
    return aliases.get(compact, str(value).strip())


def _instance_identity(row: Mapping[str, object]) -> str:
    digest = str(row.get("instance_sha256", "")).strip()
    if digest:
        return digest.lower()
    path = str(row.get("instance_path", "")).strip()
    if path:
        return str(Path(path).resolve()).lower()
    return str(row.get("instance", "")).strip().lower()


def aggregate_rows(results: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in results:
        groups[(_canonical_label(row["algorithm_label"]), int(row["jobs"]))].append(row)

    algorithm_order = {name: index for index, name in enumerate(EXPECTED_ALGORITHMS)}
    summaries: list[dict[str, object]] = []
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            item[0][1],
            algorithm_order.get(item[0][0], len(algorithm_order)),
            item[0][0],
        ),
    )
    for (algorithm, jobs), rows in ordered_groups:
        objective = _mean_std([float(row["total_completion_time"]) for row in rows])
        tis = _mean_std([float(row["average_time_in_system"]) for row in rows])
        cmax = _mean_std([float(row["c_max"]) for row in rows])
        utilization1 = _mean_std(
            [float(row["utilization_stage1_percent"]) for row in rows]
        )
        utilization2 = _mean_std(
            [float(row["utilization_stage2_percent"]) for row in rows]
        )
        runtime = _mean_std([float(row["runtime_seconds"]) for row in rows])
        summaries.append(
            {
                "algorithm": algorithm,
                "jobs": jobs,
                "instance_count": len(rows),
                "total_completion_time_mean": objective[0],
                "total_completion_time_std": objective[1],
                "total_completion_time_paper": _paper(*objective),
                "average_time_in_system_mean": tis[0],
                "average_time_in_system_std": tis[1],
                "average_time_in_system_paper": _paper(*tis),
                "c_max_mean": cmax[0],
                "c_max_std": cmax[1],
                "c_max_paper": _paper(*cmax),
                "utilization_stage1_percent_mean": utilization1[0],
                "utilization_stage1_percent_std": utilization1[1],
                "utilization_stage1_percent_paper": _paper(*utilization1),
                "utilization_stage2_percent_mean": utilization2[0],
                "utilization_stage2_percent_std": utilization2[1],
                "utilization_stage2_percent_paper": _paper(*utilization2),
                "runtime_seconds_mean": runtime[0],
                "runtime_seconds_std": runtime[1],
                "runtime_seconds_paper": _paper(*runtime),
                "feasible_percent": 100.0
                * sum(bool(row["feasible"]) for row in rows)
                / max(1, len(rows)),
            }
        )
    return summaries


def load_ibscabc_results(result_path: Path) -> list[dict[str, object]]:
    """Normalize the completed IBS_CABC JSONL rows to the baseline schema."""
    if not result_path.is_file():
        raise FileNotFoundError(f"IBS_CABC result file not found: {result_path}")
    normalized: list[dict[str, object]] = []
    for line_number, line in enumerate(
        result_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error"):
            raise ExperimentIntegrityError(
                f"IBS_CABC row {line_number} is a failure record: {row.get('error')}"
            )
        normalized.append(
            {
                "task_id": row.get("task_id", f"ibscabc:{line_number}"),
                "algorithm": "ibscabc",
                "algorithm_label": "IBS_CABC",
                "instance": row.get("instance", ""),
                "instance_path": row.get("instance_path", ""),
                "instance_sha256": row.get("instance_sha256", ""),
                "jobs": int(row["jobs"]),
                "seed": int(row.get("seed", 101)),
                "time_limit_seconds": row.get("budget_elapsed_seconds", ""),
                "evaluations": row.get("evaluations", ""),
                "runtime_seconds": float(
                    row.get("wall_runtime_seconds", row.get("runtime_seconds", 0.0))
                ),
                "process_cpu_runtime_seconds": row.get(
                    "process_cpu_runtime_seconds", ""
                ),
                "total_completion_time": int(
                    row["total_completion_time"]
                    if "total_completion_time" in row
                    else row["objective"]
                ),
                "average_time_in_system": float(row["average_time_in_system"]),
                "c_max": int(row.get("makespan", row.get("c_max", 0))),
                "utilization_stage1_percent": 100.0
                * float(row["utilization_stage1"]),
                "utilization_stage2_percent": 100.0
                * float(row["utilization_stage2"]),
                "feasible": bool(row.get("feasible", False)),
                "history": list(row.get("history", [])),
                "history_evaluations": list(row.get("history_evaluations", [])),
            }
        )
    return normalized


def validate_complete_four_algorithm_results(
    results: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Enforce the 4 algorithms x 4 sizes x 20 paired-instance contract."""
    rows = list(results)
    errors: list[str] = []
    if len(rows) != EXPECTED_TOTAL_RESULTS:
        errors.append(f"result count is {len(rows)}, expected {EXPECTED_TOTAL_RESULTS}")

    groups: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        try:
            label = _canonical_label(row["algorithm_label"])
            jobs = int(row["jobs"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"row {index} has invalid algorithm/jobs fields: {error}")
            continue
        groups[(label, jobs)].append(row)
        if not bool(row.get("feasible", False)):
            errors.append(f"{label}/J{jobs}/{row.get('instance')} is infeasible")
        for metric in REQUIRED_METRICS:
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{label}/J{jobs}/{row.get('instance')} lacks numeric {metric}")
                continue
            if not math.isfinite(value):
                errors.append(f"{label}/J{jobs}/{row.get('instance')} has non-finite {metric}")
        for metric in ("utilization_stage1_percent", "utilization_stage2_percent"):
            if metric in row:
                value = float(row[metric])
                if not 0.0 <= value <= 100.0 + 1e-9:
                    errors.append(
                        f"{label}/J{jobs}/{row.get('instance')} has invalid {metric}={value}"
                    )

    expected_groups = {
        (algorithm, jobs)
        for algorithm in EXPECTED_ALGORITHMS
        for jobs in EXPECTED_JOB_COUNTS
    }
    observed_groups = set(groups)
    for missing in sorted(expected_groups - observed_groups):
        errors.append(f"missing group {missing[0]}/J{missing[1]}")
    for extra in sorted(observed_groups - expected_groups):
        errors.append(f"unexpected group {extra[0]}/J{extra[1]}")

    reference_instances: dict[int, set[str]] = {}
    for jobs in EXPECTED_JOB_COUNTS:
        for algorithm in EXPECTED_ALGORITHMS:
            members = groups.get((algorithm, jobs), [])
            identities = [_instance_identity(row) for row in members]
            if len(members) != EXPECTED_INSTANCES_PER_GROUP:
                errors.append(
                    f"{algorithm}/J{jobs} has {len(members)} results, "
                    f"expected {EXPECTED_INSTANCES_PER_GROUP}"
                )
            duplicate_count = len(identities) - len(set(identities))
            if duplicate_count:
                errors.append(f"{algorithm}/J{jobs} has {duplicate_count} duplicate instances")
            identity_set = set(identities)
            if algorithm == EXPECTED_ALGORITHMS[0]:
                reference_instances[jobs] = identity_set
            elif jobs in reference_instances and identity_set != reference_instances[jobs]:
                errors.append(f"{algorithm}/J{jobs} does not use the IBS_CABC instance set")

    seeds = {int(row.get("seed", -1)) for row in rows}
    if seeds != {101}:
        errors.append(f"seed set is {sorted(seeds)}, expected [101]")

    if errors:
        preview = "; ".join(errors[:12])
        suffix = f"; ... and {len(errors) - 12} more" if len(errors) > 12 else ""
        raise ExperimentIntegrityError(preview + suffix)

    return {
        "status": "PASS",
        "result_count": len(rows),
        "algorithms": list(EXPECTED_ALGORITHMS),
        "job_counts": list(EXPECTED_JOB_COUNTS),
        "instances_per_algorithm_size_group": EXPECTED_INSTANCES_PER_GROUP,
        "group_count": len(expected_groups),
        "seed_set": [101],
        "all_feasible": True,
        "paired_instance_sets": True,
        "required_metrics": list(REQUIRED_METRICS),
    }


def _write_instance_csv(path: Path, results: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INSTANCE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _write_summary_files(
    tables: Path,
    stem: str,
    summaries: list[dict[str, object]],
) -> None:
    if summaries:
        with (tables / f"{stem}.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    (tables / f"{stem}.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "|算法|规模|实例数|ΣCj2均值（标准差）|平均TIS（标准差）|"
        "Cmax（标准差）|阶段1利用率/%|阶段2利用率/%|运行时间/s|可行率/%|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "|{algorithm}|J{jobs}|{instance_count}|{total_completion_time_paper}|"
            "{average_time_in_system_paper}|{c_max_paper}|"
            "{utilization_stage1_percent_paper}|{utilization_stage2_percent_paper}|"
            "{runtime_seconds_paper}|{feasible_percent:.2f}|".format(**row)
        )
    (tables / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(
    output_dir: Path, results: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Write the three-baseline instance and summary reports."""
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    ordered_results = sorted(
        results,
        key=lambda row: (
            int(row["jobs"]),
            _canonical_label(row["algorithm_label"]),
            str(row["instance"]),
        ),
    )
    _write_instance_csv(tables / "instance_results.csv", ordered_results)
    summaries = aggregate_rows(ordered_results)
    _write_summary_files(tables, "fixed_scale_summary", summaries)
    return summaries


def write_four_algorithm_reports(
    output_dir: Path,
    baseline_results: list[dict[str, object]],
    ibscabc_result_paths: Path | Iterable[Path],
) -> list[dict[str, object]]:
    """Validate and write the formal, paired four-algorithm comparison table."""
    paths = (
        [ibscabc_result_paths]
        if isinstance(ibscabc_result_paths, Path)
        else list(ibscabc_result_paths)
    )
    ibscabc_results = [
        row for path in paths for row in load_ibscabc_results(path)
    ]
    combined = ibscabc_results + list(baseline_results)
    integrity = validate_complete_four_algorithm_results(combined)
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        combined,
        key=lambda row: (
            int(row["jobs"]),
            EXPECTED_ALGORITHMS.index(_canonical_label(row["algorithm_label"])),
            str(row["instance"]),
        ),
    )
    _write_instance_csv(tables / "four_algorithm_instance_results.csv", ordered)
    summaries = aggregate_rows(ordered)
    _write_summary_files(tables, "four_algorithm_summary", summaries)
    (tables / "four_algorithm_integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summaries


def print_summary(
    summaries: Iterable[Mapping[str, object]],
    *,
    title: str = "固定规模对比实验汇总",
) -> None:
    print(f"\n{title}（括号内为同组实例的样本标准差）")
    print(
        f"{'算法':<14} {'规模':<6} {'N':<4} {'ΣCj2均值（标准差）':<28} "
        f"{'平均TIS':<22} {'Cmax':<22} {'阶段1利用率/%':<22} {'阶段2利用率/%':<22}"
    )
    for row in summaries:
        print(
            f"{str(row['algorithm']):<14} J{int(row['jobs']):<5} "
            f"{int(row['instance_count']):<4} "
            f"{str(row['total_completion_time_paper']):<28} "
            f"{str(row['average_time_in_system_paper']):<22} "
            f"{str(row['c_max_paper']):<22} "
            f"{str(row['utilization_stage1_percent_paper']):<22} "
            f"{str(row['utilization_stage2_percent_paper']):<22}"
        )
