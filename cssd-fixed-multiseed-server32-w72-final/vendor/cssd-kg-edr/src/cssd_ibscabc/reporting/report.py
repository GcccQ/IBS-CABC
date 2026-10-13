from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

_GROUP_ORDER = {
    "small_j25": 0,
    "real365": 1,
    "fixed_j100": 2,
    "fixed_j250": 3,
    "fixed_j500": 4,
    "fixed_j1000": 5,
}

_METRICS = (
    "objective",
    "paper_gap_lb_percent",
    "average_time_in_system",
    "p95_time_in_system",
    "makespan",
    "utilization_stage1",
    "utilization_stage2",
    "waiting_stage1",
    "waiting_stage2",
    "mean_fill_stage1",
    "mean_fill_stage2",
    "total_batch_cycles",
    "evaluations",
    "iterations",
    "runtime_seconds",
)

_DEFAULT_REFERENCE: dict[str, dict[str, float]] = {
    "small_j25": {
        "published_gap_lb_percent": 9.74,
        "published_runtime_seconds": 9.81,
    },
    "real365": {
        "published_gap_lb_percent": 12.70,
        "published_average_time_in_system": 477.0,
        "published_average_time_in_system_std": 66.0,
        "published_objective": 310_390.0,
        "published_objective_std": 125_830.0,
        "published_makespan": 1_693.0,
        "published_makespan_std": 105.0,
        "published_utilization_stage1_percent": 39.07,
        "published_utilization_stage2_percent": 56.87,
    },
    "fixed_j100": {
        "published_gap_lb_percent": 13.25,
        "published_average_time_in_system": 481.0,
        "published_average_time_in_system_std": 56.0,
        "published_objective": 129_202.0,
        "published_objective_std": 5_754.0,
        "published_makespan": 1_615.0,
        "published_makespan_std": 74.0,
        "published_utilization_stage1_percent": 19.45,
        "published_utilization_stage2_percent": 29.14,
    },
    "fixed_j250": {
        "published_gap_lb_percent": 7.85,
        "published_average_time_in_system": 481.0,
        "published_average_time_in_system_std": 56.0,
        "published_objective": 297_876.0,
        "published_objective_std": 14_445.0,
        "published_makespan": 1_664.0,
        "published_makespan_std": 118.0,
        "published_utilization_stage1_percent": 36.35,
        "published_utilization_stage2_percent": 51.95,
    },
    "fixed_j500": {
        "published_gap_lb_percent": 12.35,
        "published_average_time_in_system": 468.0,
        "published_average_time_in_system_std": 119.0,
        "published_objective": 638_982.0,
        "published_objective_std": 59_192.0,
        "published_makespan": 1_929.0,
        "published_makespan_std": 247.0,
        "published_utilization_stage1_percent": 65.84,
        "published_utilization_stage2_percent": 82.73,
    },
    "fixed_j1000": {
        "published_gap_lb_percent": 34.15,
        "published_average_time_in_system": 918.0,
        "published_average_time_in_system_std": 251.0,
        "published_objective": 1_724_170.0,
        "published_objective_std": 249_863.0,
        "published_makespan": 2_993.0,
        "published_makespan_std": 487.0,
        "published_utilization_stage1_percent": 83.41,
        "published_utilization_stage2_percent": 89.70,
    },
}


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


def _collapse_seed_medians(
    records: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        if bool(record.get("feasible", False)):
            groups[
                (
                    str(record.get("protocol")),
                    str(record.get("instance_path")),
                )
            ].append(record)
    collapsed: list[dict[str, object]] = []
    for members in groups.values():
        representative = {
            key: value
            for key, value in members[0].items()
            if key not in ("task_id", "seed", "diagnostics", "history", "history_evaluations")
        }
        for metric in _METRICS:
            values = [float(member[metric]) for member in members if metric in member]
            if values:
                representative[metric] = median(values)
        representative["seed_runs"] = len(members)
        collapsed.append(representative)
    return collapsed


def _summary_row(
    group: str,
    members: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset_group": group,
        "instance_count": len(members),
        "seed_runs_min": min((int(item.get("seed_runs", 1)) for item in members), default=0),
        "seed_runs_max": max((int(item.get("seed_runs", 1)) for item in members), default=0),
        "protocol": str(members[0].get("protocol", "")) if members else "",
    }
    for metric in _METRICS:
        values = [float(item[metric]) for item in members if metric in item]
        if not values:
            continue
        scale = 100.0 if metric in ("utilization_stage1", "utilization_stage2") else 1.0
        row[f"{metric}_mean"] = fmean(values) * scale
        row[f"{metric}_std"] = (stdev(values) if len(values) > 1 else 0.0) * scale
    return row


def _group_summaries(
    instances: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in instances:
        grouped[str(record.get("dataset_group", "unknown"))].append(record)
    return [
        _summary_row(group, grouped[group])
        for group in sorted(grouped, key=lambda value: _GROUP_ORDER.get(value, 99))
    ]


def _comparison_rows(
    summaries: Sequence[Mapping[str, object]],
    reference: Mapping[str, Mapping[str, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary in summaries:
        group = str(summary["dataset_group"])
        published = dict(reference.get(group, {}))
        if not published:
            continue
        row: dict[str, object] = {
            "dataset_group": group,
            "instance_count": summary["instance_count"],
            "ibscabc_gap_lb_percent": summary.get("paper_gap_lb_percent_mean"),
            "published_ga_gap_lb_percent": published.get("published_gap_lb_percent"),
            "ibscabc_objective_mean": summary.get("objective_mean"),
            "ibscabc_objective_std": summary.get("objective_std"),
            "published_ga_objective_mean": published.get("published_objective"),
            "published_ga_objective_std": published.get("published_objective_std"),
            "ibscabc_runtime_seconds_mean": summary.get("runtime_seconds_mean"),
            "published_ga_runtime_seconds_mean": published.get("published_runtime_seconds"),
            "ibscabc_average_time_in_system_mean": summary.get(
                "average_time_in_system_mean"
            ),
            "ibscabc_average_time_in_system_std": summary.get(
                "average_time_in_system_std"
            ),
            "published_ga_average_time_in_system_mean": published.get(
                "published_average_time_in_system"
            ),
            "published_ga_average_time_in_system_std": published.get(
                "published_average_time_in_system_std"
            ),
            "ibscabc_makespan_mean": summary.get("makespan_mean"),
            "ibscabc_makespan_std": summary.get("makespan_std"),
            "published_ga_makespan_mean": published.get("published_makespan"),
            "published_ga_makespan_std": published.get("published_makespan_std"),
            "ibscabc_utilization_stage1_percent_mean": summary.get(
                "utilization_stage1_mean"
            ),
            "published_ga_utilization_stage1_percent_mean": published.get(
                "published_utilization_stage1_percent"
            ),
            "ibscabc_utilization_stage2_percent_mean": summary.get(
                "utilization_stage2_mean"
            ),
            "published_ga_utilization_stage2_percent_mean": published.get(
                "published_utilization_stage2_percent"
            ),
        }
        own_objective = summary.get("objective_mean")
        published_objective = published.get("published_objective")
        if own_objective is not None and published_objective:
            row["objective_gain_vs_published_ga_percent"] = (
                100.0
                * (float(published_objective) - float(own_objective))
                / float(published_objective)
            )
        own_gap = summary.get("paper_gap_lb_percent_mean")
        published_gap = published.get("published_gap_lb_percent")
        if own_gap is not None and published_gap is not None:
            row["gap_lb_reduction_percentage_points"] = float(published_gap) - float(own_gap)
        rows.append(row)
    return rows


def _paper_contract(
    manifest: Mapping[str, object],
    instances: Sequence[Mapping[str, object]],
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    experiment = dict(dict(manifest.get("config", {})).get("experiment", {}))
    counts: dict[str, int] = defaultdict(int)
    for record in instances:
        counts[str(record.get("dataset_group"))] += 1
    required = {
        "small_j25": 20,
        "real365": 365,
        "fixed_j100": 20,
        "fixed_j250": 20,
        "fixed_j500": 20,
        "fixed_j1000": 20,
    }
    for group, expected in required.items():
        if counts[group] != expected:
            reasons.append(f"{group}_requires_{expected}_instances")
    if int(manifest.get("instance_count", 0)) != 465:
        reasons.append("paper_requires_465_instances")
    selection = dict(experiment.get("selection", {}))
    if Path(str(selection.get("split", ""))).name.lower() != "paper_all465.txt":
        reasons.append("paper_split_not_exact")
    algorithms = [dict(item) for item in experiment.get("algorithms", [])]
    if [str(item.get("name")) for item in algorithms] != ["ibscabc"]:
        reasons.append("experiment_must_contain_only_ibscabc")
    seeds = [int(seed) for seed in experiment.get("seeds", [])]
    if len(seeds) != 1:
        reasons.append("published_path_requires_one_run_per_instance")
    protocols = [dict(item) for item in experiment.get("protocols", [])]
    if len(protocols) != 1:
        reasons.append("published_path_requires_one_protocol")
    elif (
        str(protocols[0].get("mode")) != "official_time"
        or float(protocols[0].get("tau", 0.0)) != 1.0
    ):
        reasons.append("published_time_formula_tau1_required")
    return not reasons, tuple(reasons)


def _diagnostic_summary(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    fields = (
        "intensification_objective_gain",
        "intensification_accepted_moves",
        "assignment_repair_objective_gain",
        "assignment_repair_runtime_seconds",
        "assignment_repair_solver_runtime_seconds",
        "assignment_repair_max_solver_call_runtime_seconds",
        "decode_calls",
        "decode_cache_hits",
    )
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        if bool(record.get("feasible", False)):
            grouped[str(record.get("dataset_group", "unknown"))].append(record)
    rows: list[dict[str, object]] = []
    for group in sorted(grouped, key=lambda value: _GROUP_ORDER.get(value, 99)):
        row: dict[str, object] = {"dataset_group": group, "run_count": len(grouped[group])}
        for field in fields:
            values = []
            for record in grouped[group]:
                diagnostics = record.get("diagnostics", {})
                if isinstance(diagnostics, Mapping) and field in diagnostics:
                    values.append(float(diagnostics[field]))
            if values:
                row[f"{field}_mean"] = fmean(values)
                row[f"{field}_positive_runs"] = sum(value > 0 for value in values)
        rows.append(row)
    return rows


def _save_figures(
    records: Sequence[Mapping[str, object]],
    comparisons: Sequence[Mapping[str, object]],
    figures: Path,
) -> None:
    figures.mkdir(exist_ok=True)
    progress_grid = np.linspace(0.0, 1.0, 101)
    curves = []
    for record in records:
        if not bool(record.get("feasible", False)):
            continue
        values = record.get("history", [])
        checkpoints = record.get("history_evaluations", [])
        if not isinstance(values, list) or not values:
            continue
        if not isinstance(checkpoints, list) or len(checkpoints) != len(values):
            checkpoints = list(range(len(values)))
        denominator = max(1.0, float(checkpoints[-1]))
        progress = np.asarray(checkpoints, dtype=float) / denominator
        normalized = np.asarray(values, dtype=float) / max(1.0, float(values[0]))
        curves.append(np.interp(progress_grid, progress, normalized))
    plt.figure(figsize=(7, 4))
    if curves:
        matrix = np.vstack(curves)
        plt.plot(progress_grid, np.median(matrix, axis=0), label="median")
        plt.fill_between(
            progress_grid,
            np.quantile(matrix, 0.25, axis=0),
            np.quantile(matrix, 0.75, axis=0),
            alpha=0.2,
            label="IQR",
        )
        plt.legend()
    plt.xlabel("Fraction of consumed search budget")
    plt.ylabel("Best objective / initial objective")
    plt.tight_layout()
    plt.savefig(figures / "ibscabc_convergence.png", dpi=180)
    plt.close()

    objective_rows = [
        row
        for row in comparisons
        if row.get("ibscabc_objective_mean") is not None
        and row.get("published_ga_objective_mean") is not None
    ]
    plt.figure(figsize=(8, 4))
    if objective_rows:
        labels = [str(row["dataset_group"]) for row in objective_rows]
        positions = np.arange(len(labels))
        width = 0.38
        plt.bar(
            positions - width / 2,
            [float(row["published_ga_objective_mean"]) for row in objective_rows],
            width,
            label="Published GA",
        )
        plt.bar(
            positions + width / 2,
            [float(row["ibscabc_objective_mean"]) for row in objective_rows],
            width,
            label="IBS-CABC",
        )
        plt.xticks(positions, labels, rotation=20)
        plt.legend()
    plt.ylabel("Mean sum of stage-2 completion times")
    plt.tight_layout()
    plt.savefig(figures / "objective_vs_published_ga.png", dpi=180)
    plt.close()


def build_report(
    experiment_directory: str | Path,
    report_config: str | Path | None = None,
) -> Path:
    root = Path(experiment_directory).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest_preview = json.loads(manifest_path.read_text(encoding="utf-8"))
        experiment_preview = dict(
            dict(manifest_preview.get("config", {})).get("experiment", {})
        )
        if experiment_preview.get("report_template"):
            from .paper_experiment_report import (
                build_paper_experiment_report,
            )

            return build_paper_experiment_report(root)
        if experiment_preview.get("planning_modes"):
            from .case_study_report import build_case_study_report

            return build_case_study_report(root, report_config)
    reference: dict[str, dict[str, float]] = {
        group: dict(values) for group, values in _DEFAULT_REFERENCE.items()
    }
    if report_config is not None:
        loaded = yaml.safe_load(Path(report_config).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("report configuration must be a mapping")
        loaded_reference = loaded.get("published_reference", {})
        if not isinstance(loaded_reference, dict):
            raise ValueError("published_reference must be a mapping")
        for group, values in loaded_reference.items():
            if not isinstance(values, dict):
                raise ValueError(f"published_reference.{group} must be a mapping")
            reference[str(group)] = {
                str(key): float(value) for key, value in values.items()
            }

    records = _load_records(root / "raw" / "results.jsonl")
    instances = _collapse_seed_medians(records)
    report_dir = root / "report"
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    aggregate_dir = root / "aggregate"
    for directory in (report_dir, tables_dir, figures_dir, aggregate_dir):
        directory.mkdir(exist_ok=True)

    _write_csv(
        tables_dir / "all_runs.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key not in ("diagnostics", "history", "history_evaluations")
            }
            for row in records
        ],
    )
    _write_csv(aggregate_dir / "instance_results.csv", instances)
    summaries = _group_summaries(instances)
    comparisons = _comparison_rows(summaries, reference)
    _write_csv(
        tables_dir / "paper_table3_small.csv",
        [row for row in summaries if row["dataset_group"] == "small_j25"],
    )
    _write_csv(
        tables_dir / "paper_table4_real365.csv",
        [row for row in summaries if row["dataset_group"] == "real365"],
    )
    _write_csv(
        tables_dir / "paper_table5_fixed_size.csv",
        [row for row in summaries if str(row["dataset_group"]).startswith("fixed_j")],
    )
    _write_csv(tables_dir / "comparison_to_published_ga.csv", comparisons)
    _write_csv(tables_dir / "module_diagnostics.csv", _diagnostic_summary(records))
    _save_figures(records, comparisons, figures_dir)

    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    expected_records = int(manifest.get("task_count_total", -1))
    execution_complete = (
        expected_records == len(records) == 465
        and all(bool(record.get("feasible", False)) for record in records)
        and int(manifest.get("task_count_pending", 0)) == 0
        and int(manifest.get("task_count_failed", 0)) == 0
    )
    contract_passed, contract_failures = _paper_contract(manifest, instances)
    status = "PASS" if execution_complete and contract_passed else "INCOMPLETE"
    performance_summary = {
        "status": status,
        "records": len(records),
        "expected_records": expected_records,
        "execution_complete": execution_complete,
        "paper_contract_passed": contract_passed,
        "paper_contract_failures": list(contract_failures),
        "seed_aggregation": "median",
        "comparison_scope": "published group means; not paired statistical evidence",
        "summaries": summaries,
        "comparisons_to_published_ga": comparisons,
    }
    (report_dir / "performance_summary.json").write_text(
        json.dumps(performance_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        "# IBS-CABC 论文同路径实验报告",
        "",
        f"- 执行状态：**{status}**",
        f"- 完成记录：{len(records)} / {expected_records if expected_records >= 0 else '未知'}",
        "- 数据路径：J25×20、真实日实例×365、J100/J250/J500/J1000 各×20",
        "- 预算：参考论文时间公式，τ=1，单实例最长 1800 秒",
        "- 主目标：阶段二完工时间总和",
        "- GapLB：100 × (目标值 − 论文下界) / 目标值",
        "",
    ]
    if contract_failures:
        summary_lines.append("- 尚未满足：" + "、".join(contract_failures))
        summary_lines.append("")
    summary_lines.extend(
        [
            "## 对照边界",
            "",
            "论文只公布了分组均值/标准差，未公布每个实例的随机种子结果。"
            "因此 `comparison_to_published_ga.csv` 是同数据集上的分组均值对照，"
            "不能替代重新运行作者源码后形成的逐实例配对统计。",
            "",
            "运行时间受硬件和实现语言影响；论文结论应以主目标值为主要比较依据。",
        ]
    )
    output = report_dir / "summary_zh.md"
    output.write_text("\n".join(summary_lines) + "\n", encoding="utf-8-sig")

    comparison_fields = list(comparisons[0]) if comparisons else []
    header_html = "<tr>" + "".join(
        f"<th>{html.escape(field)}</th>" for field in comparison_fields
    ) + "</tr>"
    rows_html = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(field, '')))}</td>"
            for field in comparison_fields
        )
        + "</tr>"
        for row in comparisons
    )
    (report_dir / "performance_report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>IBS-CABC paper-aligned report</title>"
        f"<h1>IBS-CABC 论文同路径实验报告</h1><p>状态：{status}</p>"
        f"<table border='1'>{header_html}{rows_html}</table>",
        encoding="utf-8",
    )
    return output
