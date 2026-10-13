from __future__ import annotations

import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

_PUBLISHED_TABLE8_GA = {
    "DBD": {
        "time_in_system_mean": 602.0,
        "time_in_system_std": 254.0,
        "waiting_stage1_mean": 118.0,
        "waiting_stage1_std": 174.0,
        "waiting_stage2_mean": 60.0,
        "waiting_stage2_std": 84.0,
        "utilization_stage1_percent": 56.95,
        "utilization_stage2_percent": 59.96,
    },
    "AIO": {
        "time_in_system_mean": 797.0,
        "time_in_system_std": 771.0,
        "waiting_stage1_mean": 144.0,
        "waiting_stage1_std": 333.0,
        "waiting_stage2_mean": 188.0,
        "waiting_stage2_std": 597.0,
        "utilization_stage1_percent": 53.57,
        "utilization_stage2_percent": 59.00,
    },
}

_PUBLISHED_TABLE8_ALL: dict[str, dict[str, float | None]] = {
    "CSSD": {
        "time_in_system_mean": 1876.0,
        "time_in_system_std": 3101.0,
        "waiting_stage1_mean": 318.0,
        "waiting_stage1_std": 2231.0,
        "waiting_stage2_mean": 1135.0,
        "waiting_stage2_std": 2162.0,
        "utilization_stage1_percent": None,
        "utilization_stage2_percent": None,
    },
    "FFCBSPT_DBD": {
        "time_in_system_mean": 678.0,
        "time_in_system_std": 257.0,
        "waiting_stage1_mean": 103.0,
        "waiting_stage1_std": 170.0,
        "waiting_stage2_mean": 116.0,
        "waiting_stage2_std": 130.0,
        "utilization_stage1_percent": 57.25,
        "utilization_stage2_percent": 62.12,
    },
    "FFCBSPT_AIO": {
        "time_in_system_mean": 763.0,
        "time_in_system_std": 627.0,
        "waiting_stage1_mean": 108.0,
        "waiting_stage1_std": 178.0,
        "waiting_stage2_mean": 192.0,
        "waiting_stage2_std": 572.0,
        "utilization_stage1_percent": 53.27,
        "utilization_stage2_percent": 58.85,
    },
    "GA_DBD": _PUBLISHED_TABLE8_GA["DBD"],
    "GA_AIO": _PUBLISHED_TABLE8_GA["AIO"],
}

_PUBLISHED_TABLE9_FFCBSPT_AIO = {
    "base": _PUBLISHED_TABLE8_ALL["FFCBSPT_AIO"],
    "stage2_m3": {
        "time_in_system_mean": 2445.0,
        "time_in_system_std": 4728.0,
        "waiting_stage1_mean": 108.0,
        "waiting_stage1_std": 178.0,
        "waiting_stage2_mean": 1874.0,
        "waiting_stage2_std": 4685.0,
        "utilization_stage1_percent": 53.27,
        "utilization_stage2_percent": 97.60,
    },
    "stage2_m4": {
        "time_in_system_mean": 972.0,
        "time_in_system_std": 892.0,
        "waiting_stage1_mean": 108.0,
        "waiting_stage1_std": 178.0,
        "waiting_stage2_mean": 401.0,
        "waiting_stage2_std": 835.0,
        "utilization_stage1_percent": 53.27,
        "utilization_stage2_percent": 73.64,
    },
    "stage2_m6": {
        "time_in_system_mean": 726.0,
        "time_in_system_std": 611.0,
        "waiting_stage1_mean": 108.0,
        "waiting_stage1_std": 178.0,
        "waiting_stage2_mean": 155.0,
        "waiting_stage2_std": 558.0,
        "utilization_stage1_percent": 53.27,
        "utilization_stage2_percent": 49.10,
    },
    "stage2_m7": {
        "time_in_system_mean": 717.0,
        "time_in_system_std": 611.0,
        "waiting_stage1_mean": 108.0,
        "waiting_stage1_std": 178.0,
        "waiting_stage2_mean": 147.0,
        "waiting_stage2_std": 557.0,
        "utilization_stage1_percent": 53.27,
        "utilization_stage2_percent": 42.08,
    },
    "stage1_m10": {
        "time_in_system_mean": 814.0,
        "time_in_system_std": 682.0,
        "waiting_stage1_mean": 188.0,
        "waiting_stage1_std": 306.0,
        "waiting_stage2_mean": 163.0,
        "waiting_stage2_std": 577.0,
        "utilization_stage1_percent": 63.93,
        "utilization_stage2_percent": 58.81,
    },
    "stage1_m11": {
        "time_in_system_mean": 779.0,
        "time_in_system_std": 639.0,
        "waiting_stage1_mean": 138.0,
        "waiting_stage1_std": 215.0,
        "waiting_stage2_mean": 179.0,
        "waiting_stage2_std": 572.0,
        "utilization_stage1_percent": 58.12,
        "utilization_stage2_percent": 58.66,
    },
    "stage1_m13": {
        "time_in_system_mean": 761.0,
        "time_in_system_std": 622.0,
        "waiting_stage1_mean": 85.0,
        "waiting_stage1_std": 151.0,
        "waiting_stage2_mean": 212.0,
        "waiting_stage2_std": 575.0,
        "utilization_stage1_percent": 49.18,
        "utilization_stage2_percent": 59.02,
    },
    "stage1_m14": {
        "time_in_system_mean": 764.0,
        "time_in_system_std": 641.0,
        "waiting_stage1_mean": 67.0,
        "waiting_stage1_std": 128.0,
        "waiting_stage2_mean": 233.0,
        "waiting_stage2_std": 601.0,
        "utilization_stage1_percent": 45.66,
        "utilization_stage2_percent": 59.05,
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


def _pooled_moment(
    records: Sequence[Mapping[str, object]],
    metric: str,
) -> tuple[float, float, int]:
    count = 0
    total = 0.0
    total_squares = 0.0
    for record in records:
        moments = record.get("metric_moments", {})
        if not isinstance(moments, Mapping):
            continue
        values = moments.get(metric, {})
        if not isinstance(values, Mapping):
            continue
        count += int(values.get("count", 0))
        total += float(values.get("sum", 0.0))
        total_squares += float(values.get("sum_squares", 0.0))
    mean = total / count if count else 0.0
    numerator = total_squares - total * total / count if count else 0.0
    sample_std = math.sqrt(max(0.0, numerator / (count - 1))) if count > 1 else 0.0
    return mean, sample_std, count


def _sum_hourly(
    records: Sequence[Mapping[str, object]],
    field: str,
) -> list[int]:
    length = max(
        (
            len(value)
            for record in records
            if isinstance((value := record.get(field)), list)
        ),
        default=24,
    )
    total = [0] * length
    for record in records:
        values = record.get(field, [])
        if not isinstance(values, list) or len(values) != length:
            continue
        for index, value in enumerate(values):
            total[index] += int(value)
    return total


def _aggregate_group(
    scenario: str,
    mode: str,
    seed: int,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    row: dict[str, object] = {
        "algorithm": "IBS-CABC",
        "scenario": scenario,
        "planning_mode": mode,
        "seed": seed,
        "daily_run_count": len(records) if mode == "DBD" else 0,
        "optimization_run_count": len(records),
        "device_count": sum(int(record.get("jobs", 0)) for record in records),
        "calendar_objective": sum(
            int(record.get("calendar_objective", record.get("objective", 0)))
            for record in records
        ),
        "runtime_seconds": sum(float(record.get("runtime_seconds", 0.0)) for record in records),
        "utilization_stage1_percent": 100.0
        * fmean(float(record.get("utilization_stage1", 0.0)) for record in records),
        "utilization_stage2_percent": 100.0
        * fmean(float(record.get("utilization_stage2", 0.0)) for record in records),
        "machines_stage1": int(records[0].get("machines_stage1", 0)),
        "machines_stage2": int(records[0].get("machines_stage2", 0)),
    }
    # AIO contains the whole case in one optimization record; DBD contains one
    # record per day. In both cases pooled moments count each device exactly once.
    for metric in ("time_in_system", "waiting_stage1", "waiting_stage2"):
        mean, std, count = _pooled_moment(records, metric)
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
        row[f"{metric}_device_count"] = count
        row[f"{metric}_paper"] = f"{mean:.2f} ({std:.2f})"
    row["utilization_stage1_paper"] = (
        f"{float(row['utilization_stage1_percent']):.2f}%"
    )
    row["utilization_stage2_paper"] = (
        f"{float(row['utilization_stage2_percent']):.2f}%"
    )
    row["cpu_time_paper"] = f"{float(row['runtime_seconds']):.2f}"
    return row


def _comparison_rows(
    rows: Sequence[Mapping[str, object]],
    reference: Mapping[str, Mapping[str, float]],
) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    for row in rows:
        if str(row.get("scenario")) != "base":
            continue
        mode = str(row["planning_mode"])
        published = reference.get(mode)
        if not published:
            continue
        comparison: dict[str, object] = {
            "planning_mode": mode,
            "algorithm": "IBS-CABC",
            "dataset_scope": "locked_91_day_proxy",
        }
        for field, published_value in published.items():
            own_value = float(row.get(field, 0.0))
            comparison[f"ibscabc_{field}"] = own_value
            comparison[f"published_ga_{field}"] = published_value
            comparison[f"difference_{field}"] = own_value - published_value
        comparisons.append(comparison)
    return comparisons


def _machine_comparison_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    for row in rows:
        scenario = str(row["scenario"])
        published = _PUBLISHED_TABLE9_FFCBSPT_AIO.get(scenario)
        if published is None:
            continue
        comparison: dict[str, object] = {
            "scenario": scenario,
            "machines_stage1": row["machines_stage1"],
            "machines_stage2": row["machines_stage2"],
        }
        for field, published_value in published.items():
            if published_value is None:
                continue
            own_value = float(row.get(field, 0.0))
            comparison[f"ibscabc_{field}"] = own_value
            comparison[f"published_ffcbspt_aio_{field}"] = published_value
            comparison[f"difference_{field}"] = own_value - published_value
        comparisons.append(comparison)
    return comparisons


def _save_hourly_figures(
    grouped_records: Mapping[tuple[str, str, int], Sequence[Mapping[str, object]]],
    figures: Path,
) -> None:
    base = {
        mode: records
        for (scenario, mode, _seed), records in grouped_records.items()
        if scenario == "base" and mode in {"AIO", "DBD"}
    }
    if not base:
        return
    hours = np.arange(24)
    arrivals = _sum_hourly(next(iter(base.values())), "arrivals_hourly")
    for stage in (1, 2):
        plt.figure(figsize=(9, 4.5))
        plt.plot(hours, arrivals, marker="o", linewidth=1.3, label="arrivals")
        for mode in ("DBD", "AIO"):
            if mode not in base:
                continue
            completed = _sum_hourly(
                base[mode],
                f"completed_stage{stage}_hourly",
            )
            plt.plot(
                hours,
                completed,
                marker="o",
                linewidth=1.3,
                label=f"IBS-CABC {mode}",
            )
        plt.xticks(hours)
        plt.xlabel("Hour of day")
        plt.ylabel("Number of medical devices")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / f"case_hourly_stage{stage}.png", dpi=180)
        plt.close()


def build_case_study_report(
    experiment_directory: str | Path,
    report_config: str | Path | None = None,
) -> Path:
    root = Path(experiment_directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    experiment = dict(dict(manifest.get("config", {})).get("experiment", {}))
    records = _load_records(root / "raw" / "results.jsonl")
    reference = {mode: dict(values) for mode, values in _PUBLISHED_TABLE8_GA.items()}
    if report_config is not None:
        loaded = yaml.safe_load(Path(report_config).read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
            supplied = loaded.get("published_table8_ga", {})
            if isinstance(supplied, Mapping):
                for mode, values in supplied.items():
                    if isinstance(values, Mapping):
                        reference[str(mode).upper()] = {
                            str(field): float(value) for field, value in values.items()
                        }

    report_dir = root / "report"
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    aggregate_dir = root / "aggregate"
    for directory in (report_dir, tables_dir, figures_dir, aggregate_dir):
        directory.mkdir(exist_ok=True)

    feasible = [record for record in records if bool(record.get("feasible", False))]
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for record in feasible:
        grouped[
            (
                str(record.get("scenario", "base")),
                str(record.get("planning_mode", "")).upper(),
                int(record.get("seed", 0)),
            )
        ].append(record)
    rows = [
        _aggregate_group(scenario, mode, seed, members)
        for (scenario, mode, seed), members in sorted(grouped.items())
    ]
    base_rows = [row for row in rows if row["scenario"] == "base"]
    machine_rows = [row for row in rows if row["planning_mode"] == "AIO"]
    comparisons = _comparison_rows(base_rows, reference)
    machine_comparisons = _machine_comparison_rows(machine_rows)

    _write_csv(
        tables_dir / "all_runs.csv",
        [
            {
                key: value
                for key, value in record.items()
                if key not in (
                    "diagnostics",
                    "history",
                    "history_evaluations",
                    "metric_moments",
                    "source_files",
                )
            }
            for record in records
        ],
    )
    _write_csv(aggregate_dir / "case_aggregate.csv", rows)
    _write_csv(tables_dir / "paper_table8_ibscabc_aio_dbd.csv", base_rows)
    _write_csv(
        tables_dir / "paper_table8_published_reference.csv",
        [
            {"published_method": method, **values}
            for method, values in _PUBLISHED_TABLE8_ALL.items()
        ],
    )
    _write_csv(
        tables_dir / "comparison_to_published_table8_ga.csv",
        comparisons,
    )
    _write_csv(tables_dir / "paper_table9_ibscabc_machine_sensitivity.csv", machine_rows)
    _write_csv(
        tables_dir / "paper_table9_published_ffcbspt_aio_reference.csv",
        [
            {"scenario": scenario, **values}
            for scenario, values in _PUBLISHED_TABLE9_FFCBSPT_AIO.items()
        ],
    )
    _write_csv(
        tables_dir / "comparison_to_published_table9_ffcbspt_aio.csv",
        machine_comparisons,
    )
    _save_hourly_figures(grouped, figures_dir)

    expected = int(manifest.get("task_count_total", -1))
    execution_complete = (
        len(records) == expected
        and all(bool(record.get("feasible", False)) for record in records)
        and int(manifest.get("task_count_pending", 0)) == 0
        and int(manifest.get("task_count_failed", 0)) == 0
    )
    expected_days = int(dict(experiment.get("case_study", {})).get("days", 91))
    contract_failures: list[str] = []
    if [dict(item).get("name") for item in experiment.get("algorithms", [])] != [
        "ibscabc"
    ]:
        contract_failures.append("only_ibscabc_is_permitted")
    planning_modes = {
        str(mode).upper() for mode in experiment.get("planning_modes", [])
    }
    if planning_modes == {"AIO", "DBD"}:
        dbd_records = [
            record
            for record in feasible
            if record.get("scenario") == "base"
            and str(record.get("planning_mode")).upper() == "DBD"
        ]
        aio_records = [
            record
            for record in feasible
            if record.get("scenario") == "base"
            and str(record.get("planning_mode")).upper() == "AIO"
        ]
        if len(dbd_records) != expected_days:
            contract_failures.append(f"DBD_requires_{expected_days}_daily_runs")
        if len(aio_records) != 1:
            contract_failures.append("AIO_requires_one_full_horizon_run")
    if int(manifest.get("instance_count", 0)) != expected_days:
        contract_failures.append(f"case_window_requires_{expected_days}_source_days")
    status = "PASS" if execution_complete and not contract_failures else "INCOMPLETE"

    summary = {
        "status": status,
        "execution_complete": execution_complete,
        "records": len(records),
        "expected_records": expected,
        "contract_failures": contract_failures,
        "dataset_scope": "locked contiguous 91-day proxy; not the paper's unavailable raw case period",
        "aggregation": {
            "device_metrics": "pooled device-level moments",
            "DBD_utilization": "arithmetic mean of daily active-horizon utilization",
            "AIO_utilization": "full-horizon active-horizon utilization",
            "objective": "calendar-aligned sum of stage-2 completion times",
        },
        "rows": rows,
        "comparisons_to_published_ga": comparisons,
    }
    (report_dir / "performance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# IBS-CABC AIO/DBD 三个月案例实验报告",
        "",
        f"- 执行状态：**{status}**",
        f"- 完成记录：{len(records)} / {expected}",
        "- 算法：仅 IBS-CABC。",
        "- DBD：每天独立求解，日初系统为空，不允许跨日组批。",
        "- AIO：连续 91 天统一时间轴一次求解，允许满足释放时间和容量约束的跨日组批。",
        "- 指标：系统时间、阶段一等待、阶段二等待均按器械级观测计算均值与样本标准差；利用率按阶段有效跨度计算。",
        "- 主目标：两种策略均换算到同一日历时间轴后的阶段二完工时间总和。",
        "",
        "## 数据边界",
        "",
        "论文三个月案例原始逐日文件未随当前数据集提供。本实验锁定 RealWorldInstances365 中 seed 233–323 的连续 91 天作为代理窗口；"
        "该窗口共 22,927 件器械，最接近论文公开的 22,934 件规模。所有选择规则在运行前固定，报告不得将其表述为原表 8 的精确复现。",
        "",
        "## 输出",
        "",
        "- `tables/paper_table8_ibscabc_aio_dbd.csv`：本算法 AIO/DBD 结果。",
        "- `tables/comparison_to_published_table8_ga.csv`：与论文公开 GA 汇总值的描述性对照。",
        "- `tables/paper_table9_ibscabc_machine_sensitivity.csv`：AIO 机器数量敏感性。",
        "- `tables/comparison_to_published_table9_ffcbspt_aio.csv`：与论文表 9 的描述性对照。",
        "- `figures/case_hourly_stage1.png`、`case_hourly_stage2.png`：按小时到达/处理曲线。",
    ]
    if contract_failures:
        lines.extend(["", "## 尚未满足", "", "、".join(contract_failures)])
    output = report_dir / "case_study_summary_zh.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    fields = list(comparisons[0]) if comparisons else []
    header = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields
        )
        + "</tr>"
        for row in comparisons
    )
    (report_dir / "case_study_report.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>IBS-CABC AIO/DBD case study</title>"
        f"<h1>IBS-CABC AIO/DBD 案例实验</h1><p>状态：{status}</p>"
        f"<table border='1'><tr>{header}</tr>{body}</table>",
        encoding="utf-8",
    )
    return output
