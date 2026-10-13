"""Read-only audit of completed DBD runs; writes derived publication artifacts only."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The report directory is <package-root>/paper_artifacts/dbd_10seeds_20260831.
# Resolve all experiment inputs from the package root so the archive is portable.
BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
KRAUL = BASE / "kraul-dbd-reproduction/outputs/dbd_proxy91_10seeds_ffcbspt_kraul_ga_20260826T064333748261Z"
THREE = BASE / "cssd-dbd-case-study/outputs/dbd_case_proxy91_10seeds_three_algorithms_20260827T083126700163Z"
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE / "kraul-dbd-reproduction/src"))
from kraul_dbd_reproduction import merge

METHODS = merge.METHOD_ORDER
METRICS = ("time_in_system", "waiting_stage1", "waiting_stage2")
FIELDS = [f"{m}_mean_min" for m in METRICS] + ["stage1_utilization_percent", "stage2_utilization_percent"]
LABELS = ["系统停留时间/min", "阶段1等待/min", "阶段2等待/min", "阶段1利用率/%", "阶段2利用率/%"]
BOOT_SEED = 20260831
RESAMPLES = 10000


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def audit_raw(seed_table):
    groups = defaultdict(list)
    hashes_by_day = defaultdict(set)
    source_audit = []
    seen = set()
    for root in (KRAUL, THREE):
        source = root / "raw/results.jsonl"
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        for r in records:
            key = (r["algorithm_label"], r["seed"], r["case_day_index"])
            assert key not in seen, ("duplicate task", key)
            seen.add(key)
            assert r["feasible"] is True, key
            assert r.get("status", "success") == "success", key
            assert r["machines_stage1"] == 12 and r["machines_stage2"] == 5, key
            hashes_by_day[r["case_day_index"]].add(r["instance_sha256"])
            groups[key[:2]].append(r)
        source_audit.append({"path": str(source), "records": len(records), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    assert len(seen) == 3731
    assert set(hashes_by_day) == set(range(91))
    assert all(len(hashes) == 1 for hashes in hashes_by_day.values())
    expected = {("FFCBSPT", None)} | {(m, s) for m in METHODS[1:] for s in range(1, 11)}
    assert set(groups) == expected
    saved = {(r["method"], merge._seed_value(r["random_seed"])): r for r in seed_table}
    result = []
    max_error = 0.0
    for method in METHODS:
        for seed in ([None] if method == "FFCBSPT" else range(1, 11)):
            records = groups[(method, seed)]
            assert {r["case_day_index"] for r in records} == set(range(91)) and len(records) == 91
            jobs = sum(r["jobs"] for r in records)
            assert jobs == 22927
            row = {"method": method, "seed": seed, "days": 91, "jobs": jobs}
            for metric in METRICS:
                moments = [r.get("metric_moments", r)[metric] for r in records]
                assert all(m["count"] == r["jobs"] for m, r in zip(moments, records))
                total = sum(m["sum"] for m in moments)
                assert math.isfinite(total) and total >= 0
                row[f"{metric}_mean_min"] = total / jobs
            assert math.isclose(row["time_in_system_mean_min"], sum(r["objective"] - r["release_sum"] for r in records) / jobs, abs_tol=1e-9)
            for stage in (1, 2):
                for r in records:
                    denominator = r[f"machines_stage{stage}"] * r[f"stage{stage}_active_horizon_minutes"]
                    expected_util = r[f"stage{stage}_processing_minutes"] / denominator if denominator else 0.0
                    assert math.isclose(expected_util, r[f"utilization_stage{stage}"], abs_tol=1e-12)
                row[f"stage{stage}_utilization_percent"] = float(np.mean([r[f"utilization_stage{stage}"] for r in records]) * 100)
            row["sum_Cj2_local"] = sum(r["objective"] for r in records)
            row["runtime_seconds_sum_daily_runs"] = sum(r["runtime_seconds"] for r in records)
            for field in FIELDS:
                error = abs(row[field] - float(saved[(method, seed)][field]))
                max_error = max(error, max_error)
                assert error < 1e-8, (method, seed, field, error)
            result.append(row)
    return result, {"raw_sources": source_audit, "records": len(seen), "complete_seed_runs": 40, "deterministic_runs": 1, "max_seed_aggregate_error": max_error, "duplicate_tasks": 0, "missing_tasks": 0, "infeasible_tasks": 0, "all_daily_instance_hashes_match": True}


def build_curves(seed_rows):
    print("Recovering final schedules (decoding saved solutions only; no optimization)...", flush=True)
    intervals = merge._queue_intervals_from_csv(KRAUL, merge.QUEUE_METHODS_KRAUL)
    intervals.update(merge._three_algorithm_queue_intervals(THREE))
    expected = {(r["method"], r["seed"]): r for r in seed_rows}
    long_rows, summary, diagnostics = [], [], []
    matrices = {}
    rng = np.random.default_rng(BOOT_SEED)
    for method in METHODS:
        seeds = [None] if method == "FFCBSPT" else list(range(1, 11))
        assert set(intervals[method]) == set(seeds)
        # Resample whole three-month seed curves, retaining their cross-hour dependence.
        bootstrap_indices = rng.integers(0, len(seeds), size=(RESAMPLES, len(seeds)))
        for stage in (1, 2):
            matrix = []
            for seed in seeds:
                values = intervals[method][seed]
                assert len(values) == 22927
                hourly = np.zeros(24)
                total_wait = 0.0
                for release, start1, end1, start2 in values:
                    a, b = (release, start1) if stage == 1 else (end1, start2)
                    assert 0 <= a <= b
                    hourly += merge._interval_minutes_by_hour(a, b, day_length_minutes=1440, bin_minutes=60)
                    total_wait += b - a
                area_error = abs(hourly.sum() - total_wait)
                table_error = abs(total_wait - expected[(method, seed)][f"waiting_stage{stage}_mean_min"] * 22927)
                assert area_error < 1e-7 and table_error < 1e-6
                curve = hourly / (91 * 60)
                matrix.append(curve)
                diagnostics.append({"method": method, "seed": seed, "stage": stage, "queue_area_error_minutes": area_error, "table_wait_sum_error_minutes": table_error})
                long_rows.extend({"method": method, "seed": seed, "stage": stage, "hour": h, "average_queue_jobs": float(v)} for h, v in enumerate(curve))
            matrix = np.asarray(matrix)
            matrices[(method, stage)] = matrix
            means = matrix.mean(axis=0)
            stds = matrix.std(axis=0, ddof=1) if len(seeds) > 1 else [None] * 24
            quantiles = np.quantile(matrix[bootstrap_indices].mean(axis=1), [0.025, 0.975], axis=0)
            for hour in range(24):
                summary.append({"method": method, "stage": stage, "hour": hour, "n_seed_runs": len(seeds), "mean": float(means[hour]), "sample_sd": None if len(seeds) == 1 else float(stds[hour]), "ci95_low": None if len(seeds) == 1 else float(quantiles[0, hour]), "ci95_high": None if len(seeds) == 1 else float(quantiles[1, hour])})
        print(f"Verified queues: {method}", flush=True)
    write_csv("hourly_queue_by_seed.csv", long_rows)
    write_csv("hourly_queue_mean_sd_ci95.csv", summary)
    write_csv("queue_area_validation.csv", diagnostics)
    return summary, matrices, diagnostics


def draw_figures(summary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "pdf.fonttype": 42, "svg.fonttype": "none"})
    for stage in (1, 2):
        fig, ax = plt.subplots(figsize=(10.8, 5.4))
        for method in METHODS:
            data = [r for r in summary if r["stage"] == stage and r["method"] == method]
            hours, values = [r["hour"] for r in data], [r["mean"] for r in data]
            if method != "FFCBSPT":
                ax.fill_between(hours, [r["ci95_low"] for r in data], [r["ci95_high"] for r in data], color=merge.COLORS[method], alpha=0.23, linewidth=0, zorder=1)
            focal = method == "IBS-CABC"
            ax.plot(hours, values, color=merge.COLORS[method], marker=merge.MARKERS[method], markersize=4 if focal else 3.5, linewidth=2.0 if focal else 1.25, linestyle="-", label=method, zorder=4 if focal else 3)
        ax.set(xlim=(-0.3, 23.3), ylim=(0, None), xlabel="Hour of day", ylabel="Average queued jobs", title=f"Three-month DBD average queue before stage {stage}")
        ax.set_xticks(range(24))
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(ncol=3, loc="upper left", frameon=False, fontsize=10)
        fig.text(0.5, 0.012, "Solid lines: mean of 10 complete runs; shading: pointwise 95% bootstrap CI. FFCBSPT: deterministic reference.", ha="center", fontsize=8, color="#444444")
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        for extension in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"figure_{5+stage}_stage{stage}_10seeds.{extension}", dpi=400, facecolor="white")
        plt.close(fig)


def main():
    manifest, _ = merge._validate_pair(KRAUL, THREE)
    seed_table = merge._combined_seed_table(KRAUL, THREE, manifest)
    saved_summary = {r["method"]: r for r in merge._combined_table(KRAUL, THREE, manifest)}
    seed_rows, audit = audit_raw(seed_table)
    write_csv("table9_by_seed.csv", seed_rows)
    table, paper = [], []
    for method in METHODS:
        members = [r for r in seed_rows if r["method"] == method]
        row = {"method": method, "n_complete_runs": len(members), "days_per_run": 91, "jobs_per_run": 22927}
        cells = [method + " (DBD)"]
        for field in FIELDS:
            values = np.array([r[field] for r in members])
            row[field] = float(values.mean())
            row[field + "_sample_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
            assert abs(row[field] - float(saved_summary[method][field])) < 1e-8
            sd_field = field.replace("_mean_min", "_across_seed_std_min") if field.endswith("_mean_min") else field.replace("_percent", "_across_seed_std_percentage_points")
            if len(values) > 1:
                assert abs(row[field + "_sample_sd"] - float(saved_summary[method][sd_field])) < 1e-8
            cells.append(f"{row[field]:.2f} ({row[field + '_sample_sd']:.2f})" if len(values) > 1 else f"{row[field]:.2f} (—)")
        table.append(row)
        paper.append(cells)
    write_csv("table9_mean_sample_sd.csv", table)
    write_csv("table9_paper.csv", [dict(zip(["方法"] + LABELS, cells)) for cells in paper])
    print("Table 10 independently recomputed from all 3731 raw task records.", flush=True)
    summary, matrices, diagnostics = build_curves(seed_rows)
    draw_figures(summary)
    queue_stats = []
    for method in METHODS:
        for stage in (1, 2):
            matrix = matrices[(method, stage)]
            mean_curve = matrix.mean(axis=0)
            queue_stats.append({"method": method, "stage": stage, "full_day_mean_queue": float(mean_curve.mean()), "peak_of_mean_curve": float(mean_curve.max()), "peak_hour": int(mean_curve.argmax()), "mean_queue_20_to_24": float(mean_curve[20:24].mean()), "queue_at_23": float(mean_curve[23])})
    write_csv("queue_summary.csv", queue_stats)
    ibs = next(r for r in table if r["method"] == "IBS-CABC")
    improvements = [{"reference": r["method"], **{f: 100*(r[f]-ibs[f])/r[f] for f in FIELDS[:3]}} for r in table if r["method"] != "IBS-CABC"]
    write_csv("ibs_relative_reductions_percent.csv", improvements)
    paired = []
    rng = np.random.default_rng(BOOT_SEED + 1)
    ibs_members = [r for r in seed_rows if r["method"] == "IBS-CABC"]
    for reference in METHODS[1:-1]:
        members = [r for r in seed_rows if r["method"] == reference]
        for field in FIELDS[:3]:
            diffs = np.array([r[field] - i[field] for r, i in zip(members, ibs_members)])
            ci = np.quantile(diffs[rng.integers(0, 10, (RESAMPLES, 10))].mean(axis=1), [0.025, 0.975])
            paired.append({"reference": reference, "metric": field, "mean_reference_minus_ibs": float(diffs.mean()), "sample_sd_of_difference": float(diffs.std(ddof=1)), "ci95_low": float(ci[0]), "ci95_high": float(ci[1]), "ibs_wins": int((diffs > 0).sum()), "ties": int((diffs == 0).sum()), "ibs_losses": int((diffs < 0).sum())})
    write_csv("paired_seed_differences.csv", paired)
    audit.update({"status": "PASS", "dataset_sha256": manifest["audit"]["dataset_sha256"], "window": "RealWorldInstances365 instance seeds 233..323; 91-day proxy window; 22927 jobs", "optimizer_seeds": list(range(1, 11)), "ci_method": "pointwise percentile bootstrap of complete seed curves", "bootstrap_resamples": RESAMPLES, "bootstrap_rng_seed": BOOT_SEED, "maximum_queue_area_error_minutes": max(r["queue_area_error_minutes"] for r in diagnostics), "maximum_queue_table_waiting_error_minutes": max(r["table_wait_sum_error_minutes"] for r in diagnostics), "generated_utc": datetime.now(timezone.utc).isoformat(), "algorithm_code_modified": False, "experiments_rerun": False})
    write_json("audit.json", audit)
    make_report(table, paper, seed_rows, queue_stats, improvements, paired, audit)
    write_json("reader_data.json", {"table": table, "paper": paper, "seed_rows": seed_rows, "hourly_summary": summary, "queue_summary": queue_stats, "improvements": improvements, "paired": paired, "audit": audit})
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# DBD 10-seed reproducible audit\n", "Run the saved report script. It reads completed runs, validates every daily task, reconstructs saved final schedules, and writes derived reports/figures only. No optimizer is run.\n"]}, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["from pathlib import Path\n", "import runpy\n", f"runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')\n"]}]}
    write_json("reproduce.ipynb", notebook)
    print(json.dumps({"output": str(OUT), "table": paper, "queue_summary": queue_stats, "paired_stage2_kraul": next(r for r in paired if r["reference"] == "Kraul-GA" and r["metric"] == "waiting_stage2_mean_min")}, ensure_ascii=False), flush=True)


def make_report(table, paper, seeds, queues, improvements, paired, audit):
    qs = {(r["method"], r["stage"]): r for r in queues}
    gain = {r["reference"]: r for r in improvements}
    diff = next(r for r in paired if r["reference"] == "Kraul-GA" and r["metric"] == "waiting_stage2_mean_min")
    lines = ["# 表9及图6—图7：三个月DBD十随机种子完整报告", "", "## 核心结果", "", "四种随机算法均完整覆盖种子1—10，每个种子覆盖91个日实例、22,927个作业。重新读取并核验3,731条日任务原始记录（四种随机算法3,640条，FFCBSPT确定性规则91条），无缺失、重复或不可行记录。下文全部数据来自本轮已完成实验，未重新求解、未改动算法或原始结果。", "", f"IBS-CABC的系统停留时间为425.53 (0.38) min，阶段2等待时间为57.52 (0.27) min。与Kraul-GA相比，二者分别降低{gain['Kraul-GA']['time_in_system_mean_min']:.2f}%和{gain['Kraul-GA']['waiting_stage2_mean_min']:.2f}%。但其阶段1等待为51.00 min，高于ALNS的49.89 min，不能继续写成该指标与ALNS持平或全部指标最优。", "", "## 表9：三个月案例期DBD模式下的运营绩效", "", md_table(["方法"] + LABELS, paper), "", "注：四种随机算法的数值为10次完整三个月运行指标的算术均值（跨种子样本标准差，ddof=1）。时间单位为min；利用率列均值单位为%，括号内标准差单位为百分点。FFCBSPT为确定性方法，完整窗口仅运行一次，跨种子标准差不适用，以—表示。旧表括号内的作业间标准差不得与本表混用。", "", "## 统计口径与公式", "", r"对算法$a$和种子$s$，先将91日的22,927个作业合并，计算每个完整运行的时间指标：", "", r"$$x_{a,s}=\frac{1}{22\,927}\sum_{d=1}^{91}\sum_{j\in J_d}x_{a,s,d,j}.$", "", r"再对10个完整运行等权汇总：", "", r"$$\bar{x}_a=\frac{1}{10}\sum_{s=1}^{10}x_{a,s},\qquad s_a=\sqrt{\frac{1}{9}\sum_{s=1}^{10}(x_{a,s}-\bar{x}_a)^2}.$", "", "系统停留时间为第二阶段完成时刻减释放时刻；阶段1等待为阶段1开工减释放；阶段2等待为阶段2开工减阶段1完工。并非对910个日任务直接计算标准差。", "", r"利用率保留实验原有口径：先计算每个日实例的活动时域利用率$u_{a,s,d,k}=B_{a,s,d,k}/(m_kH_{a,s,d,k})$，再计算$U_{a,s,k}=100\times\frac{1}{91}\sum_{d=1}^{91}u_{a,s,d,k}$，最后跨种子计算均值和样本标准差。其中$B$为设备加工分钟之和，$H$为该日该阶段的活动时域长度。此值不是以全天24小时为分母，也不是将全部日实例的分子、分母分别求和后的比值。", "", "## 图6与图7", "", "中心线全部为实线。四种随机算法的中心线为10个完整种子运行的均值；阴影为对整条种子曲线重采样10,000次得到的均值百分位Bootstrap 95%逐点置信区间（重采样随机种子20260831）。FFCBSPT不绘制置信阴影。置信区间较窄是跨种子波动较小的结果，未人为放大。", "", "图6 三个月DBD案例期阶段1前的小时平均排队作业数（10随机种子均值及95%逐点置信区间）。", "", "![图6](figure_6_stage1_10seeds.png)", "", "图7 三个月DBD案例期阶段2前的小时平均排队作业数（10随机种子均值及95%逐点置信区间）。", "", "![图7](figure_7_stage2_10seeds.png)", "", r"排队曲线采用时间积分，不是小时到达/完工作业数量。阶段1为区间$[r_j,S_{j1})$，阶段2为$[C_{j1},S_{j2})$。每个种子的每小时平均队长等于全部等待区间与该钟点小时区间的重叠分钟之和除以$91\times60$；跨午夜等待按实际钟点折回0—23时，窗口最后一天作业的后续等待亦保留。", "", "已对全部41个完整运行、两个阶段逐一检查：曲线面积与原始总等待时间守恒，且由曲线反推的平均等待时间与表9一致。", "", md_table(["方法", "阶段", "全天平均队长", "均值曲线峰值", "峰值时段起点", "20—24时平均队长", "23时队长"], [[r['method'], r['stage'], f"{r['full_day_mean_queue']:.3f}", f"{r['peak_of_mean_curve']:.3f}", f"{r['peak_hour']}:00", f"{r['mean_queue_20_to_24']:.3f}", f"{r['queue_at_23']:.3f}"] for r in queues]), "", "注意：表中峰值是“跨种子均值曲线的峰值”，不是“各次运行峰值的平均值”。", "", "## 可用于论文的结果说明", "", f"表9表明，在10个随机种子的完整三个月DBD运行中，IBS-CABC的平均系统停留时间为425.53 min，相较FFCBSPT、ALNS、TS和Kraul-GA分别降低{gain['FFCBSPT']['time_in_system_mean_min']:.2f}%、{gain['ALNS']['time_in_system_mean_min']:.2f}%、{gain['TS']['time_in_system_mean_min']:.2f}%和{gain['Kraul-GA']['time_in_system_mean_min']:.2f}%。其阶段2平均等待时间为57.52 min，在比较方法中最低，较Kraul-GA降低{gain['Kraul-GA']['waiting_stage2_mean_min']:.2f}%，同时阶段2利用率达到74.80%。阶段1等待时间为51.00 min，略高于ALNS的49.89 min，但低于其余比较方法，表明整体改善并非以大幅增加阶段1等待为代价。跨种子样本标准差显示上述汇总指标在本案例窗口内波动较小，但该结果不能单独用于归因各搜索组件的独立贡献。", "", f"图6和图7进一步显示，IBS-CABC的阶段1全天平均排队作业数为{qs[('IBS-CABC',1)]['full_day_mean_queue']:.2f}件，略高于ALNS的{qs[('ALNS',1)]['full_day_mean_queue']:.2f}件，远低于Kraul-GA的{qs[('Kraul-GA',1)]['full_day_mean_queue']:.2f}件。阶段2全天平均排队作业数为{qs[('IBS-CABC',2)]['full_day_mean_queue']:.2f}件，低于Kraul-GA的{qs[('Kraul-GA',2)]['full_day_mean_queue']:.2f}件；在23时，IBS-CABC的平均排队作业数为{qs[('IBS-CABC',2)]['queue_at_23']:.2f}件，Kraul-GA为{qs[('Kraul-GA',2)]['queue_at_23']:.2f}件。上述队列形态与系统停留时间及阶段等待指标一致。阴影表示给定案例窗口下均值的跨种子不确定性，不能解读为95%的单次运行轨迹覆盖范围，也不是全曲线同时置信带。", "", "## 配对种子补充核验", "", f"对相同种子编号配对，Kraul-GA减IBS-CABC的阶段2平均等待差为{diff['mean_reference_minus_ibs']:.4f} min，95%配对Bootstrap区间为[{diff['ci95_low']:.4f}, {diff['ci95_high']:.4f}] min，IBS-CABC在{diff['ibs_wins']}/10个种子中更低。该检查比仅比较两个算法各自置信带是否重叠更直接；它仅描述本固定91日窗口的跨种子结果，未实施多重比较校正，不推断其他医院或其他日期窗口的效果。", "", "## 数据来源、适用范围及边界", "", "本实验实际使用公开RealWorldInstances365中实例编号seed233—323的连续91日代理窗口，共22,927个作业；实例文件中的seed编号与优化器随机种子1—10是两回事。该窗口不是原论文未公开的22,934件原始三个月窗口。窗口在算法运行前锁定。设备配置为阶段1共12台、阶段2共5台。DBD中各日独立调度；小时曲线折算跨午夜等待并不意味着仿真了跨日设备冲突或滚动调度。", "", "本次95%区间反映固定实例窗口下优化算法随机性带来的均值不确定性，不包含日期抽样、加工时间扰动、医院差异或模型误设的不确定性。10个种子增加了结果的可信度，但不替代更大范围的外部验证。", "", f"Kraul-GA与FFCBSPT源目录：{KRAUL}", "", f"ALNS、TS与IBS-CABC源目录：{THREE}", "", f"数据集SHA-256：{audit['dataset_sha256']}", "", "## 十次完整三个月运行逐种子指标", ""]
    for method in METHODS[1:]:
        lines.extend([f"### {method}", "", md_table(["种子"] + LABELS, [[r['seed']] + [f"{r[f]:.4f}" for f in FIELDS] for r in seeds if r['method'] == method]), ""])
    lines = [
        line.replace("表9", "表10")
        .replace(str(KRAUL), "kraul-dbd-reproduction/outputs/dbd_proxy91_10seeds_ffcbspt_kraul_ga_20260826T064333748261Z")
        .replace(str(THREE), "cssd-dbd-case-study/outputs/dbd_case_proxy91_10seeds_three_algorithms_20260827T083126700163Z")
        for line in lines
    ]
    lines.extend(["## 交付与复核记录", "", "完整精度汇总（历史文件名保留）：table9_mean_sample_sd.csv；逐种子数据：table9_by_seed.csv；逐种子小时曲线：hourly_queue_by_seed.csv；小时均值/标准差/95%区间：hourly_queue_mean_sd_ci95.csv；复核：audit.json与queue_area_validation.csv；成对差异：paired_seed_differences.csv。", "", "图6和图7同时提供400 dpi PNG、矢量PDF和SVG。rebuild_report.py与reproduce.ipynb可从原始完成结果重建全部派生产物，不执行优化实验。"])
    # Display math must have paired delimiters (the prose above uses inline $...$).
    lines = [line + "$" if line.startswith("$$") and line.endswith(".$") else line for line in lines]
    (OUT / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
