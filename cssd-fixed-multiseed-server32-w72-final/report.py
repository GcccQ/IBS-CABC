"""Two-level aggregation. Never flatten seed runs into inferential instances."""
from __future__ import annotations

import bootstrap
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

from contracts import IBSCABC_DIAGNOSTIC_FIELDS, METRICS, METHODS, SCALES, atomic_json, digest, load_checkpoints, read_json


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_instances(tasks, records):
    expected, observed = defaultdict(list), defaultdict(list)
    for task in tasks:
        expected[(task["jobs"], task["instance_sha256"], task["algorithm"])].append(task)
    for row in records:
        observed[(row["jobs"], row["instance_sha256"], row["algorithm"])].append(row)
    if set(observed) - set(expected):
        raise ValueError("results contain unplanned instances/algorithms")
    rows, coverage = [], []
    for key, planned in sorted(expected.items()):
        selected = observed.get(key, [])
        expected_seeds = {t["seed"] for t in planned}
        actual_seeds = [r["seed"] for r in selected]
        if len(actual_seeds) != len(set(actual_seeds)) or not set(actual_seeds) <= expected_seeds:
            raise ValueError(f"duplicate/unexpected seeds: {key}")
        complete = set(actual_seeds) == expected_seeds
        coverage.append({"jobs": key[0], "instance_sha256": key[1], "algorithm": key[2], "expected_runs": len(expected_seeds), "completed_runs": len(actual_seeds), "complete": complete, "missing_seeds": ";".join(map(str, sorted(expected_seeds - set(actual_seeds))))})
        if not complete:
            continue
        selected = sorted(selected, key=lambda r: r["seed"])
        row = {"jobs": key[0], "instance": planned[0]["instance"], "instance_sha256": key[1], "algorithm": key[2], "method": METHODS[key[2]], "repeat_count": len(selected), "seeds": ";".join(map(str, sorted(expected_seeds)))}
        for metric in METRICS:
            values = np.array([r[metric] for r in selected], dtype=float)
            if len(values) < 2 or not np.isfinite(values).all():
                raise ValueError("sample SD requires at least two finite repetitions")
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_within_sd"] = float(values.std(ddof=1))
        row["objective_best"] = min(r["objective"] for r in selected)
        row["objective_worst"] = max(r["objective"] for r in selected)
        row["objective_sum"] = sum(r["objective"] for r in selected)
        row["runtime_overrun_seconds_mean"] = float(np.mean([r.get("runtime_overrun_seconds", 0.0) for r in selected]))
        row["runs_exceeding_nominal_budget"] = sum(r.get("runtime_overrun_seconds", 0.0) > 0 for r in selected)
        rows.append(row)
    return rows, coverage


def aggregate_scales(instance_rows, expected_instances):
    tables, complete_scales = [], []
    for scale in sorted(expected_instances):
        expected_ids = expected_instances[scale]
        by_method = {a: [r for r in instance_rows if r["jobs"] == scale and r["algorithm"] == a] for a in METHODS}
        if any({r["instance_sha256"] for r in rows} != expected_ids or len(rows) != len(expected_ids) for rows in by_method.values()):
            continue
        scale_rows = []
        for algorithm, members in by_method.items():
            row = {"jobs": scale, "algorithm": algorithm, "method": METHODS[algorithm], "instances": len(members), "repeats_per_instance": members[0]["repeat_count"]}
            if len({r["repeat_count"] for r in members}) != 1:
                raise ValueError("unequal repetition counts within one scale")
            for metric in METRICS:
                values = np.array([r[metric + "_mean"] for r in members])
                row[metric + "_mean"] = float(values.mean())
                row[metric + "_between_instance_sd"] = float(values.std(ddof=1))
                row[metric + "_mean_within_instance_sd"] = float(np.mean([r[metric + "_within_sd"] for r in members]))
            row["runtime_overrun_seconds_mean"] = float(np.mean([r["runtime_overrun_seconds_mean"] for r in members]))
            row["runs_exceeding_nominal_budget"] = sum(r["runs_exceeding_nominal_budget"] for r in members)
            scale_rows.append(row)
        minimum = min(r["objective_mean"] for r in scale_rows)
        ga = next(r["objective_mean"] for r in scale_rows if r["algorithm"] == "kraul_ga")
        for row in scale_rows:
            row["rpr"] = row["objective_mean"] / minimum
            row["rpr_denominator"] = minimum
            row["rpr_scope"] = ";".join(METHODS.values())
            row["gain_over_kraul_ga_percent_ratio_of_means"] = 100 * (ga - row["objective_mean"]) / ga
        tables.extend(scale_rows)
        complete_scales.append(scale)
    return tables, complete_scales


def aggregate_cp_sat(records, complete_scales, expected_instances, expected_seeds, parameters):
    """Aggregate transparent CP-SAT accounting without mixing it into quality tests."""

    rows = []
    for scale in complete_scales:
        selected = [
            row
            for row in records
            if row["jobs"] == scale and row["algorithm"] == "ibscabc"
        ]
        expected_runs = len(expected_instances[scale]) * len(expected_seeds[scale])
        if len(selected) != expected_runs:
            raise ValueError(f"J{scale}: incomplete IBS-CABC CP-SAT diagnostics")
        cp_runtime = np.array(
            [row["assignment_repair_runtime_seconds"] for row in selected],
            dtype=float,
        )
        wall_runtime = np.array([row["runtime_seconds"] for row in selected], dtype=float)
        rows.append(
            {
                "jobs": scale,
                "runs": len(selected),
                "applicable_runs": sum(row["assignment_repair_applicable"] for row in selected),
                "cp_sat_per_model_time_limit_seconds": float(parameters["assignment_repair_seconds_per_family"]),
                "mean_model_calls_per_run": float(np.mean([row["assignment_repair_model_calls"] for row in selected])),
                "mean_rounds_per_run": float(np.mean([row["assignment_repair_rounds"] for row in selected])),
                "mean_cp_wall_seconds_per_run": float(cp_runtime.mean()),
                "aggregate_cp_wall_time_share_percent": float(100.0 * cp_runtime.sum() / wall_runtime.sum()),
                "max_single_solver_call_seconds": float(max(row["assignment_repair_max_solver_call_runtime_seconds"] for row in selected)),
                "mean_cp_objective_gain_per_run": float(np.mean([row["assignment_repair_objective_gain"] for row in selected])),
                "mean_final_reserve_seconds": float(np.mean([row["final_repair_reserved_seconds"] for row in selected])),
                "mean_final_repair_seconds": float(np.mean([row["final_repair_runtime_seconds"] for row in selected])),
                "mean_returned_search_seconds": float(np.mean([row["post_final_repair_search_seconds"] for row in selected])),
                "mean_returned_search_evaluations": float(np.mean([row["post_final_repair_search_evaluations"] for row in selected])),
                "mean_returned_search_objective_gain": float(np.mean([row["post_final_repair_search_objective_gain"] for row in selected])),
            }
        )
    return rows


def holm(p_values):
    order = np.argsort(p_values)
    result = np.empty(len(p_values))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(p_values) - rank) * p_values[index]))
        result[index] = previous
    return result.tolist()


def paired_statistics(instance_rows, complete_scales, resamples=10000, bootstrap_seed=20260831):
    paired = []
    for scale in complete_scales:
        ibs = {r["instance_sha256"]: r for r in instance_rows if r["jobs"] == scale and r["algorithm"] == "ibscabc"}
        ga = {r["instance_sha256"]: r for r in instance_rows if r["jobs"] == scale and r["algorithm"] == "kraul_ga"}
        if set(ibs) != set(ga):
            raise ValueError("unpaired instance hashes")
        for sha in sorted(ibs):
            left, right = ga[sha], ibs[sha]
            if left["repeat_count"] != right["repeat_count"] or left["seeds"] != right["seeds"]:
                raise ValueError("paired instances must have matching repetition sets")
            difference = (left["objective_sum"] - right["objective_sum"]) / right["repeat_count"]
            # Inputs are sums of integer completion times; remove floating roundoff only.
            if abs(difference) < 1e-8:
                difference = 0.0
            paired.append({"jobs": scale, "instance": right["instance"], "instance_sha256": sha, "repeat_count": right["repeat_count"], "ibscabc_mean_objective": right["objective_mean"], "kraul_ga_mean_objective": left["objective_mean"], "objective_difference_ga_minus_ibs": difference, "gain_percent": 100 * difference / left["objective_mean"]})
    scopes = [(f"J{s}", [r for r in paired if r["jobs"] == s]) for s in complete_scales]
    all_scales = set(complete_scales) == set(SCALES)
    if all_scales:
        scopes.append(("Overall", paired))
    rows, rng = [], np.random.default_rng(bootstrap_seed)
    for label, members in scopes:
        differences = np.array([r["objective_difference_ga_minus_ibs"] for r in members])
        gains = np.array([r["gain_percent"] for r in members])
        n = len(members)
        if label == "Overall":
            # Preserve the predeclared 20/20/20/20 scale composition when bootstrapping.
            sampled = np.concatenate([gains[[i for i, r in enumerate(members) if r["jobs"] == s]][rng.integers(0, sum(r["jobs"] == s for r in members), (resamples, sum(r["jobs"] == s for r in members)))] for s in SCALES], axis=1)
        else:
            sampled = gains[rng.integers(0, n, (resamples, n))]
        low, high = np.quantile(np.median(sampled, axis=1), [0.025, 0.975])
        nonzero = differences[differences != 0]
        p = float(wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto").pvalue) if len(nonzero) else 1.0
        ranks = rankdata(np.abs(nonzero), method="average")
        effect = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()) if len(nonzero) else 0.0
        wins, ties, losses = int((differences > 0).sum()), int((differences == 0).sum()), int((differences < 0).sum())
        rows.append({"scale": label, "instances": n, "median_gain_percent": float(np.median(gains)), "bootstrap_ci95_low_percent": float(low), "bootstrap_ci95_high_percent": float(high), "wilcoxon_p_two_sided": p, "holm_p_five_tests": None, "rank_biserial": effect, "wins": wins, "ties": ties, "losses": losses, "W_T_L": f"{wins}/{ties}/{losses}", "statistical_unit": "instance represented by mean across repetitions", "wilcoxon_input": "raw paired mean-objective differences", "holm_family_complete": all_scales})
    if all_scales:
        for row, adjusted in zip(rows, holm([r["wilcoxon_p_two_sided"] for r in rows])):
            row["holm_p_five_tests"] = adjusted
    return rows, paired


def build_report(roots, output):
    output = Path(output).resolve()
    roots = [Path(r).resolve() for r in roots]
    if output in roots or any(output == r / "checkpoints" for r in roots):
        raise ValueError("report output must not be a raw experiment directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest = None
    tasks, records, seen = [], [], set()
    for root in roots:
        current = read_json(root / "manifest.json")
        if digest(current["core"]) != current["study_hash"]:
            raise ValueError("experiment manifest study checksum mismatch")
        # ``run_contract`` is created before host-observation metadata is
        # attached in ``prepare``.  Reconstruct exactly that signed payload;
        # observed_environment is diagnostic metadata, not part of the run
        # contract used by resume validation.
        contract_fields = {
            k: v
            for k, v in current.items()
            if k not in {"created_utc", "run_contract", "observed_environment"}
        }
        if digest(contract_fields) != current["run_contract"]:
            raise ValueError("experiment manifest run checksum mismatch")
        if manifest is None:
            manifest = current
        elif current["study_hash"] != manifest["study_hash"]:
            raise ValueError("cannot merge different configurations, code versions, data or environments")
        planned = read_json(root / "task_manifest.json")
        if digest(planned) != current["task_manifest_hash"]:
            raise ValueError("task manifest checksum mismatch")
        ids = {t["task_id"] for t in planned}
        if ids & seen:
            raise ValueError("overlapping run directories; do not count any task twice")
        seen.update(ids)
        tasks.extend(planned)
        records.extend(load_checkpoints(root, planned).values())
    if manifest is None:
        raise ValueError("no runs supplied")
    expected = defaultdict(set)
    for row in manifest["core"]["instances"]:
        expected[row["jobs"]].add(row["instance_sha256"])
    # Planned subsets are not allowed to masquerade as full-repeat aggregates.
    expected_seeds = {(r["jobs"], r["instance_sha256"]): set(r["expected_seeds"]) for r in manifest["core"]["instances"]}
    expected_seeds_by_scale = {}
    for (scale, _), seeds in expected_seeds.items():
        if scale in expected_seeds_by_scale and expected_seeds_by_scale[scale] != seeds:
            raise ValueError(f"J{scale}: instances use different repetition sets")
        expected_seeds_by_scale[scale] = seeds
    planned_groups = defaultdict(set)
    for t in tasks:
        planned_groups[(t["jobs"], t["instance_sha256"], t["algorithm"])].add(t["seed"])
    for key, seeds in planned_groups.items():
        if seeds != expected_seeds[key[:2]]:
            raise ValueError("task plan does not contain the entire required seed set")
    instances, coverage = aggregate_instances(tasks, records)
    # Include not-yet-launched algorithms/scales in the coverage audit.
    for (scale, sha), seeds in expected_seeds.items():
        for algorithm in METHODS:
            if (scale, sha, algorithm) not in planned_groups:
                coverage.append({"jobs": scale, "instance_sha256": sha, "algorithm": algorithm, "expected_runs": len(seeds), "completed_runs": 0, "complete": False, "missing_seeds": ";".join(map(str, sorted(seeds)))})
    tables, complete_scales = aggregate_scales(instances, expected)
    stats = manifest["core"]["config"]["statistics"]
    tests, pairs = paired_statistics(instances, complete_scales, stats["bootstrap_resamples"], stats["bootstrap_seed"])
    cp_rows = aggregate_cp_sat(
        records,
        complete_scales,
        expected,
        expected_seeds_by_scale,
        manifest["core"]["config"]["algorithms"]["ibscabc"],
    )
    write_csv(output / "coverage.csv", coverage)
    write_csv(output / "instance_mean_within_sd.csv", instances)
    write_csv(output / "tables4_7_numeric.csv", tables)
    write_csv(output / "table8_paired_statistics.csv", tests)
    write_csv(output / "table8_paired_instance_means.csv", pairs)
    write_csv(output / "cp_sat_diagnostics.csv", cp_rows)
    raw_fields = (
        "task_id",
        "jobs",
        "instance",
        "instance_sha256",
        "algorithm",
        "seed",
        "time_limit_seconds",
        "solver_reported_runtime_seconds",
        "process_cpu_seconds",
        "runtime_overrun_seconds",
        "runtime_underrun_seconds",
        "termination_reason",
        "lower_bound",
        *METRICS,
        *IBSCABC_DIAGNOSTIC_FIELDS,
        "assignment_repair_time_share_percent",
    )
    raw_rows = [{key: row.get(key, "") for key in raw_fields} for row in records]
    write_csv(output / "all_repetitions.csv", raw_rows)
    atomic_json(output / "algorithm_parameters.json", manifest["core"]["config"]["algorithms"])
    mode = manifest["core"]["config"]["mode"]
    full = set(complete_scales) == set(SCALES)
    audit = {"status": "PASS" if full else "PARTIAL", "mode": mode, "publication_ready": full and mode == "formal", "completed_runs": len(records), "expected_runs": sum(len(s) * len(METHODS) for s in expected_seeds.values()), "complete_scales": complete_scales, "missing_or_incomplete_instance_algorithm_groups": sum(not r["complete"] for r in coverage), "source_runs": list(map(str, roots)), "study_hash": manifest["study_hash"], "sd_between_ddof": 1, "sd_within_ddof": 1, "within_sd_aggregation": "arithmetic mean of instance sample SDs; not pooled SD", "wilcoxon_n": {r["scale"]: r["instances"] for r in tests}, "rpr_definition": "ratio of scale-level means, minimum over four stochastic methods", "cp_sat_time_share_definition": "sum of IBS-CABC CP-SAT wall seconds divided by sum of IBS-CABC complete solver wall seconds", "unused_final_cp_reserve_policy": "return to the same BFC/BSC search; repaired incumbent retained by elite archive", "ffcBFCFS_status": "no verified per-instance implementation/results found; excluded, not imputed", "generated_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(output / "integrity.json", audit)
    lines = ["# 固定规模多随机种子实验统计", "", f"状态：{audit['status']}；模式：{mode}；完成{len(records)}/{audit['expected_runs']}次。只有全部规定重复均完成的实例进入汇总；只有四种算法的全部实例齐全的规模生成正式规模表。smoke结果不得用于论文。", "", "## 统计口径", "", r"每个实例$i$、算法$a$先计算$\overline{TC}_{ia}=R^{-1}\sum_{r=1}^{R}TC_{iar}$与$s_{ia}=\sqrt{\sum_r(TC_{iar}-\overline{TC}_{ia})^2/(R-1)}$。J100/J250的$R=30$；J500/J1000的$R=20$。", "", r"表4—7中$\overline{TC}_a=20^{-1}\sum_i\overline{TC}_{ia}$；括号内为20个实例均值之间的样本标准差$S_a=\sqrt{\sum_i(\overline{TC}_{ia}-\overline{TC}_a)^2/19}$。脚注另列平均实例内标准差$\overline{s}_a=20^{-1}\sum_i s_{ia}$，不将二者混淆。TIS、Cmax、利用率采用同样的两层汇总。", "", r"相对性能比$\mathrm{RPR}_a=\overline{TC}_a/\min_{b\in\mathcal{A}}\overline{TC}_b$。这里$\mathcal{A}$为ALNS、TS、Kraul-GA和IBS-CABC；1为本规模最优。它是总体均值之比，不是逐实例比值的平均，亦不是最优性差距。", ""]
    formatted = []
    for scale in complete_scales:
        selected = [r for r in tables if r["jobs"] == scale]
        lines.extend([f"## 表{4 + SCALES.index(scale)} J{scale}", "", "| 算法 | RPR | 平均TIS/min | Cmax/min | ΣCj2均值（实例间标准差） | 阶段1利用率/% | 阶段2利用率/% |", "|---|---:|---:|---:|---:|---:|---:|"])
        for r in selected:
            def cell(metric):
                return f"{r[metric+'_mean']:,.2f} ({r[metric+'_between_instance_sd']:,.2f})"
            values = {"算法": r["method"], "RPR": f"{r['rpr']:.4f}", "平均TIS/min": cell("average_time_in_system"), "Cmax/min": cell("makespan"), "ΣCj2均值（实例间标准差）": cell("objective"), "阶段1利用率/%": cell("utilization_stage1_percent"), "阶段2利用率/%": cell("utilization_stage2_percent")}
            lines.append("| " + " | ".join(values.values()) + " |")
            formatted.append({"规模": f"J{scale}", **values})
        lines.extend(["", f"注：每算法{selected[0]['instances']}个实例，每实例{selected[0]['repeats_per_instance']}次运行。括号内为实例间样本标准差；利用率的标准差单位为百分点。目标值的算法内部平均标准差分别为：" + "；".join(f"{r['method']}：{r['objective_mean_within_instance_sd']:,.2f}" for r in selected) + "。", ""])
    write_csv(output / "tables4_7_paper.csv", formatted)
    lines.extend(["## 表8 配对统计", "", "Wilcoxon双侧符号秩检验以相同实例两算法的重复均值之差为输入；每规模n=20，Overall n=80。中位增益Bootstrap按实例成对重采样10,000次；Overall在规模内分层重采样以保持20/20/20/20构成。Holm同时校正四规模与Overall五个检验，全部规模未完成时不填写校正p值。零差值按wilcox规则处理。", "", "| 规模 | 实例数 | 中位增益/% | Bootstrap 95% CI/% | Wilcoxon p | Holm p | Rank-biserial | W/T/L |", "|---|---:|---:|---|---:|---:|---:|---|"])
    for r in tests:
        adjusted = f"{r['holm_p_five_tests']:.4g}" if r["holm_p_five_tests"] is not None else "待全部规模完成"
        lines.append(f"| {r['scale']} | {r['instances']} | {r['median_gain_percent']:.3f} | [{r['bootstrap_ci95_low_percent']:.3f}, {r['bootstrap_ci95_high_percent']:.3f}] | {r['wilcoxon_p_two_sided']:.4g} | {adjusted} | {r['rank_biserial']:.3f} | {r['W_T_L']} |")
    lines.extend(["", "该检验支持的是跨实例的算法性能差异，不意味着每个给定实例均显著更优。每个实例的运行间稳定性应查看instance_mean_within_sd.csv中的目标值标准差；不能将80×R个运行视为独立实例。Overall混合规模，效应解释仍以分规模结果为主。", "", "## CP-SAT调用与时间占比", "", "CP-SAT时间占比按同一规模全部IBS-CABC运行的CP-SAT墙钟时间之和除以完整求解墙钟时间之和计算。单模型时间上限来自锁定配置；未用完的末次修复预留时间归还BFC/BSC主搜索，精英档案保留修复所得解。", "", "| 规模 | 运行数 | 适用运行数 | 平均调用次数 | 单模型上限/s | 平均CP时间/s | CP总时间占比/% | 最大单次调用/s | 平均修复增益 | 平均归还搜索时间/s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in cp_rows:
        lines.append(f"| J{row['jobs']} | {row['runs']} | {row['applicable_runs']} | {row['mean_model_calls_per_run']:.2f} | {row['cp_sat_per_model_time_limit_seconds']:.2f} | {row['mean_cp_wall_seconds_per_run']:.3f} | {row['aggregate_cp_wall_time_share_percent']:.3f} | {row['max_single_solver_call_seconds']:.3f} | {row['mean_cp_objective_gain_per_run']:.2f} | {row['mean_returned_search_seconds']:.3f} |")
    lines.extend(["", "## 参考基线与运行时间", "", "目前未取得FFCBFCFS的可核验逐实例结果，本多种子主统计不混入文献聚合均值。若在论文保留该行，必须明确为文献参考；若补齐同80实例的确定性结果，只需每实例一次，并统一重新计算RPR的分母集合。", "", f"所有方法保持预先锁定的参数和名义墙钟预算，加载/解析在求解器计时外；记录完整求解调用的实际墙钟时间、CPU时间、预算超时和欠用时间。IBS-CABC仅修正了CP-SAT提前结束后剩余预留时间未归还的问题，未改变任何基线参数或结果筛选规则。外层{manifest['core']['config']['workers']}进程，内部数值线程与CP-SAT线程数均1；运行时避免同时启动多个正式批次挤占资源。", "", "all_repetitions.csv为运行级数据；instance_mean_within_sd.csv为实例级重复均值与标准差；tables4_7_numeric.csv包含全部精度、两种标准差、运行时间、RPR和额外Gap/Gain指标；cp_sat_diagnostics.csv报告CP-SAT调用及总时间占比；algorithm_parameters.json锁定四算法参数；coverage.csv记录完整性，integrity.json记录统计口径与来源。原始检查点包含最终调度和搜索轨迹，后续可以据此汇总多种子收敛曲线。"])
    (output / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"[report] {output / 'report_zh.md'} ({audit['status']}, {len(records)}/{audit['expected_runs']} runs)", flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_report(args.runs, args.output)


if __name__ == "__main__":
    main()
