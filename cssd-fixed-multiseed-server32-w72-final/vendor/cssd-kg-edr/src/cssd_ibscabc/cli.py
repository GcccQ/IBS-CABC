from __future__ import annotations

import argparse

from .experiments import (
    ExperimentRunner,
    apply_selected_parameters,
    load_experiment_config,
)
from .experiments.splits import build_published_splits
from .io import audit_dataset
from .reporting import build_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cssd-ibscabc", description="IBS-CABC CSSD 批调度实验工具"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="审计九列数据集")
    audit.add_argument("--data-root", required=True)
    split = commands.add_parser("build-splits", help="生成论文表 3--5 的精确实例划分")
    split.add_argument("--data-root", required=True)
    split.add_argument("--output", default="data/splits")
    run = commands.add_parser("run", help="运行可续跑实验并显示进度条")
    run.add_argument("--config", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--output", default="outputs")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--resume")
    run.add_argument(
        "--parameter-file",
        help="E1 report generated selected_parameters.yaml for E2-E6",
    )
    report = commands.add_parser("report", help="生成论文表 3--5 同结构的中文报告")
    report.add_argument("--experiment", required=True)
    report.add_argument("--report-config")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        result = audit_dataset(args.data_root)
        print(
            f"实例={result.instance_count}, 作业={result.job_count}, "
            f"规模={result.job_min}..{result.job_max}, Family={result.family_min}..{result.family_max}"
        )
        if result.invalid_files:
            print("数据审计失败：")
            print("\n".join(result.invalid_files))
            return 1
        print("数据审计通过。")
        return 0
    if args.command == "build-splits":
        splits = build_published_splits(args.data_root, args.output)
        print(", ".join(f"{name}={len(paths)}" for name, paths in splits.items()))
        return 0
    if args.command == "run":
        config = load_experiment_config(args.config)
        if args.parameter_file:
            config = apply_selected_parameters(config, args.parameter_file)
        runner = ExperimentRunner(
            config, args.data_root, args.output, args.workers
        )
        output = runner.run(args.resume)
        print(f"实验完成：{output}")
        return 0
    if args.command == "report":
        output = build_report(args.experiment, args.report_config)
        print(f"报告已生成：{output}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
