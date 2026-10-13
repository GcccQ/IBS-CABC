from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cssd_ibscabc.experiments import ExperimentRunner, load_experiment_config
from cssd_ibscabc.reporting import build_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "e7_cp_sat_ablation.yaml"
EXPERIMENT_PREFIX = "e7_cp_sat_ablation_"


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_directories(output_root: Path) -> list[Path]:
    if not output_root.is_dir():
        return []
    return sorted(
        (
            path
            for path in output_root.glob(f"{EXPERIMENT_PREFIX}*")
            if path.is_dir() and (path / "manifest.json").is_file()
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )


def _is_complete(run_directory: Path) -> bool:
    manifest = _read_json(run_directory / "manifest.json")
    progress = _read_json(run_directory / "progress.json")
    return (
        int(manifest.get("task_count_pending", -1)) == 0
        and int(manifest.get("task_count_failed", -1)) == 0
        and str(progress.get("status", "")) == "finished"
    )


def _latest_incomplete(output_root: Path) -> Path | None:
    return next(
        (path for path in _run_directories(output_root) if not _is_complete(path)),
        None,
    )


def _latest_complete(output_root: Path) -> Path | None:
    return next(
        (path for path in _run_directories(output_root) if _is_complete(path)),
        None,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or automatically resume the isolated E7 CP-SAT ablation. "
            "Completed task rows are durable checkpoints."
        )
    )
    parser.add_argument("--data-root", default=str(PROJECT_ROOT.parent / "dataset"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--new",
        action="store_true",
        help="start a new run even if an earlier checkpoint exists",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    os.chdir(PROJECT_ROOT)
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_experiment_config(CONFIG_PATH)
    runner = ExperimentRunner(
        config,
        Path(args.data_root).resolve(),
        output_root,
        workers=args.workers,
    )

    resume_directory: Path | None = None
    if not args.new:
        resume_directory = _latest_incomplete(output_root)
        if resume_directory is None:
            completed = _latest_complete(output_root)
            if completed is not None:
                report = build_report(completed)
                print(f"[complete] 已存在完整实验：{completed}")
                print(f"[report] {report}")
                print("如需重新开始一轮，请增加 --new。")
                return 0

    if resume_directory is not None:
        print(f"[resume] 自动续跑：{resume_directory}")
    else:
        print("[new] 未发现可续跑检查点，创建新实验。")

    try:
        run_directory = runner.run(resume_directory)
    except KeyboardInterrupt:
        checkpoint = resume_directory or _latest_incomplete(output_root)
        print("\n[interrupted] 已完成任务均已落盘。")
        if checkpoint is not None:
            print(f"[checkpoint] {checkpoint}")
        print("再次执行同一命令即可自动续跑；正在运行但未完成的单个任务会重跑。")
        return 130

    report = build_report(run_directory)
    print(f"[complete] {run_directory}")
    print(f"[report] {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
