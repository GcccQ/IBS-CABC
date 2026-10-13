from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cssd_baselines.reporting import (  # noqa: E402
    ExperimentIntegrityError,
    print_summary,
    write_four_algorithm_reports,
    write_reports,
)
from cssd_baselines.convergence import (  # noqa: E402
    ConvergenceDataError,
    write_convergence_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    results_path = output / "raw" / "results.jsonl"
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline_summaries = write_reports(output, rows)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    formal_protocol = (
        set(manifest.get("algorithms", [])) == {"standard_abc", "alns", "ts"}
        and manifest.get("max_evaluations_override") is None
        and manifest.get("max_instances_per_size") is None
        and float(manifest.get("time_scale", 0.0)) == 1.0
    )
    if formal_protocol and int(manifest.get("task_count", 0)) != 300:
        print(
            "正式结果完整性检查失败：当前五规模协议要求300条基线结果；"
            f"该目录清单为{manifest.get('task_count', 0)}条。",
            file=sys.stderr,
        )
        return 2
    formal_contract = formal_protocol and int(manifest.get("task_count", 0)) == 300
    if formal_contract:
        reference_manifests = [
            Path(str(value))
            for value in manifest.get(
                "reference_manifests", [manifest["reference_manifest"]]
            )
        ]
        try:
            reference_result_paths = [
                reference.parent / "raw" / "results.jsonl"
                for reference in reference_manifests
            ]
            summaries = write_four_algorithm_reports(
                output,
                rows,
                reference_result_paths,
            )
            convergence_audit = write_convergence_reports(
                output, rows, reference_result_paths
            )
        except (ExperimentIntegrityError, ConvergenceDataError) as error:
            print(f"正式结果完整性检查失败：{error}", file=sys.stderr)
            return 2
        print_summary(summaries, title="四算法固定规模正式实验汇总")
        print(f"convergence_status={convergence_audit['status']}")
    else:
        print_summary(baseline_summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
