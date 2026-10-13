from pathlib import Path

from cssd_ibscabc.experiments import ExperimentRunner, load_experiment_config

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = ExperimentRunner(
        load_experiment_config(root / "configs" / "smoke.yaml"),
        root.parent / "dataset",
        root / "outputs",
        1,
    ).run()
    print(f"冒烟测试完成：{output}")
