# ALNS与TS基线实现

本目录保留ALNS与TS的独立实现、测试及原始开发入口。`cssd-dbd-case-study`调用这里的算法实现；固定规模正式实验则使用`../cssd-fixed-multiseed-server32-w72-final/`中锁定的vendored版本，以确保四种算法共享同一任务合同、解码器、时间预算和单线程设置。

最新版论文的固定规模证据是J100、J250各20个实例×30个种子，J500、J1000各20个实例×20个种子，共8,000次运行。正式结果、配置和审计均不位于本目录，请以包根目录的`README_REPRODUCE.md`和`PAPER_RESULT_MAPPING.md`为准。旧单种子输出已从交付包移除，避免与当前多种子结果混用。

安装与测试：

```bash
python -m pip install -e .
python -m pytest -q
```

核心源码位于`src/cssd_baselines/`。`configs/`和`scripts/`保留为开发阶段入口，不构成最新版论文结果的证据来源。
