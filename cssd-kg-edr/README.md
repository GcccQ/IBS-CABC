# IBS-CABC：CSSD 两阶段批调度

本工程只包含本文提出的 IBS-CABC，不再内置任何对比算法。作者的
Kraul-GA 源码应在独立目录中复现，再按实例标识与本工程结果配对。

主目标保持为最小化阶段二完工时间总和 `ΣCj2`，不增加多目标或额外惩罚项。

## 算法结构

IBS-CABC 使用完整两阶段直接批结构：

`X = (Π1, G1, Σ1, Π2, G2, Σ2)`

- `Πk`：阶段 `k` 的族内作业次序；
- `Gk`：批边界；
- `Σk`：全局批优先级。

同一食物源种群上严格交替执行批形成邻域 BFC 与批排序邻域 BSC，并使用
瓶颈感知路由、共享多样性精英档案、关键批强化、跨阶段对齐和完整两阶段
统一解码。阶段二初始批由释放期—拥堵权衡动态规划生成，但采用延迟门控：
只有在主搜索完成后仍优于当前解时才注入，避免启发式种子破坏搜索轨迹。
J≤600 时，主搜索后可执行 IBS-CABC 自身的联合批修复与族分解分配修复；
每个候选都重新计算完整指标并接受独立可行性校验。

## 安装与检查

```powershell
cd <package-root>\cssd-kg-edr
python -m pip install -e ".[dev]"
python scripts\validate_dataset.py --data-root ..\dataset
python scripts\build_splits.py --data-root ..\dataset --output data\splits
python scripts\run_smoke.py
pytest -q
```

数据审计应得到 465 个实例、126,931 个作业。三个论文集合分别为
20、365、80 个实例。

## 论文同路径主实验

参考论文没有报告独立随机重复次数，因此主配置按每个实例一次运行：

```powershell
cd <package-root>\cssd-kg-edr
$Data = "..\dataset"
$Workers = 8

python scripts\run_experiment.py `
  --config configs\paper_main.yaml `
  --data-root $Data `
  --output outputs `
  --workers $Workers
```

任务总数为 465。每个实例使用论文时间预算：

`min((5000 + J² × (m1+m2)/2)/1000, 1800)` 秒。

中断后续跑时，配置、代码、数据和 worker 数必须与首次运行完全一致：

```powershell
$Run = ".\outputs\paper_aligned_all465_<实际时间>"

python scripts\run_experiment.py `
  --config configs\paper_main.yaml `
  --data-root $Data `
  --output outputs `
  --workers $Workers `
  --resume $Run
```

## 生成论文对照表

```powershell
python scripts\analyze_results.py `
  --experiment $Run `
  --report-config configs\reporting.yaml
```

输出目录：

```text
outputs/<experiment_id>/
├─ manifest.json
├─ progress.json
├─ raw/results.jsonl
├─ aggregate/instance_results.csv
├─ tables/
│  ├─ paper_table3_small.csv
│  ├─ paper_table4_real365.csv
│  ├─ paper_table5_fixed_size.csv
│  ├─ comparison_to_published_ga.csv
│  └─ module_diagnostics.csv
├─ figures/
└─ report/
   ├─ summary_zh.md
   ├─ performance_summary.json
   └─ performance_report.html
```

`comparison_to_published_ga.csv` 使用论文公开的分组均值作直观对照。
逐实例显著性检验必须使用你另外运行的作者源码结果，不能用一张论文均值表
代替配对样本。

## E1-E6 独立实验

与《实验设计.pdf》对应的参数标定、25 作业、365 实例、固定规模、机器网格以及
AIO/DBD 案例实验已经拆分为独立配置。服务器命令、续跑方法和每张论文表的回填
文件见 `docs/E1-E6服务器实验运行与回填说明.md`。

正式实验前建议先运行 `configs/local_precheck.yaml`。原论文逐项对齐结论、已修正
的下界/计时口径、本地 PowerShell 命令及预检输出解释见
`docs/原论文实验对齐审计与本地预检.md`。

## E7：CP-SAT 配对消融

审稿用 CP-SAT 消融与其他实验隔离，比较完整版 IBS-CABC 和仅关闭
`assignment_repair_enabled` 的受控变体。J100、J250、J500 各包含 20 个实例，
两版本共享实例、种子和总墙钟预算，任务按实例成对排列：

```powershell
cd <package-root>\cssd-kg-edr

python scripts\run_cp_sat_ablation.py `
  --data-root ..\dataset `
  --output outputs `
  --workers 8
```

每个完成任务都会立即写入 `raw/results.jsonl`。若关机、关闭终端或按
`Ctrl+C` 中断，再次执行完全相同的命令即可自动找到最新未完成目录并续跑；
当前尚未完成的单个任务会重新执行。为防止证据混合，配置、代码、数据和
`--workers` 必须与首轮保持一致。若完整实验已经存在，脚本只重新生成报告；
确需另起一轮时增加 `--new`。

脚本在全部任务完成后会自动生成表 8-1。也可手工重新生成：

```powershell
$Run = ".\outputs\e7_cp_sat_ablation_<实际时间>"
python scripts\analyze_results.py --experiment $Run
```

正文回填文件为 `tables/table8_1_cp_sat_ablation_zh.csv`；英文详细表、实例级
配对数据、调用与计时诊断、Bootstrap 95% 置信区间及判读说明会同时生成。
