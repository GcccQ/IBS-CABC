# Kraul FFCBSPT / GA 的独立复现

本项目只负责原作者的 **FFCBSPT** 与 **Kraul-GA**，不导入、修改或调用现有
`cssd-dbd-case-study` 中的 IBS-CABC、ALNS、TS 实现。两个实验完成后，可用只读合并脚本生成同一口径的五算法表格和阶段 1/阶段 2 曲线。

除 Real365 的 DBD 案例外，项目还提供与 `cssd-kg-edr` 隔离的固定规模
Kraul-GA 入口，用于 J25/J100/J250/J500/J1000 共 100 个公开实例的实例级配对统计和收敛分析。

实现依据为原作者公开的
[Genetic_Algorithm_Example.ipynb](https://github.com/SebastianKraul/A-two-stage-hybrid-flow-shop-formulation-for-sterilization-processes-in-hospitals)
及[正式论文](https://www.sciencedirect.com/science/article/pii/S0957417424024916)。正式配置会记录并校验本地作者 Notebook、配置、源代码和 91 日数据窗口的 SHA-256，防止实验过程中口径漂移。

## 已锁定的复现口径

- DBD：每天单独读取、调度并清零机器状态，不传递前一日排程。
- 实例：公开 RealWorldInstances365 中 seed 233–323 的连续 91 日代理窗口，共 22,927 件器械；它不是论文未公开的 22,934 件原始三个月窗口。
- 设备：阶段 1 为 12 台、容量 15；阶段 2 为 5 台、容量 11,000。
- 加工时间：按照作者的 Sule 规则，将对应阶段家族准备时间加到每个作业加工时间。
- FFCBSPT：以 ERD/FCFS 作业序列进入作者的共同解码器，按家族首次适配装批，并使用动态 SPT 规则将批分配给并行设备。
- Kraul-GA：种群 100、轮盘赌、TPX、插入变异、交叉率 0.95、变异率 0.40、固定 25% 精英、改进时插入局部搜索、50 代重启，正式时限为
  `min((5000 + J²(m1+m2)/2)/1000, 1800)` 秒/日。
- 单种子复现配置使用随机种子101；论文表10与图6—图7的多种子实验统一使用随机种子1—10。

公开 Notebook 示例常量是交叉率 0.80 与 flip 变异，而论文最终标定值是交叉率 0.95 与 insert 变异。项目分别保留 `notebook_example` 和 `paper_final` 两个配置，正式实验只允许后者。

还有两处源代码事实不能隐去：

1. 公开 Notebook 的轮盘赌概率直接使用 `objective / sum(objective)`，对最小化问题会提高较差解的抽中概率。正式复现保留这一行为，并在 manifest 中标注；`corrected_selection` 只用于诊断，不能进入正式配置。
2. Notebook 的 `calculate_lower_bound` 末行截断为 `lower_bound = processin`。该下界不参与 FFCBSPT/GA 的候选接受与最终排程，本项目不猜测补写它。

另外，Notebook 用值为 0 的矩阵单元填充不存在的批，并按“释放时刻不等于 0”过滤填充位；公开数据本身却有真实的 0 时刻释放作业。重构使用显式批列表，因此会正常调度真实的 0 时刻批，避免把作业静默丢失。这是保证模型可行性的表示修复，manifest 会记录它。

更完整的源码到实现映射见 [reproduction_contract.md](docs/reproduction_contract.md)。

## 安装与自检

在 PowerShell 中运行：

```powershell
cd <package-root>\kraul-dbd-reproduction
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q
python scripts\validate_paper.py --data ..\dataset\RealWorldInstances365 --output outputs\paper_table4_validation.json
```

365 个公开实例的 FFCBSPT 交叉验证应返回 `"status": "PASS"`。当前公开压缩包得到的 TIS 样本标准差与论文舍入值存在约 5.15 分钟差异；目标均值、TIS 均值、完工期和两阶段利用率均在严格容差内，验证文件会完整保留该差异而不隐藏。

## 先跑冒烟实验

```powershell
cd <package-root>\kraul-dbd-reproduction
python scripts\run_dbd.py --config configs\smoke.yaml
```

它只运行两天、四个任务，通常几秒内完成，并自动生成全部表格和曲线。

## 正式运行 91 日 DBD 实验

先审核任务契约：

```powershell
python scripts\run_dbd.py --config configs\dbd_proxy91.yaml --dry-run
```

确认输出为 91 日、22,927 件、182 个任务、91 个 GA 任务、8 workers 后开始：

```powershell
python scripts\run_dbd.py --config configs\dbd_proxy91.yaml
```

表 10、图 6 和图 7 的多种子实验使用以下配置。FFCBSPT 为确定性规则，仅执行一次；
Kraul-GA 使用随机种子 1--10：

```powershell
python scripts\run_dbd.py --config configs\dbd_proxy91_10seeds.yaml --dry-run
python scripts\run_dbd.py --config configs\dbd_proxy91_10seeds.yaml
```

任务合同包含 91 个 FFCBSPT 任务和 `91 x 10 = 910` 个 Kraul-GA 任务，共 1001 个。

控制台在每个任务完成时输出一行；即使某个 GA 日实例长时间未完成，也会每 15 秒输出一次 heartbeat。磁盘上的 `progress.json` 是权威进度，不受 Windows 控制台“选择文本后暂停显示”的影响。

正式配置的 GA 名义总预算约 57,966.54 秒，8 进程理想下界约 2.01 小时；局部搜索、代末时限检查、系统负载和进程竞争会使实际时间更长。

若手动中断，已经落盘且哈希一致的任务不会重算。使用终端最后打印的目录恢复：

```powershell
python scripts\run_dbd.py --config configs\dbd_proxy91.yaml --resume outputs\你的实验目录
```

多种子实验恢复时必须使用 `configs\dbd_proxy91_10seeds.yaml` 和原实验目录。

如只需重新生成报告：

```powershell
python scripts\report_dbd.py --experiment outputs\你的实验目录
```

## 固定规模 Kraul-GA 实验

固定规模实验只运行 Kraul-GA，不重复运行其他项目中的算法。正式口径如下：

- 实例：J25/J100/J250/J500/J1000 各 20 个，共 100 个；
- 设备与容量：直接读取各公开实例的六值文件头；J25 为 3/2 台设备，其余规模为 12/5 台，容量均为 15/11000；
- 目标：`sum(C_j2)`；
- 时间预算：`min((5000 + J²(m1+m2)/2)/1000, 1800)` 秒，`tau=1`；
- 随机种子：101；
- Kraul-GA：未修改的 `paper_final` 参数；
- 并行度：8 个独立进程，每个进程内部数值库线程数固定为 1。

先做合同检查：

```powershell
cd <package-root>\kraul-dbd-reproduction
python scripts\run_fixed_scale.py --config configs\fixed_scale_all.yaml --dry-run
```

应显示 100 个实例和 100 个任务，五个规模各 20 个。名义串行预算为
84,656.25 秒，8 进程理想下界约 2.94 小时。确认后正式运行：

```powershell
python scripts\run_fixed_scale.py --config configs\fixed_scale_all.yaml
```

每完成一个实例会立即输出一行；长任务期间每 15 秒输出 heartbeat，并同步写入
`progress.json`。若中断，使用终端打印的实验目录恢复：

```powershell
python scripts\run_fixed_scale.py `
  --config configs\fixed_scale_all.yaml `
  --resume outputs\你的固定规模实验目录
```

正式运行完成后自动生成：

- `tables/instance_level_kraul_ga.csv`：按实例聚合的数据，可与 IBS-CABC 按实例 ID 配对；
- `tables/scale_summary_kraul_ga.csv`：五个规模的目标、TIS、Cmax、利用率等均值和样本标准差；
- `tables/paper_table5_kraul_ga_comparison.csv`：与原论文表 5 的逐规模数值对照；
- `tables/ga_convergence_raw.csv`：逐实例逐代的实际用时、评价次数、当代最优和历史最优；
- `tables/ga_convergence_by_scale.csv`：0%–100% 名义预算网格上的五规模平均收敛数据；
- `figures/kraul_ga_convergence_fixed_scale.png` 与 `.pdf`：论文用固定规模收敛图；
- `report/paper_table5_kraul_ga_validation.json`：原论文结果接近程度及 J250 TIS 口径审计；
- `report/integrity.json`：实例数、可行性、论文接近程度和收敛历史完整性检查。

收敛图先在每次运行内计算相对于首次完整种群评价后最优值的历史最优改善率，再按规模求均值；横轴为已消耗名义时间预算比例。这样不会让 J1000 的绝对目标量级掩盖 J25 的变化。需要重新生成表格和图片时运行：

```powershell
python scripts\report_fixed_scale.py --experiment outputs\你的固定规模实验目录
```

## 正式实验会生成什么

- `tables/table_4_9_ffcbspt_kraul_ga.csv`：表 4-9 所需两行聚合数据；
- `tables/table_9_ffcbspt_kraul_ga_seed_summary.csv`：Kraul-GA 的10种子均值与跨种子样本标准差；
- `tables/day_level_runs.csv`：逐日目标、等待、利用率、耗时、评价次数；
- `tables/job_level_schedules.csv`：每件器械的两个阶段开始/完成、机器和批次；
- `tables/hourly_stage1_dbd.csv`、`hourly_stage2_dbd.csv`：总量与每天平均量；
- `tables/ga_convergence.csv`：每个日实例的逐代最优值；
- `figures/figure_4_7_*.png`、`figure_4_8_*.png`：阶段 1/阶段 2 小时曲线；
- `raw/results.jsonl`：可断点恢复的原始记录和完整批/作业排程；
- `report/integrity.json`：完整性、配对和指标恒等式审计结果。

## 与 IBS-CABC、ALNS、TS 合并

FFCBSPT/Kraul-GA 正式实验结束后运行：

```powershell
python scripts\merge_five_algorithms.py `
  --kraul-experiment outputs\你的正式实验目录 `
  --three-experiment ..\cssd-dbd-case-study\outputs\你的三算法10种子实验目录 `
  --output outputs\five_algorithm_merged_10seeds
```

合并前会强制核对两边的完整性状态、数据 SHA-256、91 日实例顺序、每个实例 SHA-256、设备配置、作业数、日长度和种子集合。只有全部一致才会输出五算法表 4-9、图 4-7、图 4-8 及 `merge_integrity.json`。
表 10 中随机算法的括号为10个完整三个月运行之间的样本标准差。阶段 1、阶段 2 排队图的
中心线为10种子均值，阴影为对种子均值进行10,000次百分位 Bootstrap 得到的95%置信区间；
确定性 FFCBSPT 不绘制置信带。
