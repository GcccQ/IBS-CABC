# CSSD 三个月 DBD 案例实验

这是一个与既有固定规模实验隔离的案例实验项目，只负责以下三种 DBD
（day-by-day）方案：

- IBS-CABC
- ALNS
- TS

项目不实现 FFCBSPT 或 Kraul-GA。FFCBSPT 可由原作者代码单独复现，随后将结果并入表 4-9。

## 隔离边界

本目录拥有独立的配置、案例日清单、运行清单、原始结果、统计表和图片。算法实现以只读依赖方式从
`../cssd-kg-edr/src` 和 `../cssd-comparison-baselines/src` 导入；运行器不会修改这两个项目的源码、
配置或结果目录。所有新结果默认写入本项目的 `outputs/`。

每个 DBD 任务只处理一个日实例，日初设备为空，不跨日组批；外层进程池负责并行，单个任务的内部
数值线程固定为 1。

## 正式协议

[`configs/dbd_proxy91.yaml`](configs/dbd_proxy91.yaml) 锁定：

- 连续 91 日：RealWorldInstances365 的 seed 233–323；
- 22,927 件器械，阶段 1/2 设备数为 12/5；
- 三种算法均使用随机种子 101；
- 每个“日 × 算法”任务采用第 4.2 节时间预算：
  `min((5000 + J^2 × (m1 + m2) / 2) / 1000, 1800)` 秒；
- 共 `91 × 3 × 1 = 273` 个独立任务。

91 日正式预算合计约为每种算法 57,966.5 秒，三种算法共 48.3 个任务小时；8 个外层进程的理论下限
约为 6.04 小时，实际耗时还会受任务负载不均和机器性能影响。

公开的 RealWorldInstances365 不包含原论文三个月案例的逐日原始文件。该配置使用最接近论文规模的
连续 91 日代理窗口，不能表述为论文 22,934 件案例数据的精确复现。

## 运行

在新的 PowerShell 窗口中进入隔离项目；若当前环境尚未安装依赖，再执行可编辑安装：

```powershell
Set-Location <package-root>\cssd-dbd-case-study
python -m pip install -e ".[dev]"
```

先执行快速完整性检查和冒烟实验：

```powershell
python scripts/run_dbd.py --config configs/dbd_proxy91.yaml --dry-run
python scripts/run_dbd.py --config configs/smoke.yaml --workers 1
```

确认无误后启动正式实验：

```powershell
python scripts/run_dbd.py --config configs/dbd_proxy91.yaml
```

论文表 10 与图 6/图 7 的多随机种子版本使用种子 1--10：

```powershell
python scripts/run_dbd.py --config configs/dbd_proxy91_10seeds.yaml --dry-run
python scripts/run_dbd.py --config configs/dbd_proxy91_10seeds.yaml
```

该配置运行 IBS-CABC、ALNS 和 TS，共 `91 x 3 x 10 = 2730` 个任务。每个种子先独立
汇总完整 91 日窗口，再在种子之间计算均值与样本标准差。

运行中会持续更新 `progress.json`，成功任务逐条追加到 `raw/results.jsonl`。中断后必须使用原配置、
相同代码和相同进程数恢复：

```powershell
python scripts/run_dbd.py --config configs/dbd_proxy91.yaml `
  --workers 8 --resume outputs/<实验目录>
```

多种子实验恢复时将配置替换为 `configs/dbd_proxy91_10seeds.yaml`，并保持原实验的
8 个 workers 不变。

运行目录会在程序启动时立即打印。正式运行期间不要修改配置、算法源码或 91 个实例文件，也不要让
计算机进入睡眠状态；时间预算采用单调墙钟计时，睡眠时间也会消耗预算。同一实验目录只能有一个
写入进程，重复执行相同的 `--resume` 会被目录锁拒绝。

建议在第二个 PowerShell 窗口中用轮询方式查看原子更新的进度文件：

```powershell
$run = Get-ChildItem .\outputs -Directory -Filter 'dbd_case_proxy91_three_algorithms_*' |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty FullName

while ($true) {
  Clear-Host
  Get-Content "$run\progress.json" -Raw -Encoding UTF8
  Start-Sleep -Seconds 15
}
```

按 `Ctrl+C` 可以停止计算或停止监控。计算进程收到中断后会保存 `interrupted` 状态；恢复命令只会
提交尚未成功的任务。若实例内容、配置、代码、依赖版本或 worker 数发生变化，恢复会被拒绝，必须
重新建立实验目录。

也可对已有实验目录重新生成报告：

```powershell
python scripts/report_dbd.py --experiment outputs/<实验目录>
```

报告命令会重新认证 `task_manifest.json` 与每一条原始结果；仅当
`report/integrity.json` 的状态为 `PASS` 时返回退出码 0，否则仍保留诊断报告并返回退出码 2。

## 结果文件

- `report/integrity.json`：任务配对、实例哈希、预算、可行性、指标恒等式和小时计数审计；
- `task_manifest.json`：273 个任务的实例、算法、参数、种子和预算清单；
- `tables/table_4_9_dbd_three_algorithms.csv`：表 4-9 中三种算法的 DBD 行；
- `tables/table_9_dbd_three_algorithms_seed_summary.csv`：种子 1--10 的三个月指标均值与跨种子样本标准差；
- `tables/day_level_runs.csv`：逐日审计表；
- `tables/hourly_stage1_dbd.csv`、`hourly_stage2_dbd.csv`：小时总量与日均量；
- `figures/figure_4_7_dbd_stage1.png`、`figure_4_8_dbd_stage2.png`：按原论文总量口径绘制；
- `report/dbd_case_summary_zh.md`：可直接核对的中文汇总。

时间均值与样本标准差由 91 日全部器械的充分统计量精确合并，不对“每日均值”再次求平均。表4-9的
阶段利用率按每日活动时间窗利用率做算术平均，以对齐原论文；汇总 CSV 另提供
`*_utilization_pooled_active_horizon_percent`，即总加工分钟数除以设备数与各日活动时间窗之和，可用于
核对式（4-10）。小时 CSV 中的 `*_average_per_day` 列对应式（4-11）和式（4-12）；图片采用三个月
小时总量，以保持原论文图形口径。
