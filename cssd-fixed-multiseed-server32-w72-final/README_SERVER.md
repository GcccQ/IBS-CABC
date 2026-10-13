# 32核服务器运行说明（8,000次固定规模多随机种子实验）

该文件夹是自包含服务器包，只包含四个求解器的锁定源码、80个正式实例、实验配置、统计脚本和测试，不包含本地旧结果或冒烟测试结果。

## 固定实验协议

- J100、J250：20实例 × 30种子 × 4算法，各2,400次；
- J500、J1000：20实例 × 20种子 × 4算法，各1,600次；
- 合计8,000次，每算法2,000次；
- 外层并发72进程，每个算法任务仅使用1个数值线程，CP-SAT也固定为1线程；
- 进程池每完成至多576个任务（72个进程 × 每进程8个任务）后重建一次，以限制长时间运行中的内存高水位；
- 所有任务使用与论文相同的墙钟预算、实例、种子和算法参数。

72进程下名义预算的理想下界约28.02小时（1.17天），实际耗时会受机器负载、进程调度、求解器预算超时和I/O影响。本次正式运行完成8,000次任务，日志记录的总耗时为1,746.5 min。

## 首次安装

建议使用Python 3.10或3.11：

```bash
cd /path/to/IBS-CABC_reproducibility_package/cssd-fixed-multiseed-server32-w72-final
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
python -m pytest tests -q
python run.py --config configs/fixed_30_20_server32.yaml --dry-run
```

`--dry-run`必须显示：`tasks: 8000`、`workers: 72`，四个规模任务数依次为2,400、2,400、1,600和1,600。该命令只解析和核验数据，不运行优化算法。

## 后台启动

```bash
chmod +x run_server.sh
mkdir -p logs
nohup bash ./run_server.sh > logs/fixed_30_20_server32_w72.log 2>&1 &
echo $! > logs/fixed_30_20_server32_w72.pid
```

查看日志和结构化进度：

```bash
tail -f logs/fixed_30_20_server32_w72.log
cat outputs/fixed_30_20_server32_w72/progress.json
```

如服务器提供`tmux`，也可在`tmux`会话中执行`./run_server.sh`。不要同时启动第二个正式进程；输出目录锁会拒绝并发写入。

## 中断与续跑

Ctrl+C、SSH断开或服务器重启后，再次执行完全相同的命令即可：

```bash
source .venv/bin/activate
nohup bash ./run_server.sh > logs/fixed_30_20_server32_w72.log 2>&1 &
```

脚本检测到`outputs/fixed_30_20_server32_w72/manifest.json`后会自动进入`--resume`模式。已原子保存并通过SHA-256校验的任务不会重算；中断时尚未完成的一次求解会从该任务起点重新执行。开始实验后不得修改配置、源码、依赖版本、数据、进程数或输出目录。

## 完成与汇总

全部8,000次完成后会自动生成：

```text
outputs/fixed_30_20_server32_w72/report/
```

也可重新执行只读汇总：

```bash
python report.py \
  --runs outputs/fixed_30_20_server32_w72 \
  --output outputs/fixed_30_20_server32_w72/report
```

完成后请完整下载输出目录，包括`manifest.json`、`task_manifest.json`、`checkpoints/`、`progress.json`和`report/`。不要只下载汇总CSV，否则无法审计逐次运行和断点完整性。

## 当前交付归档的范围

本交付包保存了正式配置、锁定源码、80个实例、运行清单、日志以及
`report/all_repetitions.csv`中的8,000条逐次指标，并可据此核对论文表4—表8。
服务器端逐任务`checkpoints/`未包含在当前本地归档中，因此该归档不能恢复每次运行的完整批结构与排程；
若仍可访问原服务器，应补充下载
`outputs/fixed_30_20_server32_w72/checkpoints/`。
