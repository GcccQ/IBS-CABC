# IBS-CABC 论文代码与复现

本仓库公开论文正式实验版本的核心算法、对比算法、配置、实例划分、历史结果与统计核验脚本。核心求解代码保持原样，MIT 许可证适用于本仓库原创代码；第三方数据与参考 Notebook 从原作者固定版本下载并核验，不包含在 Git 仓库内。

论文正文及图表图片/PDF不上传到Git仓库。绘图脚本仅在本地 `reruns/` 生成图表，GitHub自动检查只上传核验日志；发布范围由 `tools/check_publication_scope.py` 检查。

## 安装与核验（Windows PowerShell）

建议使用64位 Python 3.10 或 3.11。在仓库根目录执行：

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt -c requirements-constraints.txt
& .\.venv\Scripts\python.exe -m pip install -e ./cssd-kg-edr -e ./cssd-comparison-baselines -e ./cssd-dbd-case-study -e ./kraul-dbd-reproduction
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe tools/prepare_data.py
& .\.venv\Scripts\python.exe tools/prepare_results.py
& .\.venv\Scripts\python.exe tools/verify_files.py
& .\.venv\Scripts\python.exe tools/check_delivery.py --tests
& .\.venv\Scripts\python.exe tools/rebuild_saved.py --target all
```

首次下载需要联网，547项实验输入均按原实验 SHA-256 校验。已保存原始结果经过无损压缩，解压后再次核验原始哈希。最后两条命令核验论文结果并重新生成图5、表10及图6–7，输出到 `reruns/`。

## 先运行小实验

```powershell
& .\.venv\Scripts\python.exe tools/run_experiment.py --experiment fixed --smoke
& .\.venv\Scripts\python.exe tools/run_experiment.py --experiment dbd-three --smoke
& .\.venv\Scripts\python.exe tools/run_experiment.py --experiment dbd-kraul --smoke
```

完整重跑时将 `--smoke` 去掉；CP-SAT移除对照使用 `--experiment cp`。正式运行前可用 `--dry-run` 核验固定规模、DBD三算法和DBD规则/GA任务数，依次应为8,000、2,730、1,001。CP-SAT正式对照共120次运行，不支持此参数。

历史固定规模实验使用72个外层进程；便利入口使用24个外层进程。原始72进程配置和直接调用方法见 [English README](README.md)。固定规模历史完整实验约耗时1,746.5分钟。

## 可复现范围

- 历史8,000条固定规模运行指标、3,731条DBD记录和120条消融记录可核验。保留的结果足以复算论文统计，但当前归档缺少固定规模逐次排程的服务器 `checkpoints/`。
- 随机种子、实例、参数和依赖版本固定；墙钟预算算法在不同机器与负载下可能得到不同数值，不能承诺逐项一致。
- Kraul参考实现沿用NumPy默认非稳定排序；相同键值的排列顺序也可能随CPU变化，因此FFCBSPT的确定性限于同一运行环境。仓库保留原算法行为，精确解码回归测试使用无并列键的小实例；论文数值以归档结果核验。
- DBD是Real365公开数据seed 233–323的91日代理窗口，共22,927件作业；按日独立调度，不模拟跨日机器占用。
- 源码位置见 [核心代码说明](docs/CORE_CODE.md)，表图关系见 [结果映射](docs/PAPER_RESULTS.md)，本次实际验证见 [验证记录](VALIDATION.md)。
