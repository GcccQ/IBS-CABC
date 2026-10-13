# 论文同路径数据划分

原始实例只从 `G:\article\dataset` 读取，程序不会复制或改写数据集。

- `splits/paper_small25.txt`：20 个 J25 实例，对应论文表 3；
- `splits/paper_real365.txt`：365 个真实日实例，对应论文表 4；
- `splits/paper_fixed.txt`：J100、J250、J500、J1000 各 20 个，对应论文表 5；
- `splits/paper_all465.txt`：上述三个集合的无重复并集；
- `splits/smoke.txt`：只用于快速检查，不产生论文结论。

`cssd-ibscabc build-splits` 会检查各组数量及并集关系。当前公开
`RealWorldInstances365` 数据的实际规模范围为 70–431；论文正文写作
70–437。所有实验统计以文件内容、实例 SHA-256 和实际范围为准。
