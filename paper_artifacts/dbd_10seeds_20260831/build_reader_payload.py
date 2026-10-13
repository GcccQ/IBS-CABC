"""Package the validated local report for the optional in-app reader."""
import base64
import json
import re
import sqlite3
from pathlib import Path

root = Path(__file__).resolve().parent
data = json.loads((root / "reader_data.json").read_text(encoding="utf-8"))
report = (root / "report_zh.md").read_text(encoding="utf-8-sig")
title = report.splitlines()[0][2:]
sections = re.split(r"(?m)^## ", report)
blocks = [{"id": "title", "type": "markdown", "body": "# " + title}]
paper_fields = ["method", "tis", "wait1", "wait2", "util1", "util2"]
paper_rows = [{"order": i + 1, **dict(zip(paper_fields, row))} for i, row in enumerate(data["paper"])]
source_id = "completed-dbd-audit"
for i, section in enumerate(sections[1:]):
    name, body = section.split("\n", 1)
    if name.startswith("表9："):
        note = body[body.index("注："):]
        blocks.append({"id": f"section-{i}", "type": "markdown", "body": "## " + name})
        blocks.append({"id": "table9-block", "type": "table", "tableId": "table9"})
        blocks.append({"id": "table9-notes", "type": "markdown", "body": note, "sourceId": source_id})
    elif name == "图6与图7":
        parts = re.split(r"!\[图[67]\]\([^\n]+\)", body)
        blocks.append({"id": f"section-{i}", "type": "markdown", "body": "## " + name + "\n" + parts[0], "sourceId": source_id})
        for stage in (1, 2):
            svg = base64.b64encode((root / f"figure_{5+stage}_stage{stage}_10seeds.svg").read_bytes()).decode()
            blocks.append({"id": f"figure-{5+stage}", "type": "html", "body": f'<figure style="margin:0"><img style="display:block;width:100%;height:auto" alt="阶段{stage}前小时平均队长；实线为10种子均值，阴影为95%逐点置信区间" src="data:image/svg+xml;base64,{svg}" /></figure>', "sourceId": source_id})
            blocks.append({"id": f"figure-{5+stage}-notes", "type": "markdown", "body": parts[stage], "sourceId": source_id})
    else:
        blocks.append({"id": f"section-{i}", "type": "markdown", "body": "## " + name + "\n" + body, "sourceId": source_id})
    if name == "配对种子补充核验":
        blocks.append({"id": "seed-scatter", "type": "chart", "chartId": "seed-stage2"})

source = {"id": source_id, "label": "已完成DBD原始实验及逐种子复核", "path": str(root / "audit.json"), "query": {"description": "读取两组3731条已完成日任务，按方法/种子汇总91日，再计算跨种子均值与ddof=1标准差。最终解重放仅用于恢复排队区间，不重新优化。", "tables_used": [str(s["path"]) for s in data["audit"]["raw_sources"]], "filters": ["RealWorldInstances365实例seed233—323", "优化器种子1—10；FFCBSPT确定性参考", "可行且完整任务，无缺失或重复"], "metric_definitions": {"time_metrics": "每个完整种子运行按22927个作业加权平均，再对10个种子等权平均；样本标准差分母9", "utilization": "各日活动时域利用率的算术平均；非24小时分母；再跨种子汇总", "hourly_queue": "等待区间在钟点小时内的重叠分钟/91/60，再对10条完整种子曲线取均值", "ci95": "整条种子曲线重采样10000次，均值2.5%和97.5%分位数；逐点区间，非同时置信带"}}}
payload = {"surface": "report", "manifest": {"version": 1, "surface": "report", "title": title, "description": "完整91日、种子1—10的表9统计及图6/图7，含逐种子指标与审计口径。", "generatedAt": data["audit"]["generated_utc"], "sources": [source], "blocks": blocks, "tables": [{"id": "table9", "title": "表9 10次完整三个月运行的均值（跨种子样本标准差）", "dataset": "table9", "sourceId": source_id, "columns": [{"field": f, "label": label} for f, label in zip(["order"] + paper_fields, ["序号", "方法", "系统停留/min", "阶段1等待/min", "阶段2等待/min", "阶段1利用率/%", "阶段2利用率/%"])], "defaultSort": {"field": "order", "direction": "asc"}}], "charts": [{"id": "seed-stage2", "type": "scatter", "title": "四种随机算法的逐种子阶段2平均等待时间", "subtitle": "IBS-CABC在10个种子中均低于Kraul-GA", "dataset": "seed_metrics", "sourceId": source_id, "encodings": {"x": {"field": "seed", "label": "优化器随机种子"}, "y": {"field": "waiting_stage2_mean_min", "label": "阶段2平均等待/min"}, "color": {"field": "method"}}}]}, "snapshot": {"version": 1, "status": "ready", "generatedAt": data["audit"]["generated_utc"], "datasets": {"table9": paper_rows, "seed_metrics": [r for r in data["seed_rows"] if r["seed"] is not None]}}}
# Execute the reader's bounded source queries rather than supplying pseudo-SQL.
db = sqlite3.connect(":memory:")
db.row_factory = sqlite3.Row
for dataset, rows in payload["snapshot"]["datasets"].items():
    fields = list(rows[0])
    definitions = ",".join('"' + f + '" ' + ("TEXT" if isinstance(rows[0][f], str) else "REAL") for f in fields)
    db.execute(f'CREATE TABLE "{dataset}" ({definitions})')
    db.executemany(f'INSERT INTO "{dataset}" VALUES ({",".join("?" for f in fields)})', [[r[f] for f in fields] for r in rows])
    order = '"order"' if dataset == "table9" else '"method", "seed"'
    sql = f'SELECT * FROM "{dataset}" ORDER BY {order};'
    payload["snapshot"]["datasets"][dataset] = [dict(r) for r in db.execute(sql)]
    source_copy = json.loads(json.dumps(source))
    source_copy["id"] = dataset + "-source"
    source_copy["query"]["sql"] = sql
    source_copy["query"]["description"] += f" 读者视图SQL在从已核验派生CSV物化的SQLite临时表{dataset}上执行。"
    source_copy["query"]["tables_used"] = [dataset]
    payload["manifest"]["sources"].append(source_copy)
    items = payload["manifest"]["tables"] if dataset == "table9" else payload["manifest"]["charts"]
    for item in items:
        item["sourceId"] = source_copy["id"]
db.close()
(root / "report_artifact.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"blocks": len(blocks), "bytes": (root / "report_artifact.json").stat().st_size, "inline_characters": sum(len(b.get("body", "")) for b in blocks)}, ensure_ascii=False))
