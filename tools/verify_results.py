"""Audit saved evidence without running an optimizer or rewriting original results."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]
FIXED=ROOT/'cssd-fixed-multiseed-server32-w72-final/fixed_30_20_server32_w72_report_bundle/outputs/fixed_30_20_server32_w72'
THREE=ROOT/'cssd-dbd-case-study/outputs/dbd_case_proxy91_10seeds_three_algorithms_20260827T083126700163Z'
KRAUL=ROOT/'kraul-dbd-reproduction/outputs/dbd_proxy91_10seeds_ffcbspt_kraul_ga_20260826T064333748261Z'
CP=ROOT/'cssd-kg-edr/outputs/e7_cp_sat_ablation_20260825T161433740372Z'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

def raw_rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)

def main():
    fixed=csv_rows(FIXED/'report/all_repetitions.csv')
    tasks=json.loads((FIXED/'task_manifest.json').read_text(encoding='utf-8'))
    key=lambda r:(int(r['jobs']),r['instance_sha256'],r['algorithm'],int(r['seed']))
    assert len(fixed)==len(tasks)==8000
    assert len({key(r) for r in fixed})==8000
    assert {key(r) for r in fixed}=={key(r) for r in tasks}
    assert len({r['task_id'] for r in fixed})==8000
    groups=defaultdict(list)
    for r in fixed:
        assert math.isfinite(float(r['objective'])) and float(r['objective'])>0
        groups[key(r)[:3]].append(r)
    assert len(groups)==320
    for (jobs,instance,algorithm),rows in groups.items():
        assert set(int(r['seed']) for r in rows)==set(range(1,31 if jobs in {100,250} else 21))
    counts=Counter(int(r['jobs']) for r in fixed)
    assert counts=={100:2400,250:2400,500:1600,1000:1600}
    saved=csv_rows(FIXED/'report/tables4_7_numeric.csv')
    for summary in saved:
        jobs=int(summary['jobs']); algorithm=summary['algorithm']
        members=[rows for k,rows in groups.items() if k[0]==jobs and k[2]==algorithm]
        assert len(members)==20
        mean=sum(sum(float(r['objective']) for r in rows)/len(rows) for rows in members)/20
        assert math.isclose(mean,float(summary['objective_mean']),rel_tol=1e-12,abs_tol=1e-7)
    data_root=ROOT/'cssd-fixed-multiseed-server32-w72-final/data/formal/instances'
    for t in tasks:
        assert digest(data_root/t['relative_instance_path'])==t['instance_sha256']

    dbd_groups=defaultdict(list); dbd_ids=set(); hashes=defaultdict(set)
    total=0; max_identity_error=0.0
    for source in [THREE/'raw/results.jsonl',KRAUL/'raw/results.jsonl']:
        for r in raw_rows(source):
            total+=1
            assert r['task_id'] not in dbd_ids; dbd_ids.add(r['task_id'])
            assert r['feasible'] is True and r.get('status','success')=='success'
            method=r['algorithm_label']; seed=None if method=='FFCBSPT' else int(r['seed'])
            day=int(r['case_day_index']); hashes[day].add(r['instance_sha256'])
            assert (r['machines_stage1'],r['machines_stage2'])==(12,5)
            moments=r.get('metric_moments',r)
            tis=moments['time_in_system']
            error=abs(float(tis['sum'])-(float(r['objective'])-float(r['release_sum'])))
            max_identity_error=max(max_identity_error,error)
            assert error<1e-6
            dbd_groups[(method,seed)].append({'day':day,'jobs':int(r['jobs']),'moments':moments})
    expected={('FFCBSPT',None)}|{(m,s) for m in ['ALNS','TS','Kraul-GA','IBS-CABC'] for s in range(1,11)}
    assert total==3731 and set(dbd_groups)==expected
    assert set(hashes)==set(range(91)) and all(len(h)==1 for h in hashes.values())
    for members in dbd_groups.values():
        assert len(members)==91 and {r['day'] for r in members}==set(range(91))
        assert sum(r['jobs'] for r in members)==22927
        for r in members:
            assert all(r['moments'][m]['count']==r['jobs'] for m in ['time_in_system','waiting_stage1','waiting_stage2'])

    cp=list(raw_rows(CP/'raw/results.jsonl'))
    assert len(cp)==120 and len({r['task_id'] for r in cp})==120
    assert all(r['feasible'] is True and int(r['seed'])==101 for r in cp)
    cp_pairs=defaultdict(list)
    for r in cp:cp_pairs[(int(r['jobs']),r['instance_sha256'])].append(r)
    assert len(cp_pairs)==60 and all(len(rs)==2 and len({r['algorithm'] for r in rs})==2 for rs in cp_pairs.values())
    assert Counter(int(r['jobs']) for r in cp)=={100:40,250:40,500:40}
    for item in json.loads((ROOT/'results_for_advisor/RESULT_FILE_MAPPING.json').read_text(encoding='utf-8')):
        assert digest(ROOT/item['presentation_file'])==digest(ROOT/item['original_file'])==item['sha256']
    print(json.dumps({'status':'PASS','fixed_runs':8000,'fixed_instance_algorithm_groups':320,'fixed_scales':dict(counts),'dbd_records':3731,'dbd_complete_window_runs':41,'dbd_days':91,'dbd_jobs_per_window':22927,'dbd_max_TIS_identity_error':max_identity_error,'cp_sat_records':120,'cp_sat_pairs':60,'presentation_files_match_originals':True,'optimizer_executed':False},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
