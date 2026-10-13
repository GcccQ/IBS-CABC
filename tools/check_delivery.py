"""One-command integrity and runnable-entry checks; --tests adds existing unit tests."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
FIXED='cssd-fixed-multiseed-server32-w72-final'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as s:
        for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify_copied():
    manifest=json.loads((ROOT/'validation/SOURCE_COPY_MANIFEST.json').read_text(encoding='utf-8'))
    for r in manifest['files']:
        assert digest(ROOT/r['destination'])==r['sha256'],r['destination']
    print(f"[PASS] {len(manifest['files'])} copied files unchanged",flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests',action='store_true')
    args=parser.parse_args()
    verify_copied()
    commands=[
        ('saved_evidence',ROOT,['tools/verify_results.py']),
        ('fixed_windows24_dry_run',ROOT/FIXED,['run.py','--config','configs/fixed_30_20_windows24.yaml','--dry-run']),
        ('dbd_three_dry_run',ROOT/'cssd-dbd-case-study',['scripts/run_dbd.py','--config','configs/dbd_proxy91_10seeds.yaml','--dry-run']),
        ('dbd_kraul_dry_run',ROOT/'kraul-dbd-reproduction',['scripts/run_dbd.py','--config','configs/dbd_proxy91_10seeds.yaml','--dry-run']),
        ('table8_independent',ROOT,['paper_artifacts/validation/verify_table8.py',f'{FIXED}/fixed_30_20_server32_w72_report_bundle/outputs/fixed_30_20_server32_w72/report/all_repetitions.csv']),
        ('table10_independent',ROOT,['paper_artifacts/validation/verify_dbd_table10.py','cssd-dbd-case-study/outputs/dbd_case_proxy91_10seeds_three_algorithms_20260827T083126700163Z/raw/results.jsonl','kraul-dbd-reproduction/outputs/dbd_proxy91_10seeds_ffcbspt_kraul_ga_20260826T064333748261Z/raw/results.jsonl']),
    ]
    if args.tests:
        for project in [FIXED,'cssd-kg-edr','cssd-comparison-baselines','cssd-dbd-case-study','kraul-dbd-reproduction']:
            commands.append(('tests_'+project,ROOT/project,['-m','pytest','tests','-q','-p','no:cacheprovider']))
    out=ROOT/'reruns/delivery_checks'; out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    checks=[]
    for name,cwd,argv in commands:
        print(f'[check] {name}',flush=True)
        result=subprocess.run([sys.executable,*argv],cwd=cwd,env=env,text=True,encoding='utf-8',errors='replace',stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/(name+'.log')).write_text(result.stdout,encoding='utf-8')
        checks.append({'name':name,'cwd':str(cwd.relative_to(ROOT)),'argv':['python',*argv],'exit_code':result.returncode,'log':str((out/(name+'.log')).relative_to(ROOT))})
        if result.returncode:
            print(result.stdout[-7000:]); raise SystemExit(result.returncode)
        print('[PASS] '+name,flush=True)
    verify_copied()
    summary={'status':'PASS','python':sys.version,'full_experiment_executed':False,'unit_tests_may_execute_short_solver_fixtures':args.tests,'checks':checks}
    (out/'check_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','checks':len(checks),'logs':str(out)},ensure_ascii=False))

if __name__=='__main__':main()
