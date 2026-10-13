"""Rebuild derived reports from saved runs into reruns/, preserving all saved inputs."""
from pathlib import Path
import argparse
import ast
import importlib.util
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]

def load_report(path):
    spec=importlib.util.spec_from_file_location('handoff_dbd_report',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def dbd(out):
    module=load_report(ROOT/'paper_artifacts/dbd_10seeds_20260831/rebuild_report.py')
    module.OUT=out
    module.main()
    notebook_path=out/'reproduce.ipynb'
    notebook=json.loads(notebook_path.read_text(encoding='utf-8'))
    notebook['cells'][1]['source']=[
        'from pathlib import Path\n', 'import subprocess, sys\n',
        'package = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "tools/rebuild_saved.py").is_file())\n',
        'subprocess.run([sys.executable, str(package / "tools/rebuild_saved.py"), "--target", "dbd"], check=True)\n',
    ]
    notebook_path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')

def figure5(out):
    path=ROOT/'paper_artifacts/fixed_scale_figure5/build_figure5.py'
    tree=ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    changed=0
    for statement in tree.body:
        if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OUT_DIR' for t in statement.targets):
            statement.value=ast.Name(id='_handoff_output',ctx=ast.Load()); changed+=1
    assert changed==1,'Publication script output declaration changed; re-audit wrapper.'
    ast.fix_missing_locations(tree)
    exec(compile(tree,str(path),'exec'),{'__file__':str(path),'__name__':'__main__','_handoff_output':out})

def main():
    os.environ.setdefault('MPLBACKEND','Agg')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',choices=['dbd','figure5','all'],default='all')
    parser.add_argument('--output',default='reruns/rebuilt_reports')
    args=parser.parse_args()
    target=Path(args.output)
    if not target.is_absolute():target=ROOT/target
    target=target.resolve()
    if target==ROOT or (ROOT in target.parents and not target.is_relative_to(ROOT/'reruns')):
        parser.error('Output inside the delivery package must be under reruns/.')
    for label,function in [('figure5',figure5),('dbd',dbd)]:
        if args.target in [label,'all']:
            out=target/label; out.mkdir(parents=True,exist_ok=True)
            function(out)
            print('[output] '+str(out),flush=True)

if __name__=='__main__':main()
