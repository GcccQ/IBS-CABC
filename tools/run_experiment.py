"""Convenient launcher; new experiments go under reruns/, never saved outputs/."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=['fixed','dbd-three','dbd-kraul','cp'],required=True)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',help='Actual printed interrupted run directory, relative to the package or absolute')
    args=p.parse_args()
    if args.experiment=='cp' and (args.dry_run or args.smoke or args.resume):
        p.error('CP launcher starts a NEW formal run only; see commands.md for auto-resume without --new.')
    projects={'fixed':'cssd-fixed-multiseed-server32-w72-final','dbd-three':'cssd-dbd-case-study','dbd-kraul':'kraul-dbd-reproduction','cp':'cssd-kg-edr'}
    project=ROOT/projects[args.experiment]
    if args.experiment=='cp':
        argv=['scripts/run_cp_sat_ablation.py','--data-root',str(ROOT/'dataset'),'--output',str(ROOT/'reruns/cp_sat_ablation'),'--workers','8','--new']
    else:
        if args.smoke:
            config='smoke.yaml'
        elif args.experiment=='fixed':
            config='fixed_30_20_windows24.yaml'
        else:
            config='dbd_proxy91_10seeds.yaml'
        script='run.py' if args.experiment=='fixed' else 'scripts/run_dbd.py'
        argv=[script,'--config','configs/'+config]
        if args.dry_run:
            argv.append('--dry-run')
        elif args.resume:
            run=Path(args.resume)
            if not run.is_absolute():run=ROOT/run
            argv+=['--resume',str(run.resolve())]
        else:
            label=args.experiment+('_smoke' if args.smoke else '_formal')
            argv+=['--output',str(ROOT/'reruns'/label)]
        if args.smoke and args.experiment=='dbd-three':argv+=['--workers','1']
    print('[cwd] '+str(project),flush=True)
    print('[command] '+str([sys.executable,*argv]),flush=True)
    raise SystemExit(subprocess.call([sys.executable,*argv],cwd=project))

if __name__=='__main__':main()
