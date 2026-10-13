"""Reject manuscript and figure files from the Git publication set."""
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.pdf', '.png', '.jpg', '.jpeg', '.svg', '.doc', '.docx', '.tex'}

def main():
    result = subprocess.run(
        ['git', 'ls-files', '-z'], cwd=ROOT, check=True, capture_output=True,
    )
    paths = result.stdout.decode('utf-8').split('\0')
    forbidden = [name for name in paths if name and (
        PurePosixPath(name).suffix.lower() in EXCLUDED
        or 'manuscript' in (part.lower() for part in PurePosixPath(name).parts)
    )]
    if forbidden:
        raise SystemExit('Excluded publication files are tracked:\n' + '\n'.join(forbidden))
    print('PASS: no manuscript or figure files in the Git publication set.')

if __name__ == '__main__':
    main()
