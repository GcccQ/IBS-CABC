"""Verify the advisor delivery SHA256SUMS.txt; ignores later reruns and virtualenvs."""
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=ROOT/'SHA256SUMS.txt'
    if not manifest.is_file():raise SystemExit('Missing SHA256SUMS.txt')
    count=0
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        expected,name=line.split('  ',1)
        path=ROOT/name
        if not path.is_file():raise SystemExit('Missing: '+name)
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=expected:raise SystemExit('Hash mismatch: '+name)
        count+=1
    print(f'PASS: {count} delivery files match SHA-256.')

if __name__=='__main__':main()
