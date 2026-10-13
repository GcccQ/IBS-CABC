"""Restore losslessly compressed historical records, checking their paper hashes."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    for row in json.loads((ROOT/'validation/RESULT_ARCHIVES.json').read_text(encoding='utf-8')):
        target = ROOT / row['path']
        if not target.exists():
            temporary = target.with_suffix('.jsonl.unpacking')
            with gzip.open(ROOT/row['archive'], 'rb') as inp, temporary.open('wb') as out:
                shutil.copyfileobj(inp, out)
            if digest(temporary) != row['sha256'] or temporary.stat().st_size != row['bytes']:
                temporary.unlink()
                raise ValueError('Historical result checksum mismatch: ' + row['path'])
            temporary.replace(target)
        if digest(target) != row['sha256']:
            raise ValueError('Historical result has changed: ' + row['path'])
        print('PASS: ' + row['path'])

if __name__ == '__main__':
    main()
