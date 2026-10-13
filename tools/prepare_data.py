"""Fetch pinned upstream benchmarks and verify every experimental input."""
from pathlib import Path
from io import BytesIO
from urllib.request import Request, urlopen
from urllib.parse import quote
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def fetch(url):
    print('Downloading ' + url, flush=True)
    with urlopen(Request(url, headers={'User-Agent': 'IBS-CABC-reproduction'}), timeout=120) as response:
        return response.read()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify local inputs without downloading')
    args = parser.parse_args()
    contract = json.loads((ROOT/'validation/EXTERNAL_ASSETS.json').read_text(encoding='utf-8'))
    base = 'https://raw.githubusercontent.com/' + contract['upstream_repository'] + '/' + contract['upstream_commit'] + '/'
    payloads = {}
    for row in contract['files']:
        rel = row['path']
        target = ROOT / rel
        if target.exists():
            if sha(target.read_bytes()) != row['sha256']:
                raise ValueError('Existing input differs from paper input: ' + rel)
            continue
        if args.check:
            raise FileNotFoundError(rel + ': run python tools/prepare_data.py first')
        if rel.endswith('.ipynb'):
            key = 'Genetic_Algorithm_Example.ipynb'
            if key not in payloads:
                payloads[key] = fetch(base + key)
            data = payloads[key]
        else:
            if rel.startswith('dataset/'):
                member = rel[len('dataset/'):]
            else:
                member = rel.split('/data/formal/instances/', 1)[1]
            folder = member.split('/')[0]
            if folder not in payloads:
                archive = zipfile.ZipFile(BytesIO(fetch(base + quote(folder + '.zip'))))
                payloads[folder] = {n.replace('\\', '/'): archive.read(n) for n in archive.namelist() if not n.endswith('/')}
                archive.close()
            candidates = [v for n, v in payloads[folder].items() if n == member or n.endswith('/' + member) or n == member.split('/', 1)[1]]
            if len(candidates) != 1:
                raise ValueError('Expected one upstream archive member: ' + member)
            data = candidates[0]
        if sha(data) != row['sha256']:
            raise ValueError('Upstream SHA-256 mismatch for ' + rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.download')
        temporary.write_bytes(data)
        temporary.replace(target)
    print(f"PASS: {len(contract['files'])} pinned experimental inputs verified")

if __name__ == '__main__':
    main()
