"""Explicitly replay frozen numerical stages in a fresh isolated workspace.

This wrapper has not been run end to end. It does not generate new narrative
reports or change protocols. No work is performed without --execute.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA256 = '967e41e04fc306fe604e2a693f488995a8b41e5047418f8a5c8e4abd6deca784'
STAGES = {
    'v1': ('olist-logistic', 'olist-logistic-2026-10-16'),
    'v2': ('olist-calibration', 'olist-calibration-v2-2026-10-16'),
    'v3': ('olist-lightgbm', 'olist-lightgbm-v3-2026-10-16'),
}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def snapshot(workspace, stages):
    result = {}
    for stage in stages:
        work, output = STAGES[stage]
        for folder in (workspace / 'work' / work, workspace / 'outputs' / output):
            for path in sorted(folder.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    result[path.relative_to(workspace).as_posix()] = sha256(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Explicitly authorize numerical replay')
    args = parser.parse_args()
    if not args.execute:
        parser.error('No experiments were run. Numerical replay requires --execute.')
    data = args.data.resolve()
    workspace = args.run_dir.resolve()
    runs = (ROOT / 'runs').resolve()
    if not workspace.is_relative_to(runs) or workspace == runs:
        parser.error('--run-dir must be a new subdirectory inside this repository\'s runs/')
    if workspace.exists():
        parser.error('Run directory already exists; choose a new one. Nothing was overwritten.')
    if not data.is_file() or sha256(data) != DATA_SHA256:
        parser.error('Missing dataset or archive SHA-256 mismatch; nothing was run.')
    # Verify the publication snapshot before creating a replay workspace.
    subprocess.run([sys.executable, str(ROOT / 'scripts/verify_package.py')], check=True)
    workspace.mkdir(parents=True, exist_ok=False)
    scout = workspace / 'work/dataset-scout'
    scout.mkdir(parents=True)
    shutil.copy2(data, scout / 'olist.zip')
    shutil.copy2(ROOT / 'analysis/audit/sample_audit.py', scout / 'sample_audit.py')
    for stage, (work, output) in STAGES.items():
        target = workspace / 'work' / work
        target.mkdir(parents=True)
        for source in (ROOT / 'analysis' / stage).glob('*.py'):
            shutil.copy2(source, target / source.name)
        shutil.copy2(ROOT / 'configs' / stage / 'protocol.json', target / 'protocol.json')
        destination = workspace / 'outputs' / output
        destination.mkdir(parents=True)
        shutil.copy2(target / 'protocol.json', destination / 'protocol.json')
    save_json(workspace / 'replay_record.json', {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'purpose': 'Replay of already completed exploratory protocols; not historical preregistration',
        'source_archive_sha256': DATA_SHA256,
        'python': sys.version,
        'narrative_reports_regenerated': False,
    })

    def run(relative):
        print(f'Running frozen stage: {relative}', flush=True)
        environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable, str(workspace / relative)], cwd=workspace,
                       env=environment, check=True)

    run('work/dataset-scout/sample_audit.py')
    run('work/olist-logistic/features.py')
    run('work/olist-logistic/run_models.py')
    for stage, previous, record, count_field in [
        ('v2', ['v1'], 'v1_before.json', 'v1_file_count'),
        ('v3', ['v1', 'v2'], 'previous_before.json', 'preserved_files'),
    ]:
        work, _ = STAGES[stage]
        target = workspace / 'work' / work
        before = snapshot(workspace, previous)
        save_json(target / record, before)
        save_json(target / 'registration.json', {
            'registered_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'purpose': 'Replay stage lock; original results already known',
            'protocol_sha256': sha256(target / 'protocol.json'),
            count_field: len(before),
        })
        run(f'work/{work}/run_fixed.py')
    print(f'Numerical replay complete: {workspace}')


if __name__ == '__main__':
    main()
