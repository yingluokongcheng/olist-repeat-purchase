"""Read-only checks of the publication snapshot; never imports or fits models."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import ast
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8-sig'))


def rows(relative, stage):
    with (ROOT / relative).open(encoding='utf-8-sig', newline='') as stream:
        return {row['model']: row for row in csv.DictReader(stream) if row['stage'] == stage}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets.extend(value for name, value in attrs if name in ('href', 'src') and value)


def main():
    manifest = read_json('package_manifest.json')
    for relative, expected in manifest['files'].items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT), f'Out-of-repository manifest path: {relative}')
        require(path.is_file() and digest(path) == expected, f'Hash mismatch: {relative}')
    packaged = [p for p in ROOT.rglob('*') if p.is_file()
                and not {'runs', '.git', '.venv', '__pycache__'}.intersection(p.relative_to(ROOT).parts)
                and not (p.is_relative_to(ROOT / 'data/raw') and p.name != '.gitkeep')]
    expected_paths = set(manifest['files']) | {'package_manifest.json'}
    require({p.relative_to(ROOT).as_posix() for p in packaged} == expected_paths,
            'Unexpected or missing repository files')
    provenance = read_json('docs/provenance.json')
    for item in provenance:
        require(digest(ROOT / item['packaged']) == item['packaged_sha256'],
                f'Provenance mismatch: {item["packaged"]}')
        if item['mode'].startswith('byte-identical'):
            require(item['source_sha256'] == item['packaged_sha256'], 'Original-copy hash mismatch')
    links_checked = 0
    for path in packaged:
        require(path.suffix.lower() not in ('.zip', '.pkl', '.pickle', '.joblib', '.parquet'),
                f'Unexpected raw or customer-level artifact: {path.name}')
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        if path.suffix == '.csv':
            with path.open(encoding='utf-8-sig', newline='') as stream:
                header = next(csv.reader(stream), [])
            require(not {'customer_id', 'customer_unique_id', 'order_id'}.intersection(header),
                    f'Customer/order-level table included: {path.name}')
        if path.suffix in {'.md', '.html', '.json', '.py', '.txt', '.log', '.csv'}:
            text = path.read_text(encoding='utf-8-sig')
            # Narrow checks, not a guarantee of detecting every possible secret.
            require(not re.search(r'[A-Za-z]:[/\\]Users[/\\]', text), f'Personal absolute path: {path.name}')
            require(not re.search(r'(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)', text),
                    f'Credential-like text: {path.name}')
        targets = []
        if path.suffix == '.md':
            body = re.sub(r'```.*?```', '', text, flags=re.S)
            targets = re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)', body)
        elif path.suffix == '.html':
            parser = Links()
            parser.feed(text)
            targets = parser.targets
        for target in targets:
            url = urlsplit(target.strip('<>'))
            if url.scheme or url.netloc or not url.path:
                continue
            local = (path.parent / unquote(url.path)).resolve()
            require(local.is_relative_to(ROOT) and local.exists(), f'Broken local link in {path.name}: {target}')
            links_checked += 1

    audit = read_json('reports/audit/audit_results.json')
    require((audit['main']['customers'], audit['main']['repeat90']) == (51981, 567), 'Audit count drift')
    for stage in ('v1', 'v2', 'v3'):
        require((ROOT / f'configs/{stage}/protocol.json').read_bytes() ==
                (ROOT / f'reports/{stage}/protocol.json').read_bytes(), f'Protocol drift: {stage}')
    v1 = rows('reports/v1/model_metrics.csv', 'test')
    v2 = rows('reports/v2/metrics.csv', 'evaluation')
    v3 = rows('reports/v3/metrics.csv', 'evaluation')
    for group in (v1, v2, v3):
        for row in group.values():
            require((int(row['n']), int(row['events'])) == (7893, 87), 'Evaluation count drift')
    require([float(v1[m]['top10_hits']) for m in ('M1', 'M2', 'M3')] == [17, 19, 19], 'V1 headline drift')
    require(v2['B']['AP'] == v2['C']['AP'] and float(v2['C']['log_loss']) > float(v2['B']['log_loss']), 'V2 finding drift')
    ranking = read_json('reports/v2/results.json')['ranking']['evaluation']
    require(ranking['full_rank_identical'] and ranking['top10_identical'], 'V2 ranking record drift')
    require([float(v3[m]['top_hits']) for m in ('LR', 'G1', 'G2')] == [11, 13, 18], 'V3 headline drift')
    require(float(v3['LR']['AP']) > float(v3['G2']['AP']) > float(v3['G1']['AP']), 'V3 AP finding drift')
    require(all(v3['LR'][key] == v2['B'][key] for key in ('AP', 'log_loss', 'Brier', 'mean_prediction')), 'Frozen LR mismatch')
    locked = read_json('reports/v3/locked_models_before_evaluation.json')
    require(locked['evaluation_predictions_generated'] is False, 'Invalid historical locking record')
    for model, iteration in [('G1', 1), ('G2', 146)]:
        info = locked['models'][model]
        require(info['best_iteration'] == iteration, 'Early-stopping record drift')
        require(digest(ROOT / f'reports/v3/{model}_model.txt') == info['model_sha256'], 'Model-lock hash mismatch')
    tie = next(item for item in read_json('reports/v3/tie_diagnostics.json') if item['model'] == 'G1')
    require((tie['boundary_tie_customers'], tie['boundary_tie_events']) == (1548, 22), 'Tie record drift')
    print(json.dumps({'status': 'PASS', 'manifest_files': len(manifest['files']),
                      'provenance_records': len(provenance), 'local_links_checked': links_checked,
                      'checks': ['snapshot hashes', 'source syntax', 'local file links',
                                 'recorded headline metrics', 'protocol copies', 'model lock hashes',
                                 'excluded-data checks', 'narrow credential-pattern scan'],
                      'models_fitted': 0, 'end_to_end_replay_performed': False}, indent=2))


if __name__ == '__main__':
    main()
