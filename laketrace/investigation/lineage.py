"""Offline cross-directory lineage checks, before any formal model request."""
import json
from pathlib import Path
from laketrace.util import sha256_file, stable_hash


def inventory(root):
    root = Path(root)
    manifest = json.loads((root/'MANIFEST.json').read_text())
    if isinstance(manifest.get('files'), dict):
        for name, digest in manifest['files'].items():
            path=(root/name).resolve()
            if not path.is_relative_to(root.resolve()) or sha256_file(path)!=digest:
                raise ValueError('lineage dataset differs from its content manifest')
    truths = [json.loads(line) for line in (root/'truth/exam_truth.jsonl').read_text().splitlines() if line]
    tasks = {t['instance_id']:t for t in map(json.loads,(root/'tasks/instances.jsonl').read_text().splitlines())}
    tokens = set()
    for truth in truths:
        for key in ('event_cluster_id','background_id','event_fingerprint','background_fingerprint'):
            if truth.get(key):
                tokens.add(key+':'+str(truth[key]))
        field = tasks[truth['instance_id']]['frame_id']
        for path in (root/'fields').glob('*/'+field+'.csv'):
            tokens.add('field:'+sha256_file(path))
    tokens.add('manifest:'+sha256_file(root/'MANIFEST.json'))
    if manifest.get('parent_manifest_sha256'):
        tokens.add('manifest:'+manifest['parent_manifest_sha256'])
    return tokens


def validate_registry(registry):
    """Every dataset gets a split; sharing lineage is allowed only within a split."""
    if not isinstance(registry, dict) or set(registry) != {'datasets'}:
        raise ValueError('registry requires datasets with explicit root and split')
    assigned = {}
    summaries = []
    for row in registry['datasets']:
        if set(row) != {'root','split'} or row['split'] not in {'development','heldout'}:
            raise ValueError('invalid dataset split entry')
        tokens = inventory(row['root'])
        for token in tokens:
            prior = assigned.setdefault(token, row['split'])
            if prior != row['split']:
                raise ValueError('event/background/content lineage crosses development and heldout')
        summaries.append({'root':row['root'],'split':row['split'],'lineage_tokens':len(tokens)})
    return {'validated':True,'datasets':summaries,'registry_sha256':stable_hash(registry)}


def validate_formal(config, root):
    if not config.get('formal_evaluation'):
        return None
    path = config.get('split_registry')
    if not path:
        raise ValueError('Formal evaluation requires cross-version split_registry')
    registry = json.loads(Path(path).read_text())
    manifest=json.loads((Path(root)/'MANIFEST.json').read_text())
    if manifest.get('role')!='reserved_controlled_holdout':
        raise ValueError('dataset was not generated as reserved heldout data')
    if any(json.loads(line).get('split')!='heldout' for line in (Path(root)/'truth/exam_truth.jsonl').read_text().splitlines() if line):
        raise ValueError('dataset task splits do not match formal heldout role')
    target = Path(root).resolve()
    if not any(Path(d['root']).resolve()==target and d['split']=='heldout' for d in registry['datasets']):
        raise ValueError('Formal evaluation data must be registered as heldout')
    if not any(d['split']=='development' for d in registry['datasets']):
        raise ValueError('Formal registry must include all registered development datasets')
    catalog=Path(__file__).resolve().parents[2]/'configs/investigation/data_registry.json'
    if catalog.exists():
        required={Path(d['root']).resolve() for d in json.loads(catalog.read_text())['datasets'] if d['split']=='development'}
        provided={Path(d['root']).resolve() for d in registry['datasets'] if d['split']=='development'}
        if not required<=provided:
            raise ValueError('Formal registry omits previously registered development datasets')
    return validate_registry(registry)
