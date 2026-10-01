#!/usr/bin/env python3
"""Verify the packaged tasks and isolated computation without a model request."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        raise RuntimeError('Python 3.12 or newer is required')
    out = args.output.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError('Use a new output directory outside the benchmark package')
    out.mkdir(parents=True, exist_ok=False)

    import numpy as np
    from laketrace.data.store import DataStore
    from laketrace.investigation.runner import load_settings, manifest
    from laketrace.investigation.workbench import compute
    from laketrace.util import dump_json

    checked = 0
    for line in (ROOT / 'SHA256SUMS.txt').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Package checksum mismatch: ' + relative)
        checked += 1

    config_path = ROOT / 'configs/investigation/models/local_v22_survey_v6_gate3.json'
    config, protocol, public = load_settings(config_path)
    store = DataStore(ROOT / config['data_root'], evaluator=True)
    tasks = store.tasks()
    identifiers = [t['instance_id'] for t in tasks]
    definition = manifest(config, protocol, public, identifiers, 0)
    if set(public.get('tasks', public)) != set(identifiers):
        raise ValueError('Public task identifiers differ from the task manifest')

    counts = Counter()
    domains = Counter()
    node_counts = []
    for task in tasks:
        iid = task['instance_id']
        frame = store.frame(iid)
        values = np.array([
            [*node.xyz, node.bed_depth, *node.channels.values(), *node.flow.values()]
            for node in frame.nodes
        ])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite field values: ' + iid)
        start = tuple(task['start_position'][k] for k in 'xyz')
        if start not in frame.by_xyz:
            raise ValueError('Start is outside the navigation grid: ' + iid)
        with np.load(store.root / ('environment_' + iid + '.npz'), allow_pickle=False) as env:
            shape = (len(frame.nodes), len(task['available_channels']))
            if env['clean'].shape != shape or env['noise'].shape != shape:
                raise ValueError('Environment array shape mismatch: ' + iid)
            if not np.isfinite(env['clean']).all() or not np.isfinite(env['noise']).all():
                raise ValueError('Nonfinite environment arrays: ' + iid)
        with np.load(store.root / ('evidence_' + iid + '.npz'), allow_pickle=False) as evidence:
            if evidence['signal'].shape != (len(frame.nodes),) or evidence['flow'].shape != (len(frame.nodes), 3):
                raise ValueError('Evidence array shape mismatch: ' + iid)
            if not all(np.isfinite(evidence[key]).all() for key in evidence.files):
                raise ValueError('Nonfinite evidence arrays: ' + iid)
        truth = store.truth(iid)
        counts[task['waterbody'] + ':' + ('source' if truth['source_present'] else 'control')] += 1
        domains[task['domain_id']] += 1
        node_counts.append(len(frame.nodes))
    if len(tasks) != 48 or sum(v for k, v in counts.items() if k.endswith(':source')) != 36:
        raise ValueError('Expected 48 tasks with 36 sources and 12 controls')

    code = 'assert float(np.mean([1., 2., 3.])) == 2.; assert len(df) == 1; print(2.)'
    data = {'samples': [], 'target_nodes': [{'target_value': 1.}]}
    result = compute(code, data, out / 'workbench', 1, api=3)
    if not result['ok']:
        raise RuntimeError('Scientific computation failed: ' + str(result))
    replay = compute(code, data, out / 'workbench', 1, replay=True, api=3)
    if replay != result:
        raise RuntimeError('Recorded computation replay differs')
    receipt = {
        'passed': True, 'python': sys.version, 'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'scipy', 'pandas', 'matplotlib', 'jsonschema')},
        'package_files_checked': checked, 'tasks_checked': len(tasks),
        'site_task_counts': dict(counts), 'domain_task_counts': dict(domains),
        'nodes_per_task_range': [min(node_counts), max(node_counts)],
        'isolated_computation_passed': True, 'recorded_computation_replay_passed': True,
        'data_manifest_sha256': definition['data_manifest_sha256'],
        'provider_requests': 0,
    }
    dump_json(out / 'verification.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
