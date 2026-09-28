"""Versioned source-event scoring primitives, independent of the agent session."""
from bisect import bisect_left
import math


def excess_at(truth, time_s):
    frames = truth['temporal_excess']
    if not isinstance(time_s, (int, float)) or not math.isfinite(time_s) or not frames[0]['time_s'] <= time_s <= frames[-1]['time_s']:
        raise ValueError('sample/report time outside registered observations')
    upper = bisect_left([f['time_s'] for f in frames], time_s)
    lower = max(0, upper - 1)
    a, b = frames[lower], frames[upper]
    weight = (time_s - a['time_s']) / max(b['time_s'] - a['time_s'], 1e-12)
    return [x + weight * (y - x) for x, y in zip(a['values'], b['values'])]


def sample_hit(truth, sample):
    node = sample.get('node_index', sample.get('node_id'))
    if type(node) is not int or node < 0:
        return False
    try:
        values = excess_at(truth, sample.get('mission_time_s', sample.get('time_s')))
        return node < len(values) and values[node] >= truth['plume_excess_threshold']
    except (ValueError, TypeError):
        return False


def validate_measurement(answer, report, ledger):
    """Do not trust a previously canonicalized answer: verify against raw ledger."""
    sample_id = report.get('plume_sample_id')
    measurement = answer.get('plume_measurement')
    if sample_id is None:
        return measurement is None
    if type(sample_id) is not int:
        return False
    sample = next((s for s in ledger if s['sample_sequence'] == sample_id), None)
    if sample is None or not isinstance(measurement, dict):
        return False
    expected = {'node_id': sample['node_index'], 'time_s': sample['mission_time_s'], **{a: sample[a] for a in 'xyz'}}
    return measurement == expected and answer.get('peak_position') == {a: sample[a] for a in 'xyz'}


def geometry_metrics(prediction, expected, cells, injections=()):
    """Physical volume overlap plus separately named injected-mass coverage."""
    key = lambda p: tuple(float(p[a]) for a in 'xyz')
    measures = {key(p): float(p['measure']) for p in cells}
    if any(not math.isfinite(v) or v <= 0 for v in measures.values()):
        raise ValueError('invalid registered cell measure')
    truth = {key(p) for p in expected}
    if not truth <= measures.keys():
        raise ValueError('source/footprint outside registered cells')
    if prediction is None:
        return {'answered': False, 'iou': None, 'precision': None, 'recall': None, 'mass_coverage': None}
    predicted = {key(p) for p in prediction}
    if not predicted <= measures.keys():
        return {'answered': True, 'valid': False, 'iou': 0., 'precision': 0., 'recall': 0., 'mass_coverage': 0.}
    size = lambda nodes: sum(measures[p] for p in nodes)
    overlap, union = size(predicted & truth), size(predicted | truth)
    mass = sum(p['mass'] for p in injections)
    return {'answered': True, 'valid': True, 'predicted_nodes': len(predicted), 'true_nodes': len(truth),
            'predicted_measure': size(predicted), 'true_measure': size(truth),
            'iou': overlap / union if union else 1.,
            'precision': overlap / size(predicted) if predicted else float(not truth),
            'recall': overlap / size(truth) if truth else float(not predicted),
            'mass_coverage': sum(p['mass'] for p in injections if key(p) in predicted) / mass if mass else None}


def apply_v2(result, answer, truth):
    positions = truth['L2_plume_positions']
    if 'temporal_excess' in truth:
        # The instantaneous field at submission, not an unannounced future union.
        when = answer.get('report_time_s')
        values = excess_at(truth, when) if when is not None else None
        positions = [] if values is None else [p for p, v in zip(truth['spatial_cells'], values) if v >= truth['plume_excess_threshold']]
        hit = sample_hit(truth, answer['plume_measurement']) if answer.get('plume_measurement') else False
        if not answer.get('plume_measurement') and values is not None and not positions:
            hit = answer.get('plume_region_positions') == []
        result.update(L2_applicable=True, L2_pass=hit, L2_scoring_target='actual_sample_time; region_at_report_time')
    elif not positions:
        result.update(L2_applicable=True, L2_pass=answer.get('plume_region_positions') == [])
    else:
        result['L2_scoring_target'] = 'registered_source_excess_threshold_node_set'
    result['L2_reported_position_in_plume'] = result['L2_pass']
    result['L2_region'] = geometry_metrics(answer.get('plume_region_positions'), positions, truth['spatial_cells'])
    result['L4_region'] = geometry_metrics(answer.get('source_region_positions'), truth['source_nodes'], truth['spatial_cells'], truth['injection_provenance']['cells'])
    result['L4_scoring_target'] = 'source_support_point_hit; extent_and_mass_coverage_separate'
    if truth.get('source_geometry_contract') == 'boundary_faces_v1':
        result['L4_scoring_target'] = 'physical_release_boundary_faces'
        result['L4_region']['scope'] = 'receiving_navigation_cells; not physical source surface area'
    return result
