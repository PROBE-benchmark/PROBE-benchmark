"""Measured-only spatial memory and structured retrieval over the one SQL ledger."""
from collections import defaultdict
import json
import math
from statistics import mean, median
from pathlib import Path

from laketrace.util import sha256_file

VIEWS = ['grid', 'samples', 'summary', 'profile', 'timeseries', 'correlation', 'artifact']
PROPERTIES = {
    'view': {'type': 'string', 'enum': VIEWS},
    'channel': {'type': 'string'},
    'channels': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 12},
    'node_ids': {'type': 'array', 'items': {'type': 'integer'}, 'maxItems': 128},
    'sample_ids': {'type': 'array', 'items': {'type': 'integer'}, 'maxItems': 128},
    'bounds': {'type': 'object', 'properties': {a: {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2} for a in 'xyz'}, 'additionalProperties': False},
    'time_window_s': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2},
    'limit': {'type': 'integer', 'minimum': 1, 'maximum': 128},
    'offset': {'type': 'integer', 'minimum': 0},
    'analysis_id': {'type': 'integer', 'minimum': 1},
    'artifact': {'type': 'string', 'maxLength': 120},
}


def records(session):
    session._sync()
    cursor = session.db.execute('SELECT * FROM samples ORDER BY sample_id')
    return [dict(zip(session.measurement_columns, row)) for row in cursor]


def sample_row(row, channel):
    return {k: row[k] for k in ('sample_id', 'node_id', 'x', 'y', 'z', 'time_s', channel, 'u', 'v', 'w')}


def grid(session, rows, bins=4):
    """Never collapse depth; per-node latest samples do not erase raw time series."""
    nav = session.env.graph.positions
    lows = [min(p[i] for p in nav) for i in range(2)]
    spans = [max(p[i] for p in nav) - lows[i] for i in range(2)]
    def cell(p):
        return tuple(min(bins - 1, int((p[i] - lows[i]) / max(spans[i], 1e-12) * bins)) for i in range(2)) + (p[2],)
    capacities = defaultdict(int)
    for p in nav:
        capacities[cell(p)] += 1
    grouped = defaultdict(list)
    for row in rows:
        grouped[cell([row[a] for a in 'xyz'])].append(row)
    channel = session.target['target_channel']
    columns = ['ix', 'iy', 'z', 'visited_nodes', 'available_nodes', 'samples', 'first_t_s', 'last_t_s', 'minimum', 'median', 'maximum', 'max_sample_id', 'latest_sample_id', 'mean_u', 'mean_v', 'mean_w']
    cells = []
    for key, observations in sorted(grouped.items()):
        latest = {r['node_id']: r for r in observations}
        values = list(latest.values())
        high = max(values, key=lambda r: (r[channel], -r['sample_id']))
        cells.append([*key, len(values), capacities[key], len(observations), observations[0]['time_s'], observations[-1]['time_s'],
                      min(r[channel] for r in values), median(r[channel] for r in values), high[channel], int(high['sample_id']),
                      int(observations[-1]['sample_id']), *[mean(r[a] for r in values) for a in ('u', 'v', 'w')]])
    return {'channel': channel, 'horizontal_bins': [bins, bins], 'origin_xy': lows, 'span_xy': spans,
            'depths': sorted({p[2] for p in nav}), 'columns': columns, 'rows': cells,
            'visited_cells': len(cells), 'available_cells': len(capacities),
            'unmeasured_cells': len(capacities) - len(cells),
            'scope': 'Absent cells are UNKNOWN. Depth is exact, never pooled. Statistics use latest measured value per node; times can differ. No interpolation, source labels or fabricated multichannel vectors. One visited node does not cover an entire cell.'}


def memory(session, compact=False):
    if session.protocol.get('native_grid_version'):
        from .native_grid import memory as native_memory
        return native_memory(session, compact)
    rows = records(session)
    channel = session.target['target_channel']
    now = session.env.budget.used_s
    current = session.env.graph.positions[session.env.current]
    watercolumn = [r for r in rows if r['x'] == current[0] and r['y'] == current[1]]
    latest_column = list({r['z']: r for r in watercolumn}.values())
    # Deterministic range representatives include background, not only maxima.
    ordered = sorted({r['node_id']: r for r in rows}.values(), key=lambda r: (r[channel], r['node_id']))
    anchors = [ordered[round((len(ordered)-1)*q)] for q in (0, .25, .5, .75, 1)] if ordered else []
    ledger = session.env.sample_ledger
    last_group = ledger[-1]['provider_round'] if ledger else None
    ids = {s['sample_sequence'] for s in ledger if s['provider_round'] == last_group}
    route = [r for r in rows if r['sample_id'] in ids]
    result = {'clock_s': now, 'grid': grid(session, rows, 2 if compact else 4),
              'current_column': [sample_row(r, channel) for r in latest_column],
              'evidence_anchors': [sample_row(r, channel) for r in anchors],
              'last_route': {'samples': len(route), 'sample_id_range': [route[0]['sample_id'], route[-1]['sample_id']] if route else [],
                             'tail': [sample_row(r, channel) for r in route[-(2 if compact else 6):]]},
              'retrieval': 'inspect(view=grid/samples/profile/timeseries, bounds/node_ids/sample_ids/time_window_s, offset, limit). Raw sample IDs persist across context truncation.',
              'time_semantics': 'Measurement age = clock_s - time_s. Latest values at different nodes need not be simultaneous. Use time_window_s for aligned analysis.'}
    # Large depth sets stay exact and retrievable rather than silently pooled.
    cap = 32 if compact else 96
    cells = result['grid']['rows']
    result['grid'].update(total_rows=len(cells), omitted_rows=max(0, len(cells)-cap))
    if len(cells) > cap:
        result['grid']['rows'] = [cells[round(i*(len(cells)-1)/(cap-1))] for i in range(cap)]
        result['grid']['selection'] = 'Evenly spaced spatial rows; retrieve full grid via inspect with pagination.'
    return result


def artifact(session, args):
    number, name = args.get('analysis_id'), args.get('artifact')
    if type(number) is not int or not 1 <= number <= session.compute_count or not isinstance(name, str) or Path(name).name != name:
        raise ValueError('artifact requires a completed analysis_id and its exact recorded filename')
    directory = session.workspace / f'analysis-{number:03d}'
    receipt = json.loads((directory/'receipt.json').read_text())
    path = directory/name
    if name not in receipt['files'] or path.is_symlink() or sha256_file(path) != receipt['files'][name]:
        raise ValueError('artifact identity/integrity mismatch')
    if path.suffix not in {'.txt', '.json', '.csv'}:
        raise ValueError('Only text, JSON and CSV artifacts are readable; PNGs are human inspection artifacts')
    lines = path.read_text().splitlines()
    start, limit = args.get('offset', 0), args.get('limit', 64)
    content = '\n'.join(lines[start:start+limit])
    return {'analysis_id': number, 'artifact': name, 'sha256': receipt['files'][name],
            'content': content[:16000], 'truncated': len(content)>16000 or start+limit<len(lines),
            'next_offset': start+limit if start+limit<len(lines) else None,
            'scope': 'Recorded model computation, not a water observation.'}


def inspect(session, args):
    if set(args) - PROPERTIES.keys() or args.get('view') not in VIEWS:
        raise ValueError('inspect requires a declared view and supported filter parameters')
    limit, offset = args.get('limit', 64), args.get('offset', 0)
    if type(limit) is not int or not 1 <= limit <= 128 or type(offset) is not int or offset < 0:
        raise ValueError('limit must be 1-128; offset must be a nonnegative integer')
    if args['view'] == 'grid' and session.protocol.get('native_grid_version'):
        from .native_grid import page
        return page(session, args)
    if args['view'] == 'artifact':
        return artifact(session, args)
    channel = args.get('channel', session.target['target_channel'])
    channels = args.get('channels', [channel])
    if channel not in session.channels or not isinstance(channels, list) or not channels or any(c not in session.channels for c in channels):
        raise ValueError('Use available measured channel names')
    rows = records(session)
    for field, column in [('node_ids', 'node_id'), ('sample_ids', 'sample_id')]:
        if field in args:
            ids = args[field]
            if not isinstance(ids, list) or len(ids)>128 or any(type(x) is not int for x in ids):
                raise ValueError('IDs must be a list of integers, at most 128')
            rows = [r for r in rows if r[column] in ids]
    ranges = dict(args.get('bounds', {}))
    if set(ranges)-set('xyz'):
        raise ValueError('bounds uses x/y/z intervals')
    if 'time_window_s' in args:
        ranges['time_s'] = args['time_window_s']
    for key, limits in ranges.items():
        if not isinstance(limits, list) or len(limits)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in limits) or limits[0]>limits[1]:
            raise ValueError('Ranges are [finite minimum, maximum]')
        rows = [r for r in rows if limits[0] <= r[key] <= limits[1]]
    view = args['view']
    if view == 'grid':
        result = grid(session, rows)
        result.update(rows=result['rows'][offset:offset+limit], total_rows=result['visited_cells'], next_offset=offset+limit if offset+limit<result['visited_cells'] else None)
        return result
    if view in {'samples', 'timeseries'}:
        output = [{k:r[k] for k in dict.fromkeys(['sample_id','node_id','x','y','z','time_s',*channels,'u','v','w'])} for r in rows]
        return {'rows': output[offset:offset+limit], 'total_rows':len(output), 'next_offset':offset+limit if offset+limit<len(output) else None, 'scope':'Raw measured samples; no interpolation.'}
    if view == 'correlation':
        pairs=[]
        for i,c in enumerate(channels):
            for d in channels[i+1:]:
                xs,ys=[r[c] for r in rows],[r[d] for r in rows]
                mx,my=(mean(xs),mean(ys)) if rows else (0,0)
                denom=math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
                pairs.append({'channels':[c,d],'n':len(rows),'pearson':sum((x-mx)*(y-my) for x,y in zip(xs,ys))/denom if denom else None})
        return {'pairs':pairs,'scope':'Same-sample descriptive correlations; adaptive/repeated samples are dependent. Correlation is not a mechanism label.'}
    groups=defaultdict(list)
    for row in rows:
        groups[row['z'] if view=='profile' else 'selected'].append(row)
    return {'groups':[{'group':key,'samples':len(group),'nodes':len({r['node_id'] for r in group}),
                      'time_range_s':[min(r['time_s'] for r in group),max(r['time_s'] for r in group)],
                      'channels':{c:{'min':min(r[c] for r in group),'median':median(r[c] for r in group),'max':max(r[c] for r in group),
                                     'max_sample_id':int(max(group,key=lambda r:r[c])['sample_id'])} for c in channels}}
                     for key,group in sorted(groups.items())],
            'scope':'Raw selected samples, grouped at exact depth for profiles. Time spans and repeats remain explicit; not representative whole-lake statistics.'}
