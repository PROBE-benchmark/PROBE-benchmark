"""Lossless native-node views over the append-only measurement ledger."""
from collections import Counter


def page(session, args=None, *, limit=None):
    from .evidence import records
    args = args or {}
    channels = args.get('channels', [args['channel']] if 'channel' in args else session.channels)
    if not channels or any(c not in session.channels for c in channels):
        raise ValueError('Use available measured channel names')
    import math
    for bounds in [*args.get('bounds', {}).values(), *([args['time_window_s']] if 'time_window_s' in args else [])]:
        if len(bounds) != 2 or any(not math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
            raise ValueError('Ranges require finite minimum <= maximum')
    rows = records(session)
    counts = Counter(r['node_id'] for r in rows)
    latest = {}
    for r in rows:
        if 'sample_ids' in args and r['sample_id'] not in args['sample_ids']:
            continue
        if 'time_window_s' in args and not args['time_window_s'][0] <= r['time_s'] <= args['time_window_s'][1]:
            continue
        latest[r['node_id']] = r
    columns = ['node_id', 'x', 'y', 'z', 'ever_visited', 'visit_count', 'sample_id', 'time_s', *channels, 'u', 'v', 'w']
    output = []
    for node, pos in enumerate(session.env.graph.positions):
        if 'node_ids' in args and node not in args['node_ids']:
            continue
        bounds = args.get('bounds', {})
        if any(not bounds[a][0] <= pos[i] <= bounds[a][1] for i, a in enumerate('xyz') if a in bounds):
            continue
        r = latest.get(node)
        output.append([node, *pos, bool(counts[node]), counts[node],
                       *([r[k] for k in ['sample_id', 'time_s', *channels, 'u', 'v', 'w']] if r else [None] * (len(channels)+5))])
    offset = args.get('offset', 0)
    limit = limit if limit is not None else args.get('limit', 64)
    return {'representation': 'native_3d_nodes', 'columns': columns, 'rows': output[offset:offset+limit],
            'total_rows': len(output), 'next_offset': offset+limit if offset+limit < len(output) else None,
            'scope': 'One row per valid native navigation node. No spatial aggregation or rounding. Null is unobserved in the selected time/sample window. ever_visited and visit_count cover all history. All channel and velocity values in a row belong to its sample_id. Earlier measurements remain in samples; different nodes may have different acquisition times.'}


def memory(session, compact=False):
    n = len(session.env.graph.positions)
    full = not compact
    args = {}
    if not full:
        current = session.env.graph.positions[session.env.current]
        # Geometry-only window: no hidden water values or maxima select evidence.
        args['node_ids'] = sorted(range(n), key=lambda i: (sum((a-b)**2 for a,b in zip(session.env.graph.positions[i],current)), i))[:32]
    result = page(session, args, limit=n)
    result.update(context_scope='complete' if full else 'explicit_local_window', total_navigation_nodes=n,
                  omitted_nodes=n-len(result['rows']))
    ledger = session.env.sample_ledger
    # Receipts record exactly which samples each completed response added.
    previous = session.feedback_receipts[-2]['sample_count'] if len(session.feedback_receipts)>1 else 1
    last = session.feedback_receipts[-1]['sample_count'] if session.feedback_receipts else 1
    delta = ledger[previous:last]
    earlier = {s['node_index'] for s in ledger[:previous]}
    changed = list(dict.fromkeys(s['node_index'] for s in delta))
    requested_pages = [o['result'] for o in session.feedback_receipts[-1]['outputs'] if o['tool']=='inspect' and o['result'].get('representation')=='native_3d_nodes'] if session.feedback_receipts else []
    return {'clock_s': session.env.budget.used_s, 'grid': result,
            'requested_grid_pages': requested_pages,
            'last_response_delta': {'sample_id_after': previous, 'sample_id_through': last,
                                    'new_samples': last-previous, 'changed_node_ids': changed,
                                    'new_nodes': len(set(changed)-earlier),
                                    'repeat_samples': len(delta)-len(set(changed)-earlier)},
            'visited_nodes': len({s['node_index'] for s in ledger}), 'raw_samples': len(ledger),
            'instruction': 'Read this current native grid before choosing the next action. For a target outside an explicit_local_window, inspect its region first. survey defaults to exploration and skips visited targets; revisit=true allows deliberate return. Transit through visited nodes remains legal. Use sample_id/time_s and inspect(samples/timeseries) to compare repeated observations; latest values are not a simultaneous field.',
            'retrieval': 'inspect(view=grid, bounds/node_ids, offset, limit, channels, time_window_s) reads exact node pages including unknown nodes. inspect(view=samples, node_ids, offset, limit, channels) retrieves full history. query navigation/samples provides full geometry/history.'}


def plan_goto(session, **point):
    """Fewest edges, then fewest visited arrivals; never ban necessary transit."""
    import heapq
    from laketrace.env.v20_3 import GOTO_DIRECTION_ORDER
    env = session.env
    target = env.resolve_goto_target(**point)
    visited = {s['node_index'] for s in env.sample_ledger}
    costs = {env.current: (0, 0)}
    parents = {}
    queue = [(0, 0, env.current)]
    while queue:
        steps, repeats, node = heapq.heappop(queue)
        if costs[node] != (steps, repeats):
            continue
        if node == target:
            break
        for direction in GOTO_DIRECTION_ORDER:
            edge = env.graph.neighbor_map[node].get(direction)
            if edge is None:
                continue
            cost = (steps+1, repeats+int(edge.target in visited))
            if cost < costs.get(edge.target, (float('inf'), float('inf'))):
                costs[edge.target] = cost
                parents[edge.target] = (node, direction)
                heapq.heappush(queue, (*cost, edge.target))
    route = []
    node = target
    while node != env.current:
        node, direction = parents[node]
        route.append(direction)
    return list(reversed(route))[:env.max_steps], target
