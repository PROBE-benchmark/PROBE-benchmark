"""Offline source-event contract checks; never imported by a model tool."""
import math

from laketrace.util import stable_hash

CONTRACT = "source_event_v1"


def validate_case(task, truth, positions):
    if truth.get("case_contract") not in {CONTRACT, "source_event_v2"}:
        raise ValueError("unsupported source-event contract")
    if task["instance_id"] != truth["instance_id"] or type(truth["source_present"]) is not bool:
        raise ValueError("case identity or source label mismatch")
    if truth["detectable_source"] != truth["source_present"]:
        raise ValueError("legacy event alias must not relabel weak sources as negatives")
    window = task["observation_contract"]["event_window_s"]
    observation = task["observation_contract"]["snapshot_time_s"]
    if len(window) != 2 or not window[0] <= observation <= window[1]:
        raise ValueError("observation outside declared event window")
    nodes = {tuple(p) for p in positions}
    injections = truth["injection_provenance"]["cells"]
    support = {tuple(p[a] for a in "xyz") for p in truth["source_nodes"]}
    actual = {tuple(p[a] for a in "xyz") for p in injections}
    if actual != support or not actual <= nodes:
        raise ValueError("source geometry must include every actual injection cell")
    if truth.get('source_geometry_contract') == 'boundary_faces_v1':
        validate_boundary_injections(truth, nodes)
    if bool(injections) != truth["source_present"]:
        raise ValueError("physical event label disagrees with injected mass")
    for p in injections:
        if not math.isfinite(p["mass"]) or p["mass"] <= 0 or not window[0] <= p["time_s"] <= observation:
            raise ValueError("invalid mass or injection outside observable history")
    if truth["source_present"] and abs(sum(p["mass"] for p in injections) - truth["injection_provenance"]["mass"]) > 1e-7:
        raise ValueError("injection mass ledger mismatch")
    footprint = {tuple(p[a] for a in "xyz") for p in truth["L2_plume_positions"]}
    if not footprint <= nodes:
        raise ValueError("footprint outside public geometry")
    if truth["evidence_stratum"] not in {"ordinary", "weak", "information_insufficient", "unverified"}:
        raise ValueError("invalid predeclared evidence stratum")
    if truth.get('case_contract') == 'source_event_v2':
        cells = truth['spatial_cells']
        if {tuple(p[a] for a in 'xyz') for p in cells} != nodes or any(not math.isfinite(p['measure']) or p['measure']<=0 for p in cells):
            raise ValueError('physical cell measures must cover exactly the public geometry')
        if not truth.get('event_fingerprint') or not truth.get('background_fingerprint'):
            raise ValueError('v2 requires content lineage fingerprints')
    return {"contract": truth['case_contract'], "instance_id": task["instance_id"],
            "source_cells": len(actual), "footprint_nodes": len(footprint),
            "truth_sha256": stable_hash(truth), "validated": True,
            "scope": "time/injection/footprint consistency, not proof of identifiability"}


def validate_boundary_injections(truth, nodes):
    """Link physical release faces to receiving cells without conflating them."""
    geometry = truth.get('injection_geometry', {})
    faces = geometry.get('faces', [])
    if not truth['source_present']:
        if faces:
            raise ValueError('absent event has release faces')
        return
    from laketrace.scoring.geometry import boundary_support_error
    boundary_support_error((0., 0., 0.), geometry, {'horizontal_m': 1., 'vertical_m': 1.})
    receivers = set()
    face_ids = set()
    for face in faces:
        identifier = face['face_id']
        if identifier in face_ids:
            raise ValueError('duplicate release face')
        face_ids.add(identifier)
        node = tuple(face['receiving_node'][a] for a in 'xyz')
        if node not in nodes:
            raise ValueError('boundary receiver is not navigable')
        receivers.add(node)
        b = face['bounds']
        if face['boundary_type'] == 'bed':
            if b['z'][0] != b['z'][1] or b['z'][0] < node[2]:
                raise ValueError('bed face must be below receiving node')
            if not b['x'][0] <= node[0] <= b['x'][1] or not b['y'][0] <= node[1] <= b['y'][1]:
                raise ValueError('bed receiver outside footprint')
        elif face['boundary_type'] != 'open_window_release':
            raise ValueError('unsupported physical boundary type')
    expected = {tuple(p[a] for a in 'xyz') for p in truth['source_nodes']}
    if receivers != expected:
        raise ValueError('release faces and receiving support differ')
    injections = truth['injection_provenance']['cells']
    by_id = {f['face_id']: tuple(f['receiving_node'][a] for a in 'xyz') for f in faces}
    for injection in injections:
        if by_id.get(injection.get('face_id')) != tuple(injection[a] for a in 'xyz'):
            raise ValueError('injection mass not linked to its release face')


def validate_split(truths):
    """An event or shared background cannot cross development/test boundaries."""
    groups = {}
    for truth in truths:
        for key in ("event_cluster_id", "background_id"):
            group = (key, truth[key])
            prior = groups.setdefault(group, truth["split"])
            if prior != truth["split"]:
                raise ValueError("shared event/background crosses dataset splits")
    return True
