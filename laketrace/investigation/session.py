"""Truth-blind investigation tools over measured data and public geometry only."""
from __future__ import annotations

import copy
import json
import math
import re
import sqlite3
import tempfile
from pathlib import Path
from statistics import median
from typing import Any

from laketrace.env.v20_3 import EnvironmentV20_3
from laketrace.env.graph import DIRECTIONS
from laketrace.util import stable_hash

MECHANISMS = ["external_point", "external_diffuse", "internal_release", "physical_resuspension"]
REPORT_FIELDS = {
    "source_present": {"type": ["boolean", "null"]},
    "source_presence_probability": {"type": "number", "minimum": 0, "maximum": 1},
    "plume_node_id": {"type": ["integer", "null"]},
    "mechanism": {"type": ["string", "null"], "enum": [*MECHANISMS, None]},
    "source_position": {"type": ["object", "null"], "properties": {a: {"type": "number"} for a in "xyz"},
                        "required": list("xyz"), "additionalProperties": False},
    "source_candidate_id": {"type": ["string", "null"]},
    "source_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "decision": {"type": "string", "enum": ["declare", "cannot_determine"], "description": "declare means affirm a detectable pollution source AT source_position. It never means declaring absence. Use cannot_determine when absent or location is uncertain; source_present separately records presence/absence."},
    "evidence_node_ids": {"type": "array", "items": {"type": "integer"}, "maxItems": 20},
    "explanation": {"type": "string", "maxLength": 1600},
}
REQUIRED_REPORT = {"source_present", "source_presence_probability", "plume_node_id", "mechanism", "source_position", "source_confidence", "decision"}


def tool(name, description, properties, required=()):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {**properties,
                "notes": {"type": "string", "description": "Optional replacement of your persistent notes; keep evidence and plans you need later."}},
                "required": list(required), "additionalProperties": False}}}


TOOLS = [
    tool("goto", "Travel toward a selected coordinate along a geometry-only shortest route, at most 32 measured edges. Call again if the target is still distant.", {a: {"type": "number"} for a in "xyz"}, list("xyz")),
    tool("move", "Travel 1-32 adjacent edges in a chosen direction; every arrival is measured.", {"direction": {"type": "string", "enum": list(DIRECTIONS)}, "steps": {"type": "integer", "minimum": 1, "maximum": 32}}, ["direction", "steps"]),
    tool("query", "Read or calculate on all your exact measured history and public geometry using SQLite SELECT. Aggregates, joins, window functions, arithmetic, sqrt and pow are available. Never reads unmeasured water values. LIMIT/OFFSET paginate results.", {"sql": {"type": "string"}}, ["sql"]),
    tool("report", "Save or revise any report fields. Set final=true only when ready to end. At the end you can still revise all fields; null denotes unknown/inapplicable. No answer is inferred for you.", {**REPORT_FIELDS, "final": {"type": "boolean"}}, ["final"]),
]

SYSTEM = """You investigate a possible pollution source with a mobile sensor in a 3-D lake.
Use the task's known monitoring target and public scoring contract. The presence of a
high raw reading, a source candidate, or natural variability alone does not establish
an injected target source. Classify external point input, distributed external input,
internal sediment release, or physical sediment resuspension when supported.
You choose all measurements and all conclusions. No source or unmeasured water value
is available from the tools. z is depth positive down. Native geometry is public.
Reason normally, then use tools. You may use query for arbitrary read-only numerical
analysis of collected evidence. Tool calls execute in listed order. Queries cost no
movement allowance. No physical travel cutoff applies; compute and movement budgets
are separate. Data and all depths remain exactly queryable for the entire episode.
Recent conversation is retained; older exchanges may leave the context. Persistent
notes and report drafts survive, and all measurements remain in SQL. Save your own
notes when needed. Only measured values are observations; fitted values are predictions.
Report source_presence_probability and source_confidence as probabilities, not percentages.
source_confidence means probability that your proposed source coordinate is within the
published spatial tolerance, not probability that some source exists. You can infer an
unvisited source. L2 must cite an actually measured node. A null answer is allowed when
unknown and is scored as missing only where that level applies. Do not fabricate an
answer for a control. Save partial reports if useful; you may revise every field until
final=true. decision=declare ONLY claims that a detectable pollution source exists at
your source_position. Never use declare to announce absence; report source_present=false
and decision=cannot_determine for an absence conclusion. In report-only mode complete/revise the report using the existing evidence.
No private reasoning transcript is used to judge scientific correctness.
"""


def system_prompt(protocol):
    if protocol.get("scientific_workbench"):
        header = "Clocked task: L2 must cite plume_sample_id (sample_id in samples), not just a node. The static fit_source tool is unavailable; use acquisition times in analyze.\n" if protocol.get('temporal_observations') else ''
        if protocol.get('complete_survey_version'):
            header += ("For a complete initial geometric overview, survey(pattern='coarse_3d_full', fit_after=false) "
                       "executes all 18 public-geometry waypoints in one tool call (up to 48 movement chunks). "
                       "This is optional and not exhaustive lake coverage; target follow-up columns and depths from observed evidence.\n")
        if protocol.get('workbench_api',1)>=2:
            header += "Python analyze provides df (pandas DataFrame of target_nodes), rows (the same rows as dicts), np, pd and data. df already joins background_median/lod. Start with print(df.columns.tolist()) or print(df.nlargest(5,'target_value').to_string(index=False)); no manual calibration join is needed.\n"
        if protocol.get('progress_version', 0) >= 2:
            if protocol.get('evidence_mode') == 'fixed':
                header += "EVIDENCE MODE: fixed. All permitted samples are already supplied; use existing observations.\n"
            elif protocol.get('temporal_observations'):
                header += "EVIDENCE MODE: active, time-varying. Movement acquires spatial measurements at their acquisition times.\n"
            else:
                header += "EVIDENCE MODE: active, static spatial field. You initially know only measured nodes. Moving through unmeasured nodes acquires NEW water-quality and flow measurements. Static means frozen in time, NOT fully observed or unable to sample. Unknown grid values are null, not zero.\n"
            header += "Use df as the provided DataFrame. Column names are case-sensitive: ph, not pH; do not overwrite df with a list. Batch related spatial/depth/channel comparisons into one analyze call and reuse its output.\n"
        prompt = header + """Investigate a possible water pollution event. Source existence, mechanism, location
and release history are unknown. Answer L1 existence, L2 affected location, L3 mechanism
and L4 source location independently, with evidence and uncertainty. A concentration
maximum is not necessarily the source. Target channel and candidate catalogs are
monitoring information, not proof of a source or its mechanism. z is depth positive down.

Choose observations that distinguish competing explanations. survey batches geometry-only
travel; inspect remaining_points before assuming a survey is complete. query retrieves
exact observed data; analyze runs Python (np and data are provided; scipy/pandas/matplotlib
available) for statistics, comparisons and inverse hypotheses. Query/compute never create
measurements. Use actual flow observations with their uncertainty; do not invent known
release ages or physical coefficients. fit_source only tests an explicit point-pulse
hypothesis. Model fits, plots and residuals are predictions, not new observations or truth.

Read the public temporal and scoring contract. In frozen-F1 tasks, moving/revisiting cannot
reveal environmental evolution. Broad calibration is not a matched current background and
LoD is a noise scale, not a decision threshold. Compare spatial, depth and channel evidence.
Mechanism classes describe registered interventions; real processes can be ambiguous.

Use short notes and report drafts to preserve evidence and competing hypotheses. Old
conversation may be dropped; all acquired data remain queryable. Computations start fresh.
Stop when supported or the remaining budget cannot resolve the question. In fixed-evidence
mode, sampling is already complete; analyze the shared observations. Review allows analysis
and report; report-only allows report. Several ordered tools may share a response.

source_present is true/false/null (unknown); uncertainty does not erase supported tentative
L2/L3/L4 estimates. source_presence_probability is in [0,1]. source_confidence is the joint
probability of presence and a correct location, and cannot exceed presence probability.
L2/evidence IDs must be measured nodes; source coordinates can be unvisited. Optional
region IDs predict navigation-node sets: [] asserts empty, null is unknown. declare affirms
a present located source; cannot_determine may accompany a tentative coordinate. Submit
all required report fields with final=true. Tools only correct interface errors, never
scientific answers; no private reasoning transcript is scored.
"""
        if protocol.get('progress_version', 0) >= 2:
            prompt = prompt.replace('In fixed-evidence\nmode, sampling is already complete; analyze the shared observations. ', '')
        if protocol.get('stability_version'):
            prompt = prompt.replace('all required report fields with final=true', 'all required fields using submit_report')
            prompt += '\nUse save_report to persist tentative fields and submit_report to finish; neither accepts a final argument. Prefer inspect for standard summaries/profiles/time series, analyze for calculations not covered by inspect.\n'
        return prompt
    if protocol.get("independent_levels"):
        return """Investigate a possible pollution event in a 3-D lake with a mobile sensor.
The monitoring channel is given, but source existence, mechanism, position, release
history and transport parameters are NOT given. Assess four questions independently:
L1: does the evidence support a target source event, absence, or insufficient evidence?
L2: which actually measured node best represents an elevated target footprint?
L3: which mechanism hypothesis best explains the observations?
L4: where is the source most plausibly located, allowing an unvisited coordinate?
An uncertain or negative L1 does not erase supported tentative L2/L3/L4 hypotheses.
Missing source coordinates do not establish absence. Do not invent unsupported answers.

Start with spatial AND depth contrasts, using survey(pattern="coarse_3d") or your
own points. Inspect remaining_points and measured coverage before assuming the survey
is complete. Use query to inspect exact values and competing explanations. Compare
same-depth horizontal contrasts, vertical profiles, other channels and measured flow.
The compact evidence summary is descriptive, not a significance test or source label.
Broad historical calibration is not a matched current no-source counterfactual; its
LoD is a sensor noise scale. Neither exceeding it nor staying below its median decides L1.
All observations sample one frozen F1 field: revisits can reduce sensor noise but
cannot reveal event evolution, release age or persistence. Time spent moving is not
new environmental time. z is depth positive down.

Consider localized external input (external_point), spatially distributed external
input (external_diffuse), sediment chemical release (internal_release), and physical
sediment mobilization (physical_resuspension). Compare plausible alternatives using
spatial/depth/multichannel evidence; a target channel or candidate type alone does
not identify mechanism. These are registered scenario classes, not exclusive laws
of nature. A static snapshot may leave mechanisms or source positions ambiguous.
Choose follow-up samples that could distinguish your leading hypotheses, including
background comparisons. Concentration maxima need not be injection locations.
fit_source is optional: it tests a point-pulse hypothesis only, requires explicitly
chosen unknown physics, and is neither a general lake solver nor an L1/L3 classifier.
A low residual cannot establish that assumed age, flow or diffusivity is correct.

Work efficiently. Batch movements with survey; several ordered tools may share a
response. Queries cannot collect new observations. Save concise notes and a report
draft as useful evidence develops; all measured data remain queryable. Stop when
supported or further sampling cannot resolve ambiguity within budget.
Give source_present=true, false or null explicitly; null is not an absence conclusion.
source_presence_probability is your probability of source presence. source_confidence
is the JOINT probability of source existence and your coordinate meeting tolerance.
Both are in [0,1]. A tentative coordinate with low confidence and cannot_determine is
allowed. declare affirms a present source AT that coordinate; it never declares absence.
L2 and evidence IDs must be measured nodes; L4 coordinates may be inferred.
In explanation briefly connect the four answers to measured evidence and remaining
alternatives. No private reasoning transcript is scored. Tools never fill your report.
Review mode permits analysis/report only; report-only mode permits report alone.
Finalize with final=true and all missing fields; rejected reports remain revisable.
"""
    if protocol.get("efficient_tools"):
        design = """For an initial overview, survey(pattern=\"coarse_3d\") provides geometry-only transects
at three depths; you may instead choose explicit points. Native grid spacing is public:
nearby requested coordinates may snap to the same node and add no new observation.
When a transport model is publicly registered, survey automatically refreshes the
numerical fit after sampling (fit_after=false disables it). Use the updated fit, not
an earlier fit made before new measurements. Reports remain entirely your choice.
""" if protocol.get("survey_design") else ""
        return design + """Use measured evidence and declared physics to investigate a source in 3-D (z is depth, positive down).
The task explicitly states whether source existence is given or must be inferred. In
source_confirmed tasks, focus on localization; existence and source type are not scored
as discoveries. Otherwise distinguish absence from insufficient evidence.
Work efficiently: use survey for several spatial/depth transects in one call; use query
for exact measured data and fit_source for numerical source hypotheses when its physical
assumptions apply. Calls in one response execute in order, so survey then fit_source can
share a response. The next call sees all measurements acquired by earlier calls.
fit_source uses only measured values and public geometry. Inspect training AND held-out
residuals, competing positions and domain coverage. Its output is a prediction, not truth
or a calibrated confidence probability. A poor fit or a broad candidate set calls for
discriminating measurements or an uncertain report. Concentration maximum is not generally
the source: transport can displace it. You choose sampling and the final scientific answer;
tools never fill your report. Do not assume undeclared lake physics is known.
Save a provisional report and short notes as useful evidence develops. All observations
remain queryable, while only recent conversation and the latest fit remain in context.
Stop when evidence supports a decision, or further observations are unlikely to resolve
ambiguity within budget. Do not spend calls repeating unchanged queries or self-confirmation.
L2 cites an actually measured high-value node. L4 may infer an unvisited coordinate.
source_confidence is the joint probability that a source exists and the proposed coordinate
meets tolerance (conditional on the given existence fact in source_confirmed tasks).
Use probabilities in [0,1]. A tentative coordinate can accompany cannot_determine and low
confidence; unsupported fields may be null. declare affirms a present source AT the coordinate.
review_only allows query, fit_source and report; report_only allows report alone. Finalize
with final=true, supplying missing fields; corrections remain possible after a rejected report.
No private reasoning transcript is scored. Fitted values never become sensor observations.
"""
    if not protocol.get("evidence_review"):
        return SYSTEM
    return """Investigate a possible pollution source with a mobile sensor in a 3-D lake.
Use the public monitoring target, background references and scoring contract.
Weigh evidence for source presence and absence symmetrically. A high reading alone
does not prove a source, and failure to locate a source does not prove absence.
source_present=false is an absence conclusion; null means evidence is insufficient.
Assess the measured plume location, mechanism hypothesis and source-coordinate
hypothesis separately from your presence decision. Report supported tentative
estimates with appropriate uncertainty; do not automatically leave all downstream
fields null after a negative/uncertain L1 decision. Do not invent unsupported answers.
decision=declare affirms a present source at your coordinate. cannot_determine may
accompany a tentative coordinate; it does not require erasing your best estimate.
source_presence_probability is the probability of a detectable target source.
source_confidence is the joint probability that a source exists and your proposed
coordinate meets the published tolerance, not a probability conditional on presence.
Both probabilities use 0-1. Location confidence may be null without a coordinate.
L2 must cite an actually measured node; L4 can infer an unvisited coordinate.
Use all measured evidence, including depth, spatial contrasts and flow where useful.
The supplied summaries only describe your measurements; they are not source labels,
plume memberships or matched no-source counterfactuals. z is depth positive down.
query performs read-only SQLite calculations on all exact historical observations
and public geometry. Its target_nodes view joins measured node averages to broad
depth-layer calibration; values at unvisited nodes are never available.
Choose all measurements and conclusions yourself. Model-derived values are predictions.
Tool calls execute in order. Querying uses compute but no movement allowance.
Recent conversation may be dropped; your notes, report draft and measurements persist.
Save a provisional report as evidence develops; every field remains revisable.
In review_only mode you can query existing measurements and revise reports, but cannot
move. In report_only mode issue a concise final report with all missing fields.
No private reasoning transcript is used to judge scientific correctness.
"""


class Session:
    def __init__(self, store, instance_id, seed, protocol, target, *, workspace=None, replay=False):
        if protocol.get("independent_levels") and (
                protocol.get("task_track") != "end_to_end" or set(target) != {"target_channel"}):
            raise ValueError("main L1-L4 condition permits only target_channel; existence and physics must not be supplied")
        environment_options = {}
        if protocol.get("physics_protocol_relative_path"):
            environment_options["physics_protocol_path"] = store.root / protocol["physics_protocol_relative_path"]
        if protocol.get('budget_policy_version') and protocol.get('temporal_observations'):
            environment_options['observation_horizon_only'] = True
        environment = EnvironmentV20_3
        if protocol.get('temporal_observations'):
            from .temporal import TemporalEnvironment
            environment = TemporalEnvironment
        if protocol.get('closure_version'):
            from .sensors import MeasuredEnvironment, ClockedMeasuredEnvironment
            environment = ClockedMeasuredEnvironment if protocol.get('temporal_observations') else MeasuredEnvironment
        self.env = environment(store, instance_id, seed, physical_budget_enabled=protocol.get("physical_budget_enabled", False), **environment_options)
        self.protocol = copy.deepcopy(protocol)
        self.target = copy.deepcopy(target)
        self.env.control_protocol["resource_control"]["max_successful_active_actions"] = protocol["max_movement_actions"]
        self.env.control_protocol["resource_control"]["provider_call_hard_guard"] = protocol["max_provider_calls"] + 10
        self.env.unlimited_provider_calls = bool(protocol.get('unlimited_provider_calls'))
        self.env.separate_provider_actions = bool(protocol.get('cockpit_version'))
        self.report: dict[str, Any] = {}
        self.notes = ""
        self.notebook = []
        self.feedback_receipts = []
        self.final = False
        self.report_only = False
        self.review_only = False
        self.last_analysis = None
        self.analysis_count = 0
        self.compute_count = 0
        self.last_compute = None
        self.survey_progress = None
        self.replay = replay
        self._temporary_workspace = None
        self.workspace = Path(workspace) if workspace is not None else None
        self.confirmed = protocol.get("task_track") == "source_confirmed"
        if self.confirmed and target.get("source_confirmed") is not True:
            raise ValueError("source_confirmed track requires an explicitly registered public confirmation")
        self.db = sqlite3.connect(":memory:")
        self.db.enable_load_extension(False)
        self.channels = sorted(self.env.task["available_channels"])
        self.measurement_columns = ["sample_id", "node_id", "x", "y", "z", "bed_depth", "time_s", "visit", *self.channels, "u", "v", "w"]
        self.db.execute("CREATE TABLE samples(" + ",".join(c + " REAL" for c in self.measurement_columns) + ")")
        self.db.execute("CREATE TABLE navigation(node_id INTEGER PRIMARY KEY,x REAL,y REAL,z REAL)")
        self.db.executemany("INSERT INTO navigation VALUES(?,?,?,?)", [(i, *p) for i, p in enumerate(self.env.graph.positions)])
        self.db.execute("CREATE TABLE calibration(channel TEXT,layer TEXT,background_median REAL,lod REAL)")
        self.db.executemany("INSERT INTO calibration VALUES(?,?,?,?)", [(c, l, *self.env.limits[(self.env.task['waterbody'], c, l)]) for c in self.channels for l in ["surface", "middle", "bottom"]])
        self.db.execute("CREATE VIEW nodes AS SELECT CAST(node_id AS INTEGER) AS node_id,x,y,z,bed_depth,COUNT(*) AS sample_count,MAX(time_s) AS last_time_s," + ",".join(f"AVG({c}) AS {c}" for c in [*self.channels, "u", "v", "w"]) + " FROM samples GROUP BY node_id")
        if protocol.get('temporal_observations') or protocol.get('native_grid_version'):
            self.db.execute('DROP VIEW nodes')
            self.db.execute('CREATE VIEW nodes AS WITH ranked AS (SELECT *, COUNT(*) OVER (PARTITION BY node_id) AS sample_count, ROW_NUMBER() OVER (PARTITION BY node_id ORDER BY time_s DESC,sample_id DESC) AS rank FROM samples) SELECT CAST(node_id AS INTEGER) AS node_id,x,y,z,bed_depth,sample_count,time_s AS last_time_s,' + ','.join([*self.channels,'u','v','w']) + ' FROM ranked WHERE rank=1')
        if protocol.get("evidence_review"):
            channel = target["target_channel"]
            if channel not in self.channels:
                raise ValueError("monitoring target must be an available channel")
            self.db.execute(f"""CREATE VIEW target_nodes AS
                SELECT n.*,n.{channel} AS target_value,c.layer,c.background_median,c.lod
                FROM nodes n JOIN calibration c ON c.channel='{channel}' AND c.layer=
                CASE WHEN n.z/n.bed_depth<1.0/3 THEN 'surface'
                     WHEN n.z/n.bed_depth<2.0/3 THEN 'middle' ELSE 'bottom' END""")
        self.loaded = 0
        self._sync()
        if protocol.get("evidence_mode") == "fixed":
            # Identical public geometry-only acquisition for every model, with
            # ordinary measurement noise, traversal accounting and no labels.
            self.survey({"pattern": "coarse_3d", "max_actions": protocol.get("fixed_survey_actions", 8)})
            self.review_only = True

    def close(self):
        self.db.close()
        if self._temporary_workspace is not None:
            self._temporary_workspace.cleanup()

    def required_report(self):
        fields = {"plume_node_id", "source_position", "source_confidence", "decision"} if self.confirmed else set(REQUIRED_REPORT)
        if self.protocol.get('temporal_observations'):
            fields = (fields - {'plume_node_id'}) | {'plume_sample_id'}
        return fields

    def geometry_summary(self):
        axes = {a: sorted({float(p[i]) for p in self.env.graph.positions}) for i,a in enumerate("xyz")}
        return {"nodes": len(self.env.graph.positions), "axes": {a: {"levels": len(v), "min": v[0], "max": v[-1],
                "minimum_spacing_m": min((b-a for a,b in zip(v,v[1:])), default=None)} for a,v in axes.items()},
                "sampling_note": "Coordinates snap to reachable native nodes; sub-grid offsets can repeat the same location without a new measurement."}

    def coarse_survey_points(self):
        axes = [sorted({float(p[i]) for p in self.env.graph.positions}) for i in range(3)]
        xs, ys, zs = axes
        design = self.protocol.get("survey_design")
        if design in {"boundary_depth_interleaved_v2", "boundary_depth_interleaved_v3"}:
            points = []
            depths = sorted({zs[0], zs[len(zs)//2], zs[-1]})
            # Geometry alone sets the same boundary/interior/depth targets for
            # every task. v3 spreads the first action-limited pass across all
            # horizontal bands instead of exhausting the first boundary.
            for band, y in enumerate(sorted({ys[0], ys[len(ys)//2], ys[-1]})):
                for depth, z in enumerate(depths if band % 2 == 0 else depths[::-1]):
                    ends = [xs[0], xs[-1]] if (band * len(depths) + depth) % 2 == 0 else [xs[-1], xs[0]]
                    points.extend({"x": x, "y": y, "z": z} for x in dict.fromkeys(ends))
            if design == "boundary_depth_interleaved_v3":
                points = [points[i + band * 2 * len(depths)]
                          for i in range(2 * len(depths)) for band in range(3)]
            return points
        # Three horizontal transects at each of three depths: 18 waypoints. The
        # ordinary action limit still applies and unfinished waypoints are returned.
        return [{"x": x, "y": y, "z": z} for z in sorted({zs[0], zs[len(zs)//2], zs[-1]})
                for y in [ys[round((len(ys)-1)*q)] for q in (.2,.5,.8)]
                for x in [xs[round((len(xs)-1)*q)] for q in (.2,.8)]]

    def _sync(self):
        new = self.env.sample_ledger[self.loaded:]
        self.db.executemany("INSERT INTO samples VALUES(" + ",".join("?" for _ in self.measurement_columns) + ")",
            [(s["sample_sequence"], s["node_index"], s["x"], s["y"], s["z"], s["bed_depth_m"], s["mission_time_s"], s["visit_index"],
              *(s["channels"][c] for c in self.channels), *(s["flow"][c] for c in ["u_m_s", "v_m_s", "w_m_s"])) for s in new])
        self.loaded = len(self.env.sample_ledger)

    def query(self, sql):
        if not isinstance(sql, str) or len(sql) > 16000:
            raise ValueError("SQL must be a string of at most 16000 characters")
        if self.protocol.get("deterministic_query_version") and re.search(
                r"\b(?:random|randomblob|date|time|datetime|julianday|strftime|unixepoch|changes|total_changes|last_insert_rowid)\s*\(|\bCURRENT_(?:DATE|TIME|TIMESTAMP)\b",
                sql, re.IGNORECASE):
            raise ValueError("Non-deterministic SQL is unavailable for exact replay; use ORDER BY node_id or deterministic coordinate filters")
        self._sync()
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
        def authorize(action, arg1, arg2, database, source):
            if action not in allowed or (action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() in {"load_extension", "readfile", "writefile"}):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        ticks = 0
        def progress():
            nonlocal ticks
            ticks += 1000
            return int(ticks > self.protocol["query_max_instructions"])
        self.db.set_authorizer(authorize)
        self.db.set_progress_handler(progress, 1000)
        try:
            cursor = self.db.execute(sql)
            rows = cursor.fetchmany(self.protocol["query_max_rows"] + 1)
            truncated = len(rows) > self.protocol["query_max_rows"]
            rows = rows[:self.protocol["query_max_rows"]]
            while rows and len(json.dumps(rows).encode()) > self.protocol["query_max_bytes"]:
                rows.pop()
                truncated = True
            return {"columns": [c[0] for c in cursor.description], "rows": rows, "truncated": truncated,
                    "data_scope": "measured samples and public navigation only"}
        finally:
            self.db.set_authorizer(None)
            self.db.set_progress_handler(None, 0)

    def sql_tables(self):
        tables = {"samples": self.measurement_columns,
                  "nodes": ["node_id", "x", "y", "z", "bed_depth", "sample_count", "last_time_s", *self.channels, "u", "v", "w"],
                  "navigation": ["node_id", "x", "y", "z"],
                  "calibration": ["channel", "layer", "background_median", "lod"]}
        if self.protocol.get("evidence_review"):
            tables["target_nodes"] = [*tables["nodes"], "target_value", "layer", "background_median", "lod"]
        return tables

    def evidence_summary(self):
        """Bounded descriptive views, computed exclusively from measured nodes."""
        result = {
            "scope": "Measured node averages only; adaptive sampling is not representative of the whole lake. Ranges and extremes are not source/plume labels.",
            "by_depth": self.query("SELECT z,COUNT(*) AS measured_nodes,MIN(target_value) AS minimum,AVG(target_value) AS mean,MAX(target_value) AS maximum FROM target_nodes GROUP BY z ORDER BY z"),
            "extremes": self.query("""WITH ranked AS (
                SELECT node_id,x,y,z,layer,target_value,background_median,lod,u,v,w,
                    ROW_NUMBER() OVER (ORDER BY target_value DESC,node_id) AS high_rank,
                    ROW_NUMBER() OVER (ORDER BY target_value ASC,node_id) AS low_rank
                FROM target_nodes)
                SELECT node_id,x,y,z,layer,target_value,background_median,lod,u,v,w
                FROM ranked WHERE high_rank<=3 OR low_rank<=3 ORDER BY target_value,node_id""")}
        if self.protocol.get('recovery_version'):
            result['scope'] = 'Latest complete measured sample at each native node; depth means average across those nodes, not across repeat samples. Adaptive sampling is not representative of the whole lake. Ranges and extremes are not source/plume labels.'
        if self.protocol.get("independent_levels"):
            result["contrasts"] = self.measured_contrasts()
        return result

    def measured_contrasts(self):
        """No truth, full-field statistics, assumed physics or significance labels."""
        self._sync()
        columns = ["node_id", "x", "y", "z", "layer", "target_value", *self.channels]
        rows = [dict(zip(columns, row)) for row in self.db.execute(
            "SELECT " + ",".join(columns) + " FROM target_nodes ORDER BY node_id")]
        depth_groups = {}
        for row in rows:
            depth_groups.setdefault(row["z"], []).append(row)
        ranked = []
        for depth, group in depth_groups.items():
            values = [r["target_value"] for r in group]
            center = median(values)
            scale = max(1.4826 * median([abs(v-center) for v in values]), .002 * max(abs(center), 1))
            if len(group) >= 4:
                ranked.extend({k: r[k] for k in ("node_id", "x", "y", "z", "target_value")} |
                              {"sample_depth_contrast": (r["target_value"]-center)/scale,
                               "reference_nodes": len(group)} for r in group)
        multichannel = []
        for layer in ("surface", "middle", "bottom"):
            group = [r for r in rows if r["layer"] == layer]
            if group:
                multichannel.append({"layer": layer, "nodes": len(group),
                    "channel_min_median_max": {c: [min(r[c] for r in group), median([r[c] for r in group]),
                                                   max(r[c] for r in group)] for c in self.channels}})
        return {"coverage": {"measured_nodes": len(rows), "navigation_nodes": len(self.env.graph.positions),
                    "depths_measured": sorted(depth_groups),
                    "measured_bounds": {a: [min(r[a] for r in rows), max(r[a] for r in rows)] for a in "xyz"}},
                "largest_sample_depth_contrasts": sorted(ranked, key=lambda r: (-r["sample_depth_contrast"], r["node_id"]))[:5],
                "multichannel": multichannel,
                "interpretation": "Ranks use measured same-depth median/MAD (at least 4 nodes), not the hidden full-field L2 threshold. Adaptive sample coverage and ranges cannot establish absence. Channel ranges describe co-sampled nodes, not correlation or mechanism."}

    def context(self):
        obs = self.env.observation()
        result = {"instance_id": self.env.task["instance_id"], "monitoring_target": self.target,
                "contract": {k: self.protocol[k] for k in ["condition", "L1", "L2", "L3", "L4", "L5"]},
                "source_tolerance_m": self.env.store.protocol["tolerances"][self.env.task["waterbody"]],
                "domain_bounds": self.env.domain_bounds, "depth_axis": "positive_down",
                "candidates": obs["candidates"], "candidate_semantics": obs.get("candidate_catalog_semantics"),
                "calibration": self.query("SELECT * FROM calibration"),
                "calibration_note": "Values use the same units as measurements. LoD is a background-MAD-derived noise scale; transit sensor sigma=LoD, initial sigma=LoD/3. Background median is a broad waterbody/depth reference, not the current local background. Layer is z/bed_depth: surface<1/3, middle<2/3, bottom otherwise. Flow noise sigma=0.0025m/s; w has low SNR.",
                "sql_tables": self.sql_tables(),
                "sql_note": "samples retains every raw observation; nodes averages repeat measurements at each exact 3D node. Channels are columns, not rows: SELECT node_id,x,y,z,fdom FROM nodes ORDER BY fdom DESC LIMIT 20. There is no channel column in samples/nodes. SQLite has no STDDEV; use sqrt(MAX(0,AVG(fdom*fdom)-AVG(fdom)*AVG(fdom))). navigation contains geometry only. SELECT ... LIMIT 128 OFFSET n retrieves another page. Queries cannot collect new water evidence; move/goto to sample more nodes."}
        if self.protocol.get("evidence_review"):
            result["sql_note"] = "Use one SELECT/WITH statement per query call; multiple query tool calls may share a response. Use CTEs within a statement instead of temporary tables. target_nodes contains your monitoring target as target_value, with layer/background_median/lod already joined. bed_depth is only in measured tables; layer is in target_nodes/calibration. navigation has only node_id,x,y,z. No STDDEV; use sqrt(MAX(0,AVG(target_value*target_value)-AVG(target_value)*AVG(target_value))). LIMIT/OFFSET paginate; null division by zero is not zero evidence."
            result["query_examples"] = [
                "SELECT node_id,x,y,z,target_value,layer,u,v,w FROM target_nodes ORDER BY target_value DESC,node_id LIMIT 20",
                "SELECT layer,COUNT(*) AS n,MIN(target_value),AVG(target_value),MAX(target_value) FROM target_nodes GROUP BY layer",
                "SELECT node_id,x,y,z,(target_value-background_median)/NULLIF(lod,0) AS reference_contrast FROM target_nodes ORDER BY reference_contrast DESC LIMIT 20"]
        if self.protocol.get("deterministic_query_version"):
            result["sql_note"] += " Queries must replay exactly: RANDOM() and time-dependent SQLite functions are rejected. Use ORDER BY node_id or deterministic coordinate filters."
        if self.protocol.get("efficient_tools"):
            result["workflow"] = "survey spatial/depth contrasts -> fit_source if appropriate -> compare residuals/alternatives -> discriminating survey if needed -> report. Multiple ordered tools can share a response."
            result["physics"] = self.target.get("transport_model")
            result["source_existence"] = "Given by task registration; not evaluated as model detection" if self.confirmed else "Not given; evaluate event presence separately from location"
            result["fit_note"] = "fit_source accepts the public transport_model by default; without it you must explicitly supply a hypothetical model. Neither measured flow nor a broad reference automatically establishes the full transport model."
            actual_flow_noise = self.env.store.protocol["sensor_noise"]["flow_sigma_m_s"]
            result["calibration_note"] = result["calibration_note"].replace("Flow noise sigma=0.0025m/s", f"Flow noise sigma={actual_flow_noise:g}m/s")
        if self.protocol.get("survey_design"):
            result["navigation_geometry"] = self.geometry_summary()
            result["survey_pattern"] = "coarse_3d uses three horizontal transects at three native depths, chosen from geometry alone. max_actions may stop it early; resume with the returned remaining_points. Auto-fit uses only publicly registered transport, never inferred hidden physics."
        if self.protocol.get("independent_levels"):
            result["workflow"] = "Survey spatial/depth contrasts -> assess L1/L2/L3/L4 independently -> sample to distinguish alternatives -> report evidence, hypotheses and uncertainty. fit_source is optional and requires hypothetical physics."
            if self.protocol.get("survey_design") == "boundary_depth_interleaved_v3":
                result["survey_pattern"] = "coarse_3d interleaves boundary/interior/boundary bands and depths using public geometry only, so an action-limited first pass spans the site. Check complete and resume remaining_points if needed, or choose discriminating points yourself."
            else:
                result["survey_pattern"] = "coarse_3d alternates depth transects across boundary/interior/boundary bands; all points use geometry only. Resume remaining_points if incomplete, or choose discriminating points yourself."
            result["observation_time"] = "Frozen F1 snapshot; movement/revisits do not advance the environmental field."
        if self.protocol.get("scientific_workbench"):
            result["science"] = {
                "evidence_mode": self.protocol.get("evidence_mode", "active"),
                "analysis": "analyze(code) executes isolated Python with numpy as np and data as a dict of row lists: samples, nodes, target_nodes, navigation, calibration. scipy, pandas and matplotlib are installed. Print concise numerical evidence. Plots are saved as artifacts; they are not automatically sent to the model. No network, private files or unmeasured values. Each call starts fresh; only measured records persist.",
                "example": "rows=data['target_nodes']; print([(r['node_id'], r['target_value']) for r in sorted(rows, key=lambda r:r['target_value'], reverse=True)[:5]])",
                "limits": {"calls": self.protocol.get("max_compute_calls", 6), "cpu_seconds_per_call": 10, "wall_seconds_per_call": 15},
                "acquisition": "Fixed geometry-only evidence already acquired; sampling is closed." if self.protocol.get("evidence_mode") == "fixed" else "Choose your own observations; batching charges every traversed edge.",
            }
        if self.protocol.get("spatial_reports"):
            result["spatial_reports"] = "Optional plume_region_node_ids and source_region_node_ids predict sets of public navigation node IDs, including unmeasured nodes. [] asserts an empty set; null means unknown. Set precision/recall and IoU are scored separately from a single-point hit."
            if self.protocol.get('source_boundary_version'):
                result['spatial_reports'] += ' In this version source_region_node_ids describe receiving water cells only; physical source_position is scored against the release boundary surface, which may lie between navigation nodes.'
        if "observation_contract" in self.env.task:
            result["observation_contract"] = copy.deepcopy(self.env.task["observation_contract"])
        if self.protocol.get('native_grid_version'):
            result['sql_note'] += ' nodes/target_nodes use the latest complete sample per native node, not repeat averages. samples preserves every earlier observation; navigation includes all unknown nodes.'
        if self.protocol.get('temporal_observations'):
            result['observation_time'] = 'Dynamic field: travel and wait advance mission time. Samples retain exact acquisition times. nodes/target_nodes use the latest sample at each node, NEVER an average across different times; spatial summaries may combine different times. Use samples for time-series analysis.'
            result['temporal_reporting'] = 'L2 plume_sample_id cites one actually acquired sample, or null when unknown. Its source contribution is scored at its acquisition time. Optional plume_region_node_ids predict the affected-node union over the registered observation window. wait(seconds) acquires a repeat at the current node and charges environment time.'
        if self.protocol.get('closure_version'):
            result.pop('query_examples', None)
            result['workflow'] = 'Use spatial_memory and inspect for measured evidence; choose discriminating sampling or analyze for custom calculations; retain alternatives and report all four judgments. No forced order.'
            result['science']['analysis'] = 'analyze(code): Python with df/rows/data/np/pd. Syntax checked before imports. Recorded text/JSON/CSV artifacts can be read later with inspect. No private files or network.'
            result['science']['limits'] = {'cpu_s':10, 'startup_wall_s':30, 'execution_wall_s':15}
            if self.protocol.get('temporal_observations'):
                result['temporal_reporting'] = 'L2 point cites an actual plume_sample_id. Region IDs predict source-excess extent at report/mission time, not future union. Unknown is null; explicit empty is [].'
            result['observation_contract'] = copy.deepcopy(self.env.task.get('observation_contract', {}))
        return result

    def state(self, remaining_calls, cost):
        self.env.apply_resource_stage()
        position = self.env._position_snapshot()
        result = {"position": position, "measured_nodes": len(self.env.measurements), "raw_samples": self.loaded,
                "mode": "report_only" if self.report_only else self.env.mode,
                "instruction": ("Investigation has ended. Call report with final=true and ALL missing_report_fields explicitly supplied, using null for unknown fields. In particular source_present must be true, false or null; probability alone does not supply it." if self.report_only else "Investigate, keep your own notes, and save/revise a report when useful."),
                "movement_actions_used": self.env.successful_action_count,
                "movement_actions_limit": self.protocol["max_movement_actions"],
                "provider_calls_remaining": remaining_calls, "cost_cny_proxy": cost,
                "saved_report": self.report, "missing_report_fields": sorted(self.required_report() - self.report.keys()),
                "notes": self.notes, "recent_measurements": self.query("SELECT * FROM samples ORDER BY sample_id DESC LIMIT 4")}
        if self.protocol.get("survey_progress_version"):
            result["survey_progress"] = self.survey_progress or {"status": "not_started"}
        if self.protocol.get("evidence_review"):
            result["evidence_summary"] = self.evidence_summary()
            if self.review_only and not self.report_only:
                result["mode"] = "review_only"
                result["instruction"] = "Sampling has ended. Query existing evidence if needed and save or finalize your report now. All fields remain revisable; final requests are reserved for reporting and correction."
            elif not self.report_only:
                result["instruction"] = "Investigate and save a provisional report as evidence develops. Assess each level separately; unknown source location does not imply source absence."
        if self.protocol.get("efficient_tools"):
            result["latest_source_fit"] = self.last_analysis
            result["source_fit_has_new_unfitted_samples"] = bool(self.last_analysis and self.last_analysis["sample_count_at_fit"] != self.loaded)
            result["numerical_fits_used"] = self.analysis_count
            if self.confirmed:
                result["source_existence_given"] = True
                result["instruction"] = "Call report with final=true and the missing fields." if self.report_only else "Localize the confirmed source. Batch spatial/depth sampling, fit only declared assumptions, assess residuals and ambiguity, then report."
        if self.protocol.get("scientific_workbench"):
            result["latest_computation"] = None if self.last_compute is None else {
                k: (v[:1600] if k in {'stdout','stderr'} else v) for k,v in self.last_compute.items()}
            result["computation_has_new_samples"] = bool(self.last_compute and self.last_compute["observation_samples"] != self.loaded)
            result["computation_calls_remaining"] = self.protocol.get("max_compute_calls", 6) - self.compute_count
        if self.protocol.get('closure_version') and self.protocol.get('spatial_memory_enabled',True):
            from .evidence import memory
            result.pop('evidence_summary', None)
            result['spatial_memory'] = memory(self)
            result['computation_artifacts'] = [{'analysis_id':i, 'receipt':'recorded; inspect(view=artifact, analysis_id, artifact, offset, limit)'} for i in range(1, self.compute_count+1)]
        if self.protocol.get('budget_policy_version'):
            result['sampling_budget'] = {
                'endurance_limit': None, 'range_limit': None,
                'samples_used': len(self.env.sample_ledger),
                'observation_horizon_s': getattr(self.env, 'frame_times', [None])[-1],
                'mission_time_s': self.env.budget.used_s,
                'note': 'Every arrival is measured, including repeat visits. Movement count is an engineering guard; temporal coverage is a data boundary.'}
        return result

    def tools(self):
        available = list(TOOLS)
        if self.protocol.get("deterministic_query_version"):
            available = copy.deepcopy(available)
            next(t["function"] for t in available if t["function"]["name"] == "query")["description"] += " Exact replay requires deterministic SQL: RANDOM() and time-dependent functions are rejected; sort by node_id or coordinates."
        if self.protocol.get('closure_version') and self.protocol.get('structured_inspect_enabled',True):
            from .evidence import PROPERTIES
            available += [tool('inspect', 'Read measured spatial grid, samples, depth profiles, time series, descriptive correlations or recorded computation artifacts without writing code. Filter by bounds/node_ids/sample_ids/time_window_s; paginate with offset/limit. No hidden values or automatic scientific conclusions.', PROPERTIES, ['view'])]
        if self.protocol.get("efficient_tools"):
            xyz = {"type": "object", "properties": {a: {"type": "number"} for a in "xyz"}, "required": list("xyz"), "additionalProperties": False}
            vector = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
            physics = {"type": "object", "properties": {"model": {"type": "string", "enum": ["uniform_flow_gaussian_pulse"]}, "age_s": {"type": "number", "exclusiveMinimum": 0}, "diffusivity_m2_s": vector, "velocity_m_s": vector}, "required": ["model", "age_s", "diffusivity_m2_s", "velocity_m_s"], "additionalProperties": False}
            available += [tool("survey", "Sample an ordered list of up to 12 coordinates along measured shortest routes. Reaches each target using 32-edge chunks, up to max_actions total. Returns compact evidence; exact readings remain in query. No unobserved values guide travel.", {"points": {"type": "array", "items": xyz, "minItems": 1, "maxItems": 12}, "max_actions": {"type": "integer", "minimum": 1, "maximum": 12}}, ["points"]),
                          tool("fit_source", "Fit a point-pulse transport hypothesis to all measured nodes (at least 12). Return competing source coordinates, fitted mass/background, held-out residuals and suggested measurements. Does not classify mechanism or fill the report. Use task transport_model by default; otherwise supply explicit hypothetical physics. Candidate bounds are optional and never read unobserved water values.", {"physics": physics, "bounds": {"type": "object", "properties": {"min": xyz, "max": xyz}, "required": ["min", "max"], "additionalProperties": False}})]
            if self.protocol.get("survey_design"):
                survey = next(t["function"] for t in available if t["function"]["name"] == "survey")
                survey["description"] = "Sample explicit points OR pattern=coarse_3d for a geometry-only initial survey at three depths. At most 12 movement actions; returns remaining_points if unfinished. A public transport model is automatically refitted after sampling; fit_after=false disables this. Unknown physics is never supplied automatically."
                survey["parameters"]["required"] = []
                survey["parameters"]["properties"]["points"]["maxItems"] = 18
                survey["parameters"]["properties"].update(pattern={"type":"string","enum":["coarse_3d"]}, fit_after={"type":"boolean"})
                if self.protocol.get("complete_survey_version"):
                    survey["description"] = ("Sample explicit points (max_actions up to 48), a partial geometry-only "
                        "coarse_3d pass, or coarse_3d_full: all 18 geometry-only waypoints in one call, "
                        "up to 48 movement chunks regardless of max_actions. Every traversed node is measured. "
                        "Returns compact evidence and remaining_points; exact observations stay in query. "
                        "Use discriminating follow-up columns/depths after the overview. fit_after=false disables a point-pulse fit.")
                    survey["parameters"]["properties"]["pattern"]["enum"].append("coarse_3d_full")
                    survey["parameters"]["properties"]["max_actions"]["maximum"] = 48
        if self.protocol.get("scientific_workbench"):
            available += [tool("analyze", "Execute Python on all your measured evidence. data contains row lists for SQL tables; np is NumPy; scipy/pandas/matplotlib imports allowed. Print concise findings. No private files/network/unmeasured water data. Saved plots are artifacts, not new observations. Fresh process each call, CPU 10s, wall 15s.", {"code": {"type": "string", "maxLength": 16000}}, ["code"])]
        if self.protocol.get('temporal_observations'):
            available = [t for t in available if t['function']['name']!='fit_source']
            available += [tool('wait', 'Wait at the current node, advancing the dynamic environment and acquiring one observation. Charges physical time and one movement/observation action.', {'seconds':{'type':'number','minimum':1,'maximum':300}}, ['seconds'])]
        if self.protocol.get('closure_version') and not self.protocol.get('solver_enabled',True):
            available=[t for t in available if t['function']['name']!='fit_source']
        selected = copy.deepcopy([t for t in available if (t["function"]["name"] == "report" if self.report_only else
                t["function"]["name"] in ({"query", "fit_source", "analyze", "report", "inspect"} if self.protocol.get("continuity_version") else {"query", "fit_source", "analyze", "report"}) if self.review_only or self.env.mode == "submit_only" else True)])
        if self.protocol.get("spatial_reports"):
            for item in selected:
                if item["function"]["name"] == "report":
                    for name in ("plume_region_node_ids", "source_region_node_ids"):
                        item["function"]["parameters"]["properties"][name] = {"type": ["array", "null"], "items": {"type": "integer"}, "maxItems": 4096, "description": "Predicted public navigation node set, allowing unmeasured nodes. [] asserts empty; null is unknown. Scope and false coverage are evaluated."}
        if self.protocol.get('temporal_observations'):
            for item in selected:
                if item['function']['name']=='report':
                    item['function']['parameters']['properties'].pop('plume_node_id',None)
                    item['function']['parameters']['properties']['plume_sample_id']={'type':['integer','null'],'description':'Actual sample_id from samples, identifying the affected node AND acquisition time; null if unknown.'}
        if self.report_only:
            selected[0]["function"]["parameters"]["required"] = ["final", *sorted(self.required_report() - self.report.keys())]
        if self.protocol.get("nullable_location_confidence", False):
            for item in selected:
                if item["function"]["name"] == "report":
                    field = item["function"]["parameters"]["properties"]["source_confidence"]
                    field["type"] = ["number", "null"]
                    field["description"] = "Probability that a proposed coordinate is correct. May be null when source_position is null; a proposed coordinate requires a numeric probability."
        if self.protocol.get("evidence_review"):
            for item in selected:
                if item["function"]["name"] == "report":
                    fields = item["function"]["parameters"]["properties"]
                    fields["mechanism"]["description"] = "Best-supported source mechanism hypothesis, assessed independently of L1. null if unknown."
                    fields["source_position"]["description"] = "Best-supported tentative coordinate, even with uncertain presence or cannot_determine. null if no supported estimate."
                    fields["source_confidence"]["description"] = "Joint probability that a source exists and this coordinate is within tolerance, not conditional on existence. null is allowed only without a coordinate."
        if self.confirmed and self.protocol.get("focused_report_schema"):
            for item in selected:
                if item["function"]["name"] == "report":
                    fields = item["function"]["parameters"]["properties"]
                    for name in ("source_present", "source_presence_probability", "mechanism", "source_candidate_id"):
                        fields.pop(name, None)
                    fields["plume_node_id"]["description"] = "An actually measured node, for example measured_target_maximum.node_id when supported. Never cite an unmeasured source_grid_node_id. null if unknown."
                    fields["source_position"]["description"] = "Your source-coordinate hypothesis; may be unmeasured. Source coordinates and measured plume/evidence IDs are different objects."
        if self.protocol.get('closure_version'):
            for item in selected:
                function = item['function']
                if function['name']=='analyze':
                    function['description']='Execute Python with measured df/rows/data/np/pd. Syntax validated before imports. Each call starts fresh: startup wall 30s, execution wall 15s, total CPU 10s. Inspect saved text/JSON/CSV artifacts by analysis_id. No private files/network.'
                if function['name']=='fit_source':
                    function['description']='Compare explicit Gaussian point-pulse hypotheses on measured nodes and public geometry in bounded chunks (12-65536 measured nodes, <=65536 candidates). Requires your hypothetical physics, never true parameters. This point-source approximation is not a validated mechanism classifier.'
        if self.protocol.get('continuity_version'):
            for item in selected:
                function=item['function']
                if function['name']=='inspect':
                    function['parameters']['properties']['view']['enum'].append('feedback')
                    function['parameters']['properties']['receipt_id']={'type':'integer','minimum':1}
                    function['description'] += ' feedback retrieves an exact retained tool receipt by receipt_id; offset/limit count JSON lines for this view.'
                if function['name']=='report' and self.report_only:
                    function['parameters']['properties']['final']['const']=True
        if self.protocol.get('native_grid_version'):
            for item in selected:
                f = item['function']
                if f['name'] == 'survey':
                    f['parameters']['properties']['revisit'] = {'type':'boolean', 'description':'Default false: skip already measured targets, including incidental arrivals. True: deliberately return; already at target does not create a new sample. Use wait for a later dynamic observation.'}
                    f['description'] = 'Explore explicit points or pattern=coarse_3d using native nodes. Every traversed arrival is sampled. Skip visited targets by default; revisit=true permits deliberate return. Shortest routes prefer fewer visited arrivals when equally short; necessary transit is legal. reached confirms position, not a new sample; complete confirms all requested targets were handled.'
                if f['name'] == 'inspect':
                    f['description'] += ' grid returns exact native nodes including unknown nodes, all channels by default, without binning. ever_visited/visit_count refer to full history, values to latest selected sample. Read pages before planning outside the current window.'
        if self.protocol.get('stability_version'):
            expanded = []
            for item in selected:
                if item['function']['name'] != 'report':
                    if item['function']['name'] == 'inspect':
                        properties = item['function']['parameters']['properties']
                        properties['channel']['enum'] = list(self.channels)
                        properties['channels']['items']['enum'] = list(self.channels)
                    expanded.append(item)
                    continue
                for name in ('save_report', 'submit_report'):
                    if self.report_only and name == 'save_report':
                        continue
                    entry = copy.deepcopy(item)
                    f = entry['function']
                    f['name'] = name
                    f['description'] = ('Save tentative report fields without ending the investigation.' if name == 'save_report' else
                        'Submit your saved report, optionally updating fields. Ends only after validation. Unknown scientific answers must be explicit nulls.')
                    f['parameters']['properties'].pop('final', None)
                    f['parameters']['required'] = (sorted(self.required_report() - self.report.keys())
                        if name == 'submit_report' and self.protocol.get('delivery_version') else [])
                    expanded.append(entry)
            selected = expanded
        return selected

    def plan_goto(self, **point):
        if self.protocol.get('native_grid_version'):
            from .native_grid import plan_goto
            return plan_goto(self, **point)
        return self.env.plan_goto(**point)

    def survey(self, args):
        args = dict(args)
        revisit = args.pop("revisit", False) if self.protocol.get("native_grid_version") else False
        designed = bool(self.protocol.get("survey_design"))
        allowed = {"points", "max_actions", "pattern", "fit_after"} if designed else {"points", "max_actions"}
        if set(args) - allowed:
            raise ValueError("survey accepts points and max_actions only")
        points, limit = args.get("points"), args.get("max_actions", 8)
        full_pattern = bool(designed and self.protocol.get("complete_survey_version") and
                            args.get("pattern") == "coarse_3d_full")
        fit_after = args.get("fit_after", self.protocol.get("auto_fit_public_physics", False))
        if type(fit_after) is not bool:
            raise ValueError("fit_after must be boolean")
        if designed and "pattern" in args:
            patterns = {"coarse_3d", "coarse_3d_full"} if self.protocol.get("complete_survey_version") else {"coarse_3d"}
            if "points" in args or args["pattern"] not in patterns:
                raise ValueError("use either explicit points or a registered coarse_3d pattern")
            points = self.coarse_survey_points()
        if full_pattern:
            limit = 48
        max_points = 18 if designed else 12
        max_actions = 48 if self.protocol.get("complete_survey_version") else 12
        if not isinstance(points, list) or not 1 <= len(points) <= max_points or type(limit) is not int or not 1 <= limit <= max_actions:
            raise ValueError(f"survey needs 1-{max_points} points and max_actions in 1-{max_actions}")
        for point in points:
            if not isinstance(point, dict) or set(point) != set("xyz") or any(type(v) not in (int, float) or not math.isfinite(v) for v in point.values()):
                raise ValueError("each survey point requires finite x,y,z")
        if self.review_only or self.env.mode == "submit_only":
            raise ValueError("sampling is closed; analyze existing data or report")
        before, actions, reached = self.loaded, 0, []
        stop = None
        skipped = []
        before_nodes = {s['node_index'] for s in self.env.sample_ledger}
        for point in points:
            if self.protocol.get('native_grid_version') and not revisit:
                target = self.env.resolve_goto_target(**point)
                if target in {s['node_index'] for s in self.env.sample_ledger}:
                    skipped.append({'requested': point, 'node_id': target, 'reason':'already_measured'})
                    continue
            while True:
                route, target = self.plan_goto(**point)
                if not route:
                    reached.append({"requested": point, "node_id": target})
                    break
                if actions >= limit or self.env.successful_action_count >= self.protocol["max_movement_actions"]:
                    stop = "action_limit"
                    break
                result = self.env.move(route)
                self._sync()
                if not result["ok"] or not result.get("completed_edges"):
                    stop = result.get("code", "route_stopped")
                    break
                actions += 1
            if stop:
                break
        result = {"ok": True, "reached": reached, "requested_points": len(points), "actions": actions,
                "new_samples": self.loaded - before, "stop_reason": stop, "position": self.env._position_snapshot(),
                "evidence_summary": self.evidence_summary(), "retrieval": f"SELECT * FROM samples WHERE sample_id>{before} ORDER BY sample_id LIMIT 128"}
        if designed:
            if full_pattern:
                result["survey_pattern"] = "coarse_3d_full"
                result["max_actions_effective"] = limit
            result["remaining_points"] = points[len(reached)+len(skipped):]
            result["distinct_reached_nodes"] = len({r["node_id"] for r in reached})
            if self.protocol.get("survey_progress_version"):
                self.survey_progress = {
                    "status": "incomplete" if result["remaining_points"] else "complete",
                    "remaining_points": result["remaining_points"],
                    "message": "Unfinished waypoints have not been sampled. Continue them or select a different discriminating route before interpreting missing coverage as absence."
                    if result["remaining_points"] else "All requested survey waypoints were handled."
                }
            if fit_after and self.target.get("transport_model") is not None and self.loaded > before:
                # Numerical assistance is declared and uses only public physics.
                # A failed fit never erases a successful sampling result.
                try:
                    result["updated_source_fit"] = self.fit({})
                except (ValueError, TypeError, KeyError) as exc:
                    result["fit_feedback"] = str(exc)[:600]
            else:
                result["fit_feedback"] = "No automatic fit: disabled, no newly collected samples, or no public transport model."
        if self.protocol.get('native_grid_version'):
            new_nodes = {s['node_index'] for s in self.env.sample_ledger} - before_nodes
            result.update(skipped_visited=skipped, complete=stop is None,
                          new_nodes=len(new_nodes), repeated_samples=self.loaded-before-len(new_nodes))
        return result

    def fit(self, args):
        from .inverse import fit_source
        if self.protocol.get('temporal_observations'):
            raise ValueError('static point-pulse fitter cannot pool dynamic observations; use analyze with explicit acquisition times')
        if set(args) - {"physics", "bounds"}:
            raise ValueError("fit_source accepts physics and bounds only")
        if self.analysis_count >= self.protocol.get("max_numerical_fits", 12):
            raise ValueError("numerical fit allowance exhausted; query or report remains available")
        physics = args.get("physics", self.target.get("transport_model"))
        bounds = args.get("bounds")
        self._sync()
        sql, params = "SELECT node_id,x,y,z FROM navigation", []
        if bounds is not None:
            if not isinstance(bounds, dict) or set(bounds) != {"min", "max"}:
                raise ValueError("bounds require min and max xyz objects")
            for side in bounds.values():
                if not isinstance(side, dict) or set(side) != set("xyz") or any(type(v) not in (int, float) or not math.isfinite(v) for v in side.values()):
                    raise ValueError("bounds must contain finite xyz coordinates")
            if any(bounds["min"][a] > bounds["max"][a] for a in "xyz"):
                raise ValueError("bounds min exceeds max")
            sql += " WHERE " + " AND ".join(f"{a} BETWEEN ? AND ?" for a in "xyz")
            params = [bounds[s][a] for a in "xyz" for s in ("min", "max")]
        nav = self.db.execute(sql + (" ORDER BY node_id" if self.protocol.get("closure_version") else " ORDER BY node_id LIMIT 4097"), params).fetchall()
        samples = self.db.execute("SELECT node_id,x,y,z,target_value FROM target_nodes ORDER BY node_id" + ("" if self.protocol.get("closure_version") else " LIMIT 4097")).fetchall()
        result = fit_source(samples, nav, physics, result_schema=self.protocol.get("fit_result_schema", 1), **({"max_candidates":65536, "max_observations":65536} if self.protocol.get("closure_version") else {}))
        result["assumptions_origin"] = "model_supplied_hypothesis" if "physics" in args else "public_task_registration"
        result["candidate_bounds"] = bounds
        result["sample_count_at_fit"] = self.loaded
        self.last_analysis = result
        self.analysis_count += 1
        return {"ok": True, **result}

    def _validate_report_field(self, key, value):
        if self.protocol.get('temporal_observations') and key=='plume_sample_id':
            if value is not None and (type(value) is not int or not self.db.execute('SELECT 1 FROM samples WHERE sample_id=?',(value,)).fetchone()):
                raise ValueError('plume_sample_id must identify an acquired sample or be null')
            return
        if self.protocol.get("spatial_reports") and key in {"plume_region_node_ids", "source_region_node_ids"}:
            if value is not None and (not isinstance(value, list) or len(value) > 4096 or any(type(i) is not int or not 0 <= i < len(self.env.graph.positions) for i in value) or len(set(value)) != len(value)):
                raise ValueError("region must be null or up to 4096 distinct public navigation node IDs")
            return
        if key not in REPORT_FIELDS:
            raise ValueError("unknown field")
        if key in {"source_presence_probability", "source_confidence"}:
            if key == "source_confidence" and value is None and self.protocol.get("nullable_location_confidence", False):
                return
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("must be a finite probability from 0 to 1")
        elif key == "source_present":
            if value is not None and type(value) is not bool:
                raise ValueError("must be boolean or null")
        elif key == "plume_node_id":
            if value is not None and (type(value) is not int or value not in self.env.measurements):
                raise ValueError("must identify an actually measured node or be null")
        elif key == "source_position":
            if value is not None and (not isinstance(value, dict) or set(value) != set("xyz") or any(type(v) not in (int, float) or not math.isfinite(v) for v in value.values())):
                raise ValueError("requires finite x,y,z or null")
        elif key == "mechanism" and value not in [*MECHANISMS, None]:
            raise ValueError("unknown mechanism")
        elif key == "decision" and value not in ["declare", "cannot_determine"]:
            raise ValueError("unknown decision")
        elif key == "source_candidate_id" and value is not None and not isinstance(value, str):
            raise ValueError("candidate ID must be a string or null")
        elif key == "evidence_node_ids" and (not isinstance(value, list) or len(value) > 20 or any(type(i) is not int or i not in self.env.measurements for i in value)):
            raise ValueError("cite at most 20 actually measured nodes")
        elif key == "explanation" and (not isinstance(value, str) or len(value) > 1600):
            raise ValueError("explanation must be at most 1600 characters")

    def execute(self, name, args):
        if self.final:
            return {"ok": False, "error": "episode finalized"}
        args = copy.deepcopy(args)
        try:
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
            if self.protocol.get('closure_version'):
                from jsonschema import Draft202012Validator
                declared = next((t['function']['parameters'] for t in self.tools() if t['function']['name']==name), None)
                if declared is None:
                    raise ValueError('tool unavailable in the current phase')
                corrections = []
                if self.protocol.get('stability_version') and name in {'save_report', 'submit_report'}:
                    from .stability import normalize_report
                    args, corrections = normalize_report(args, declared)
                errors = list(Draft202012Validator(declared).iter_errors(args))
                if errors:
                    if self.protocol.get('stability_version'):
                        return {'ok': False, 'error': 'tool_schema', 'field_errors': [
                            {'path': list(e.absolute_path), 'message': e.message[:400]} for e in errors],
                            'normalizations': corrections, 'saved_report': copy.deepcopy(self.report)}
                    raise ValueError('tool schema: '+errors[0].message[:400])
            if self.protocol.get('stability_version') and name in {'save_report', 'submit_report'}:
                args['final'] = name == 'submit_report'
                name = 'report'

            notes = args.pop("notes", None)
            if notes is not None:
                if not isinstance(notes, str) or len(notes) > self.protocol["notes_max_chars"]:
                    raise ValueError("notes exceed the declared character limit")
                self.notes = notes
                if self.protocol.get('continuity_version') and notes and (not self.notebook or self.notebook[-1]['text'] != notes):
                    self.notebook.append({'provider_call':self.env.provider_call_count,
                        'samples':len(self.env.sample_ledger),'text':notes})
            if self.report_only and name != "report":
                raise ValueError("report-only phase: revise and finalize the report")
            if name=='wait' and self.protocol.get('temporal_observations'):
                if self.review_only or self.env.successful_action_count>=self.protocol['max_movement_actions']:
                    raise ValueError('sampling is closed')
                if set(args)!={'seconds'}:
                    raise ValueError('wait requires exactly seconds')
                result=self.env.wait_and_measure(args['seconds'])
                self._sync()
                return result
            if name == 'inspect' and self.protocol.get('closure_version'):
                if self.protocol.get('continuity_version') and args.get('view')=='feedback':
                    from .continuity import retrieve
                    return retrieve(self,args)
                from .evidence import inspect
                return {'ok': True, **inspect(self, args)}
            if name == "analyze" and self.protocol.get("scientific_workbench"):
                from .workbench import compute
                if set(args) != {"code"}:
                    raise ValueError("analyze requires exactly code")
                if self.compute_count >= self.protocol.get("max_compute_calls", 6):
                    raise ValueError("computation allowance exhausted; query/report remain available")
                self._sync()
                data = {}
                for table in self.sql_tables():
                    cursor = self.db.execute("SELECT * FROM " + table + " ORDER BY 1")
                    data[table] = [dict(zip([c[0] for c in cursor.description], row)) for row in cursor]
                if self.workspace is None:
                    self._temporary_workspace = tempfile.TemporaryDirectory(prefix="laketrace-analysis-")
                    self.workspace = Path(self._temporary_workspace.name)
                self.compute_count += 1
                result = compute(args["code"], data, self.workspace, self.compute_count, replay=self.replay, api=self.protocol.get('workbench_api',1))
                self.last_compute = result
                return result
            if name in {"survey", "fit_source"} and self.protocol.get("efficient_tools"):
                return self.survey(args) if name == "survey" else self.fit(args)
            if name == "query":
                if set(args) != {"sql"}:
                    raise ValueError("query requires exactly sql")
                return {"ok": True, **self.query(args["sql"])}
            if name in {"move", "goto"}:
                if self.review_only:
                    raise ValueError("review-only phase: sampling has ended; query and report remain available")
                if self.env.mode == "submit_only":
                    raise ValueError("movement budget exhausted; query and reporting remain available")
                if name == "move":
                    if set(args) != {"direction", "steps"} or args["direction"] not in DIRECTIONS or type(args["steps"]) is not int or not 1 <= args["steps"] <= 32:
                        raise ValueError("move requires direction and integer steps 1-32")
                    route = [args["direction"]] * args["steps"]
                else:
                    if set(args) != set("xyz") or any(type(v) not in (int, float) or not math.isfinite(v) for v in args.values()):
                        raise ValueError("goto requires finite x,y,z")
                    route, target = self.plan_goto(**args)
                    if not route:
                        raise ValueError("already at the nearest reachable target")
                before = len(self.env.sample_ledger)
                result = self.env.move(route)
                self._sync()
                return {"ok": result["ok"], "completed_edges": result.get("completed_edges", 0),
                        "stop_reason": result.get("route_stop_reason"), "position": self.env._position_snapshot(),
                        "new_samples": self.query(f"SELECT * FROM samples WHERE sample_id>{before} ORDER BY sample_id"),
                        "event_sha256": stable_hash(result)}
            if name == "report":
                final = args.pop("final", None)
                if type(final) is not bool:
                    raise ValueError("report requires final=true or false")
                errors = {}
                for key, value in args.items():
                    try:
                        self._validate_report_field(key, value)
                        self.report[key] = value
                    except ValueError as exc:
                        errors[key] = str(exc)
                missing = sorted(self.required_report() - self.report.keys())
                if self.confirmed and (("source_present" in self.report and self.report["source_present"] is not True) or
                                       ("source_presence_probability" in self.report and self.report["source_presence_probability"] != 1)):
                    errors["source_present"] = "source existence is given in this track; omit detection fields or use true/probability 1"
                if final and self.protocol.get("nullable_location_confidence", False) and self.report.get("source_position") is not None and self.report.get("source_confidence") is None:
                    errors["source_confidence"] = "a proposed source coordinate requires a numeric probability; null is allowed only without a coordinate"
                if final and self.report.get("decision") == "declare" and (
                    (not self.confirmed and self.report.get("source_present") is not True) or self.report.get("source_position") is None):
                    errors["decision"] = "declare affirms a present, located pollution source. For absence or no source coordinate use cannot_determine; source_present separately records presence/absence."
                if final and self.protocol.get("probability_coherence"):
                    confidence = self.report.get("source_confidence")
                    presence = self.report.get("source_presence_probability")
                    if confidence is not None and presence is not None and confidence > presence:
                        errors["source_confidence"] = "joint existence-and-location probability cannot exceed source_presence_probability"
                if final and not errors and not missing:
                    self.final = True
                    self.env.submit(self.canonical_answer())
                return {"ok": not errors and (not final or not missing), "errors": errors, "missing_fields": missing,
                        "saved_report": copy.deepcopy(self.report), "finalized": self.final,
                        **({'normalizations': corrections} if self.protocol.get('stability_version') else {})}
            raise ValueError("unknown tool")
        except (ValueError, TypeError, KeyError, sqlite3.Error) as exc:
            result = {"ok": False, "error": str(exc)[:600]}
            if name == "query" and self.protocol.get("evidence_review"):
                result["sql_tables"] = self.sql_tables()
                result["hint"] = "Use one SELECT/WITH statement per call; no temporary tables. target_nodes already joins the measured target to layer/calibration. Group aggregates before applying another aggregate."
            return result

    def canonical_answer(self):
        node = self.report.get("plume_node_id")
        position = dict(zip("xyz", self.env.graph.positions[node])) if node is not None else None
        present = True if self.confirmed else self.report.get("source_present")
        answer = {"anomaly": "anomaly" if present is True else "no_anomaly" if present is False else None,
                "peak_position": position, "mechanism": self.report.get("mechanism"),
                "source_position": self.report.get("source_position"), "source_candidate_id": self.report.get("source_candidate_id"),
                "decision": self.report.get("decision")}
        if self.protocol.get("spatial_reports"):
            for name in ("plume", "source"):
                nodes = self.report.get(name + "_region_node_ids")
                answer[name + "_region_positions"] = None if nodes is None else [dict(zip("xyz", self.env.graph.positions[i])) for i in nodes]
        if self.protocol.get('temporal_observations'):
            sample=self.report.get('plume_sample_id')
            row=self.db.execute('SELECT node_id,time_s,x,y,z FROM samples WHERE sample_id=?',(sample,)).fetchone() if sample is not None else None
            answer['plume_measurement']=None if row is None else dict(zip(('node_id','time_s','x','y','z'),row))
            answer['peak_position']=None if row is None else dict(zip('xyz',row[2:]))
        if self.protocol.get('closure_version'):
            answer['report_time_s'] = self.env.budget.used_s
        return answer
