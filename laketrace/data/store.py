from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from laketrace.util import jsonl, load_config


CHANNELS = {
    "controlled": ("chl", "do", "ph", "turb", "temp", "cond", "fdom"),
    "dahuofang": ("temp", "cond", "ph", "turb", "tds", "do", "fdom", "chl"),
    "xingkai": ("chl", "cond", "ph", "turb", "do", "fdom", "temp"),
    "donghu": ("chl", "do", "ph", "turb", "temp", "cond", "fdom"),
}


@dataclass(frozen=True)
class Node:
    index: int
    x: float
    y: float
    z: float
    bed_depth: float
    channels: dict[str, float]
    flow: dict[str, float]
    datetime: str

    @property
    def xyz(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


class Frame:
    def __init__(self, path: Path, waterbody: str):
        self.path = path
        self.waterbody = waterbody
        self.frame_id = path.stem
        rows: list[Node] = []
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            required = {
                "x_m_east", "y_m_north", "depth_m", "bed_depth_m",
                "u_m_s", "v_m_s", "w_m_s", "datetime", *CHANNELS[waterbody],
            }
            missing = required - columns
            if missing:
                raise ValueError(f"{path} missing columns: {sorted(missing)}")
            for index, row in enumerate(reader):
                rows.append(Node(
                    index=index,
                    x=float(row["x_m_east"]), y=float(row["y_m_north"]),
                    z=float(row["depth_m"]), bed_depth=float(row["bed_depth_m"]),
                    channels={key: float(row[key]) for key in CHANNELS[waterbody]},
                    flow={key: float(row[key]) for key in ("u_m_s", "v_m_s", "w_m_s")},
                    datetime=row["datetime"],
                ))
        if not rows:
            raise ValueError(f"empty frame: {path}")
        self.nodes = tuple(rows)
        self.by_xyz = {node.xyz: node for node in rows}
        if len(self.by_xyz) != len(rows):
            raise ValueError(f"duplicate node coordinates: {path}")


class DataStore:
    def __init__(self, root: str | Path, *, evaluator: bool = False):
        self.root = Path(root).resolve()
        required = [
            self.root / "tasks" / "instances.jsonl",
            self.root / "config" / "protocol.yaml",
            self.root / "config" / "access_points.yaml",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError(f"incomplete LakeTrace v2 data root: {missing}")
        self._tasks = {row["instance_id"]: row for row in jsonl(required[0])}
        self.evaluator = bool(evaluator)
        truth_path = self.root / "truth" / "exam_truth.jsonl"
        self._truth = {row["instance_id"]: row for row in jsonl(truth_path)} if evaluator else None
        if self._truth is not None and set(self._tasks) != set(self._truth):
            raise ValueError("task/truth instance sets differ")
        self.protocol = load_config(self.root / "config" / "protocol.yaml")
        self.access_points = load_config(self.root / "config" / "access_points.yaml")

    def tasks(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._tasks[key] for key in sorted(self._tasks))

    def task(self, instance_id: str) -> dict[str, Any]:
        return self._tasks[instance_id]

    def truth(self, instance_id: str) -> dict[str, Any]:
        if self._truth is None:
            raise PermissionError("truth is unavailable in public runtime mode")
        return self._truth[instance_id]

    @lru_cache(maxsize=64)
    def frame(self, instance_id: str, phase: str = "F1") -> Frame:
        task = self.task(instance_id)
        path = self.root / "fields" / task["waterbody"] / f"{instance_id}-{phase}.csv"
        return Frame(path, task["waterbody"])
