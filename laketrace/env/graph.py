from __future__ import annotations

import heapq
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


DIRECTIONS = ("north", "south", "east", "west", "up", "down")
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east", "up": "down", "down": "up"}


@dataclass(frozen=True)
class Edge:
    direction: str
    source: int
    target: int
    distance_m: float


class NodeGraph:
    def __init__(self, positions: Iterable[tuple[float, float, float]], max_gap_ratio: float = 1.25):
        supplied = tuple(map(tuple, positions))
        self.positions = tuple(sorted(set(supplied)))
        if not self.positions or len(self.positions) != len(supplied):
            raise ValueError("graph needs unique coordinates")
        self.index_by_xyz = {xyz: index for index, xyz in enumerate(self.positions)}
        self.neighbor_map: list[dict[str, Edge]] = [dict() for _ in self.positions]
        self.spacing = self._spacing()
        specs = ((0, (1, 2), "east"), (1, (0, 2), "north"), (2, (0, 1), "down"))
        for axis, grouping_axes, direction in specs:
            nominal = self.spacing[axis]
            if nominal is None:
                continue
            groups: dict[tuple[float, float], list[tuple[float, float, float]]] = defaultdict(list)
            for xyz in self.positions:
                groups[(xyz[grouping_axes[0]], xyz[grouping_axes[1]])].append(xyz)
            maximum = nominal * max_gap_ratio + 1e-8
            for values in groups.values():
                values.sort(key=lambda item: item[axis])
                for left, right in zip(values, values[1:]):
                    if right[axis] - left[axis] <= maximum:
                        self._connect(left, right, direction)

    @classmethod
    def from_frame(cls, frame: object) -> "NodeGraph":
        return cls(node.xyz for node in frame.nodes)  # type: ignore[attr-defined]

    def _spacing(self) -> tuple[float | None, float | None, float | None]:
        values: list[float | None] = []
        for axis in range(3):
            ordered = sorted({xyz[axis] for xyz in self.positions})
            gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
            values.append(statistics.median(gaps) if gaps else None)
        return tuple(values)  # type: ignore[return-value]

    def _connect(self, left: tuple[float, float, float], right: tuple[float, float, float], direction: str) -> None:
        a, b = self.index_by_xyz[left], self.index_by_xyz[right]
        distance = math.dist(left, right)
        self.neighbor_map[a][direction] = Edge(direction, a, b, distance)
        self.neighbor_map[b][OPPOSITE[direction]] = Edge(OPPOSITE[direction], b, a, distance)

    def neighbors(self, index: int) -> dict[str, Edge | None]:
        return {direction: self.neighbor_map[index].get(direction) for direction in DIRECTIONS}

    def trace(self, index: int, direction: str, steps: int) -> tuple[Edge, ...]:
        if direction not in DIRECTIONS:
            raise ValueError("E_DIRECTION")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("E_STEPS_RANGE")
        path: list[Edge] = []
        current = index
        for _ in range(steps):
            edge = self.neighbor_map[current].get(direction)
            if edge is None:
                raise ValueError("E_NO_NEIGHBOR")
            path.append(edge)
            current = edge.target
        return tuple(path)

    def shortest_path(self, start: int, targets: set[int]) -> tuple[float, tuple[int, ...]]:
        queue = [(0.0, start)]
        distance = {start: 0.0}
        previous: dict[int, int] = {}
        found: int | None = None
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != distance[node]:
                continue
            if node in targets:
                found = node
                break
            for edge in self.neighbor_map[node].values():
                candidate = cost + edge.distance_m
                if candidate < distance.get(edge.target, float("inf")):
                    distance[edge.target] = candidate
                    previous[edge.target] = node
                    heapq.heappush(queue, (candidate, edge.target))
        if found is None:
            raise ValueError("target unreachable")
        path = [found]
        while path[-1] != start:
            path.append(previous[path[-1]])
        path.reverse()
        return distance[found], tuple(path)

