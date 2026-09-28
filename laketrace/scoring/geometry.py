from __future__ import annotations

import math
from typing import Any


def point_segment_distance(point: tuple[float, float], left: tuple[float, float], right: tuple[float, float]) -> float:
    dx, dy = right[0] - left[0], right[1] - left[1]
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return math.dist(point, left)
    t = max(0.0, min(1.0, ((point[0] - left[0]) * dx + (point[1] - left[1]) * dy) / denominator))
    projection = left[0] + t * dx, left[1] + t * dy
    return math.dist(point, projection)


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    inside = False
    x, y = point
    for left, right in zip(polygon, polygon[1:] + polygon[:1]):
        if point_segment_distance(point, left, right) <= 1e-9:
            return True
        if (left[1] > y) != (right[1] > y):
            crossing = (right[0] - left[0]) * (y - left[1]) / (right[1] - left[1]) + left[0]
            if x < crossing:
                inside = not inside
    return inside


def horizontal_distance_to_geometry(x: float, y: float, geometry: dict[str, Any]) -> float:
    point = (x, y)
    kind = geometry["type"]
    if kind == "point":
        target = geometry["point"]
        return math.dist(point, (float(target["x"]), float(target["y"])))
    if kind == "line":
        points = [(float(item["x"]), float(item["y"])) for item in geometry["points"]]
        return min(point_segment_distance(point, a, b) for a, b in zip(points, points[1:]))
    if kind == "region":
        polygon = [(float(item["x"]), float(item["y"])) for item in geometry["polygon"]]
        if point_in_polygon(point, polygon):
            return 0.0
        return min(point_segment_distance(point, a, b) for a, b in zip(polygon, polygon[1:] + polygon[:1]))
    raise ValueError(f"unsupported geometry type: {kind}")


def injection_support_error(source, geometry, tolerances):
    """New opt-in contract: actual point/planar line/region injection support.

    Never reconstructs geometry from the old nearest-node export. Nonplanar
    supports require an explicit future contract, rather than silent flattening.
    """
    kind = geometry["type"]
    if kind == "axis_aligned_faces":
        return boundary_support_error(source, geometry, tolerances)
    vertices = [geometry["point"]] if kind == "point" else geometry.get("points") if kind == "line" else geometry.get("polygon") if kind == "region" else None
    if not vertices or (kind == "line" and len(vertices) < 2) or (kind == "region" and len(vertices) < 3):
        raise ValueError("invalid injection support geometry")
    if any(set(p) != set("xyz") or any(type(v) not in (float, int) or not math.isfinite(v) for v in p.values()) for p in vertices):
        raise ValueError("injection support requires finite xyz vertices")
    if max(p["z"] for p in vertices) - min(p["z"] for p in vertices) > 1e-9:
        raise ValueError("injection_support_v1 requires a planar support at one depth")
    h = horizontal_distance_to_geometry(source[0], source[1], geometry)
    v = abs(source[2] - vertices[0]["z"])
    return h, v


def boundary_support_error(source, geometry, tolerances):
    """Distance to a union of finite-volume boundary faces, not receivers.

    A face is a zero-thickness axis-aligned rectangle. A staircase bed uses
    multiple horizontal faces at their own depths; it is never flattened.
    The nearest admissible point on each rectangle minimizes horizontal and
    vertical errors simultaneously. Select one face for both errors.
    """
    faces = geometry.get("faces")
    if geometry.get("type") != "axis_aligned_faces" or not faces:
        raise ValueError("boundary support requires nonempty faces")
    errors = []
    for face in faces:
        bounds = face.get("bounds", {})
        if set(bounds) != set("xyz"):
            raise ValueError("boundary face needs xyz bounds")
        widths, distances = [], []
        for axis, coordinate in zip("xyz", source):
            interval = bounds[axis]
            if (len(interval) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in interval)
                    or interval[0] > interval[1]):
                raise ValueError("invalid boundary face interval")
            lo, hi = interval
            widths.append(hi-lo)
            distances.append(max(lo-coordinate, 0., coordinate-hi))
        if sum(width == 0 for width in widths) != 1:
            raise ValueError("boundary face must have exactly one zero extent")
        errors.append((math.hypot(*distances[:2]), distances[2]))
    return min(errors, key=lambda e: (max(e[0]/tolerances['horizontal_m'], e[1]/tolerances['vertical_m']), e))


def geometry_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Return the exact horizontal separation between two public geometries.

    Zero means that the two candidate geometries intersect or one contains a
    point of the other.  This helper is deliberately two-dimensional: public
    candidates do not expose depth.
    """

    def points(value: dict[str, Any]) -> list[tuple[float, float]]:
        if value["type"] == "point":
            item = value["point"]
            return [(float(item["x"]), float(item["y"]))]
        key = "points" if value["type"] == "line" else "polygon"
        return [(float(item["x"]), float(item["y"])) for item in value[key]]

    def segments(value: dict[str, Any]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        values = points(value)
        if value["type"] == "point":
            return []
        if value["type"] == "region":
            return list(zip(values, values[1:] + values[:1]))
        return list(zip(values, values[1:]))

    def segment_distance(
        a: tuple[float, float], b: tuple[float, float],
        c: tuple[float, float], d: tuple[float, float],
    ) -> float:
        # Endpoint-to-segment distance is zero for all collinear/touching
        # cases and for proper crossings one of the orientation products is
        # negative.
        def orient(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
        if o1 * o2 < 0.0 and o3 * o4 < 0.0:
            return 0.0
        return min(
            point_segment_distance(a, c, d), point_segment_distance(b, c, d),
            point_segment_distance(c, a, b), point_segment_distance(d, a, b),
        )

    left_points, right_points = points(left), points(right)
    if left["type"] == "region" and any(point_in_polygon(item, left_points) for item in right_points):
        return 0.0
    if right["type"] == "region" and any(point_in_polygon(item, right_points) for item in left_points):
        return 0.0
    if left["type"] == right["type"] == "point":
        return math.dist(left_points[0], right_points[0])
    if left["type"] == "point":
        return min(point_segment_distance(left_points[0], a, b) for a, b in segments(right))
    if right["type"] == "point":
        return min(point_segment_distance(right_points[0], a, b) for a, b in segments(left))
    return min(segment_distance(a, b, c, d) for a, b in segments(left) for c, d in segments(right))
