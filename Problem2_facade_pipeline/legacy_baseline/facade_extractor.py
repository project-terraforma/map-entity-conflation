"""Extract facade candidate edges from building geometries."""
from geometry_utils import line_bearing, line_midpoint, polygon_parts, polygon_to_facade_lines


def extract_facades_from_building(building_geom):
    """Given a shapely Polygon/MultiPolygon return list of facade dicts.

    Each facade dict includes facade_index, geometry, midpoint, length_m, and bearing_degrees.
    """
    if building_geom is None or getattr(building_geom, "is_empty", True):
        return []
    facades = []
    idx = 0
    for part_index, polygon in enumerate(polygon_parts(building_geom)):
        for edge_index, line in enumerate(polygon_to_facade_lines(polygon)):
            facades.append(
                {
                    "facade_index": idx,
                    "part_index": part_index,
                    "edge_index": edge_index,
                    "geometry": line,
                    "midpoint": line_midpoint(line),
                    "length_m": float(line.length),
                    "bearing_degrees": line_bearing(line),
                }
            )
            idx += 1
    return facades
