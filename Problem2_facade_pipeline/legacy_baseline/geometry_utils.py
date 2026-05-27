"""Geometry utilities for facade matching."""
import math

from shapely.geometry import LineString, Point


def to_metric_crs(gdf):
    """Reproject a GeoDataFrame to a local metric CRS when possible."""
    if gdf is None:
        return None, None
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326, allow_override=True)
        metric_crs = gdf.estimate_utm_crs()
        if metric_crs is None:
            metric_crs = "EPSG:3857"
        projected = gdf.to_crs(metric_crs)
        return projected, projected.crs
    except Exception:
        try:
            projected = gdf.to_crs(epsg=3857)
            return projected, projected.crs
        except Exception:
            return gdf, getattr(gdf, "crs", None)


def polygon_parts(geometry):
    """Yield polygon parts from Polygon or MultiPolygon geometry."""
    if geometry is None or getattr(geometry, "is_empty", True):
        return []
    geom_type = getattr(geometry, "geom_type", "")
    if geom_type == "Polygon":
        return [geometry]
    if geom_type == "MultiPolygon":
        return list(geometry.geoms)
    return []


def polygon_to_facade_lines(polygon):
    """Split a polygon exterior ring into individual LineString facade candidates."""
    if polygon is None or getattr(polygon, "is_empty", True):
        return []
    coords = list(polygon.exterior.coords)
    lines = []
    for i in range(len(coords) - 1):
        line = LineString([coords[i], coords[i + 1]])
        if line.length > 0:
            lines.append(line)
    return lines


def line_midpoint(line):
    """Return midpoint of a LineString as Point."""
    if line is None or getattr(line, "is_empty", True):
        return None
    return line.interpolate(0.5, normalized=True)


def line_bearing(line):
    """Return compass-like bearing in degrees for a line segment in projected coordinates."""
    if line is None or getattr(line, "is_empty", True):
        return None
    coords = list(line.coords)
    if len(coords) < 2:
        return None
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    angle = math.degrees(math.atan2(x2 - x1, y2 - y1))
    return (angle + 360.0) % 360.0


def point_to_point_bearing(a: Point, b: Point):
    """Return bearing in degrees from point a to point b in projected coordinates."""
    if a is None or b is None:
        return None
    angle = math.degrees(math.atan2(b.x - a.x, b.y - a.y))
    return (angle + 360.0) % 360.0


def safe_wkt(geometry):
    """Return WKT for a geometry or an empty string."""
    if geometry is None or getattr(geometry, "is_empty", True):
        return ""
    return geometry.wkt
