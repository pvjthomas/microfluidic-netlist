from __future__ import annotations

import math
from pathlib import Path

import ezdxf
from shapely.geometry import LineString
from shapely.ops import unary_union


def make_y_channel_polygon(width_um: float = 100.0, leg_len_um: float = 4000.0, angle_deg: float = 45.0):
    """
    Builds a Y-shaped *filled* channel polygon by buffering 3 centerlines and unioning them.

    - width_um: channel width in microns
    - leg_len_um: length of each leg (from junction) in microns
    - angle_deg: angle of each upper arm from the +Y axis (45° gives a nice symmetric Y)
    """
    w2 = width_um / 2.0
    ang = math.radians(angle_deg)

    # Junction at (0,0)
    # Stem goes downward along -Y
    stem = LineString([(0.0, 0.0), (0.0, -leg_len_um)])

    # Arms go up-left and up-right, length = leg_len_um
    dx = leg_len_um * math.sin(ang)
    dy = leg_len_um * math.cos(ang)
    left = LineString([(0.0, 0.0), (-dx, dy)])
    right = LineString([(0.0, 0.0), (dx, dy)])

    # Buffer each centerline into a rectangle-ish channel, then union
    # cap_style=2 -> flat ends, join_style=2 -> miter joins
    poly = unary_union([
        stem.buffer(w2, cap_style=2, join_style=2),
        left.buffer(w2, cap_style=2, join_style=2),
        right.buffer(w2, cap_style=2, join_style=2),
    ])

    # If union produces a MultiPolygon (rare), keep the largest piece
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)

    return poly


def write_polygon_as_lwpolyline(doc: ezdxf.Document, polygon, layer: str = "CHANNEL"):
    msp = doc.modelspace()
    exterior = list(polygon.exterior.coords)

    # ezdxf expects (x, y) tuples; ensure closed polyline (DXF "closed" flag)
    lw = msp.add_lwpolyline(exterior, dxfattribs={"layer": layer, "closed": True})

    # Optional: add a hatch so CAD viewers show it as filled (some viewers prefer hatch)
    hatch = msp.add_hatch(color=7, dxfattribs={"layer": layer})
    hatch.paths.add_polyline_path(exterior, is_closed=True)

    return lw


def main():
    out_path = Path(__file__).parent / "y_channel_100um_4mmlegs.dxf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    poly = make_y_channel_polygon(width_um=100.0, leg_len_um=4000.0, angle_deg=45.0)

    doc = ezdxf.new(dxfversion="R2010")
    doc.units = ezdxf.units.MM  # just metadata; geometry is in µm coordinates
    doc.layers.new("CHANNEL")

    write_polygon_as_lwpolyline(doc, poly, layer="CHANNEL")

    # Add a tiny reference crosshair at origin (optional)
    msp = doc.modelspace()
    msp.add_line((-200, 0), (200, 0), dxfattribs={"layer": "0"})
    msp.add_line((0, -200), (0, 200), dxfattribs={"layer": "0"})

    doc.saveas(out_path)
    print(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()

