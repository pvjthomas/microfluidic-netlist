"""DXF import and normalization."""

import ezdxf
from pathlib import Path
from shapely.geometry import Polygon
import hashlib
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np

from datetime import datetime
import logging

INSUNITS_TO_NAME = {
    0:  "Unitless",
    1:  "Inches",
    2:  "Feet",
    3:  "Miles",
    4:  "Millimeters",
    5:  "Centimeters",
    6:  "Meters",
    7:  "Kilometers",
    8:  "Microinches",
    9:  "Mils",
    10: "Yards",
    11: "Angstroms",
    12: "Nanometers",
    13: "Microns",
    14: "Decimeters",
    15: "Decameters",
    16: "Hectometers",
    17: "Gigameters",
    18: "Astronomical Units",
    19: "Light Years",
    20: "Parsecs",
}
# Scale factor to meters for each INSUNITS value
INSUNITS_TO_METERS = {
    0:  None,           # Unitless → cannot infer safely
    1:  0.0254,         # inches
    2:  0.3048,         # feet
    3:  1609.344,       # miles
    4:  0.001,          # millimeters
    5:  0.01,           # centimeters
    6:  1.0,            # meters
    7:  1000.0,         # kilometers
    8:  0.0000254,      # microinches
    9:  0.0000254,      # mils
    10: 0.9144,         # yards
    11: 1e-10,          # angstroms
    12: 1e-9,           # nanometers
    13: 1e-6,           # microns
    14: 0.1,            # decimeters
    15: 10.0,           # decameters
    16: 100.0,          # hectometers
    17: 1e9,            # gigameters
    18: 1.495978707e11,# astronomical units
    19: 9.4607e15,     # light years
    20: 3.0857e16,     # parsecs
}


# ----------------------------
# Result object
# ----------------------------

@dataclass
class DrawingUnits:
    insunits: int
    name: str
    meters_per_unit: Optional[float]
    is_fallback: bool
    warning: Optional[str]

logger = logging.getLogger(__name__)
def get_drawing_units(
    doc: Any,  # ezdxf.Document
    *,
    fallback_insunits: int = 4,  # default fallback: millimeters
) -> DrawingUnits:
    """
    Safely determine drawing units from an ezdxf document.

    - Reads $INSUNITS
    - Falls back if missing, zero, or invalid
    - Returns scale factor to meters
    """

    raw_insunits = doc.header.get("$INSUNITS", 0)

    if raw_insunits in INSUNITS_TO_NAME and raw_insunits != 0:
        return DrawingUnits(
            insunits=raw_insunits,
            name=INSUNITS_TO_NAME[raw_insunits],
            meters_per_unit=INSUNITS_TO_METERS[raw_insunits],
            is_fallback=False,
            warning=None,
        )

    # ---------- fallback ----------
    fallback_name = INSUNITS_TO_NAME[fallback_insunits]
    fallback_scale = INSUNITS_TO_METERS[fallback_insunits]

    return DrawingUnits(
        insunits=fallback_insunits,
        name=fallback_name,
        meters_per_unit=fallback_scale,
        is_fallback=True,
        warning=(
            f"$INSUNITS is {raw_insunits!r} (unitless or invalid). "
            f"Falling back to {fallback_name}."
        ),
    )

def expand_bulged_segment(p1: tuple, p2: tuple, bulge: float, num_points: int = 16) -> list:
    """
    Expand a bulged segment into arc points.
    
    Args:
        p1: Start point (x, y)
        p2: End point (x, y)
        bulge: Bulge value (tan of 1/4 of included angle). 0 = straight line.
        num_points: Number of points to generate along the arc
    
    Returns:
        List of points along the arc (excluding p1, including p2)
    """
    if abs(bulge) < 1e-10:
        # Straight line, return just the end point
        return [p2]
    
    x1, y1 = p1
    x2, y2 = p2
    
    # Calculate arc parameters from bulge
    # bulge = tan(θ/4) where θ is the included angle
    # For a bulge b, the arc center and radius can be calculated
    chord_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if chord_length < 1e-10:
        return [p2]  # Degenerate segment
    
    # Calculate sagitta (height of arc)
    # sagitta = (bulge * chord_length) / 2
    sagitta = (bulge * chord_length) / 2.0
    
    # Calculate radius
    # radius = (chord_length^2 + 4*sagitta^2) / (8*sagitta)
    if abs(sagitta) < 1e-10:
        return [p2]  # Essentially straight
    
    radius = (chord_length**2 + 4 * sagitta**2) / (8 * abs(sagitta))
    
    # Calculate center point
    # Midpoint of chord
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    
    # Direction vector from p1 to p2 (normalized)
    dx = (x2 - x1) / chord_length
    dy = (y2 - y1) / chord_length
    
    # Distance from midpoint to center along perpendicular
    # For a circular arc: dist = r - |s| where s is sagitta
    dist_to_center = radius - abs(sagitta)
    
    # Perpendicular direction: left side is (-dy, dx), right side is (dy, -dx)
    # Bulge > 0: counterclockwise (center to left of p1->p2 direction)
    # Bulge < 0: clockwise (center to right of p1->p2 direction)
    if bulge > 0:
        # Center to the left: use (-dy, dx)
        perp_x = -dy
        perp_y = dx
    else:
        # Center to the right: use (dy, -dx)
        perp_x = dy
        perp_y = -dx
    
    center_x = mid_x + perp_x * dist_to_center
    center_y = mid_y + perp_y * dist_to_center
    
    # Calculate start and end angles
    angle1 = np.arctan2(y1 - center_y, x1 - center_x)
    angle2 = np.arctan2(y2 - center_y, x2 - center_x)
    
    # Adjust angles for arc direction (bulge sign determines direction)
    if bulge < 0:
        # Clockwise: ensure angle2 < angle1 (wrapping)
        if angle2 > angle1:
            angle2 -= 2 * np.pi
    else:
        # Counterclockwise: ensure angle2 > angle1
        if angle2 < angle1:
            angle2 += 2 * np.pi
    
    # Generate points along the arc
    angles = np.linspace(angle1, angle2, num_points + 1)[1:]  # Skip first point (p1)
    arc_points = [
        (center_x + radius * np.cos(a), center_y + radius * np.sin(a))
        for a in angles
    ]
    
    return arc_points

def load_dxf(file_path: str, snap_close_tol: float = 2.0) -> dict:
    """
    Load DXF file and extract geometry.
    
    All coordinates are converted from DXF drawing units to micrometers based on $INSUNITS.
    
    Args:
        file_path: Path to DXF file
        snap_close_tol: Tolerance for auto-closing nearly-closed polylines (in original DXF units)
    
    Returns:
        Dictionary with:
        - layers: List of layer names
        - polygons: List of polygon dicts with geometry and metadata (coordinates in micrometers)
        - circles: List of circle dicts with geometry and metadata (coordinates in micrometers)
        - bounds: Bounding box dict (coordinates in micrometers)
        - source_hash: SHA256 hash of source file
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"DXF file not found: {file_path}")
    
    logger.debug(f"Loading DXF: {file_path}")
    
    # Compute file hash
    with open(path, 'rb') as f:
        file_bytes = f.read()
        source_hash = hashlib.sha256(file_bytes).hexdigest()
    
    # Load DXF
    try:
        doc = ezdxf.readfile(file_path)
        units = get_drawing_units(doc)
        logger.debug(f"Units: {units}")
    except Exception as e:
        logger.error(f"Failed to parse DXF file: {e}")
        raise ValueError(f"Failed to parse DXF file: {e}")
    
    # Convert DXF units to micrometers
    # meters_per_unit gives conversion from DXF units to meters
    # micrometers = DXF_coords × meters_per_unit × 1,000,000
    if units.meters_per_unit is None:
        raise ValueError(f"Cannot convert unitless DXF coordinates to micrometers. "
                        f"$INSUNITS={units.insunits} is unitless.")
    
    scale_to_um = units.meters_per_unit * 1_000_000  # meters → micrometers
    logger.debug(f"Converting DXF coordinates to micrometers: scale_factor={scale_to_um:.6f}")
    if units.is_fallback:
        logger.warning(f"Using fallback units: {units.name}. {units.warning}")
    
    # Convert snap_close_tol to micrometers as well
    snap_close_tol_um = snap_close_tol * scale_to_um
    logger.debug(f"snap_close_tol converted: {snap_close_tol} → {snap_close_tol_um:.3f} µm")
    
    msp = doc.modelspace()
    
    # Collect all layers
    layers = set()
    polygons = []
    circles = []
    
    # Extract polylines (closed ones become polygons)
    for entity in msp:
        layer_name = entity.dxf.layer if hasattr(entity.dxf, 'layer') else '0'
        layers.add(layer_name)
        entity_handle = entity.dxf.handle
        
        # Handle LWPOLYLINE and POLYLINE
        if entity.dxftype() in ('LWPOLYLINE', 'POLYLINE'):
            points = []
            is_closed = False
            
            if entity.dxftype() == 'LWPOLYLINE':
                # LWPOLYLINE has points directly, get with bulge values
                raw_points = entity.get_points('xyb')
                is_closed = entity.closed
                
                # Expand bulged segments into arc points
                expanded_points = []
                for i, (x, y, bulge) in enumerate(raw_points):
                    # Scale coordinates to micrometers
                    scaled_point = (x * scale_to_um, y * scale_to_um)
                    
                    if i == 0:
                        # First point always added
                        expanded_points.append(scaled_point)
                    else:
                        # Check if previous segment had a bulge
                        prev_x, prev_y, prev_bulge = raw_points[i - 1]
                        prev_scaled = (prev_x * scale_to_um, prev_y * scale_to_um)
                        
                        if abs(prev_bulge) > 1e-10:
                            # Expand bulged segment
                            arc_points = expand_bulged_segment(prev_scaled, scaled_point, prev_bulge)
                            expanded_points.extend(arc_points)
                        else:
                            # Straight segment, just add current point
                            expanded_points.append(scaled_point)
                
                # If closed, check if last segment has bulge
                if is_closed and len(raw_points) > 1:
                    first_x, first_y, first_bulge = raw_points[0]
                    last_x, last_y, last_bulge = raw_points[-1]
                    first_scaled = (first_x * scale_to_um, first_y * scale_to_um)
                    last_scaled = (last_x * scale_to_um, last_y * scale_to_um)
                    
                    if abs(last_bulge) > 1e-10:
                        # Expand last bulged segment
                        arc_points = expand_bulged_segment(last_scaled, first_scaled, last_bulge)
                        expanded_points.extend(arc_points)
                
                points = expanded_points
            else:
                # POLYLINE - iterate vertices (bulge is stored in vertex)
                vertex_list = []
                for vertex in entity.vertices:
                    if vertex.dxftype() == 'VERTEX':
                        x = vertex.dxf.location.x * scale_to_um
                        y = vertex.dxf.location.y * scale_to_um
                        bulge = getattr(vertex.dxf, 'bulge', 0.0)
                        vertex_list.append((x, y, bulge))
                is_closed = entity.is_closed
                
                # Expand bulged segments
                expanded_points = []
                for i, (x, y, bulge) in enumerate(vertex_list):
                    point = (x, y)
                    
                    if i == 0:
                        expanded_points.append(point)
                    else:
                        prev_point = vertex_list[i - 1][:2]
                        if abs(vertex_list[i - 1][2]) > 1e-10:
                            # Previous vertex had bulge
                            arc_points = expand_bulged_segment(prev_point, point, vertex_list[i - 1][2])
                            expanded_points.extend(arc_points)
                        else:
                            expanded_points.append(point)
                
                # If closed, check last segment
                if is_closed and len(vertex_list) > 1:
                    first_point = vertex_list[0][:2]
                    last_point = vertex_list[-1][:2]
                    last_bulge = vertex_list[-1][2]
                    if abs(last_bulge) > 1e-10:
                        arc_points = expand_bulged_segment(last_point, first_point, last_bulge)
                        expanded_points.extend(arc_points)
                
                points = expanded_points
            
            if len(points) < 3:
                continue  # Skip degenerate polylines
            
            # Check if nearly closed (snap-close) - using scaled tolerance
            if not is_closed and snap_close_tol_um > 0:
                first = points[0]
                last = points[-1]
                dist = ((first[0] - last[0])**2 + (first[1] - last[1])**2)**0.5
                if dist <= snap_close_tol_um:
                    is_closed = True
                    points.append(points[0])  # Close the loop
            
            # Only process closed polylines as polygons
            if is_closed or (not is_closed and len(points) > 2 and points[0] == points[-1]):
                try:
                    # Ensure closed
                    if points[0] != points[-1]:
                        points.append(points[0])
                    
                    # Create Shapely polygon
                    poly = Polygon(points)
                    if not poly.is_valid:
                        # Try to fix invalid polygon
                        poly = poly.buffer(0)
                    
                    if poly.is_valid and poly.area > 0:
                        # Convert to GeoJSON-like format
                        coords = [[[x, y] for x, y in poly.exterior.coords]]
                        polygons.append({
                            'polygon': {
                                'type': 'Polygon',
                                'coordinates': coords
                            },
                            'layer': layer_name,
                            'entity_handle': entity_handle,
                            'area': poly.area,
                            'bounds': {
                                'xmin': poly.bounds[0],
                                'ymin': poly.bounds[1],
                                'xmax': poly.bounds[2],
                                'ymax': poly.bounds[3]
                            }
                        })
                except Exception as e:
                    # Skip invalid polygons
                    continue
        
        # Handle circles
        elif entity.dxftype() == 'CIRCLE':
            center = (entity.dxf.center.x * scale_to_um, entity.dxf.center.y * scale_to_um)
            radius = entity.dxf.radius * scale_to_um
            
            circles.append({
                'circle': {
                    'type': 'Circle',
                    'center': list(center),
                    'radius': radius
                },
                'layer': layer_name,
                'entity_handle': entity_handle,
                'center': list(center),
                'radius': radius
            })
        
        # Handle arcs - convert to polyline points
        elif entity.dxftype() == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            start = np.deg2rad(entity.dxf.start_angle)
            end = np.deg2rad(entity.dxf.end_angle)
            
            num_pts = 64  # or adaptive
            angles = np.linspace(start, end, num_pts)
            points = [
                (
                    (center.x + radius * np.cos(a)) * scale_to_um,
                    (center.y + radius * np.sin(a)) * scale_to_um
                )
                for a in angles
            ]
            
            # Check if nearly closed (snap-close) - using scaled tolerance
            is_closed = False
            if snap_close_tol_um > 0 and len(points) >= 2:
                first = points[0]
                last = points[-1]
                dist = ((first[0] - last[0])**2 + (first[1] - last[1])**2)**0.5
                if dist <= snap_close_tol_um:
                    is_closed = True
                    points.append(points[0])  # Close the loop
            
            # Only process closed arcs as polygons
            if is_closed or (len(points) > 2 and points[0] == points[-1]):
                try:
                    # Ensure closed
                    if points[0] != points[-1]:
                        points.append(points[0])
                    
                    # Create Shapely polygon
                    poly = Polygon(points)
                    if not poly.is_valid:
                        # Try to fix invalid polygon
                        poly = poly.buffer(0)
                    
                    if poly.is_valid and poly.area > 0:
                        # Convert to GeoJSON-like format
                        coords = [[[x, y] for x, y in poly.exterior.coords]]
                        polygons.append({
                            'polygon': {
                                'type': 'Polygon',
                                'coordinates': coords
                            },
                            'layer': layer_name,
                            'entity_handle': entity_handle,
                            'area': poly.area,
                            'bounds': {
                                'xmin': poly.bounds[0],
                                'ymin': poly.bounds[1],
                                'xmax': poly.bounds[2],
                                'ymax': poly.bounds[3]
                            }
                        })
                except Exception as e:
                    # Skip invalid polygons
                    continue
    
    # Compute overall bounds
    all_bounds = []
    for poly in polygons:
        all_bounds.append(poly['bounds'])
    
    # Also include circles in bounds
    for circle in circles:
        center = circle['center']
        radius = circle['radius']
        all_bounds.append({
            'xmin': center[0] - radius,
            'ymin': center[1] - radius,
            'xmax': center[0] + radius,
            'ymax': center[1] + radius
        })
    
    if all_bounds:
        xmin = min(b['xmin'] for b in all_bounds)
        ymin = min(b['ymin'] for b in all_bounds)
        xmax = max(b['xmax'] for b in all_bounds)
        ymax = max(b['ymax'] for b in all_bounds)
        original_bounds = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
    else:
        xmin = 0.0
        ymin = 0.0
        xmax = 0.0
        ymax = 0.0
        original_bounds = {'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0}
    
    # Shift to first quadrant: translate so minimum is at origin (or small offset)
    # Store the translation offset for later use
    translation_x = -xmin
    translation_y = -ymin
    
    # Apply translation to all polygons
    for poly in polygons:
        # Shift polygon coordinates
        coords = poly['polygon']['coordinates'][0]
        shifted_coords = [[x + translation_x, y + translation_y] for x, y in coords]
        poly['polygon']['coordinates'][0] = shifted_coords
        
        # Update bounds
        poly['bounds'] = {
            'xmin': poly['bounds']['xmin'] + translation_x,
            'ymin': poly['bounds']['ymin'] + translation_y,
            'xmax': poly['bounds']['xmax'] + translation_x,
            'ymax': poly['bounds']['ymax'] + translation_y
        }
    
    # Apply translation to all circles
    for circle in circles:
        # Shift center
        center = circle['center']
        circle['center'] = [center[0] + translation_x, center[1] + translation_y]
        circle['circle']['center'] = circle['center']
    
    # Compute new bounds after translation (shifted to first quadrant)
    if all_bounds:
        # After translation by (-xmin, -ymin), the new bounds start at (0, 0)
        bounds = {
            'xmin': 0.0,  # xmin + translation_x = xmin + (-xmin) = 0
            'ymin': 0.0,  # ymin + translation_y = ymin + (-ymin) = 0
            'xmax': xmax + translation_x,
            'ymax': ymax + translation_y
        }
    else:
        bounds = {'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0}
    
    # Store transformation information
    transform = {
        'translation_x': translation_x,
        'translation_y': translation_y,
        'original_bounds': original_bounds,
        'applied': True
    }
    
    result = {
        'layers': sorted(list(layers)),
        'polygons': polygons,
        'circles': circles,
        'bounds': bounds,
        'transform': transform,
        'source_hash': source_hash,
        'source_filename': path.name,
        'import_timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    logger.debug(f"DXF loaded: {len(polygons)} polygons, {len(circles)} circles, {len(layers)} layers")
    if translation_x != 0.0 or translation_y != 0.0:
        logger.debug(f"Applied translation: ({translation_x:.3f}, {translation_y:.3f}) µm to shift to first quadrant")
        logger.debug(f"Bounds after translation: x=[{bounds['xmin']:.1f}, {bounds['xmax']:.1f}], y=[{bounds['ymin']:.1f}, {bounds['ymax']:.1f}]")
    
    return result
