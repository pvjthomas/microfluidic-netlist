"""DXF import and normalization."""

import ezdxf
from pathlib import Path
from shapely.geometry import Polygon
import hashlib
from dataclasses import dataclass
from typing import Optional, Any

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
                # LWPOLYLINE has points directly
                points = [(p[0] * scale_to_um, p[1] * scale_to_um) for p in entity.get_points('xy')]
                is_closed = entity.closed
            else:
                # POLYLINE - iterate vertices
                for vertex in entity.vertices:
                    if vertex.dxftype() == 'VERTEX':
                        points.append((vertex.dxf.location.x * scale_to_um, 
                                      vertex.dxf.location.y * scale_to_um))
                is_closed = entity.is_closed
            
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
        bounds = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
    else:
        bounds = {'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0}
    
    result = {
        'layers': sorted(list(layers)),
        'polygons': polygons,
        'circles': circles,
        'bounds': bounds,
        'source_hash': source_hash,
        'source_filename': path.name,
        'import_timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    logger.debug(f"DXF loaded: {len(polygons)} polygons, {len(circles)} circles, {len(layers)} layers")
    
    return result
