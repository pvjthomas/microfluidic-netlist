"""DXF import and normalization."""

import ezdxf
from pathlib import Path
from shapely.geometry import Polygon
import hashlib
from datetime import datetime


def load_dxf(file_path: str, snap_close_tol: float = 2.0) -> dict:
    """
    Load DXF file and extract geometry.
    
    Args:
        file_path: Path to DXF file
        snap_close_tol: Tolerance for auto-closing nearly-closed polylines
    
    Returns:
        Dictionary with:
        - layers: List of layer names
        - polygons: List of polygon dicts with geometry and metadata
        - circles: List of circle dicts with geometry and metadata
        - bounds: Bounding box dict
        - source_hash: SHA256 hash of source file
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"DXF file not found: {file_path}")
    
    # Compute file hash
    with open(path, 'rb') as f:
        file_bytes = f.read()
        source_hash = hashlib.sha256(file_bytes).hexdigest()
    
    # Load DXF
    try:
        doc = ezdxf.readfile(file_path)
    except Exception as e:
        raise ValueError(f"Failed to parse DXF file: {e}")
    
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
                points = [(p[0], p[1]) for p in entity.get_points('xy')]
                is_closed = entity.closed
            else:
                # POLYLINE - iterate vertices
                for vertex in entity.vertices:
                    if vertex.dxftype() == 'VERTEX':
                        points.append((vertex.dxf.location.x, vertex.dxf.location.y))
                is_closed = entity.is_closed
            
            if len(points) < 3:
                continue  # Skip degenerate polylines
            
            # Check if nearly closed (snap-close)
            if not is_closed and snap_close_tol > 0:
                first = points[0]
                last = points[-1]
                dist = ((first[0] - last[0])**2 + (first[1] - last[1])**2)**0.5
                if dist <= snap_close_tol:
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
            center = (entity.dxf.center.x, entity.dxf.center.y)
            radius = entity.dxf.radius
            
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
    
    return {
        'layers': sorted(list(layers)),
        'polygons': polygons,
        'circles': circles,
        'bounds': bounds,
        'source_hash': source_hash,
        'source_filename': path.name,
        'import_timestamp': datetime.utcnow().isoformat() + 'Z'
    }
