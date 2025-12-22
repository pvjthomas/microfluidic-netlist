"""Channel region selection endpoint."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal, List, Dict, Any
from shapely.geometry import Polygon, Point
from storage import get_session_dir
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


def build_channel_regions(polygons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build channel regions by subtracting contained polygons from their containers.
    
    Process:
    1. Convert polygon dicts to Shapely polygons
    2. Find containment relationships (outer contains inner)
    3. Subtract inner polygons from outer polygons (outer - inner) to create channel regions
    4. Convert resulting difference geometries back to polygon dicts
    
    Args:
        polygons: List of polygon dicts with 'polygon' key containing GeoJSON
        
    Returns:
        List of channel region polygon dicts (difference geometries)
    """
    if not polygons:
        return []
    
    if len(polygons) == 1:
        # Single polygon, return as-is
        return polygons
    
    # Convert polygon dicts to Shapely polygons
    shapely_polygons_with_data = []
    for i, poly_data in enumerate(polygons):
        coords_list = poly_data['polygon']['coordinates']
        
        # Get the exterior ring (first coordinate list) and holes (remaining lists)
        if not coords_list or len(coords_list[0]) < 3:
            continue  # Skip invalid polygons
        
        try:
            exterior_coords = coords_list[0]
            holes = coords_list[1:] if len(coords_list) > 1 else []
            
            # Construct Polygon with exterior and holes
            poly = Polygon(exterior_coords, holes=holes)
            if poly.is_valid and poly.area > 0:
                shapely_polygons_with_data.append((poly, poly_data, i))
        except Exception:
            continue
    
    if not shapely_polygons_with_data:
        return []
    
    # Sort by area (largest first) to process outer polygons first
    shapely_polygons_with_data.sort(key=lambda x: x[0].area, reverse=True)
    
    # Build containment tree: map outer polygon index to list of contained polygon indices
    containment_map = {}  # {outer_idx: [inner_idx1, inner_idx2, ...]}
    processed_indices = set()
    
    for i, (outer_poly, _, orig_idx_i) in enumerate(shapely_polygons_with_data):
        if i in processed_indices:
            continue
        
        contained_indices = []
        for j, (inner_poly, _, orig_idx_j) in enumerate(shapely_polygons_with_data):
            if i == j or j in processed_indices:
                continue
            
            # Check if inner polygon is fully contained in outer polygon
            # Use covers() which includes boundaries (doesn't exclude touching)
            if outer_poly.covers(inner_poly):
                if inner_poly.area < outer_poly.area:
                    contained_indices.append(j)
                    processed_indices.add(j)
        
        if contained_indices:
            containment_map[i] = contained_indices
            processed_indices.add(i)
    
    # Build channel regions by subtracting contained polygons from containers
    channel_regions = []
    
    for outer_idx, inner_indices in containment_map.items():
        outer_poly, outer_data, orig_outer_idx = shapely_polygons_with_data[outer_idx]
        
        # Collect all inner polygons to subtract
        inner_polygons = [shapely_polygons_with_data[inner_idx][0] for inner_idx in inner_indices]
        
        # Union all inner polygons
        from shapely.ops import unary_union
        if len(inner_polygons) == 1:
            inner_union = inner_polygons[0]
        else:
            inner_union = unary_union(inner_polygons)
        
        # Subtract inner from outer to create channel region
        try:
            channel_region = outer_poly.difference(inner_union)
            
            # Handle result (could be Polygon or MultiPolygon)
            if channel_region.is_empty:
                logger.warning(f"Channel region for polygon {orig_outer_idx} is empty after subtraction")
                continue
            
            # Convert to list of polygons (handle MultiPolygon)
            if channel_region.geom_type == 'MultiPolygon':
                region_polygons = list(channel_region.geoms)
            else:
                region_polygons = [channel_region]
            
            # Convert each region polygon to dict format
            for region_poly in region_polygons:
                if not region_poly.is_valid or region_poly.area <= 0:
                    continue
                
                # Convert to GeoJSON format
                coords = [[[x, y] for x, y in region_poly.exterior.coords[:-1]]]  # Exterior ring
                # Add interior rings (holes) if any
                for interior in region_poly.interiors:
                    coords.append([[x, y] for x, y in interior.coords[:-1]])
                
                channel_regions.append({
                    'polygon': {
                        'type': 'Polygon',
                        'coordinates': coords
                    },
                    'layer': outer_data['layer'],
                    'entity_handle': outer_data['entity_handle'],
                    'area': region_poly.area,
                    'bounds': {
                        'xmin': region_poly.bounds[0],
                        'ymin': region_poly.bounds[1],
                        'xmax': region_poly.bounds[2],
                        'ymax': region_poly.bounds[3]
                    }
                })
                
        except Exception as e:
            logger.warning(f"Failed to compute channel region for polygon {orig_outer_idx}: {e}")
            # Fallback: use outer polygon as-is
            coords = [[[x, y] for x, y in outer_poly.exterior.coords[:-1]]]
            channel_regions.append({
                'polygon': {
                    'type': 'Polygon',
                    'coordinates': coords
                },
                'layer': outer_data['layer'],
                'entity_handle': outer_data['entity_handle'],
                'area': outer_poly.area,
                'bounds': {
                    'xmin': outer_poly.bounds[0],
                    'ymin': outer_poly.bounds[1],
                    'xmax': outer_poly.bounds[2],
                    'ymax': outer_poly.bounds[3]
                }
            })
    
    # Add polygons that are not contained in any other (standalone polygons)
    for i, (poly, poly_data, orig_idx) in enumerate(shapely_polygons_with_data):
        if i not in processed_indices:
            # This is a standalone polygon (not containing others, not contained)
            coords = [[[x, y] for x, y in poly.exterior.coords[:-1]]]
            # Add interior rings (holes) if any
            for interior in poly.interiors:
                coords.append([[x, y] for x, y in interior.coords[:-1]])
            
            channel_regions.append({
                'polygon': {
                    'type': 'Polygon',
                    'coordinates': coords
                },
                'layer': poly_data['layer'],
                'entity_handle': poly_data['entity_handle'],
                'area': poly.area,
                'bounds': {
                    'xmin': poly.bounds[0],
                    'ymin': poly.bounds[1],
                    'xmax': poly.bounds[2],
                    'ymax': poly.bounds[3]
                }
            })
    
    logger.info(f"Built {len(channel_regions)} channel regions from {len(polygons)} input polygons")
    return channel_regions


class SelectChannelsRequest(BaseModel):
    session_id: str
    mode: Literal["layer", "click"]
    layers: list[str] | None = None
    point: list[float] | None = None


class SelectChannelsResponse(BaseModel):
    selected_region_ids: list[str]
    geometry: dict  # Lightweight geometry for frontend


def load_import_result(session_id: str) -> dict:
    """Load the import result from session storage."""
    session_dir = get_session_dir(session_id)
    result_path = session_dir / "import_result.json"
    
    if not result_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Import result not found for session {session_id}. Please import a DXF file first."
        )
    
    with open(result_path, 'r') as f:
        return json.load(f)


def save_selected_regions(session_id: str, selected_regions: list[dict]) -> None:
    """Save selected regions to session storage."""
    session_dir = get_session_dir(session_id)
    regions_path = session_dir / "selected_regions.json"
    
    with open(regions_path, 'w') as f:
        json.dump(selected_regions, f, indent=2)


@router.post("/select-channels", response_model=SelectChannelsResponse)
async def select_channels(req: SelectChannelsRequest):
    """Select channel regions by layer or click."""
    # Load import result
    import_result = load_import_result(req.session_id)
    polygons = import_result.get('polygons', [])
    
    # Build channel regions by subtracting contained polygons from their containers
    polygons = build_channel_regions(polygons)
    
    selected_regions = []
    selected_region_ids = []
    
    if req.mode == "layer":
        # Layer selection mode
        if not req.layers:
            raise HTTPException(
                status_code=400,
                detail="layers parameter is required for layer selection mode"
            )
        
        # Filter polygons by selected layers
        for i, poly_data in enumerate(polygons):
            if poly_data['layer'] in req.layers:
                region_id = f"R{i+1}"
                selected_regions.append({
                    'region_id': region_id,
                    'polygon_data': poly_data,
                    'index': i
                })
                selected_region_ids.append(region_id)
    
    elif req.mode == "click":
        # Click selection mode
        if not req.point or len(req.point) != 2:
            raise HTTPException(
                status_code=400,
                detail="point parameter [x, y] is required for click selection mode"
            )
        
        click_point = Point(req.point[0], req.point[1])
        
        # Find polygon containing the clicked point
        found = False
        for i, poly_data in enumerate(polygons):
            # Reconstruct Shapely polygon from coordinates
            try:
                coords = poly_data['polygon']['coordinates'][0]
                if len(coords) < 3:
                    continue  # Skip invalid polygons
                poly_geom = Polygon(coords)
                
                # Check if point is inside or on boundary
                if poly_geom.contains(click_point) or poly_geom.touches(click_point):
                    region_id = f"R{i+1}"
                    selected_regions.append({
                        'region_id': region_id,
                        'polygon_data': poly_data,
                        'index': i
                    })
                    selected_region_ids.append(region_id)
                    found = True
                    break  # For V1, just select the first containing polygon
            except Exception as e:
                # Skip polygons that can't be reconstructed
                continue
        
        if not found:
            raise HTTPException(
                status_code=404,
                detail=f"No polygon found containing point ({req.point[0]}, {req.point[1]})"
            )
    
    # Save selected regions to session
    save_selected_regions(req.session_id, selected_regions)
    
    # Build lightweight geometry for frontend rendering
    geometry_polygons = []
    for region in selected_regions:
        poly_data = region['polygon_data']
        geometry_polygons.append({
            'region_id': region['region_id'],
            'layer': poly_data['layer'],
            'coordinates': poly_data['polygon']['coordinates'],
            'bounds': poly_data['bounds']
        })
    
    # Compute combined bounds
    if geometry_polygons:
        all_bounds = [p['bounds'] for p in geometry_polygons]
        combined_bounds = {
            'xmin': min(b['xmin'] for b in all_bounds),
            'ymin': min(b['ymin'] for b in all_bounds),
            'xmax': max(b['xmax'] for b in all_bounds),
            'ymax': max(b['ymax'] for b in all_bounds)
        }
    else:
        combined_bounds = {'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0}
    
    geometry = {
        'polygons': geometry_polygons,
        'bounds': combined_bounds,
        'total_polygons': len(geometry_polygons)
    }
    
    return SelectChannelsResponse(
        selected_region_ids=selected_region_ids,
        geometry=geometry,
    )
