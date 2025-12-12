"""Channel region selection endpoint."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
from shapely.geometry import Polygon, Point
from storage import get_session_dir

router = APIRouter()


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
