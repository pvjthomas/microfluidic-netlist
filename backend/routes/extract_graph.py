"""Graph extraction endpoint."""

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from microfluidic_ir.graph_extract import extract_graph_from_polygons
from storage import get_session_dir

router = APIRouter()


class ExtractGraphRequest(BaseModel):
    session_id: str
    tolerances: dict | None = None
    width_sample_step: float = 10.0
    px_per_unit: float = 10.0
    simplify_tolerance: float = 1.0


class ExtractGraphResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]
    suggested_port_attachments: list[dict]


def load_selected_regions(session_id: str) -> list[dict]:
    """Load selected regions from session storage."""
    session_dir = get_session_dir(session_id)
    regions_path = session_dir / "selected_regions.json"
    
    if not regions_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Selected regions not found for session {session_id}. Please select channels first."
        )
    
    with open(regions_path, 'r') as f:
        return json.load(f)


def load_import_result(session_id: str) -> dict:
    """Load import result to get circles for port detection."""
    session_dir = get_session_dir(session_id)
    result_path = session_dir / "import_result.json"
    
    if not result_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Import result not found for session {session_id}"
        )
    
    with open(result_path, 'r') as f:
        return json.load(f)


@router.post("/extract-graph", response_model=ExtractGraphResponse)
async def extract_graph(req: ExtractGraphRequest):
    """Extract network graph from selected regions."""
    # Load selected regions
    selected_regions = load_selected_regions(req.session_id)
    
    if not selected_regions:
        raise HTTPException(
            status_code=400,
            detail="No regions selected. Please select channels first."
        )
    
    # Extract polygon data
    polygons = [region['polygon_data'] for region in selected_regions]
    
    # Extract graph
    try:
        graph_result = extract_graph_from_polygons(
            polygons,
            px_per_unit=req.px_per_unit,
            simplify_tolerance=req.simplify_tolerance
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract graph: {str(e)}"
        )
    
    # Load import result for port detection
    import_result = load_import_result(req.session_id)
    circles = import_result.get('circles', [])
    
    # Suggest port attachments (snap circles to nearest nodes)
    suggested_port_attachments = []
    port_snap_tolerance = 100.0  # Default: 100 units
    
    if req.tolerances and 'port_snap_tolerance' in req.tolerances:
        port_snap_tolerance = req.tolerances['port_snap_tolerance']
    
    from shapely.geometry import Point
    
    for i, circle in enumerate(circles):
        circle_center = Point(circle['center'])
        
        # Find nearest node
        nearest_node = None
        min_dist = float('inf')
        
        for node in graph_result['nodes']:
            node_point = Point(node['xy'])
            dist = circle_center.distance(node_point)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        
        if nearest_node and min_dist <= port_snap_tolerance:
            suggested_port_attachments.append({
                'port_id': f"P{i+1}",
                'circle': circle,
                'node_id': nearest_node['id'],
                'distance': float(min_dist),
                'snapped': True
            })
        else:
            # Port too far from any node
            suggested_port_attachments.append({
                'port_id': f"P{i+1}",
                'circle': circle,
                'node_id': None,
                'distance': float(min_dist) if nearest_node else None,
                'snapped': False
            })
    
    # Save graph result to session
    session_dir = get_session_dir(req.session_id)
    graph_path = session_dir / "graph_result.json"
    with open(graph_path, 'w') as f:
        json.dump(graph_result, f, indent=2)
    
    return ExtractGraphResponse(
        nodes=graph_result['nodes'],
        edges=graph_result['edges'],
        suggested_port_attachments=suggested_port_attachments,
    )
