"""Full pipeline with optional visualization."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import math
from .dxf_loader import load_dxf
from .graph_extract import extract_graph_from_polygons
from shapely.geometry import Polygon
import logging

logger = logging.getLogger(__name__)


def build_channel_regions(polygons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build channel regions by subtracting contained polygons from their containers.
    
    Process:
    1. Convert polygon dicts to Shapely polygons
    2. Union all polygons
    3. Find containment relationships (outer contains inner)
    4. Subtract inner polygons from outer polygons (outer - inner) to create channel regions
    5. Convert resulting difference geometries back to polygon dicts
    
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


def run_pipeline(
    dxf_path: str,
    minimum_channel_width: float,
    selected_layers: Optional[List[str]] = None,
    simplify_tolerance: Optional[float] = None,
    snap_close_tol: float = 2.0,
    headful: bool = False,
    show_window: bool = True,
    width_sample_step: float = 10.0,
    measure_edges: bool = True,
    circles: Optional[List[Dict[str, Any]]] = None,
    port_snap_distance: float = 50.0,
    detect_polyline_circles: bool = False,
    circle_fit_rms_tol: float = 1.0,
    default_height: float = 50.0,
    default_cross_section_kind: str = "rectangular",
    per_edge_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    simplify_tolerance_factor: float = 0.5,
    endpoint_merge_distance_factor: float = 1.0,
    enable_step_a: bool = True,
    enable_step_b: bool = True,
    enable_step_c: bool = True,
    enable_step_d: bool = True
) -> Dict[str, Any]:
    """
    Run the full pipeline: DXF load → channel selection → graph extraction.
    
    Args:
        dxf_path: Path to DXF file
        minimum_channel_width: Minimum channel width in micrometers (required).
                              Used to calculate um_per_px = ceil(minimum_channel_width / 10)
        selected_layers: List of layer names to select (None = select all)
        simplify_tolerance: Tolerance for path simplification. If None, computed as simplify_tolerance_factor * minimum_channel_width
        snap_close_tol: Tolerance for auto-closing polylines
        headful: If True, show interactive visualization window
        show_window: If True and headful=True, show window (blocking). If False, just prepare data.
        width_sample_step: Distance between width samples along centerline (default: 10.0)
        measure_edges: If True, measure length and width profile for each edge (default: True)
        circles: Optional list of circle dicts from DXF loader for port detection
        port_snap_distance: Maximum distance to attach port to node (default: 50.0 µm)
        detect_polyline_circles: If True, detect circle-like polylines as ports (default: False)
        circle_fit_rms_tol: RMS tolerance for circle fitting (default: 1.0 µm)
        default_height: Default height in micrometers (default: 50.0)
        default_cross_section_kind: Default cross-section type: "rectangular" or "trapezoid" (default: "rectangular")
        per_edge_overrides: Optional dict mapping edge_id to override dict with height/cross_section_kind
        simplify_tolerance_factor: Factor to multiply minimum_channel_width for simplify_tolerance (default: 0.5)
        endpoint_merge_distance_factor: Factor to multiply minimum_channel_width for endpoint merge distance (default: 1.0)
        enable_step_a: If True, run Step A (DXF load) (default: True)
        enable_step_b: If True, run Step B (channel selection) (default: True)
        enable_step_c: If True, allow Step C (skeleton extraction) (default: True)
        enable_step_d: If True, allow Step D (graph extraction) (default: True)
    
    Returns:
        Dictionary with:
        - dxf_result: DXF load result
        - selected_polygons: Selected channel polygons
        - graph_result: Graph extraction result (None if steps C/D disabled)
        - visualizer: PipelineVisualizer instance (if headful=True)
    """
    # Step A: Load DXF
    dxf_result = None
    if enable_step_a:
        logger.info("Step A: Loading DXF...")
        dxf_result = load_dxf(dxf_path, snap_close_tol=snap_close_tol)
    else:
        logger.info("Step A: Skipped (disabled)")
        dxf_result = {'polygons': [], 'circles': [], 'bounds': {'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0}}
    
    # Step B: Select channels
    selected_polygons = []
    if enable_step_b and enable_step_a:
        logger.info("Step B: Selecting channels...")
        if selected_layers is None:
            # Select all polygons
            candidate_polygons = dxf_result['polygons']
        else:
            # Select by layer
            candidate_polygons = [
                p for p in dxf_result['polygons']
                if p['layer'] in selected_layers
            ]
        
        # Build channel regions by subtracting contained polygons from their containers
        # This creates difference geometries (outer - inner) that represent the actual channel regions
        selected_polygons = build_channel_regions(candidate_polygons)
    else:
        logger.info("Step B: Skipped (disabled)")
    
    # Hard maximum: um_per_px must ensure at least ~10 pixels across minimum_channel_width
    # This is critical for skeleton stability and accuracy
    um_per_px_max = minimum_channel_width / 10.0
    
    # Calculate um_per_px from minimum_channel_width (rounded up, but capped at max)
    um_per_px_from_width = min(math.ceil(minimum_channel_width / 10.0), um_per_px_max)
    
    # Check if polygon size requires a larger um_per_px to fit within 8000 pixel limit
    # IMPORTANT: Use bounding box of selected_polygons, NOT dxf_result['bounds'],
    # to avoid unrelated geometry or huge coordinate offsets forcing um_per_px to explode
    max_dimension = 8000
    if selected_polygons:
        # Compute bounding box from selected_polygons (min of xmin/ymin, max of xmax/ymax)
        xmin = min(p['bounds']['xmin'] for p in selected_polygons)
        ymin = min(p['bounds']['ymin'] for p in selected_polygons)
        xmax = max(p['bounds']['xmax'] for p in selected_polygons)
        ymax = max(p['bounds']['ymax'] for p in selected_polygons)
        
        width_um = xmax - xmin
        height_um = ymax - ymin
        max_dim_um = max(width_um, height_um)
        # Add 10% padding
        padded_max_dim = max_dim_um * 1.1
        um_per_px_from_size = math.ceil(padded_max_dim / max_dimension)
        
        # Use the maximum of both to ensure we meet both constraints
        um_per_px_proposed = max(um_per_px_from_width, um_per_px_from_size)
        
        # Enforce hard maximum: um_per_px must not exceed um_per_px_max
        if um_per_px_proposed > um_per_px_max:
            error_msg = (
                f"Cannot process polygon: size constraint requires um_per_px={um_per_px_proposed:.2f} µm/pixel, "
                f"but minimum_channel_width={minimum_channel_width:.2f} µm requires um_per_px <= {um_per_px_max:.2f} µm/pixel "
                f"to maintain at least ~10 pixels across the minimum channel width.\n"
                f"Polygon bounding box: {width_um:.1f} × {height_um:.1f} µm\n"
                f"This polygon is too large to process at the required resolution. Options:\n"
                f"  1. Increase minimum_channel_width (currently {minimum_channel_width:.2f} µm)\n"
                f"  2. Enable tiling to split the polygon into smaller chunks\n"
                f"  3. Select a smaller subset of polygons to process"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        um_per_px = um_per_px_proposed
        
        if um_per_px > um_per_px_from_width:
            logger.warning(
                f"Polygon size requires um_per_px={um_per_px:.2f} (from size constraint) "
                f"which is larger than {um_per_px_from_width:.2f} (from minimum_channel_width formula). "
                f"Using {um_per_px:.2f} to fit within {max_dimension} pixel limit. "
                f"Pixels across minimum width: {minimum_channel_width / um_per_px:.1f}"
            )
        else:
            logger.info(f"Calculated um_per_px={um_per_px:.2f} from minimum_channel_width={minimum_channel_width:.2f} µm "
                       f"({minimum_channel_width / um_per_px:.1f} pixels across minimum width)")
    else:
        um_per_px = um_per_px_from_width
        # Verify it doesn't exceed maximum (shouldn't happen, but check for safety)
        if um_per_px > um_per_px_max:
            error_msg = (
                f"Computed um_per_px={um_per_px:.2f} exceeds maximum {um_per_px_max:.2f} "
                f"required for minimum_channel_width={minimum_channel_width:.2f} µm"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info(f"Calculated um_per_px={um_per_px:.2f} from minimum_channel_width={minimum_channel_width:.2f} µm "
                   f"({minimum_channel_width / um_per_px:.1f} pixels across minimum width)")
    
    # Step C & D: Extract graph (only if enabled and not headful)
    graph_result = None
    if enable_step_c and enable_step_d and not headful:
        logger.info("Steps C & D: Extracting graph...")
        graph_result = extract_graph_from_polygons(
            selected_polygons,
            minimum_channel_width=minimum_channel_width,
            um_per_px=um_per_px,
            simplify_tolerance=simplify_tolerance,
            width_sample_step=width_sample_step,
            measure_edges=measure_edges,
            circles=circles or dxf_result.get('circles'),
            port_snap_distance=port_snap_distance,
            detect_polyline_circles=detect_polyline_circles,
            circle_fit_rms_tol=circle_fit_rms_tol,
            default_height=default_height,
            default_cross_section_kind=default_cross_section_kind,
            per_edge_overrides=per_edge_overrides,
            simplify_tolerance_factor=simplify_tolerance_factor,
            endpoint_merge_distance_factor=endpoint_merge_distance_factor
        )
    else:
        if not enable_step_c or not enable_step_d:
            logger.info(f"Steps C & D: Skipped (C={enable_step_c}, D={enable_step_d})")
        else:
            logger.info("Steps C & D: Graph extraction skipped (will be computed on-demand in visualization)")
        # Create empty result for consistency
        graph_result = {'nodes': [], 'edges': [], 'ports': []}
    
    # Show visualization window if requested
    visualizer = None
    if headful:
        try:
            from .visualize import PipelineVisualizer
            visualizer = PipelineVisualizer()
            
            # Set which steps are enabled
            visualizer.enabled_steps = {
                'A': enable_step_a,
                'B': enable_step_b,
                'C': enable_step_c,
                'D': enable_step_d
            }
            
            # Set up steps A and B (computed upfront if enabled)
            if enable_step_a:
                visualizer.set_step_a(dxf_result)
                visualizer.computed_steps.add('A')
            if enable_step_b:
                visualizer.set_step_b(selected_polygons)
                visualizer.computed_steps.add('B')
            
            # Set parameters needed to compute steps C and D on-demand (if enabled)
            if enable_step_c or enable_step_d:
                visualizer.set_compute_params(
                    minimum_channel_width=minimum_channel_width,
                    um_per_px=um_per_px,
                    simplify_tolerance=simplify_tolerance,
                    width_sample_step=width_sample_step,
                    measure_edges=measure_edges,
                    default_height=default_height,
                    default_cross_section_kind=default_cross_section_kind,
                    per_edge_overrides=per_edge_overrides,
                    simplify_tolerance_factor=simplify_tolerance_factor,
                    endpoint_merge_distance_factor=endpoint_merge_distance_factor
                )
            
            enabled_list = [step for step, enabled in visualizer.enabled_steps.items() if enabled]
            logger.info(f"Visualization ready. Enabled steps: {', '.join(enabled_list)}")
            
            if show_window:
                visualizer.show(block=True)
        except ImportError:
            logger.warning("matplotlib not available, visualization disabled")
    
    return {
        'dxf_result': dxf_result,
        'selected_polygons': selected_polygons,
        'graph_result': graph_result,
        'visualizer': visualizer
    }

