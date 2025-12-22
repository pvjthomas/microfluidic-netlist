"""Extract network graph from skeleton."""

from typing import List, Dict, Any, Tuple, Optional
from shapely.geometry import Point, Polygon, LineString, MultiPoint, GeometryCollection
import networkx as nx
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
from scipy.linalg import lstsq
from .skeleton import (
    skeletonize_polygon, 
    extract_skeleton_paths,
    cluster_junction_pixels,
    compute_cluster_centroid,
    merge_close_endpoints
)
from .measure import measure_edge
from .classify import classify_width_profile
from .ports import detect_ports
from .cross_section import apply_cross_section_defaults
import logging

logger = logging.getLogger(__name__)


def snap_endpoint_to_boundary(
    endpoint_xy: Tuple[float, float],
    polygon: Polygon,
    skeleton_graph: nx.Graph,
    endpoint_node_id: Optional[int] = None,
    incident_edge_coords: Optional[List[Tuple[float, float]]] = None
) -> Tuple[float, float]:
    """
    Snap an endpoint to the polygon boundary, positioned equidistant from the closest polygon vertices.
    
    Finds the two closest polygon vertices to the endpoint and positions the endpoint
    at the midpoint between them, then projects it onto the polygon boundary.
    
    Args:
        endpoint_xy: Current endpoint coordinates (x, y)
        polygon: Shapely polygon
        skeleton_graph: Skeleton graph (unused, kept for compatibility)
        endpoint_node_id: Optional skeleton node ID (unused, kept for compatibility)
        incident_edge_coords: Optional list of (x, y) coordinates (unused, kept for compatibility)
        
    Returns:
        Snapped coordinates (x, y) on polygon boundary, equidistant from closest vertices
    """
    endpoint_point = Point(endpoint_xy)
    
    # Get all polygon vertices (exterior coordinates)
    exterior_coords = list(polygon.exterior.coords)
    # Remove duplicate last point if polygon is closed
    if len(exterior_coords) > 1 and exterior_coords[0] == exterior_coords[-1]:
        exterior_coords = exterior_coords[:-1]
    
    if len(exterior_coords) < 2:
        # Fallback: use nearest boundary point projection
        boundary_point = polygon.boundary.interpolate(
            polygon.boundary.project(endpoint_point)
        )
        return (boundary_point.x, boundary_point.y)
    
    # Find distances to all vertices
    vertex_distances = []
    for i, vertex_coord in enumerate(exterior_coords):
        vertex_point = Point(vertex_coord)
        dist = endpoint_point.distance(vertex_point)
        vertex_distances.append((i, vertex_coord, dist))
    
    # Sort by distance and get the two closest vertices
    vertex_distances.sort(key=lambda x: x[2])
    
    if len(vertex_distances) < 2:
        # Only one vertex, use nearest boundary point
        boundary_point = polygon.boundary.interpolate(
            polygon.boundary.project(endpoint_point)
        )
        return (boundary_point.x, boundary_point.y)
    
    # Get the two closest vertices
    closest_vertex1 = Point(vertex_distances[0][1])
    closest_vertex2 = Point(vertex_distances[1][1])
    
    # Compute midpoint between the two closest vertices
    midpoint = Point(
        (closest_vertex1.x + closest_vertex2.x) / 2.0,
        (closest_vertex1.y + closest_vertex2.y) / 2.0
    )
    
    # Project midpoint onto polygon boundary
    boundary_point = polygon.boundary.interpolate(
        polygon.boundary.project(midpoint)
    )
    
    return (boundary_point.x, boundary_point.y)


def compute_stabilized_normals(
    coords: List[Tuple[float, float]],
    window_size: int = 5
) -> List[Tuple[float, float]]:
    """
    Compute stabilized normal vectors using local PCA on sliding window.
    Enforces consistent orientation (if dot(n_i, n_{i-1}) < 0, flip n_i).
    
    Args:
        coords: List of (x, y) coordinates (should be smoothed)
        window_size: Size of sliding window for PCA (default: 5)
        
    Returns:
        List of normalized normal vectors (nx, ny) for each point
    """
    if len(coords) < 2:
        return [(1.0, 0.0)] * len(coords)
    
    coords_array = np.array(coords)
    normals = []
    
    for i in range(len(coords)):
        # Determine window bounds
        half_window = window_size // 2
        start_idx = max(0, i - half_window)
        end_idx = min(len(coords), i + half_window + 1)
        
        if end_idx - start_idx < 2:
            # Fallback to central difference
            if i == 0:
                dx = coords[1][0] - coords[0][0]
                dy = coords[1][1] - coords[0][1]
            elif i == len(coords) - 1:
                dx = coords[i][0] - coords[i-1][0]
                dy = coords[i][1] - coords[i-1][1]
            else:
                dx = coords[i+1][0] - coords[i-1][0]
                dy = coords[i+1][1] - coords[i-1][1]
                dx /= 2.0
                dy /= 2.0
        else:
            # Use local PCA on window
            window_coords = coords_array[start_idx:end_idx]
            # Center the window
            center = np.mean(window_coords, axis=0)
            centered = window_coords - center
            
            # Compute covariance matrix
            if len(centered) >= 2:
                cov = np.cov(centered.T)
                # Get eigenvector (principal direction)
                eigvals, eigvecs = np.linalg.eigh(cov)
                # Use eigenvector corresponding to largest eigenvalue (tangent direction)
                if eigvals[1] > eigvals[0]:
                    dx, dy = eigvecs[:, 1]
                else:
                    dx, dy = eigvecs[:, 0]
            else:
                # Fallback to central difference
                if i == 0:
                    dx = coords[1][0] - coords[0][0]
                    dy = coords[1][1] - coords[0][1]
                else:
                    dx = coords[i][0] - coords[i-1][0]
                    dy = coords[i][1] - coords[i-1][1]
        
        # Normalize
        length = np.sqrt(dx*dx + dy*dy)
        if length < 1e-6:
            nx, ny = (1.0, 0.0)
        else:
            dx /= length
            dy /= length
            # Rotate 90 degrees to get normal (pointing right)
            nx = -dy
            ny = dx
        
        # Enforce consistent orientation
        if i > 0:
            prev_nx, prev_ny = normals[i-1]
            dot = nx * prev_nx + ny * prev_ny
            if dot < 0:
                # Flip normal
                nx = -nx
                ny = -ny
        
        normals.append((nx, ny))
    
    return normals


def compute_local_normal(
    coords: List[Tuple[float, float]],
    idx: int,
    normals: Optional[List[Tuple[float, float]]] = None,
    epsilon: float = 1e-6
) -> Tuple[float, float]:
    """
    Compute local normal vector at a point in a polyline.
    
    Args:
        coords: List of (x, y) coordinates
        idx: Index of point to compute normal for
        epsilon: Small value for numerical stability
        
    If normals are provided (from compute_stabilized_normals), use them.
    Otherwise falls back to central difference method.
    
    Args:
        coords: List of (x, y) coordinates
        idx: Index of point to compute normal for
        normals: Optional pre-computed stabilized normals
        epsilon: Small value for numerical stability
        
    Returns:
        Normalized normal vector (nx, ny) pointing to the right of the direction of travel
    """
    if normals and idx < len(normals):
        return normals[idx]
    
    # Fallback to original method
    if idx == 0:
        # Use forward direction
        dx = coords[1][0] - coords[0][0]
        dy = coords[1][1] - coords[0][1]
    elif idx == len(coords) - 1:
        # Use backward direction
        dx = coords[idx][0] - coords[idx-1][0]
        dy = coords[idx][1] - coords[idx-1][1]
    else:
        # Use average of forward and backward
        dx_fwd = coords[idx+1][0] - coords[idx][0]
        dy_fwd = coords[idx+1][1] - coords[idx][1]
        dx_bwd = coords[idx][0] - coords[idx-1][0]
        dy_bwd = coords[idx][1] - coords[idx-1][1]
        dx = (dx_fwd + dx_bwd) / 2.0
        dy = (dy_fwd + dy_bwd) / 2.0
    
    length = np.sqrt(dx*dx + dy*dy)
    if length < epsilon:
        # Fallback: use unit vector if length is zero
        return (1.0, 0.0)
    
    # Normalize direction
    dx /= length
    dy /= length
    
    # Rotate 90 degrees to get normal (pointing right)
    nx = -dy
    ny = dx
    
    return (nx, ny)


def find_boundary_intersections(
    point: Point,
    normal: Tuple[float, float],
    polygon: Polygon,
    search_distance: float = 10000.0
) -> Tuple[Optional[Point], Optional[Point]]:
    """
    Find intersections of a line through point along normal with polygon boundary.
    
    Args:
        point: Center point
        normal: Normal vector (nx, ny)
        polygon: Polygon to intersect with
        search_distance: Distance to extend line in each direction
        
    Returns:
        Tuple of (intersection_left, intersection_right) where:
        - intersection_left: closest intersection with negative dot product
        - intersection_right: closest intersection with positive dot product
    """
    nx, ny = normal
    
    # Create line segment through point along normal
    p1 = Point(point.x - nx * search_distance, point.y - ny * search_distance)
    p2 = Point(point.x + nx * search_distance, point.y + ny * search_distance)
    line = LineString([p1, p2])
    
    # Intersect with boundary
    intersection = polygon.boundary.intersection(line)
    
    if intersection.is_empty:
        return (None, None)
    
    # Handle different geometry types
    points = []
    if isinstance(intersection, Point):
        points = [intersection]
    elif isinstance(intersection, MultiPoint):
        points = list(intersection.geoms)
    elif isinstance(intersection, GeometryCollection):
        for geom in intersection.geoms:
            if isinstance(geom, Point):
                points.append(geom)
            elif isinstance(geom, MultiPoint):
                points.extend(geom.geoms)
    elif isinstance(intersection, LineString):
        # Line overlaps boundary segment - use endpoints
        points = [Point(intersection.coords[0]), Point(intersection.coords[-1])]
    
    if not points:
        return (None, None)
    
    # Find closest intersection on each side using signed dot product
    left_point = None
    right_point = None
    left_dist = float('inf')
    right_dist = float('inf')
    
    for p in points:
        # Vector from center point to intersection
        vec_x = p.x - point.x
        vec_y = p.y - point.y
        
        # Signed dot product with normal (positive = right, negative = left)
        dot = vec_x * nx + vec_y * ny
        
        dist = point.distance(p)
        
        if dot < 0 and dist < left_dist:  # Left side
            left_point = p
            left_dist = dist
        elif dot > 0 and dist < right_dist:  # Right side
            right_point = p
            right_dist = dist
    
    return (left_point, right_point)


def refine_centerline_midpoint(
    coords: List[Tuple[float, float]],
    polygon: Polygon,
    search_distance: float = 10000.0,
    normals: Optional[List[Tuple[float, float]]] = None
) -> List[Tuple[float, float]]:
    """
    One iteration of midpoint refinement: for each point, project to midpoint of boundary intersections.
    
    Args:
        coords: List of (x, y) coordinates
        polygon: Polygon to refine against
        search_distance: Distance to extend line for intersection search
        
    Returns:
        Refined coordinates
    """
    if len(coords) < 2:
        return coords
    
    refined = []
    for i in range(len(coords)):
        point = Point(coords[i])
        normal = compute_local_normal(coords, i, normals=normals)
        
        left_pt, right_pt = find_boundary_intersections(point, normal, polygon, search_distance)
        
        if left_pt and right_pt:
            # Use midpoint of intersections
            midpoint = Point(
                (left_pt.x + right_pt.x) / 2.0,
                (left_pt.y + right_pt.y) / 2.0
            )
            refined.append((midpoint.x, midpoint.y))
        else:
            # Keep original point if we can't find intersections
            refined.append(coords[i])
    
    return refined


def resample_polyline_by_arclength(
    coords: List[Tuple[float, float]],
    ds_um: float
) -> List[Tuple[float, float]]:
    """
    Resample polyline uniformly by arc length.
    
    Args:
        coords: List of (x, y) coordinates
        ds_um: Target arc length step in micrometers
        
    Returns:
        Resampled coordinates
    """
    if len(coords) < 2:
        return coords
    
    line = LineString(coords)
    total_length = line.length
    
    if total_length == 0:
        return coords
    
    # Compute number of segments needed
    num_segments = max(2, int(total_length / ds_um) + 1)
    actual_ds = total_length / num_segments
    
    # Resample at uniform arc length intervals
    resampled = []
    for i in range(num_segments + 1):
        s = i * actual_ds
        point = line.interpolate(s)
        resampled.append((point.x, point.y))
    
    return resampled


def resample_polyline(
    coords: List[Tuple[float, float]],
    target_segment_length: float = 10.0
) -> List[Tuple[float, float]]:
    """
    Resample polyline to have approximately uniform segment lengths.
    
    Args:
        coords: List of (x, y) coordinates
        target_segment_length: Target length for each segment
        
    Returns:
        Resampled coordinates
    """
    if len(coords) < 2:
        return coords
    
    line = LineString(coords)
    total_length = line.length
    
    if total_length == 0:
        return coords
    
    # Compute number of segments needed
    num_segments = max(2, int(total_length / target_segment_length) + 1)
    
    # Resample at uniform intervals
    resampled = []
    for i in range(num_segments + 1):
        s = (i / num_segments) * total_length
        point = line.interpolate(s)
        resampled.append((point.x, point.y))
    
    return resampled


def smooth_polyline(
    coords: List[Tuple[float, float]],
    window_length: Optional[int] = None,
    polyorder: int = 3
) -> List[Tuple[float, float]]:
    """
    Smooth polyline using Savitzky-Golay filter on x(s) and y(s).
    
    Args:
        coords: List of (x, y) coordinates
        window_length: Window length for filter (default: auto 7-21 points)
        polyorder: Polynomial order (default: 3)
        
    Returns:
        Smoothed coordinates
    """
    if len(coords) < 3:
        return coords
    
    # Auto-determine window length (7-21 points, must be odd)
    if window_length is None:
        window_length = min(21, max(7, len(coords) // 3))
        if window_length % 2 == 0:
            window_length += 1  # Must be odd
        window_length = min(window_length, len(coords) - 1 if len(coords) % 2 == 0 else len(coords))
    
    if window_length < 3 or window_length >= len(coords):
        return coords
    
    try:
        coords_array = np.array(coords)
        x = coords_array[:, 0]
        y = coords_array[:, 1]
        
        # Apply Savitzky-Golay filter
        x_smooth = savgol_filter(x, window_length, polyorder)
        y_smooth = savgol_filter(y, window_length, polyorder)
        
        return list(zip(x_smooth, y_smooth))
    except Exception as e:
        logger.warning("Savitzky-Golay smoothing failed: %s, returning original coordinates", e)
        return coords


def spline_smooth_centerline(
    coords: List[Tuple[float, float]],
    smoothing_factor: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Smooth centerline using spline interpolation.
    
    Args:
        coords: List of (x, y) coordinates
        smoothing_factor: Smoothing factor (0 = no smoothing, higher = more smoothing)
        
    Returns:
        Smoothed coordinates
    """
    if len(coords) < 4:  # Need at least 4 points for spline
        return coords
    
    try:
        # Convert to numpy array
        coords_array = np.array(coords)
        
        # Prepare spline parameters
        tck, u = splprep([coords_array[:, 0], coords_array[:, 1]], s=smoothing_factor * len(coords), k=min(3, len(coords)-1))
        
        # Evaluate spline at original parameter values
        smoothed = splev(u, tck)
        
        # Convert back to list of tuples
        return list(zip(smoothed[0], smoothed[1]))
    except Exception as e:
        logger.warning("Spline smoothing failed: %s, returning original coordinates", e)
        return coords


def refine_centerline_with_boundary(
    coords: List[Tuple[float, float]],
    polygon: Polygon,
    num_iterations: int = 10,
    final_iterations: int = 3,
    resample_length: float = 10.0,
    search_distance: float = 10000.0,
    um_per_px: float = 20.0
) -> List[Tuple[float, float]]:
    """
    Refine centerline coordinates using polygon boundary.
    
    Process:
    1. Iterative midpoint refinement (5-15 iterations) with resampling
    2. Spline smoothing
    3. Final midpoint refinement (2-3 iterations)
    
    Args:
        coords: Initial centerline coordinates from skeleton
        polygon: Polygon to refine against
        num_iterations: Number of refinement iterations (default: 10)
        final_iterations: Number of final refinement iterations (default: 3)
        resample_length: Target segment length for resampling (default: 10.0)
        search_distance: Distance to extend line for intersection search
        
    Returns:
        Refined centerline coordinates
    """
    if len(coords) < 2:
        return coords
    
    original_length = LineString(coords).length
    
    # A) Initial smoothing and resampling to create band-limited coarse centerline
    # Decouple skeleton from geometry - skeleton is topology only
    ds_um = max(5.0, 1.0 * um_per_px)
    refined = resample_polyline_by_arclength(coords, ds_um)
    
    # Smooth using Savitzky-Golay filter
    if len(refined) >= 7:
        refined = smooth_polyline(refined, window_length=None, polyorder=3)
    
    # B) Initial refinement iterations with stabilized normals
    # Allow ALL points (including endpoints) to be refined to be equidistant from boundary
    for iteration in range(num_iterations):
        # Compute stabilized normals from current refined curve
        normals = compute_stabilized_normals(refined)
        
        # C) Refine using stabilized normals (midpoint projection)
        # This makes all points equidistant from the polygon boundary
        refined = refine_centerline_midpoint(refined, polygon, search_distance, normals=normals)
        
        # Re-smooth and re-sample every few iterations to maintain stability
        if iteration % 3 == 0 and iteration > 0 and len(refined) > 2:
            refined = resample_polyline_by_arclength(refined, ds_um)
            if len(refined) >= 7:
                refined = smooth_polyline(refined, window_length=None, polyorder=3)
    
    # Final refinement iterations with stabilized normals
    normals = compute_stabilized_normals(refined)
    for iteration in range(final_iterations):
        refined = refine_centerline_midpoint(refined, polygon, search_distance, normals=normals)
    
    # Validate: check that length didn't increase dramatically
    final_length = LineString(refined).length
    if final_length > original_length * 2.0:
        logger.warning("Refined centerline length increased dramatically: %.1f -> %.1f, using original",
                      original_length, final_length)
        return coords
    
    return refined


def re_center_junctions(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    um_per_px: float,
    tangent_estimation_distance: float = 50.0
) -> None:
    """
    Re-center junction nodes (degree >= 3) by fitting best intersection point
    of incident edge tangents using least-squares.
    
    This removes skeleton centroid bias and centers symmetric branches.
    
    Args:
        nodes: List of node dictionaries (modified in place)
        edges: List of edge dictionaries (modified in place)
        um_per_px: Micrometers per pixel for distance calculations
        tangent_estimation_distance: Distance along edge to estimate tangent (default: 50 µm)
    """
    # Build edge lookup by node
    node_edges: Dict[str, List[Dict[str, Any]]] = {}
    for edge in edges:
        u, v = edge['u'], edge['v']
        if u not in node_edges:
            node_edges[u] = []
        if v not in node_edges:
            node_edges[v] = []
        node_edges[u].append(edge)
        node_edges[v].append(edge)
    
    for node in nodes:
        # Only re-center junctions (degree >= 3)
        if node['kind'] != 'junction':
            continue
        
        node_id = node['id']
        incident_edges = node_edges.get(node_id, [])
        
        if len(incident_edges) < 3:
            continue  # Not a junction or already handled
        
        # Estimate tangent directions for each incident edge near the junction
        tangents = []
        points_on_lines = []
        
        for edge in incident_edges:
            coords = edge['centerline']['coordinates']
            if len(coords) < 2:
                continue
            
            # Determine if this edge connects to the junction at start or end
            if edge['u'] == node_id:
                # Junction is at start, use direction from start to a point along edge
                start_xy = coords[0]
                # Find point at tangent_estimation_distance along edge
                line = LineString(coords)
                if line.length > tangent_estimation_distance:
                    target_point = line.interpolate(tangent_estimation_distance)
                    tangent_dx = target_point.x - start_xy[0]
                    tangent_dy = target_point.y - start_xy[1]
                    point_on_line = (start_xy[0], start_xy[1])
                else:
                    # Edge is too short, use direction from start to end
                    if len(coords) > 1:
                        tangent_dx = coords[-1][0] - start_xy[0]
                        tangent_dy = coords[-1][1] - start_xy[1]
                        point_on_line = (start_xy[0], start_xy[1])
                    else:
                        continue
            else:  # edge['v'] == node_id
                # Junction is at end, use direction from a point along edge to end
                end_xy = coords[-1]
                line = LineString(coords)
                if line.length > tangent_estimation_distance:
                    target_point = line.interpolate(line.length - tangent_estimation_distance)
                    tangent_dx = end_xy[0] - target_point.x
                    tangent_dy = end_xy[1] - target_point.y
                    point_on_line = (end_xy[0], end_xy[1])
                else:
                    # Edge is too short, use direction from start to end
                    if len(coords) > 1:
                        tangent_dx = end_xy[0] - coords[0][0]
                        tangent_dy = end_xy[1] - coords[0][1]
                        point_on_line = (end_xy[0], end_xy[1])
                    else:
                        continue
            
            # Normalize tangent
            length = np.sqrt(tangent_dx**2 + tangent_dy**2)
            if length < 1e-6:
                continue
            
            tangent_dx /= length
            tangent_dy /= length
            
            tangents.append((tangent_dx, tangent_dy))
            points_on_lines.append(point_on_line)
        
        if len(tangents) < 2:
            continue  # Need at least 2 edges to find intersection
        
        # Fit best intersection point using least-squares
        # For each pair of lines, find intersection, then average
        
        intersections = []
        for i in range(len(tangents)):
            for j in range(i + 1, len(tangents)):
                t1, p1 = tangents[i], points_on_lines[i]
                t2, p2 = tangents[j], points_on_lines[j]
                
                # Solve: p1 + s * t1 = p2 + u * t2
                # Rearrange: s * t1 - u * t2 = p2 - p1
                # Matrix form: [t1, -t2] * [s, u]^T = p2 - p1
                
                A = np.array([
                    [t1[0], -t2[0]],
                    [t1[1], -t2[1]]
                ])
                b = np.array([
                    p2[0] - p1[0],
                    p2[1] - p1[1]
                ])
                
                try:
                    sol, residuals, rank, s = lstsq(A, b, cond=1e-10)
                    if rank == 2:  # Full rank, unique solution
                        s_val = sol[0]
                        intersection = (
                            p1[0] + s_val * t1[0],
                            p1[1] + s_val * t1[1]
                        )
                        intersections.append(intersection)
                except:
                    continue
        
        if len(intersections) == 0:
            continue  # Couldn't find any intersections
        
        # Average intersection points
        new_x = np.mean([p[0] for p in intersections])
        new_y = np.mean([p[1] for p in intersections])
        
        # Update junction node position
        old_xy = node['xy']
        node['xy'] = [new_x, new_y]
        
        # Update incident edge centerline endpoints to match new junction position
        for edge in incident_edges:
            coords = edge['centerline']['coordinates']
            if len(coords) < 2:
                continue
            
            if edge['u'] == node_id:
                # Junction is at start, update first coordinate
                coords[0] = [new_x, new_y]
            elif edge['v'] == node_id:
                # Junction is at end, update last coordinate
                coords[-1] = [new_x, new_y]
        
        logger.debug("Re-centered junction %s from (%g, %g) to (%g, %g)",
                    node_id, old_xy[0], old_xy[1], new_x, new_y)


def extract_graph_from_polygon(
    polygon: Any,  # Shapely Polygon
    minimum_channel_width: float,
    um_per_px: float,
    simplify_tolerance: Optional[float] = None,
    width_sample_step: float = 10.0,
    measure_edges: bool = True,
    default_height: float = 50.0,
    default_cross_section_kind: str = "rectangular",
    per_edge_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    simplify_tolerance_factor: float = 0.5,
    endpoint_merge_distance_factor: float = 1.0
) -> Dict[str, Any]:
    """
    Extract network graph from a polygon.
    
    Args:
        polygon: Shapely polygon
        minimum_channel_width: Minimum channel width in micrometers (required)
        um_per_px: Resolution for skeletonization (microns per pixel) (required)
        simplify_tolerance: Tolerance for path simplification. If None, computed as simplify_tolerance_factor * minimum_channel_width.
        width_sample_step: Distance between width samples along centerline (default: 10.0)
        measure_edges: If True, measure length and width profile for each edge (default: True)
        default_height: Default height in micrometers (default: 50.0)
        default_cross_section_kind: Default cross-section type (default: "rectangular")
        per_edge_overrides: Optional dict mapping edge_id to override dict
        simplify_tolerance_factor: Factor to multiply minimum_channel_width for simplify_tolerance (default: 0.5)
        endpoint_merge_distance_factor: Factor to multiply minimum_channel_width for endpoint merge distance (default: 1.0)
    
    Returns:
        Dictionary with:
        - nodes: List of node dicts (id, xy, kind, degree)
        - edges: List of edge dicts (id, u, v, centerline, length, width_profile, cross_section)
        - skeleton_graph: NetworkX graph (for debugging)
        - transform: Transform dictionary with um_per_px
    """
    # Skeletonize polygon
    logger.info("Building graph from skeleton")
    skeleton_graph, transform = skeletonize_polygon(
        polygon,
        um_per_px=um_per_px,
        simplify_tolerance=simplify_tolerance
    )
    
    if len(skeleton_graph) == 0:
        logger.info("Empty skeleton graph")
        return {
            'nodes': [],
            'edges': [],
            'skeleton_graph': skeleton_graph,
            'transform': transform
        }
    
    # Compute simplify_tolerance if not provided
    if simplify_tolerance is None:
        simplify_tolerance = simplify_tolerance_factor * minimum_channel_width
        logger.debug("extract_graph_from_polygon: computed simplify_tolerance=%.3f (factor=%.2f × width=%.2f)",
                     simplify_tolerance, simplify_tolerance_factor, minimum_channel_width)
    
    logger.debug("extract_graph_from_polygon: using minimum_channel_width=%.2f µm", minimum_channel_width)
    
    # A) Cluster junction pixels into connected components
    import time
    start = time.time()
    
    junction_clusters = cluster_junction_pixels(skeleton_graph, um_per_px)
    logger.info(f"Clustered {sum(len(v) for v in junction_clusters.values())} junction pixels into {len(junction_clusters)} junction clusters")
    
    # Build forbidden set from junction pixels (for endpoint merging guard)
    forbidden_nodes = set()
    for cluster_nodes in junction_clusters.values():
        forbidden_nodes.update(cluster_nodes)
    logger.debug(f"Built forbidden set with {len(forbidden_nodes)} junction pixels")
    
    # Identify endpoints (before merging)
    endpoint_nodes = [n for n in skeleton_graph.nodes() if skeleton_graph.degree(n) == 1]
    logger.info(f"Found {len(endpoint_nodes)} endpoint pixels")
    
    # B) Merge close endpoints using topology-aware pixel-based merging
    # Use pixel-scale: min(2.0 * um_per_px, 0.2 * minimum_channel_width) * factor
    # This prevents giant merge distances and keeps merging pixel-scale
    endpoint_merge_distance = min(2.0 * um_per_px, 0.2 * minimum_channel_width) * endpoint_merge_distance_factor
    logger.debug("extract_graph_from_polygon: endpoint_merge_distance=%.2f µm (pixel-based: min(2.0×%.2f, 0.2×%.2f)×%.2f)",
                 endpoint_merge_distance, um_per_px, minimum_channel_width, endpoint_merge_distance_factor)
    endpoint_clusters = merge_close_endpoints(
        skeleton_graph, 
        endpoint_nodes, 
        endpoint_merge_distance,
        um_per_px=um_per_px,
        forbidden_nodes=forbidden_nodes
    )
    logger.info(f"Merged {len(endpoint_nodes)} endpoints into {len(endpoint_clusters)} endpoint clusters (topology-aware)")
    
    # Build true nodes from clusters
    true_nodes = {}  # Map from internal node_id to (x, y)
    nodes = []
    node_counter = 1
    cluster_to_node_id = {}
    
    # Add junction clusters
    for cluster_id, cluster_nodes in junction_clusters.items():
        centroid = compute_cluster_centroid(skeleton_graph, cluster_nodes)
        node_id = f"N{node_counter}"
        node_counter += 1
        true_nodes[node_counter - 1] = centroid
        cluster_to_node_id[('junction', cluster_id)] = node_counter - 1
        
        nodes.append({
            'id': node_id,
            'xy': list(centroid),
            'kind': 'junction',
            'degree': len(cluster_nodes),
            'cluster_nodes': cluster_nodes  # Store for reference
        })
    
    # Add endpoint clusters
    for cluster_id, cluster_nodes in endpoint_clusters.items():
        if len(cluster_nodes) == 1:
            centroid = skeleton_graph.nodes[cluster_nodes[0]]['xy']
            endpoint_node_id = cluster_nodes[0]
        else:
            # Use centroid of merged endpoints
            centroid = compute_cluster_centroid(skeleton_graph, cluster_nodes)
            # Use the first node for direction finding
            endpoint_node_id = cluster_nodes[0]
        
        # Snap endpoint to polygon boundary (initial snap, will be refined later)
        # At this stage we don't have edge coordinates yet, so use skeleton direction
        snapped_xy = snap_endpoint_to_boundary(
            centroid,
            polygon,
            skeleton_graph,
            endpoint_node_id,
            incident_edge_coords=None
        )
        
        node_id = f"N{node_counter}"
        node_counter += 1
        true_nodes[node_counter - 1] = snapped_xy
        cluster_to_node_id[('endpoint', cluster_id)] = node_counter - 1
        
        nodes.append({
            'id': node_id,
            'xy': list(snapped_xy),
            'kind': 'endpoint',
            'degree': len(cluster_nodes),
            'cluster_nodes': cluster_nodes
        })
        
        # Log if endpoint was moved
        dist_moved = Point(centroid).distance(Point(snapped_xy))
        if dist_moved > 1.0:  # Only log if moved more than 1 µm
            logger.debug("Snapped endpoint %s from (%.1f, %.1f) to (%.1f, %.1f), moved %.1f µm",
                        node_id, centroid[0], centroid[1], snapped_xy[0], snapped_xy[1], dist_moved)
    
    node_build_time = time.time() - start
    logger.info(f"  Built {len(nodes)} true nodes ({len(junction_clusters)} junctions, {len(endpoint_clusters)} endpoints) in {node_build_time:.2f}s")
    
    # C) Rebuild edges by walking from node to node (collapsing degree-2 chains)
    start = time.time()
    edges = []
    edge_counter = 1
    
    # Create mapping: for each true node, find skeleton nodes it represents
    true_node_to_skeleton_nodes = {}
    for node in nodes:
        skeleton_nodes = node.get('cluster_nodes', [])
        if skeleton_nodes:
            true_node_id = int(node['id'][1:]) - 1  # Extract number from "N1" -> 0
            true_node_to_skeleton_nodes[true_node_id] = skeleton_nodes
    
    # Walk from each true node to find edges to other true nodes
    visited_edges = set()
    
    for i, node1 in enumerate(nodes):
        true_id1 = int(node1['id'][1:]) - 1
        skeleton_nodes1 = true_node_to_skeleton_nodes.get(true_id1, [])
        if not skeleton_nodes1:
            continue
        
        for j, node2 in enumerate(nodes[i+1:], start=i+1):
            true_id2 = int(node2['id'][1:]) - 1
            skeleton_nodes2 = true_node_to_skeleton_nodes.get(true_id2, [])
            if not skeleton_nodes2:
                continue
            
            # Try to find path between any skeleton node of node1 to any of node2
            path_coords = None
            for s1 in skeleton_nodes1:
                for s2 in skeleton_nodes2:
                    if s1 == s2:
                        continue
                    
                    try:
                        path_skeleton_nodes = nx.shortest_path(skeleton_graph, s1, s2)
                        
                        # Check if path contains only degree-2 nodes between endpoints
                        # (collapsing degree-2 chains)
                        degrees = dict(skeleton_graph.degree())
                        intermediate_are_degree2 = all(
                            degrees.get(n, 0) == 2 
                            for n in path_skeleton_nodes[1:-1]
                        )
                        
                        # Also check if we cross another true node (shouldn't happen with clustering)
                        crosses_other_node = False
                        for n in path_skeleton_nodes[1:-1]:
                            for other_node in nodes:
                                if other_node['id'] != node1['id'] and other_node['id'] != node2['id']:
                                    other_skeleton_nodes = true_node_to_skeleton_nodes.get(
                                        int(other_node['id'][1:]) - 1, []
                                    )
                                    if n in other_skeleton_nodes:
                                        crosses_other_node = True
                                        break
                            if crosses_other_node:
                                break
                        
                        if not crosses_other_node:
                            # Get initial path coordinates from skeleton (for topology only)
                            path_coords_raw = [skeleton_graph.nodes[n]['xy'] for n in path_skeleton_nodes]
                            
                            # Refine centerline using polygon boundary (not raster coordinates)
                            # Allow refinement to adjust ALL nodes to be equidistant from boundary
                            try:
                                # Refine entire path including endpoints
                                refined_path = refine_centerline_with_boundary(
                                    path_coords_raw,
                                    polygon,
                                    num_iterations=10,
                                    final_iterations=3,
                                    resample_length=10.0,
                                    search_distance=10000.0,
                                    um_per_px=um_per_px
                                )
                                path_coords = refined_path
                            except Exception as e:
                                logger.warning("Centerline refinement failed for edge %s-%s: %s, using skeleton coordinates",
                                             node1['id'], node2['id'], e)
                                path_coords = path_coords_raw
                            
                            break
                    except nx.NetworkXNoPath:
                        continue
                    except Exception as e:
                        logger.debug("Error finding path from %s to %s: %s", node1['id'], node2['id'], e)
                        continue
                if path_coords:
                    break
            
            if path_coords and len(path_coords) >= 2:
                # Check if edge already exists (ignore direction)
                edge_key = tuple(sorted([node1['id'], node2['id']]))
                if edge_key not in visited_edges:
                    edge_id = f"E{edge_counter}"
                    edge_counter += 1
                    
                    # Update node positions to match refined centerline endpoints
                    # This ensures nodes are equidistant from the polygon boundary
                    if len(path_coords) >= 2:
                        # Update node1 position to match refined centerline start
                        node1['xy'] = list(path_coords[0])
                        # Update node2 position to match refined centerline end
                        node2['xy'] = list(path_coords[-1])
                    
                    edges.append({
                        'id': edge_id,
                        'u': node1['id'],
                        'v': node2['id'],
                        'centerline': {
                            'type': 'LineString',
                            'coordinates': [[float(x), float(y)] for x, y in path_coords]
                        }
                    })
                    visited_edges.add(edge_key)
    
    edge_build_time = time.time() - start
    logger.info(f"  Built {len(edges)} edges (collapsing degree-2 chains) in {edge_build_time:.2f}s")
    
    # D) Merge close junctions (extra junctions along trunk)
    junction_merge_threshold = 3.0 * um_per_px  # 3 pixels in world units
    nodes_to_merge = []
    for i, node1 in enumerate(nodes):
        if node1['kind'] != 'junction':
            continue
        for node2 in nodes[i+1:]:
            if node2['kind'] != 'junction':
                continue
            
            # Check if there's a direct edge between them
            has_direct_edge = any(
                (e['u'] == node1['id'] and e['v'] == node2['id']) or
                (e['u'] == node2['id'] and e['v'] == node1['id'])
                for e in edges
            )
            
            if has_direct_edge:
                # Check edge length
                edge = next(
                    (e for e in edges if 
                     (e['u'] == node1['id'] and e['v'] == node2['id']) or
                     (e['u'] == node2['id'] and e['v'] == node1['id'])),
                    None
                )
                
                if edge:
                    coords = edge['centerline']['coordinates']
                    edge_length = sum(
                        ((coords[j+1][0] - coords[j][0])**2 + 
                         (coords[j+1][1] - coords[j][1])**2)**0.5
                        for j in range(len(coords) - 1)
                    )
                    
                    if edge_length < junction_merge_threshold:
                        nodes_to_merge.append((node1['id'], node2['id']))
    
    # Merge junctions (keep first, remove second, redirect edges)
    merged_node_ids = set()
    for node1_id, node2_id in nodes_to_merge:
        if node2_id in merged_node_ids:
            continue
        
        node1 = next(n for n in nodes if n['id'] == node1_id)
        node2 = next(n for n in nodes if n['id'] == node2_id)
        
        # Merge coordinates (weighted by degree or just average)
        node1['xy'] = [
            (node1['xy'][0] + node2['xy'][0]) / 2.0,
            (node1['xy'][1] + node2['xy'][1]) / 2.0
        ]
        
        # Redirect all edges from node2 to node1
        for edge in edges:
            if edge['u'] == node2_id:
                edge['u'] = node1_id
            if edge['v'] == node2_id:
                edge['v'] = node1_id
        
        # Remove duplicate edges
        edges = [e for e in edges if not (e['u'] == node1_id and e['v'] == node1_id)]
        
        # Remove node2
        nodes = [n for n in nodes if n['id'] != node2_id]
        merged_node_ids.add(node2_id)
        
        logger.debug("Merged junction %s into %s (edge length < %.1f µm)", node2_id, node1_id, junction_merge_threshold)
    
    if merged_node_ids:
        logger.info(f"  Merged {len(merged_node_ids)} close junctions")
    
    # D) Junction re-centering: fit best intersection point for symmetric branches
    # Note: This should happen AFTER centerline refinement but BEFORE measurement
    # so that junction positions are accurate for width profile calculations
    junction_count = sum(1 for n in nodes if n['kind'] == 'junction')
    if junction_count > 0:
        logger.debug("Re-centering %d junction nodes using incident edge tangents", junction_count)
        re_center_junctions(nodes, edges, um_per_px)
    
    # Measure edges (length and width profile)
    if measure_edges and edges:
        import time
        start = time.time()
        logger.info("Measuring edges (length and width profile)...")
        for edge in edges:
            centerline_coords = edge['centerline']['coordinates']
            try:
                length, width_profile = measure_edge(
                    polygon,
                    centerline_coords,
                    width_sample_step=width_sample_step
                )
                # Classify width profile
                width_profile = classify_width_profile(width_profile)
                
                edge['length'] = length
                edge['width_profile'] = width_profile
                
            except Exception as e:
                logger.warning("Failed to measure edge %s: %s", edge['id'], e)
                # Set defaults
                edge['length'] = 0.0
                edge['width_profile'] = {
                    'kind': 'sampled',
                    'samples': [],
                    'w_median': 0.0,
                    'w_min': 0.0,
                    'w_max': 0.0,
                    'w0': 0.0,
                    'w1': 0.0
                }
        
        measure_time = time.time() - start
        logger.info(f"  Measured {len(edges)} edges in {measure_time:.2f}s")
        
        # 3. Graph-level pruning: delete leaf edges with length < L_min
        logger.debug("Graph-level pruning: removing short leaf edges")
        edges_to_remove = []
        min_length_absolute = 30.0  # Absolute minimum in µm
        
        # Compute node degrees (number of incident edges)
        node_degrees = {}
        for edge in edges:
            node_degrees[edge['u']] = node_degrees.get(edge['u'], 0) + 1
            node_degrees[edge['v']] = node_degrees.get(edge['v'], 0) + 1
        
        # Find leaf edges (edges incident to degree-1 nodes)
        for edge in edges:
            u_degree = node_degrees.get(edge['u'], 0)
            v_degree = node_degrees.get(edge['v'], 0)
            
            # Check if this is a leaf edge (one endpoint has degree 1)
            is_leaf = (u_degree == 1) or (v_degree == 1)
            
            if is_leaf and edge.get('length', 0) > 0:
                # Compute L_min = 2-3 × local_width
                width_profile = edge.get('width_profile', {})
                local_width = width_profile.get('w_median', 50.0)  # Default 50 µm if no width
                l_min = max(2.5 * local_width, min_length_absolute)
                
                if edge['length'] < l_min:
                    edges_to_remove.append(edge['id'])
                    logger.debug("Marking leaf edge %s for removal: length=%.1f < L_min=%.1f (width=%.1f)",
                                edge['id'], edge['length'], l_min, local_width)
        
        # Remove short leaf edges (iterate until no more to remove)
        removed_count = 0
        max_iterations = 10
        for iteration in range(max_iterations):
            if not edges_to_remove:
                break
            
            # Remove edges
            edges = [e for e in edges if e['id'] not in edges_to_remove]
            removed_count += len(edges_to_remove)
            
            # Recompute node degrees
            node_degrees = {}
            for edge in edges:
                node_degrees[edge['u']] = node_degrees.get(edge['u'], 0) + 1
                node_degrees[edge['v']] = node_degrees.get(edge['v'], 0) + 1
            
            # Find new leaf edges to remove
            edges_to_remove = []
            for edge in edges:
                u_degree = node_degrees.get(edge['u'], 0)
                v_degree = node_degrees.get(edge['v'], 0)
                is_leaf = (u_degree == 1) or (v_degree == 1)
                
                if is_leaf and edge.get('length', 0) > 0:
                    width_profile = edge.get('width_profile', {})
                    local_width = width_profile.get('w_median', 50.0)
                    l_min = max(2.5 * local_width, min_length_absolute)
                    
                    if edge['length'] < l_min:
                        edges_to_remove.append(edge['id'])
        
        if removed_count > 0:
            logger.info(f"  Pruned {removed_count} short leaf edges")
        
        # 4. Endpoint fork collapse: remove pitchfork branches
        logger.debug("Endpoint fork collapse: removing pitchfork branches")
        # Recompute node degrees after pruning
        node_degrees = {}
        node_edges = {}  # Map node_id -> list of incident edges
        for edge in edges:
            node_degrees[edge['u']] = node_degrees.get(edge['u'], 0) + 1
            node_degrees[edge['v']] = node_degrees.get(edge['v'], 0) + 1
            if edge['u'] not in node_edges:
                node_edges[edge['u']] = []
            if edge['v'] not in node_edges:
                node_edges[edge['v']] = []
            node_edges[edge['u']].append(edge)
            node_edges[edge['v']].append(edge)
        
        fork_edges_to_remove = []
        nodes_to_reclassify = []
        
        for node in nodes:
            node_id = node['id']
            incident_edges = node_edges.get(node_id, [])
            
            if len(incident_edges) < 3:
                continue  # Need at least 3 edges for a fork
            
            # Find the longest edge (main edge)
            incident_edges_with_length = [(e, e.get('length', 0)) for e in incident_edges]
            incident_edges_with_length.sort(key=lambda x: x[1], reverse=True)
            
            main_edge = incident_edges_with_length[0][0]
            leaf_edges = [e for e, _ in incident_edges_with_length[1:]]
            
            # Check if leaf edges are all short
            main_length = main_edge.get('length', 0)
            all_leaves_short = all(
                e.get('length', 0) < main_length * 0.5 and e.get('length', 0) < 150.0
                for e in leaf_edges
            )
            
            if all_leaves_short and len(leaf_edges) >= 2:
                # This is a pitchfork - remove short leaf edges
                for leaf_edge in leaf_edges:
                    fork_edges_to_remove.append(leaf_edge['id'])
                    logger.debug("Marking pitchfork leaf edge %s for removal at node %s",
                                leaf_edge['id'], node_id)
                
                # Reclassify node as endpoint if it becomes degree 1
                if len(incident_edges) - len(leaf_edges) == 1:
                    nodes_to_reclassify.append(node_id)
        
        # Remove pitchfork edges
        if fork_edges_to_remove:
            edges = [e for e in edges if e['id'] not in fork_edges_to_remove]
            logger.info(f"  Removed {len(fork_edges_to_remove)} pitchfork leaf edges")
        
        # Reclassify nodes
        for node_id in nodes_to_reclassify:
            node = next((n for n in nodes if n['id'] == node_id), None)
            if node:
                node['kind'] = 'endpoint'
                logger.debug("Reclassified node %s from junction to endpoint (pitchfork removed)", node_id)
    
    # Final pass: reclassify degree-1 nodes as endpoints and snap ALL endpoints to boundary
    logger.debug("Final pass: reclassifying degree-1 nodes as endpoints and snapping to boundary")
    
    # Reclassify nodes with degree 1 as endpoints
    for node in nodes:
        incident_count = sum(1 for e in edges if e['u'] == node['id'] or e['v'] == node['id'])
        if incident_count == 1 and node['kind'] != 'endpoint':
            logger.debug("Reclassifying node %s from %s to endpoint (degree=1)", node['id'], node['kind'])
            node['kind'] = 'endpoint'
    
    # Snap ALL endpoint nodes to polygon boundary
    # After refinement, endpoints are equidistant from boundary, but we want them precisely on the boundary
    # Follow the edge tangent until it hits the polygon boundary
    for node in nodes:
        if node['kind'] == 'endpoint':
            endpoint_xy = tuple(node['xy'])
            # Find incident edge to get tangent direction
            incident_edges = [e for e in edges if e['u'] == node['id'] or e['v'] == node['id']]
            
            # Get edge coordinates for tangent direction
            incident_edge_coords = None
            endpoint_node_id = None
            
            if incident_edges:
                # Use first incident edge to find tangent direction
                edge = incident_edges[0]
                incident_edge_coords = [tuple(c) for c in edge['centerline']['coordinates']]
                
                # Try to find skeleton node ID from cluster_nodes for fallback
                if 'cluster_nodes' in node and node['cluster_nodes']:
                    endpoint_node_id = node['cluster_nodes'][0]
            
            # Snap to boundary by following edge tangent
            snapped_xy = snap_endpoint_to_boundary(
                endpoint_xy,
                polygon,
                skeleton_graph,
                endpoint_node_id,
                incident_edge_coords=incident_edge_coords
            )
            
            # Update node coordinates
            dist_moved = Point(endpoint_xy).distance(Point(snapped_xy))
            node['xy'] = list(snapped_xy)
            
            if dist_moved > 1.0:  # Only log if moved more than 1 µm
                logger.debug("Snapped endpoint %s to boundary, moved %.1f µm", node['id'], dist_moved)
                
                # Extend incident edge centerlines to the new endpoint position
                for edge in incident_edges:
                    coords = edge['centerline']['coordinates']
                    if len(coords) >= 2:
                        if edge['u'] == node['id']:
                            # Node is at start of edge - update first coordinate
                            coords[0] = list(snapped_xy)
                        elif edge['v'] == node['id']:
                            # Node is at end of edge - update last coordinate
                            coords[-1] = list(snapped_xy)
                        logger.debug("Extended edge %s centerline to snapped endpoint %s", edge['id'], node['id'])
    
    if not measure_edges and edges:
        # If measurement disabled, set placeholder values
        for edge in edges:
            edge['length'] = 0.0
            edge['width_profile'] = {
                'kind': 'sampled',
                'samples': [],
                'w_median': 0.0,
                'w_min': 0.0,
                'w_max': 0.0,
                'w0': 0.0,
                'w1': 0.0
            }
    
    # Apply cross-section defaults
    if edges:
        apply_cross_section_defaults(
            edges,
            default_height=default_height,
            default_cross_section_kind=default_cross_section_kind,
            per_edge_overrides=per_edge_overrides
        )
    
    logger.info(f"Graph built: {len(nodes)} nodes, {len(edges)} edges in {node_build_time + edge_build_time:.2f}s total")
    
    result = {
        'nodes': nodes,
        'edges': edges,
        'skeleton_graph': skeleton_graph,  # For debugging/visualization
        # C2 data: skeleton graph after endpoint merging
        'c2_data': {
            'skeleton_graph': skeleton_graph,  # Skeleton graph after endpoint merging
            'junction_clusters': junction_clusters,
            'endpoint_clusters': endpoint_clusters,
            'forbidden_nodes': forbidden_nodes,
            'polygon': polygon
        }
    }
    
    return result


def extract_graph_from_polygons(
    polygons: List[Dict[str, Any]],
    minimum_channel_width: float,
    um_per_px: Optional[float] = None,
    simplify_tolerance: Optional[float] = None,
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
    endpoint_merge_distance_factor: float = 1.0
) -> Dict[str, Any]:
    """
    Extract network graph from multiple polygons (union first) and optionally detect ports.
    
    Args:
        polygons: List of polygon dicts with 'polygon' key containing GeoJSON
        minimum_channel_width: Minimum channel width in micrometers (required).
                              Used to calculate um_per_px = ceil(minimum_channel_width / 10) if um_per_px not provided.
        um_per_px: Resolution for skeletonization (microns per pixel). If None, calculated as ceil(minimum_channel_width / 10).
        simplify_tolerance: Tolerance for path simplification. If None, computed as simplify_tolerance_factor * minimum_channel_width.
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
    
    Returns:
        Dictionary with nodes, edges, and optionally ports
    """
    import math
    
    # Calculate um_per_px from minimum_channel_width if not provided
    if um_per_px is None:
        um_per_px = math.ceil(minimum_channel_width / 10.0)
        logger.info(f"Calculated um_per_px={um_per_px:.2f} from minimum_channel_width={minimum_channel_width:.2f} µm")
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    
    # Convert polygons to Shapely, preserving holes
    shapely_polygons = []
    for poly_data in polygons:
        rings = poly_data['polygon']['coordinates']
        if not rings or len(rings[0]) < 3:
            continue  # Skip invalid polygons
        
        try:
            exterior = rings[0]
            holes = rings[1:] if len(rings) > 1 else []
            poly = Polygon(exterior, holes=holes)
            if poly.is_valid:
                shapely_polygons.append(poly)
        except Exception:
            continue
    
    if not shapely_polygons:
        return {'nodes': [], 'edges': [], 'ports': []}
    
    # If exactly one channel region polygon, skip union and use it directly
    # Only union multiple polygons when there are multiple disjoint channel regions
    if len(shapely_polygons) == 1:
        combined_poly = shapely_polygons[0]
    else:
        combined_poly = unary_union(shapely_polygons)
        # If union produces MultiPolygon, take largest
        if combined_poly.geom_type == 'MultiPolygon':
            combined_poly = max(combined_poly.geoms, key=lambda g: g.area)
    
    # Extract graph from combined polygon
    graph_result = extract_graph_from_polygon(
        combined_poly,
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
    
    # Optionally detect ports
    ports = []
    if circles is not None:
        polyline_circles = None
        
        # Optionally detect circle-like polylines
        if detect_polyline_circles:
            from .curve_fit import detect_circle_like_polyline
            polyline_circles = []
            for poly_data in polygons:
                coords = poly_data['polygon']['coordinates'][0]
                # Check if it's a closed polyline that might be a circle
                if len(coords) >= 8:  # Minimum points for circle detection
                    circle_result = detect_circle_like_polyline(
                        coords,
                        rms_tolerance=circle_fit_rms_tol
                    )
                    if circle_result:
                        circle_result['entity_handles'] = [poly_data.get('entity_handle', '')]
                        polyline_circles.append(circle_result)
        
        # Detect and attach ports
        ports = detect_ports(
            circles,
            graph_result['nodes'],
            snap_distance=port_snap_distance,
            polyline_circles=polyline_circles
        )
        
        graph_result['ports'] = ports
    
    return graph_result
