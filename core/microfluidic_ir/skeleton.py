"""Skeletonization of channel polygons."""

import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
import networkx as nx
from skimage.morphology import skeletonize
from skimage import measure
from typing import List, Tuple, Dict, Any
import logging
import time
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def rasterize_polygon(
    polygon: Polygon,
    px_per_unit: float = 10.0,
    padding: float = 10.0
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Rasterize a Shapely polygon to a binary image using PIL ImageDraw.
    
    Args:
        polygon: Shapely polygon to rasterize
        px_per_unit: Pixels per unit (higher = more resolution)
        padding: Padding around polygon in original units
    
    Returns:
        Tuple of (binary_image, transform_dict) where transform_dict contains:
        - origin_x, origin_y: Top-left corner in original coordinates
        - px_per_unit: Resolution
    """
    start_raster = time.time()
    
    # Get bounds and add padding
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    
    # Add padding
    padded_minx = bounds[0] - padding
    padded_miny = bounds[1] - padding
    padded_width = width + 2 * padding
    padded_height = height + 2 * padding
    
    # Calculate image dimensions
    img_width = int(np.ceil(padded_width * px_per_unit))
    img_height = int(np.ceil(padded_height * px_per_unit))
    
    # Guardrail: check raster size limits
    max_dimension = 8000
    max_pixels = 25_000_000
    
    if img_width > max_dimension or img_height > max_dimension:
        raise ValueError(
            f"Raster size {img_width}×{img_height} exceeds maximum dimension {max_dimension}. "
            f"Reduce px_per_unit (currently {px_per_unit}) or polygon size."
        )
    
    total_pixels = img_width * img_height
    if total_pixels > max_pixels:
        raise ValueError(
            f"Raster size {img_width}×{img_height} = {total_pixels} pixels exceeds maximum {max_pixels}. "
            f"Reduce px_per_unit (currently {px_per_unit}) or polygon size."
        )
    
    logger.info(f"Rasterizing: image shape ({img_height}, {img_width}), {total_pixels} pixels")
    
    # Create PIL image (black = 0, white = 1)
    img = Image.new('1', (img_width, img_height), 0)
    draw = ImageDraw.Draw(img)
    
    # Convert polygon coordinates to PIL image coordinates
    # PIL uses: (0,0) at top-left, x increases right, y increases down
    # pixel_to_coords uses: y = origin_y + (row / px_per_unit), so row 0 maps to origin_y (minimum y)
    # This means our image has y increasing upward (row increases = y increases in coords)
    # Conversion: image_x = (shapely_x - padded_minx) * px_per_unit
    #            image_y = (shapely_y - padded_miny) * px_per_unit (same direction)
    
    def shapely_to_image_coords(shapely_x: float, shapely_y: float) -> Tuple[int, int]:
        """Convert Shapely coordinates to PIL image coordinates.
        
        Note: pixel_to_coords uses y = origin_y + (row / px_per_unit), so row 0 
        corresponds to minimum y (bottom in standard view, but our origin_y is min_y).
        PIL ImageDraw expects coordinates with y increasing downward, but we'll flip
        the image vertically after drawing to match the transform semantics.
        """
        img_x = int((shapely_x - padded_minx) * px_per_unit)
        # Flip y-axis: PIL y increases down, but our transform has y increase up with row
        img_y = int(img_height - 1 - (shapely_y - padded_miny) * px_per_unit)
        return (img_x, img_y)
    
    # Fill exterior with 1 (white)
    exterior_coords = [shapely_to_image_coords(x, y) for x, y in polygon.exterior.coords[:-1]]  # Exclude last duplicate
    if len(exterior_coords) >= 3:  # Need at least 3 points for a polygon
        draw.polygon(exterior_coords, fill=1, outline=1)
    
    # Clear each interior ring (hole) with 0 (black)
    for interior in polygon.interiors:
        interior_coords = [shapely_to_image_coords(x, y) for x, y in interior.coords[:-1]]  # Exclude last duplicate
        if len(interior_coords) >= 3:
            draw.polygon(interior_coords, fill=0, outline=0)
    
    # Convert PIL image to numpy array and flip vertically to match transform semantics
    # (pixel_to_coords expects row 0 = min y, but PIL draws with y=0 at top)
    binary_img = np.array(img, dtype=bool)
    binary_img = np.flipud(binary_img)  # Flip vertically to match transform semantics
    
    raster_time = time.time() - start_raster
    num_pixels = np.sum(binary_img)
    logger.info(f"  Raster fill: {raster_time:.2f}s")
    logger.info(f"Rasterized: {num_pixels} filled pixels ({100.0 * num_pixels / total_pixels:.1f}%) in {raster_time:.2f}s total")
    
    transform = {
        'origin_x': padded_minx,
        'origin_y': padded_miny,
        'px_per_unit': px_per_unit,
        'img_width': img_width,
        'img_height': img_height
    }
    
    return binary_img, transform


def pixel_to_coords(
    pixel: Tuple[int, int],
    transform: Dict[str, float]
) -> Tuple[float, float]:
    """Convert pixel coordinates to original coordinate space."""
    row, col = pixel
    x = transform['origin_x'] + (col / transform['px_per_unit'])
    y = transform['origin_y'] + (row / transform['px_per_unit'])
    return (x, y)


def coords_to_pixel(
    x: float,
    y: float,
    transform: Dict[str, float]
) -> Tuple[int, int]:
    """Convert original coordinates to pixel coordinates."""
    col = int((x - transform['origin_x']) * transform['px_per_unit'])
    row = int((y - transform['origin_y']) * transform['px_per_unit'])
    return (row, col)


def skeletonize_polygon(
    polygon: Polygon,
    px_per_unit: float = 10.0,
    simplify_tolerance: float = 1.0
) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Skeletonize a polygon and convert to a NetworkX graph.
    
    Args:
        polygon: Shapely polygon to skeletonize
        px_per_unit: Resolution for rasterization
        simplify_tolerance: Tolerance for simplifying centerlines (in original units)
    
    Returns:
        Tuple of (skeleton_graph, metadata) where:
        - skeleton_graph: NetworkX graph with nodes as (x, y) tuples
        - metadata: Transform and processing info
    """
    # Rasterize polygon
    binary_img, transform = rasterize_polygon(polygon, px_per_unit=px_per_unit)
    
    # Crop to tight mask bounds before skeletonize (plus small margin)
    start_crop = time.time()
    # Find bounding box of filled pixels
    filled_rows, filled_cols = np.where(binary_img)
    
    if len(filled_rows) == 0:
        # No filled pixels - return empty graph
        logger.info("Empty raster - no filled pixels")
        return nx.Graph(), transform
    
    # Get tight bounds
    min_row = max(0, int(filled_rows.min()) - 2)  # 2 pixel margin
    max_row = min(binary_img.shape[0], int(filled_rows.max()) + 3)  # 3 pixel margin
    min_col = max(0, int(filled_cols.min()) - 2)  # 2 pixel margin
    max_col = min(binary_img.shape[1], int(filled_cols.max()) + 3)  # 3 pixel margin
    
    # Crop the image
    cropped_img = binary_img[min_row:max_row, min_col:max_col]
    
    # Adjust transform to account for cropping
    # pixel_to_coords: x = origin_x + (col / px_per_unit), y = origin_y + (row / px_per_unit)
    # So row offset directly affects y, and col offset directly affects x
    col_offset_in_coords = min_col / transform['px_per_unit']
    row_offset_in_coords = min_row / transform['px_per_unit']
    
    # Update transform origin
    transform['origin_x'] = transform['origin_x'] + col_offset_in_coords
    transform['origin_y'] = transform['origin_y'] + row_offset_in_coords
    transform['img_width'] = cropped_img.shape[1]
    transform['img_height'] = cropped_img.shape[0]
    
    crop_time = time.time() - start_crop
    logger.info(f"Cropped to tight bounds: ({cropped_img.shape[0]}, {cropped_img.shape[1]}) in {crop_time:.2f}s")
    
    # Skeletonize using scikit-image
    logger.info("Skeletonizing binary image")
    start_skeleton = time.time()
    skeleton_img = skeletonize(cropped_img)
    skeleton_time = time.time() - start_skeleton
    logger.info(f"  Skeletonization algorithm: {skeleton_time:.2f}s")
    
    # Find skeleton pixels (non-zero)
    start = time.time()
    skeleton_pixels = np.argwhere(skeleton_img)
    find_time = time.time() - start
    logger.info(f"  Finding skeleton pixels: {find_time:.2f}s")
    
    if len(skeleton_pixels) == 0:
        # Empty skeleton - return empty graph
        logger.info("Empty skeleton")
        return nx.Graph(), transform
    
    logger.info(f"Skeleton: {len(skeleton_pixels)} skeleton pixels in {skeleton_time + find_time:.2f}s total")
    
    # Convert skeleton pixels to coordinate space
    skeleton_coords = []
    for pixel in skeleton_pixels:
        coords = pixel_to_coords((pixel[0], pixel[1]), transform)
        skeleton_coords.append(coords)
    
    # Build graph from skeleton pixels
    # Each pixel becomes a node, connected to its 8-connected neighbors
    import time
    G = nx.Graph()
    
    # Add all skeleton pixels as nodes
    start = time.time()
    pixel_to_node = {}
    for i, pixel in enumerate(skeleton_pixels):
        coords = pixel_to_coords((pixel[0], pixel[1]), transform)
        node_id = i
        G.add_node(node_id, xy=coords)
        pixel_to_node[tuple(pixel)] = node_id
    node_time = time.time() - start
    logger.info(f"  Adding nodes: {node_time:.2f}s")
    
    # Connect neighboring pixels (8-connected)
    start = time.time()
    for pixel in skeleton_pixels:
        row, col = pixel[0], pixel[1]
        node_id = pixel_to_node[(row, col)]
        
        # Check 8 neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                neighbor = (row + dr, col + dc)
                if neighbor in pixel_to_node:
                    neighbor_id = pixel_to_node[neighbor]
                    if not G.has_edge(node_id, neighbor_id):
                        # Calculate distance in coordinate space
                        x1, y1 = G.nodes[node_id]['xy']
                        x2, y2 = G.nodes[neighbor_id]['xy']
                        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        G.add_edge(node_id, neighbor_id, weight=dist)
    edge_time = time.time() - start
    logger.info(f"  Adding edges: {edge_time:.2f}s")
    
    logger.info(f"Skeleton graph: {len(G.nodes())} nodes, {len(G.edges())} edges in {node_time + edge_time:.2f}s total")
    
    return G, transform


def extract_skeleton_paths(
    skeleton_graph: nx.Graph,
    simplify_tolerance: float = 1.0
) -> List[List[Tuple[float, float]]]:
    """
    Extract centerline paths from skeleton graph.
    
    Identifies endpoints (degree 1) and junctions (degree >= 3),
    then extracts paths between them.
    
    Args:
        skeleton_graph: NetworkX graph from skeletonization
        simplify_tolerance: Tolerance for simplifying paths (in original units)
    
    Returns:
        List of paths, where each path is a list of (x, y) coordinates
    """
    if len(skeleton_graph) == 0:
        return []
    
    # Identify nodes by degree
    endpoints = [n for n in skeleton_graph.nodes() if skeleton_graph.degree(n) == 1]
    junctions = [n for n in skeleton_graph.nodes() if skeleton_graph.degree(n) >= 3]
    
    # If no endpoints/junctions, return single path through all nodes
    if not endpoints and not junctions:
        # Simple path through all connected nodes
        if len(skeleton_graph) > 0:
            nodes = list(skeleton_graph.nodes())
            path = [skeleton_graph.nodes[n]['xy'] for n in nodes]
            return [path] if len(path) > 1 else []
        return []
    
    # Extract paths between endpoints and junctions
    paths = []
    visited_edges = set()
    
    # Start from each endpoint
    for start in endpoints:
        if start not in skeleton_graph:
            continue
        
        # Find path to nearest junction or endpoint
        targets = junctions + [ep for ep in endpoints if ep != start]
        
        for target in targets:
            if target == start:
                continue
            
            try:
                path_nodes = nx.shortest_path(skeleton_graph, start, target)
                path_coords = [skeleton_graph.nodes[n]['xy'] for n in path_nodes]
                
                # Check if we've used these edges
                path_edges = set()
                for i in range(len(path_nodes) - 1):
                    edge = tuple(sorted([path_nodes[i], path_nodes[i+1]]))
                    path_edges.add(edge)
                
                if not path_edges.intersection(visited_edges):
                    paths.append(path_coords)
                    visited_edges.update(path_edges)
                    break  # Found a path from this endpoint
            except nx.NetworkXNoPath:
                continue
    
    # Also handle paths between junctions
    for i, j1 in enumerate(junctions):
        for j2 in junctions[i+1:]:
            try:
                path_nodes = nx.shortest_path(skeleton_graph, j1, j2)
                path_coords = [skeleton_graph.nodes[n]['xy'] for n in path_nodes]
                
                path_edges = set()
                for k in range(len(path_nodes) - 1):
                    edge = tuple(sorted([path_nodes[k], path_nodes[k+1]]))
                    path_edges.add(edge)
                
                if not path_edges.intersection(visited_edges):
                    paths.append(path_coords)
                    visited_edges.update(path_edges)
            except nx.NetworkXNoPath:
                continue
    
    # Simplify paths using Douglas-Peucker-like approach
    from shapely.geometry import LineString
    simplified_paths = []
    for path in paths:
        if len(path) < 2:
            continue
        line = LineString(path)
        simplified = line.simplify(simplify_tolerance, preserve_topology=False)
        if simplified.geom_type == 'LineString':
            simplified_paths.append(list(simplified.coords))
        else:
            # MultiLineString - take longest segment
            if hasattr(simplified, 'geoms'):
                longest = max(simplified.geoms, key=lambda g: g.length)
                simplified_paths.append(list(longest.coords))
    
    return simplified_paths if simplified_paths else paths
