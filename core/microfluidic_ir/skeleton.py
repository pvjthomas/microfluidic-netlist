"""Skeletonization of channel polygons."""

import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
import networkx as nx
from skimage.morphology import skeletonize
from skimage import measure
from typing import List, Tuple, Dict, Any


def rasterize_polygon(
    polygon: Polygon,
    px_per_unit: float = 10.0,
    padding: float = 10.0
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Rasterize a Shapely polygon to a binary image.
    
    Args:
        polygon: Shapely polygon to rasterize
        px_per_unit: Pixels per unit (higher = more resolution)
        padding: Padding around polygon in original units
    
    Returns:
        Tuple of (binary_image, transform_dict) where transform_dict contains:
        - origin_x, origin_y: Top-left corner in original coordinates
        - px_per_unit: Resolution
    """
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
    
    # Create coordinate arrays
    x_coords = np.linspace(padded_minx, padded_minx + padded_width, img_width)
    y_coords = np.linspace(padded_miny, padded_miny + padded_height, img_height)
    
    # Create binary image by checking if each pixel center is inside polygon
    binary_img = np.zeros((img_height, img_width), dtype=bool)
    
    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            point = Point(x, y)
            if polygon.contains(point) or polygon.touches(point):
                binary_img[i, j] = True
    
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
    
    # Skeletonize using scikit-image
    skeleton_img = skeletonize(binary_img)
    
    # Find skeleton pixels (non-zero)
    skeleton_pixels = np.argwhere(skeleton_img)
    
    if len(skeleton_pixels) == 0:
        # Empty skeleton - return empty graph
        return nx.Graph(), transform
    
    # Convert skeleton pixels to coordinate space
    skeleton_coords = []
    for pixel in skeleton_pixels:
        coords = pixel_to_coords((pixel[0], pixel[1]), transform)
        skeleton_coords.append(coords)
    
    # Build graph from skeleton pixels
    # Each pixel becomes a node, connected to its 8-connected neighbors
    G = nx.Graph()
    
    # Add all skeleton pixels as nodes
    pixel_to_node = {}
    for i, pixel in enumerate(skeleton_pixels):
        coords = pixel_to_coords((pixel[0], pixel[1]), transform)
        node_id = i
        G.add_node(node_id, xy=coords)
        pixel_to_node[tuple(pixel)] = node_id
    
    # Connect neighboring pixels (8-connected)
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
