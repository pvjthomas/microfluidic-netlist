"""Extract network graph from skeleton."""

from typing import List, Dict, Any, Tuple
from shapely.geometry import Point
import networkx as nx
from .skeleton import skeletonize_polygon, extract_skeleton_paths


def extract_graph_from_polygon(
    polygon: Any,  # Shapely Polygon
    px_per_unit: float = 10.0,
    simplify_tolerance: float = 1.0
) -> Dict[str, Any]:
    """
    Extract network graph from a polygon.
    
    Args:
        polygon: Shapely polygon
        px_per_unit: Resolution for skeletonization
        simplify_tolerance: Tolerance for path simplification
    
    Returns:
        Dictionary with:
        - nodes: List of node dicts (id, xy, kind, degree)
        - edges: List of edge dicts (id, u, v, centerline)
        - skeleton_graph: NetworkX graph (for debugging)
    """
    # Skeletonize polygon
    skeleton_graph, transform = skeletonize_polygon(
        polygon,
        px_per_unit=px_per_unit,
        simplify_tolerance=simplify_tolerance
    )
    
    if len(skeleton_graph) == 0:
        return {
            'nodes': [],
            'edges': [],
            'skeleton_graph': skeleton_graph
        }
    
    # Extract paths
    paths = extract_skeleton_paths(skeleton_graph, simplify_tolerance=simplify_tolerance)
    
    # Identify nodes: endpoints and junctions
    endpoints = [n for n in skeleton_graph.nodes() if skeleton_graph.degree(n) == 1]
    junctions = [n for n in skeleton_graph.nodes() if skeleton_graph.degree(n) >= 3]
    
    # Build node list
    nodes = []
    node_id_map = {}  # Map from skeleton node ID to graph node ID
    
    # Add junctions first (they're more important)
    for i, junc_node in enumerate(junctions):
        node_id = f"N{i+1}"
        node_id_map[junc_node] = node_id
        nodes.append({
            'id': node_id,
            'xy': list(skeleton_graph.nodes[junc_node]['xy']),
            'kind': 'junction',
            'degree': skeleton_graph.degree(junc_node),
            'skeleton_node_id': junc_node
        })
    
    # Add endpoints
    endpoint_start_idx = len(nodes)
    for i, end_node in enumerate(endpoints):
        node_id = f"N{endpoint_start_idx + i + 1}"
        node_id_map[end_node] = node_id
        nodes.append({
            'id': node_id,
            'xy': list(skeleton_graph.nodes[end_node]['xy']),
            'kind': 'endpoint',
            'degree': 1,
            'skeleton_node_id': end_node
        })
    
    # Build edges from paths
    edges = []
    edge_counter = 1
    
    # For each path, find which nodes it connects
    for path in paths:
        if len(path) < 2:
            continue
        
        # Find start and end nodes (closest to path endpoints)
        start_point = Point(path[0])
        end_point = Point(path[-1])
        
        # Find closest nodes to path endpoints
        start_node_id = None
        end_node_id = None
        min_start_dist = float('inf')
        min_end_dist = float('inf')
        
        for node in nodes:
            node_point = Point(node['xy'])
            
            start_dist = start_point.distance(node_point)
            if start_dist < min_start_dist:
                min_start_dist = start_dist
                start_node_id = node['id']
            
            end_dist = end_point.distance(node_point)
            if end_dist < min_end_dist:
                min_end_dist = end_dist
                end_node_id = node['id']
        
        # Only create edge if we found valid nodes and they're different
        if start_node_id and end_node_id and start_node_id != end_node_id:
            # Check if edge already exists (reverse direction)
            edge_exists = any(
                (e['u'] == end_node_id and e['v'] == start_node_id) or
                (e['u'] == start_node_id and e['v'] == end_node_id)
                for e in edges
            )
            
            if not edge_exists:
                edge_id = f"E{edge_counter}"
                edge_counter += 1
                
                edges.append({
                    'id': edge_id,
                    'u': start_node_id,
                    'v': end_node_id,
                    'centerline': {
                        'type': 'LineString',
                        'coordinates': [[float(x), float(y)] for x, y in path]
                    }
                })
    
    return {
        'nodes': nodes,
        'edges': edges,
        'skeleton_graph': skeleton_graph  # For debugging/visualization
    }


def extract_graph_from_polygons(
    polygons: List[Dict[str, Any]],
    px_per_unit: float = 10.0,
    simplify_tolerance: float = 1.0
) -> Dict[str, Any]:
    """
    Extract network graph from multiple polygons (union first).
    
    Args:
        polygons: List of polygon dicts with 'polygon' key containing GeoJSON
        px_per_unit: Resolution for skeletonization
        simplify_tolerance: Tolerance for path simplification
    
    Returns:
        Dictionary with nodes and edges from combined graph
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    
    # Convert polygons to Shapely
    shapely_polygons = []
    for poly_data in polygons:
        coords = poly_data['polygon']['coordinates'][0]
        poly = Polygon(coords)
        if poly.is_valid:
            shapely_polygons.append(poly)
    
    if not shapely_polygons:
        return {'nodes': [], 'edges': []}
    
    # Union all polygons
    if len(shapely_polygons) == 1:
        combined_poly = shapely_polygons[0]
    else:
        combined_poly = unary_union(shapely_polygons)
        # If union produces MultiPolygon, take largest
        if combined_poly.geom_type == 'MultiPolygon':
            combined_poly = max(combined_poly.geoms, key=lambda g: g.area)
    
    # Extract graph from combined polygon
    return extract_graph_from_polygon(
        combined_poly,
        px_per_unit=px_per_unit,
        simplify_tolerance=simplify_tolerance
    )
