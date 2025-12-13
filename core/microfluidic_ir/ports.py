"""Port detection and attachment to graph nodes."""

from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Point
import logging

logger = logging.getLogger(__name__)


def attach_port_to_nearest_node(
    port_center: List[float],
    nodes: List[Dict[str, Any]],
    snap_distance: float = 50.0
) -> Optional[str]:
    """
    Attach a port marker to the nearest node (endpoint or junction).
    
    Args:
        port_center: [x, y] coordinates of port center
        nodes: List of node dicts with 'id', 'xy', 'kind' fields
        snap_distance: Maximum distance to attach port to node (default: 50.0 µm)
        
    Returns:
        Node ID if a node is within snap_distance, else None
    """
    port_point = Point(port_center)
    
    min_dist = float('inf')
    closest_node_id = None
    
    # Prefer endpoints, then junctions
    for node in nodes:
        node_point = Point(node['xy'])
        dist = port_point.distance(node_point)
        
        if dist < min_dist and dist <= snap_distance:
            min_dist = dist
            closest_node_id = node['id']
    
    if closest_node_id:
        logger.debug("Attached port at (%.1f, %.1f) to node %s (distance=%.1f µm)",
                    port_center[0], port_center[1], closest_node_id, min_dist)
    
    return closest_node_id


def detect_ports(
    circles: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]],
    snap_distance: float = 50.0,
    polyline_circles: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Detect ports from circles and attach them to nearest nodes.
    
    Args:
        circles: List of circle dicts from DXF loader with 'center' and 'radius'
        nodes: List of node dicts from graph extraction
        snap_distance: Maximum distance to attach port to node (default: 50.0 µm)
        polyline_circles: Optional list of detected circle-like polylines
        
    Returns:
        List of port dicts with:
        - port_id: str
        - node_id: str (or None if not attached)
        - marker: dict with 'kind': 'circle', 'center': [x, y], 'radius': r
        - source: dict with entity handles, etc.
        - label: str (default: empty, to be set by user)
        - role: str (default: 'unknown', to be set by user)
    """
    ports = []
    port_counter = 1
    
    # Process native DXF circles
    for circle_data in circles:
        center = circle_data['center']
        radius = circle_data['radius']
        
        # Find nearest node
        node_id = attach_port_to_nearest_node(center, nodes, snap_distance)
        
        port_id = f"P{port_counter}"
        port_counter += 1
        
        port = {
            'port_id': port_id,
            'node_id': node_id,
            'marker': {
                'kind': 'circle',
                'center': center,
                'radius': radius
            },
            'source': {
                'entity_handles': [circle_data.get('entity_handle', '')],
                'layer': circle_data.get('layer', '')
            },
            'label': '',  # To be set by user
            'role': 'unknown'  # To be set by user
        }
        
        ports.append(port)
        
        if node_id:
            logger.debug("Detected port %s at (%.1f, %.1f), radius=%.1f µm, attached to %s",
                        port_id, center[0], center[1], radius, node_id)
        else:
            logger.debug("Detected port %s at (%.1f, %.1f), radius=%.1f µm, not attached (no node within %.1f µm)",
                        port_id, center[0], center[1], radius, snap_distance)
    
    # Process circle-like polylines if provided
    if polyline_circles:
        for circle_data in polyline_circles:
            center = circle_data['center']
            radius = circle_data['radius']
            
            # Find nearest node
            node_id = attach_port_to_nearest_node(center, nodes, snap_distance)
            
            port_id = f"P{port_counter}"
            port_counter += 1
            
            port = {
                'port_id': port_id,
                'node_id': node_id,
                'marker': {
                    'kind': 'circle',
                    'center': center,
                    'radius': radius,
                    'rms_error': circle_data.get('rms_error', None)  # Indicate it's fitted
                },
                'source': {
                    'entity_handles': circle_data.get('entity_handles', []),
                    'detected_from': 'polyline'
                },
                'label': '',  # To be set by user
                'role': 'unknown'  # To be set by user
            }
            
            ports.append(port)
            
            if node_id:
                logger.debug("Detected port %s from polyline at (%.1f, %.1f), radius=%.1f µm, attached to %s",
                            port_id, center[0], center[1], radius, node_id)
    
    logger.info("Detected %d port(s)", len(ports))
    
    return ports

