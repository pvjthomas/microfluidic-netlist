"""Branch extraction and statistics computation."""

from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
import numpy as np
import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Branch:
    """Branch data structure."""
    id: int
    nodes: List[int]  # List of node IDs in the branch path
    length: float
    mean_width: float
    min_width: float
    max_width: float
    width_std: float
    slenderness: float
    u_terminal_type: str  # "endpoint" or "junction"
    v_terminal_type: str
    polyline: List[Tuple[float, float]]  # World coordinates


def extract_branches(
    G_px: nx.Graph,
    distance_field: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Extract branches from pixel graph G_px.
    
    A branch is a maximal path between terminals (nodes with degree != 2).
    Terminals are either endpoints (degree == 1) or junctions (degree >= 3).
    
    Args:
        G_px: NetworkX graph with pixel nodes
        distance_field: Optional dict with distance field data ('dist', 'transform' keys)
                       If provided, used to compute width statistics
    
    Returns:
        List of branch dicts with computed statistics
    """
    if len(G_px) == 0:
        return []
    
    # Import here to avoid circular imports
    from .geometry import get_distance_at_coordinate
    
    # Helper to get width at a node
    def get_node_width(node_id: int) -> float:
        """Get width (2 * radius) at a node from distance field."""
        if distance_field is None:
            return 0.0
        try:
            node_data = G_px.nodes[node_id]
            if 'xy' in node_data:
                x, y = node_data['xy']
                dist_array = distance_field.get('dist')
                transform_dict = distance_field.get('transform')
                if dist_array is not None and transform_dict is not None:
                    radius = get_distance_at_coordinate(x, y, dist_array, transform_dict)
                    return 2.0 * radius
        except Exception as e:
            logger.debug(f"Failed to get width at node {node_id}: {e}")
        return 0.0
    
    # Identify terminals (degree != 2)
    terminals = {n for n in G_px.nodes() if G_px.degree(n) != 2}
    logger.debug(f"extract_branches: found {len(terminals)} terminals")
    
    # Classify terminals
    terminal_types = {}
    for term in terminals:
        deg = G_px.degree(term)
        if deg == 1:
            terminal_types[term] = "endpoint"
        elif deg >= 3:
            terminal_types[term] = "junction"
        else:
            terminal_types[term] = "junction"  # Fallback
    
    branches = []
    visited_edges = set()
    branch_id = 0
    
    # Helper to make edge tuple
    def edge_tuple(u, v):
        return tuple(sorted([u, v]))
    
    # Helper to walk along degree-2 nodes
    def walk_branch(start_node, first_neighbor, visited_set):
        """Walk from start_node through first_neighbor along degree-2 nodes."""
        path_nodes = [start_node, first_neighbor]
        current = first_neighbor
        prev = start_node
        
        # Walk forward along degree-2 nodes
        while G_px.degree(current) == 2:
            neighbors = list(G_px.neighbors(current))
            if len(neighbors) != 2:
                break
            next_node = neighbors[1] if neighbors[0] == prev else neighbors[0]
            
            # Check if edge already visited
            edge = edge_tuple(current, next_node)
            if edge in visited_set:
                break
            
            path_nodes.append(next_node)
            prev = current
            current = next_node
        
        return path_nodes, current
    
    # Extract branches: for each terminal, walk along unvisited edges
    for terminal in terminals:
        for neighbor in G_px.neighbors(terminal):
            edge = edge_tuple(terminal, neighbor)
            if edge in visited_edges:
                continue
            
            # Walk the branch
            path_nodes, end_node = walk_branch(terminal, neighbor, visited_edges)
            
            # Mark all edges in path as visited
            for i in range(len(path_nodes) - 1):
                edge = edge_tuple(path_nodes[i], path_nodes[i+1])
                visited_edges.add(edge)
            
            # Compute branch statistics
            branch = compute_branch_stats(
                branch_id,
                path_nodes,
                G_px,
                terminal_types.get(terminal, "junction"),
                terminal_types.get(end_node, "junction"),
                get_node_width
            )
            
            if branch:
                branches.append(branch)
                branch_id += 1
    
    # Handle any remaining cycles (all nodes have degree == 2)
    remaining_edges = set(G_px.edges()) - visited_edges
    if remaining_edges:
        logger.debug(f"extract_branches: found {len(remaining_edges)} unvisited edges (cycles)")
        # Pick any unvisited edge and walk the cycle
        while remaining_edges:
            start_edge = remaining_edges.pop()
            u, v = start_edge
            path_nodes = [u, v]
            current = v
            prev = u
            
            # Walk the cycle
            while True:
                neighbors = list(G_px.neighbors(current))
                if len(neighbors) != 2:
                    break
                next_node = neighbors[1] if neighbors[0] == prev else neighbors[0]
                
                edge = edge_tuple(current, next_node)
                if edge not in remaining_edges:
                    break
                
                remaining_edges.remove(edge)
                path_nodes.append(next_node)
                prev = current
                current = next_node
                
                # Check if we've completed the cycle
                if next_node == path_nodes[0]:
                    break
            
            # Compute branch stats for cycle
            branch = compute_branch_stats(
                branch_id,
                path_nodes,
                G_px,
                "junction",  # Cycles connect to junctions
                "junction",
                get_node_width
            )
            
            if branch:
                branches.append(branch)
                branch_id += 1
    
    logger.info(f"extract_branches: extracted {len(branches)} branches")
    return branches


def compute_branch_stats(
    branch_id: int,
    path_nodes: List[int],
    G_px: nx.Graph,
    u_terminal_type: str,
    v_terminal_type: str,
    get_node_width: callable
) -> Optional[Dict[str, Any]]:
    """
    Compute statistics for a branch.
    
    Args:
        branch_id: Unique branch ID
        path_nodes: List of node IDs in the branch
        G_px: NetworkX graph
        u_terminal_type: Type of terminal at start ("endpoint" or "junction")
        v_terminal_type: Type of terminal at end ("endpoint" or "junction")
        get_node_width: Function to get width at a node ID
    
    Returns:
        Branch dict with statistics, or None if invalid
    """
    if len(path_nodes) < 2:
        return None
    
    # Compute length and collect widths
    total_length = 0.0
    widths = []
    polyline = []
    
    for i in range(len(path_nodes) - 1):
        node1 = path_nodes[i]
        node2 = path_nodes[i+1]
        
        # Get coordinates
        if 'xy' in G_px.nodes[node1]:
            x1, y1 = G_px.nodes[node1]['xy']
            polyline.append((x1, y1))
        else:
            return None  # Missing coordinates
        
        if 'xy' in G_px.nodes[node2]:
            x2, y2 = G_px.nodes[node2]['xy']
        else:
            return None
        
        # Compute edge length
        edge_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_length += edge_length
        
        # Get width at each node
        width1 = get_node_width(node1)
        width2 = get_node_width(node2)
        widths.extend([width1, width2])
    
    # Add last node coordinates
    if 'xy' in G_px.nodes[path_nodes[-1]]:
        x, y = G_px.nodes[path_nodes[-1]]['xy']
        polyline.append((x, y))
    
    if not widths or total_length == 0:
        return None
    
    # Compute width statistics
    widths_array = np.array(widths)
    mean_width = float(np.mean(widths_array))
    min_width = float(np.min(widths_array))
    max_width = float(np.max(widths_array))
    width_std = float(np.std(widths_array))
    
    # Compute slenderness
    slenderness = total_length / mean_width if mean_width > 0 else 0.0
    
    return {
        'id': branch_id,
        'nodes': path_nodes,
        'length': total_length,
        'mean_width': mean_width,
        'min_width': min_width,
        'max_width': max_width,
        'width_std': width_std,
        'slenderness': slenderness,
        'u_terminal_type': u_terminal_type,
        'v_terminal_type': v_terminal_type,
        'polyline': polyline
    }


def create_branch_graph(
    G_px: nx.Graph,
    branches: List[Dict[str, Any]]
) -> nx.Graph:
    """
    Create reduced branch graph G_br from branches.
    
    Nodes in G_br are terminals (endpoints + junctions).
    Edges in G_br are branches between terminals.
    
    Args:
        G_px: Original pixel graph
        branches: List of branch dicts from extract_branches
    
    Returns:
        NetworkX graph G_br with branch statistics as edge attributes
    """
    G_br = nx.Graph()
    
    # Identify terminals from G_px
    terminals = {n for n in G_px.nodes() if G_px.degree(n) != 2}
    
    # Add terminal nodes to G_br
    for term in terminals:
        if 'xy' in G_px.nodes[term]:
            G_br.add_node(term, xy=G_px.nodes[term]['xy'])
    
    # Add branch edges
    for branch in branches:
        if len(branch['nodes']) < 2:
            continue
        
        u = branch['nodes'][0]
        v = branch['nodes'][-1]
        
        # Only add if both terminals exist
        if u in G_br and v in G_br:
            G_br.add_edge(
                u, v,
                length=branch['length'],
                mean_width=branch['mean_width'],
                min_width=branch['min_width'],
                max_width=branch['max_width'],
                width_std=branch['width_std'],
                slenderness=branch['slenderness'],
                polyline=branch['polyline'],
                u_terminal_type=branch['u_terminal_type'],
                v_terminal_type=branch['v_terminal_type'],
                branch_id=branch['id']
            )
    
    logger.info(f"create_branch_graph: created G_br with {len(G_br.nodes())} terminals and {len(G_br.edges())} branch edges")
    return G_br

