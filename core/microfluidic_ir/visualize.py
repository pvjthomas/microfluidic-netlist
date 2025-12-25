"""Visualize pipeline steps A, B, C, D in an interactive window."""

from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.widgets import Button, RectangleSelector
from shapely.geometry import Polygon, Point, LineString
import numpy as np
import networkx as nx
import itertools
import logging
import time

logger = logging.getLogger(__name__)


class PipelineVisualizer:
    """Interactive visualization of pipeline steps A, B, C, D."""
    
    def __init__(self):
        self.current_step = 0  # 0=A, 1=B, 2=C, 3=C2, 4=D
        self.fig = None
        self.ax = None
        self.data = {}  # Store data for each step
        self.computed_steps = set()  # Track which steps have been computed
        self.compute_params = {}  # Store parameters needed to compute steps C and D
        self.enabled_steps = {'A': True, 'B': True, 'C': True, 'C2': True, 'D': True}  # Which steps are enabled
        self.original_bounds = {}  # Store original bounds for each step for zoom to fit
        self.rect_selector = None  # Rectangle selector for zoom to area
        
    def set_step_a(self, dxf_result: Dict[str, Any]):
        """Step A: DXF import & normalization."""
        self.data['A'] = {
            'polygons': dxf_result.get('polygons', []),
            'circles': dxf_result.get('circles', []),
            'bounds': dxf_result.get('bounds', {})
        }
    
    def set_step_b(self, selected_polygons: List[Dict[str, Any]]):
        """Step B: Channel region selection."""
        self.data['B'] = {
            'selected_polygons': selected_polygons
        }
    
    def set_step_c(self, skeleton_graph, transform: Dict[str, float], polygon: Polygon):
        """Step C: Skeleton → graph extraction."""
        self.data['C'] = {
            'skeleton_graph': skeleton_graph,
            'transform': transform,
            'polygon': polygon
        }
    
    def set_step_c2(self, skeleton_graph, junction_clusters: Dict[int, List[int]], 
                   endpoint_clusters: Dict[int, List[int]], forbidden_nodes: set, polygon: Polygon):
        """Step C2: Topology-aware endpoint merging."""
        self.data['C2'] = {
            'skeleton_graph': skeleton_graph,
            'junction_clusters': junction_clusters,
            'endpoint_clusters': endpoint_clusters,
            'forbidden_nodes': forbidden_nodes,
            'polygon': polygon
        }
    
    def set_step_d(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], 
                   polygon: Polygon):
        """Step D: Final graph with nodes and edges."""
        self.data['D'] = {
            'nodes': nodes,
            'edges': edges,
            'polygon': polygon
        }
    
    def _draw_step_a(self):
        """Draw Step A: DXF polygons and circles."""
        self.ax.cla()  # Faster than clear() - doesn't reset all properties
        self.ax.set_title('Step A: DXF Import & Normalization', fontsize=14, fontweight='bold')
        
        data = self.data.get('A', {})
        polygons = data.get('polygons', [])
        circles = data.get('circles', [])
        
        # Draw polygons
        for poly_data in polygons:
            coords = poly_data['polygon']['coordinates'][0]
            poly = Polygon(coords)
            patch = mpatches.Polygon(list(poly.exterior.coords), 
                                     fill=True, edgecolor='black', 
                                     facecolor='lightblue', alpha=0.6, linewidth=1)
            self.ax.add_patch(patch)
        
        # Draw circles
        for circle_data in circles:
            center = circle_data['center']
            radius = circle_data['radius']
            circle = plt.Circle(center, radius, fill=False, edgecolor='red', 
                              linewidth=2, linestyle='--')
            self.ax.add_patch(circle)
        
        # Set bounds
        bounds = data.get('bounds', {})
        if bounds:
            margin = max(bounds.get('xmax', 0) - bounds.get('xmin', 0),
                        bounds.get('ymax', 0) - bounds.get('ymin', 0)) * 0.1
            xmin = bounds.get('xmin', 0) - margin
            xmax = bounds.get('xmax', 0) + margin
            ymin = bounds.get('ymin', 0) - margin
            ymax = bounds.get('ymax', 0) + margin
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            # Store original bounds for zoom to fit
            self.original_bounds['A'] = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (µm)')
        self.ax.set_ylabel('Y (µm)')
        
        # Add info text
        info = f"Polygons: {len(polygons)}, Circles: {len(circles)}"
        self.ax.text(0.02, 0.98, info, transform=self.ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5), fontsize=10)
    
    def _draw_step_b(self):
        """Draw Step B: Selected channel regions."""
        self.ax.cla()  # Faster than clear() - doesn't reset all properties
        self.ax.set_title('Step B: Channel Region Selection', fontsize=14, fontweight='bold')
        
        data = self.data.get('B', {})
        selected_polygons = data.get('selected_polygons', [])
        
        # Draw selected polygons with highlight, including holes as voids
        for poly_data in selected_polygons:
            coords_list = poly_data['polygon']['coordinates']
            if not coords_list:
                continue
            
            # Build compound path with exterior and holes
            path_vertices = []
            path_codes = []
            
            # Process each ring (exterior is ring 0, holes are rings 1..N)
            for ring_coords in coords_list:
                if len(ring_coords) < 3:
                    continue
                
                # Convert to numpy array for easier manipulation
                ring_array = np.array(ring_coords)
                
                # Add vertices: all ring points
                path_vertices.extend(ring_array)
                
                # Add codes: MOVETO for first point, LINETO for rest
                path_codes.append(Path.MOVETO)
                path_codes.extend([Path.LINETO] * (len(ring_array) - 1))
                
                # CLOSEPOLY requires a matching vertex (dummy vertex, typically (0,0))
                path_codes.append(Path.CLOSEPOLY)
                path_vertices.append([0.0, 0.0])  # Dummy vertex for CLOSEPOLY
            
            if path_vertices:
                # Create compound path
                compound_path = Path(np.array(path_vertices), path_codes)
                
                # Create PathPatch with styling
                patch = PathPatch(compound_path,
                                facecolor='lightgreen', edgecolor='darkgreen',
                                alpha=0.7, linewidth=2)
                self.ax.add_patch(patch)
        
        # Set bounds from first polygon
        if selected_polygons:
            bounds = selected_polygons[0].get('bounds', {})
            if bounds:
                margin = max(bounds.get('xmax', 0) - bounds.get('xmin', 0),
                            bounds.get('ymax', 0) - bounds.get('ymin', 0)) * 0.1
                xmin = bounds.get('xmin', 0) - margin
                xmax = bounds.get('xmax', 0) + margin
                ymin = bounds.get('ymin', 0) - margin
                ymax = bounds.get('ymax', 0) + margin
                self.ax.set_xlim(xmin, xmax)
                self.ax.set_ylim(ymin, ymax)
                # Store original bounds for zoom to fit
                self.original_bounds['B'] = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (µm)')
        self.ax.set_ylabel('Y (µm)')
        
        info = f"Selected Regions: {len(selected_polygons)}"
        self.ax.text(0.02, 0.98, info, transform=self.ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.5), fontsize=10)
    
    def _draw_step_c(self):
        """Draw Step C: Skeleton (medial axis)."""
        self.ax.cla()  # Faster than clear() - doesn't reset all properties
        
        # Check if step is computed
        if 'C' not in self.computed_steps:
            self.ax.set_title('Step C: Skeleton Extraction (Click Next to compute)', 
                             fontsize=14, fontweight='bold')
            self.ax.text(0.5, 0.5, 'Step C not yet computed.\nClick "Next →" to compute and display.',
                        transform=self.ax.transAxes, ha='center', va='center',
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.fig.canvas.draw_idle()
            return
        
        self.ax.set_title('Step C: Skeleton Extraction (Medial Axis)', 
                         fontsize=14, fontweight='bold')
        
        data = self.data.get('C', {})
        skeleton_graph = data.get('skeleton_graph')
        polygon = data.get('polygon')
        transform = data.get('transform', {})
        
        # Draw the shaded region from Step B (selected polygons)
        step_b_data = self.data.get('B', {})
        selected_polygons = step_b_data.get('selected_polygons', [])
        
        for poly_data in selected_polygons:
            coords_list = poly_data['polygon']['coordinates']
            if not coords_list:
                continue
            
            # Build compound path with exterior and holes
            path_vertices = []
            path_codes = []
            
            # Process each ring (exterior is ring 0, holes are rings 1..N)
            for ring_coords in coords_list:
                if len(ring_coords) < 3:
                    continue
                
                # Convert to numpy array for easier manipulation
                ring_array = np.array(ring_coords)
                
                # Add vertices: all ring points
                path_vertices.extend(ring_array)
                
                # Add codes: MOVETO for first point, LINETO for rest
                path_codes.append(Path.MOVETO)
                path_codes.extend([Path.LINETO] * (len(ring_array) - 1))
                
                # CLOSEPOLY requires a matching vertex (dummy vertex, typically (0,0))
                path_codes.append(Path.CLOSEPOLY)
                path_vertices.append([0.0, 0.0])  # Dummy vertex for CLOSEPOLY
            
            if path_vertices:
                # Create compound path
                compound_path = Path(np.array(path_vertices), path_codes)
                
                # Create PathPatch with styling (slightly different from Step B to show it's the base)
                patch = PathPatch(compound_path,
                                facecolor='lightgreen', edgecolor='darkgreen',
                                alpha=0.5, linewidth=1.5, linestyle='--')
                self.ax.add_patch(patch)
        
        # Draw original polygon outline as well (if different from selected polygons)
        if polygon:
            poly_patch = mpatches.Polygon(list(polygon.exterior.coords),
                                        fill=False, edgecolor='gray',
                                        linewidth=1, linestyle=':', alpha=0.5)
            self.ax.add_patch(poly_patch)
        
        # Draw skeleton graph
        if skeleton_graph and len(skeleton_graph) > 0:
            # Batch edge plotting for performance
            edge_lines_x = []
            edge_lines_y = []
            for u, v in skeleton_graph.edges():
                edge_data = skeleton_graph.edges[u, v]
                polyline = edge_data.get('polyline')
                
                if polyline and len(polyline) >= 2:
                    # Use polyline geometry from vector graph
                    xs = [p[0] for p in polyline]
                    ys = [p[1] for p in polyline]
                    edge_lines_x.append(xs)
                    edge_lines_y.append(ys)
                else:
                    # Fallback: draw straight line between nodes
                    u_xy = skeleton_graph.nodes[u]['xy']
                    v_xy = skeleton_graph.nodes[v]['xy']
                    edge_lines_x.append([u_xy[0], v_xy[0]])
                    edge_lines_y.append([u_xy[1], v_xy[1]])
            
            # Plot all edges at once
            if edge_lines_x:
                for xs, ys in zip(edge_lines_x, edge_lines_y):
                    self.ax.plot(xs, ys, 'b-', linewidth=1.5, alpha=0.7)
            
            # Batch node plotting for performance - compute degrees once
            degrees = dict(skeleton_graph.degree())
            junction_x, junction_y = [], []
            endpoint_x, endpoint_y = [], []
            intermediate_x, intermediate_y = [], []
            
            for node_id in skeleton_graph.nodes():
                xy = skeleton_graph.nodes[node_id]['xy']
                degree = degrees.get(node_id, 0)
                if degree >= 3:
                    junction_x.append(xy[0])
                    junction_y.append(xy[1])
                elif degree == 1:
                    endpoint_x.append(xy[0])
                    endpoint_y.append(xy[1])
                else:
                    intermediate_x.append(xy[0])
                    intermediate_y.append(xy[1])
            
            # Plot all nodes of each type at once
            if junction_x:
                self.ax.plot(junction_x, junction_y, 'ro', markersize=8, 
                           markeredgecolor='darkred', markeredgewidth=1, linestyle='None')
            if endpoint_x:
                self.ax.plot(endpoint_x, endpoint_y, 'go', markersize=6,
                           markeredgecolor='darkgreen', markeredgewidth=1, linestyle='None')
            if intermediate_x:
                self.ax.plot(intermediate_x, intermediate_y, 'bo', markersize=4, 
                           alpha=0.5, linestyle='None')
            
            # Add legend for node colors
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markeredgecolor='darkred', markersize=8, markeredgewidth=1, label='Junction (degree ≥ 3)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markeredgecolor='darkgreen', markersize=6, markeredgewidth=1, label='Endpoint (degree = 1)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=4, alpha=0.5, label='Intermediate (degree = 2)')
            ]
            self.ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
                          framealpha=0.9, edgecolor='black')
        
        # Set bounds
        if polygon:
            bounds = polygon.bounds
            margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
            xmin = bounds[0] - margin
            xmax = bounds[2] + margin
            ymin = bounds[1] - margin
            ymax = bounds[3] + margin
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            # Store original bounds for zoom to fit
            self.original_bounds['C'] = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (µm)')
        self.ax.set_ylabel('Y (µm)')
        
        if skeleton_graph:
            nodes_count = len(skeleton_graph.nodes())
            edges_count = len(skeleton_graph.edges())
            info = f"Skeleton: {nodes_count} nodes, {edges_count} edges"
            self.ax.text(0.02, 0.98, info, transform=self.ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round',
                        facecolor='wheat', alpha=0.5), fontsize=10)
        
        # Use draw_idle for better performance (batches redraws)
        self.fig.canvas.draw_idle()
    
    def _draw_step_c2(self):
        """Draw Step C2: Topology-aware endpoint merging."""
        self.ax.cla()  # Faster than clear() - doesn't reset all properties
        
        # Check if step is computed
        if 'C2' not in self.computed_steps:
            self.ax.set_title('Step C2: Endpoint Merging (Click Next to compute)', 
                             fontsize=14, fontweight='bold')
            self.ax.text(0.5, 0.5, 'Step C2 not yet computed.\nClick "Next →" to compute and display.',
                        transform=self.ax.transAxes, ha='center', va='center',
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.fig.canvas.draw_idle()
            return
        
        self.ax.set_title('Step C2: Topology-Aware Endpoint Merging', 
                         fontsize=14, fontweight='bold')
        
        data = self.data.get('C2', {})
        skeleton_graph = data.get('skeleton_graph')
        junction_clusters = data.get('junction_clusters', {})
        endpoint_clusters = data.get('endpoint_clusters', {})
        forbidden_nodes = data.get('forbidden_nodes', set())
        polygon = data.get('polygon')
        
        # Draw the shaded region from Step B (selected polygons)
        step_b_data = self.data.get('B', {})
        selected_polygons = step_b_data.get('selected_polygons', [])
        
        for poly_data in selected_polygons:
            coords_list = poly_data['polygon']['coordinates']
            if not coords_list:
                continue
                
            # Build compound path with exterior and holes
            path_vertices = []
            path_codes = []
            
            # Process each ring (exterior is ring 0, holes are rings 1..N)
            for ring_coords in coords_list:
                if len(ring_coords) < 3:
                    continue
                
                # Convert to numpy array for easier manipulation
                ring_array = np.array(ring_coords)
                
                # Add vertices: all ring points
                path_vertices.extend(ring_array)
                
                # Add codes: MOVETO for first point, LINETO for rest
                path_codes.append(Path.MOVETO)
                path_codes.extend([Path.LINETO] * (len(ring_array) - 1))
                
                # CLOSEPOLY requires a matching vertex (dummy vertex, typically (0,0))
                path_codes.append(Path.CLOSEPOLY)
                path_vertices.append([0.0, 0.0])  # Dummy vertex for CLOSEPOLY
            
            if path_vertices:
                # Create compound path
                compound_path = Path(np.array(path_vertices), path_codes)
                
                # Create PathPatch with styling
                patch = PathPatch(compound_path,
                                facecolor='lightgreen', edgecolor='darkgreen',
                                alpha=0.5, linewidth=1.5, linestyle='--')
                self.ax.add_patch(patch)
        
        # Draw skeleton graph edges
        if skeleton_graph and len(skeleton_graph) > 0:
            # Batch edge plotting for performance
            edge_lines_x = []
            edge_lines_y = []
            for u, v in skeleton_graph.edges():
                edge_data = skeleton_graph.edges[u, v]
                polyline = edge_data.get('polyline')
                
                if polyline and len(polyline) >= 2:
                    # Use polyline geometry from vector graph
                    xs = [p[0] for p in polyline]
                    ys = [p[1] for p in polyline]
                    edge_lines_x.append(xs)
                    edge_lines_y.append(ys)
                else:
                    # Fallback: draw straight line between nodes
                    u_xy = skeleton_graph.nodes[u]['xy']
                    v_xy = skeleton_graph.nodes[v]['xy']
                    edge_lines_x.append([u_xy[0], v_xy[0]])
                    edge_lines_y.append([u_xy[1], v_xy[1]])
            
            # Plot all edges at once
            if edge_lines_x:
                for xs, ys in zip(edge_lines_x, edge_lines_y):
                    self.ax.plot(xs, ys, 'b-', linewidth=1.5, alpha=0.7)
            
            # Batch node plotting for performance
            degrees = dict(skeleton_graph.degree())
            
            # Collect junction nodes
            junction_x, junction_y = [], []
            for cluster_nodes in junction_clusters.values():
                for node_id in cluster_nodes:
                    if node_id in skeleton_graph:
                        xy = skeleton_graph.nodes[node_id]['xy']
                        junction_x.append(xy[0])
                        junction_y.append(xy[1])
            
            # Collect endpoint nodes (including merged clusters)
            endpoint_x, endpoint_y = [], []
            endpoint_centroid_x, endpoint_centroid_y = [], []
            endpoint_single_x, endpoint_single_y = [], []
            
            for cluster_nodes in endpoint_clusters.values():
                if len(cluster_nodes) == 1:
                    node_id = cluster_nodes[0]
                    if node_id in skeleton_graph:
                        xy = skeleton_graph.nodes[node_id]['xy']
                        endpoint_single_x.append(xy[0])
                        endpoint_single_y.append(xy[1])
                else:
                    # Collect all nodes in cluster for centroid
                    cluster_xy = []
                    for node_id in cluster_nodes:
                        if node_id in skeleton_graph:
                            xy = skeleton_graph.nodes[node_id]['xy']
                            cluster_xy.append(xy)
                            endpoint_x.append(xy[0])
                            endpoint_y.append(xy[1])
                    # Compute centroid
                    if cluster_xy:
                        centroid_x = sum(c[0] for c in cluster_xy) / len(cluster_xy)
                        centroid_y = sum(c[1] for c in cluster_xy) / len(cluster_xy)
                        endpoint_centroid_x.append(centroid_x)
                        endpoint_centroid_y.append(centroid_y)
            
            # Collect intermediate nodes
            intermediate_x, intermediate_y = [], []
            endpoint_set = set()
            for cluster_nodes in endpoint_clusters.values():
                endpoint_set.update(cluster_nodes)
            junction_set = set()
            for cluster_nodes in junction_clusters.values():
                junction_set.update(cluster_nodes)
            
            for node_id in skeleton_graph.nodes():
                if node_id in forbidden_nodes or node_id in endpoint_set or node_id in junction_set:
                    continue
                degree = degrees.get(node_id, 0)
                if degree == 2:
                    xy = skeleton_graph.nodes[node_id]['xy']
                    intermediate_x.append(xy[0])
                    intermediate_y.append(xy[1])
            
            # Plot all nodes of each type at once
            if junction_x:
                self.ax.plot(junction_x, junction_y, 'ro', markersize=8, 
                           markeredgecolor='darkred', markeredgewidth=1, linestyle='None')
            if endpoint_x:
                self.ax.plot(endpoint_x, endpoint_y, 'go', markersize=6,
                           markeredgecolor='darkgreen', markeredgewidth=1, alpha=0.6, linestyle='None')
            if endpoint_single_x:
                self.ax.plot(endpoint_single_x, endpoint_single_y, 'go', markersize=6,
                           markeredgecolor='darkgreen', markeredgewidth=1, linestyle='None')
            if endpoint_centroid_x:
                self.ax.plot(endpoint_centroid_x, endpoint_centroid_y, 'go', markersize=10,
                           markeredgecolor='darkgreen', markeredgewidth=2, linestyle='None')
            if intermediate_x:
                self.ax.plot(intermediate_x, intermediate_y, 'bo', markersize=4, 
                           alpha=0.5, linestyle='None')
            
            # Add legend for node colors
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markeredgecolor='darkred', markersize=8, markeredgewidth=1, label='Junction (degree ≥ 3)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markeredgecolor='darkgreen', markersize=6, markeredgewidth=1, label='Endpoint (degree = 1)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=4, alpha=0.5, label='Intermediate (degree = 2)')
            ]
            self.ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
                          framealpha=0.9, edgecolor='black')
        
        # Set bounds
        if polygon:
            bounds = polygon.bounds
            margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
            xmin = bounds[0] - margin
            xmax = bounds[2] + margin
            ymin = bounds[1] - margin
            ymax = bounds[3] + margin
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            # Store original bounds for zoom to fit
            self.original_bounds['C2'] = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (µm)')
        self.ax.set_ylabel('Y (µm)')
        
        if skeleton_graph:
            junction_count = sum(len(v) for v in junction_clusters.values())
            endpoint_count = sum(len(v) for v in endpoint_clusters.values())
            info = f"Junctions: {len(junction_clusters)} clusters ({junction_count} pixels), Endpoints: {len(endpoint_clusters)} clusters ({endpoint_count} pixels)"
            self.ax.text(0.02, 0.98, info, transform=self.ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round',
                        facecolor='wheat', alpha=0.5), fontsize=10)
        
        # Use draw_idle for better performance (batches redraws)
        self.fig.canvas.draw_idle()
    
    def _draw_step_d(self):
        """Draw Step D: Final graph with nodes and edges."""
        self.ax.cla()  # Faster than clear() - doesn't reset all properties
        
        # Check if step is computed
        if 'D' not in self.computed_steps:
            self.ax.set_title('Step D: Final Graph (Click Next to compute)', 
                             fontsize=14, fontweight='bold')
            self.ax.text(0.5, 0.5, 'Step D not yet computed.\nClick "Next →" to compute and display.',
                        transform=self.ax.transAxes, ha='center', va='center',
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.fig.canvas.draw_idle()
            return
        
        self.ax.set_title('Step D: Final Graph (Nodes & Edges)', 
                         fontsize=14, fontweight='bold')
        
        data = self.data.get('D', {})
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        polygon = data.get('polygon')
        
        # Draw polygon outline (faint)
        if polygon:
            poly_patch = mpatches.Polygon(list(polygon.exterior.coords),
                                        fill=False, edgecolor='lightgray',
                                        linewidth=0.5, alpha=0.3)
            self.ax.add_patch(poly_patch)
        
        # Draw edges
        for edge in edges:
            centerline = edge.get('centerline', {})
            coords = centerline.get('coordinates', [])
            if len(coords) >= 2:
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                self.ax.plot(xs, ys, 'b-', linewidth=2, alpha=0.7)
        
        # Draw nodes with labels
        for node in nodes:
            xy = node.get('xy', [0, 0])
            kind = node.get('kind', 'unknown')
            node_id = node.get('id', '?')
            
            if kind == 'junction':
                color = 'red'
                size = 10
            elif kind == 'endpoint':
                color = 'green'
                size = 8
            else:
                color = 'blue'
                size = 6
            
            self.ax.plot(xy[0], xy[1], 'o', color=color, markersize=size,
                        markeredgecolor='black', markeredgewidth=1)
            
            # Add label
            self.ax.text(xy[0], xy[1], f" {node_id}", fontsize=9,
                        verticalalignment='bottom', fontweight='bold')
        
        # Set bounds
        if polygon:
            bounds = polygon.bounds
            margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
            xmin = bounds[0] - margin
            xmax = bounds[2] + margin
            ymin = bounds[1] - margin
            ymax = bounds[3] + margin
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            # Store original bounds for zoom to fit
            self.original_bounds['D'] = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        elif nodes:
            # Use node bounds
            xs = [n.get('xy', [0, 0])[0] for n in nodes]
            ys = [n.get('xy', [0, 0])[1] for n in nodes]
            if xs and ys:
                margin_x = (max(xs) - min(xs)) * 0.1 if max(xs) > min(xs) else 1
                margin_y = (max(ys) - min(ys)) * 0.1 if max(ys) > min(ys) else 1
                xmin = min(xs) - margin_x
                xmax = max(xs) + margin_x
                ymin = min(ys) - margin_y
                ymax = max(ys) + margin_y
                self.ax.set_xlim(xmin, xmax)
                self.ax.set_ylim(ymin, ymax)
                # Store original bounds for zoom to fit
                self.original_bounds['D'] = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (µm)')
        self.ax.set_ylabel('Y (µm)')
        
        info = f"Nodes: {len(nodes)}, Edges: {len(edges)}"
        self.ax.text(0.02, 0.98, info, transform=self.ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.5), fontsize=10)
    
    def _update_display(self):
        """Update the display based on current step."""
        # Disable rectangle selector if active
        if self.rect_selector is not None:
            self.rect_selector.set_active(False)
        
        step_names = ['A', 'B', 'C', 'C2', 'D']
        step_name = step_names[self.current_step]
        
        # Check if step is enabled
        if not self.enabled_steps.get(step_name, False):
            # Show disabled message
            self.ax.cla()  # Faster than clear() - doesn't reset all properties
            self.ax.set_title(f'Step {step_name}: Disabled', fontsize=14, fontweight='bold')
            self.ax.text(0.5, 0.5, f'Step {step_name} is disabled.',
                        transform=self.ax.transAxes, ha='center', va='center',
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.fig.canvas.draw_idle()
            return
        
        if step_name == 'A':
            self._draw_step_a()
        elif step_name == 'B':
            self._draw_step_b()
        elif step_name == 'C':
            self._draw_step_c()
        elif step_name == 'C2':
            self._draw_step_c2()
        elif step_name == 'D':
            self._draw_step_d()
        
        # Use draw_idle for better performance (batches redraws)
        self.fig.canvas.draw_idle()
    
    def _prev_step(self, event):
        """Navigate to previous enabled step."""
        step_names = ['A', 'B', 'C', 'C2', 'D']
        # Find previous enabled step
        for i in range(self.current_step - 1, -1, -1):
            if self.enabled_steps.get(step_names[i], False):
                self.current_step = i
                self._update_display()
                return
    
    def _next_step(self, event):
        """Navigate to next enabled step, computing it if necessary."""
        step_names = ['A', 'B', 'C', 'C2', 'D']
        # Find next enabled step
        for i in range(self.current_step + 1, 5):
            step_name = step_names[i]
            if not self.enabled_steps.get(step_name, False):
                continue
            
            # Compute step if not already computed and enabled
            if step_name not in self.computed_steps:
                if step_name == 'C' and self.enabled_steps.get('C', False):
                    self._compute_step_c()
                elif step_name == 'C2' and self.enabled_steps.get('C2', False):
                    # C2 is computed as part of step D
                    if 'D' not in self.computed_steps:
                        self._compute_step_d()
                elif step_name == 'D' and self.enabled_steps.get('D', False):
                    self._compute_step_d()
            
            self.current_step = i
            self._update_display()
            return
    
    def _compute_step_c(self):
        """Compute Step C: Skeleton extraction."""
        if 'C' in self.computed_steps:
            return
        
        if not self.enabled_steps.get('C', False):
            logger.warning("Step C is disabled, cannot compute")
            return
        
        logger.info("Computing Step C: Skeleton extraction...")
        step_c_start = time.time()
        try:
            # Get parameters needed for computation
            selected_polygons = self.data.get('B', {}).get('selected_polygons', [])
            if not selected_polygons:
                logger.warning("No selected polygons available for Step C")
                return
            
            minimum_channel_width = self.compute_params.get('minimum_channel_width')
            um_per_px = self.compute_params.get('um_per_px')
            simplify_tolerance = self.compute_params.get('simplify_tolerance')
            
            if minimum_channel_width is None or um_per_px is None:
                logger.error("Missing required parameters for Step C computation")
                return
            
            # Import here to avoid circular imports
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
            from shapely.geometry import LineString
            
            # Convert polygons to Shapely, preserving holes
            t0 = time.time()
            shapely_polygons = []
            for poly_data in selected_polygons:
                rings = poly_data['polygon']['coordinates']
                if not rings or len(rings[0]) < 3:
                    continue
                
                # First ring is exterior, remaining rings are holes
                exterior = rings[0]
                holes = rings[1:] if len(rings) > 1 else []
                
                try:
                    poly = Polygon(exterior, holes=holes)
                    if poly.is_valid and poly.area > 0:
                        shapely_polygons.append(poly)
                except Exception as e:
                    logger.warning("Failed to create polygon with holes: %s", e)
                    # Fallback: try without holes
                    try:
                        poly = Polygon(exterior)
                        if poly.is_valid and poly.area > 0:
                            shapely_polygons.append(poly)
                    except Exception as e2:
                        logger.warning("Failed to create polygon even without holes: %s", e2)
                        continue
            t1 = time.time()
            print(f"  [Step C] Convert polygons to Shapely: {t1 - t0:.3f}s")
            
            if not shapely_polygons:
                logger.warning("No valid polygons for Step C")
                return
            
            # Union polygons
            t0 = time.time()
            if len(shapely_polygons) == 1:
                combined_poly = shapely_polygons[0]
            else:
                combined_poly = unary_union(shapely_polygons)
                if combined_poly.geom_type == 'MultiPolygon':
                    combined_poly = max(combined_poly.geoms, key=lambda g: g.area)
            t1 = time.time()
            print(f"  [Step C] Union polygons: {t1 - t0:.3f}s")
            
            # Extract raw pixel skeleton graph
            from .skeleton import skeletonize_polygon, extract_skeleton_paths
            
            # Get raw pixel skeleton graph
            t0 = time.time()
            skeleton_graph_raw, transform = skeletonize_polygon(
                combined_poly,
                um_per_px=um_per_px,
                L_spur_cutoff=minimum_channel_width,
                simplify_tolerance=None  # Don't simplify yet, we'll do it in vector conversion
            )
            t1 = time.time()
            print(f"  [Step C] Extract raw pixel skeleton graph: {t1 - t0:.3f}s")
            
            if len(skeleton_graph_raw) == 0:
                logger.warning("Empty skeleton graph from skeletonize_polygon")
                self.data['C'] = {
                    'skeleton_graph': nx.Graph(),
                    'skeleton_graph_raw': skeleton_graph_raw,
                    'transform': transform,
                    'polygon': combined_poly
                }
                self.computed_steps.add('C')
                return
            
            # Convert raw pixel graph to vector centerline graph
            logger.info("Converting raw pixel skeleton to vector centerline graph...")
            
            # Compute simplify_tolerance if not provided
            if simplify_tolerance is None:
                simplify_tolerance = 0.5 * um_per_px
            
            # Step 1: Extract skeleton paths
            t0 = time.time()
            original_paths = extract_skeleton_paths(skeleton_graph_raw, simplify_tolerance=None)
            t1 = time.time()
            print(f"  [Step C] Extract skeleton paths: {t1 - t0:.3f}s")
            logger.info("Extracted %d skeleton paths", len(original_paths))
            
            if not original_paths:
                logger.warning("No paths extracted from skeleton graph")
                self.data['C'] = {
                    'skeleton_graph': nx.Graph(),
                    'skeleton_graph_raw': skeleton_graph_raw,
                    'transform': transform,
                    'polygon': combined_poly
                }
                self.computed_steps.add('C')
                return
            
            # Step 2: Simplify each path and resample at fixed spacing
            t0 = time.time()
            resample_spacing_um = 1.5 * um_per_px  # 1-2 * um_per_px as specified
            
            vector_graph = nx.Graph()
            node_counter = itertools.count()
            node_id_map = {}  # Map (x, y) -> node_id
            
            for path_idx, path in enumerate(original_paths):
                if len(path) < 2:
                    continue
                
                # Simplify path
                try:
                    line = LineString(path)
                    simplified = line.simplify(simplify_tolerance, preserve_topology=False)
                    if simplified.geom_type == 'LineString':
                        simplified_coords = list(simplified.coords)
                    elif simplified.geom_type == 'MultiLineString':
                        # Take longest segment
                        longest = max(simplified.geoms, key=lambda g: g.length)
                        simplified_coords = list(longest.coords)
                    else:
                        simplified_coords = path
                except Exception as e:
                    logger.warning("Failed to simplify path %d: %s", path_idx, e)
                    simplified_coords = path
                
                # Resample at fixed spacing
                if len(simplified_coords) < 2:
                    continue
                
                line = LineString(simplified_coords)
                length = line.length
                
                if length < resample_spacing_um:
                    # Path too short, keep original points
                    resampled_coords = simplified_coords
                else:
                    # Resample at fixed spacing
                    num_samples = max(2, int(np.ceil(length / resample_spacing_um)) + 1)
                    distances = np.linspace(0, length, num_samples)
                    resampled_coords = []
                    for dist in distances:
                        point = line.interpolate(dist)
                        resampled_coords.append((point.x, point.y))
                
                # Build graph with resampled points
                prev_node_id = None
                for coord in resampled_coords:
                    # Check if node already exists (for path intersections)
                    node_id = node_id_map.get(coord)
                    if node_id is None:
                        node_id = next(node_counter)
                        vector_graph.add_node(node_id, xy=coord)
                        node_id_map[coord] = node_id
                    
                    # Add edge to previous node in path
                    if prev_node_id is not None:
                        if not vector_graph.has_edge(prev_node_id, node_id):
                            vector_graph.add_edge(prev_node_id, node_id)
                    
                    prev_node_id = node_id
            t1 = time.time()
            print(f"  [Step C] Simplify and resample paths, build vector graph: {t1 - t0:.3f}s")
            
            logger.info("Converted to vector graph: %d nodes (from %d pixel nodes)",
                       len(vector_graph.nodes()), len(skeleton_graph_raw.nodes()))
            
            # Store Step C data with both raw and vector graphs
            self.data['C'] = {
                'skeleton_graph': vector_graph,  # Use vector graph for visualization
                'skeleton_graph_raw': skeleton_graph_raw,  # Keep raw for reference
                'transform': transform,
                'polygon': combined_poly
            }
            self.computed_steps.add('C')
            step_c_end = time.time()
            print(f"  [Step C] Total time: {step_c_end - step_c_start:.3f}s")
            logger.info("Step C computed: vector skeleton graph with %d nodes (from %d pixel nodes)",
                       len(vector_graph.nodes()), len(skeleton_graph_raw.nodes()))
            
        except Exception as e:
            logger.error("Failed to compute Step C: %s", e, exc_info=True)
    
    def _compute_step_d(self):
        """Compute Step D: Final graph extraction."""
        if 'D' in self.computed_steps:
            return
        
        if not self.enabled_steps.get('D', False):
            logger.warning("Step D is disabled, cannot compute")
            return
        
        logger.info("Computing Step D: Final graph extraction...")
        step_d_start = time.time()
        try:
            # Ensure Step C is computed first
            t0 = time.time()
            if 'C' not in self.computed_steps:
                self._compute_step_c()
            t1 = time.time()
            if t1 - t0 > 0.001:  # Only print if it took meaningful time
                print(f"  [Step D] Ensure Step C computed: {t1 - t0:.3f}s")
            
            step_c_data = self.data.get('C', {})
            skeleton_graph = step_c_data.get('skeleton_graph')
            polygon = step_c_data.get('polygon')
            
            if skeleton_graph is None or polygon is None:
                logger.warning("Step C data not available for Step D")
                return
            
            # Get parameters
            selected_polygons = self.data.get('B', {}).get('selected_polygons', [])
            minimum_channel_width = self.compute_params.get('minimum_channel_width')
            um_per_px = self.compute_params.get('um_per_px')
            simplify_tolerance = self.compute_params.get('simplify_tolerance')
            
            # Import here to avoid circular imports
            from .graph_extract import extract_graph_from_polygon
            
            # Extract full graph
            t0 = time.time()
            graph_result = extract_graph_from_polygon(
                polygon,
                minimum_channel_width=minimum_channel_width,
                um_per_px=um_per_px,
                simplify_tolerance=simplify_tolerance,
                width_sample_step=self.compute_params.get('width_sample_step', 10.0),
                measure_edges=self.compute_params.get('measure_edges', True),
                default_height=self.compute_params.get('default_height', 50.0),
                default_cross_section_kind=self.compute_params.get('default_cross_section_kind', 'rectangular'),
                per_edge_overrides=self.compute_params.get('per_edge_overrides'),
                simplify_tolerance_factor=self.compute_params.get('simplify_tolerance_factor', 0.5),
                endpoint_merge_distance_factor=self.compute_params.get('endpoint_merge_distance_factor', 1.0),
                e_Ramer_Douglas_Peucker=self.compute_params.get('e_Ramer_Douglas_Peucker', 10.0)
            )
            t1 = time.time()
            print(f"  [Step D] Extract full graph: {t1 - t0:.3f}s")
            
            # Store Step C2 data (topology-aware endpoint merging)
            c2_data = graph_result.get('c2_data')
            if c2_data:
                self.set_step_c2(
                    c2_data['skeleton_graph'],
                    c2_data['junction_clusters'],
                    c2_data['endpoint_clusters'],
                    c2_data['forbidden_nodes'],
                    c2_data['polygon']
                )
                self.computed_steps.add('C2')
            
            # Store Step D data
            self.data['D'] = {
                'nodes': graph_result.get('nodes', []),
                'edges': graph_result.get('edges', []),
                'polygon': polygon
            }
            self.computed_steps.add('D')
            step_d_end = time.time()
            print(f"  [Step D] Total time: {step_d_end - step_d_start:.3f}s")
            logger.info("Step D computed: %d nodes, %d edges", 
                       len(graph_result.get('nodes', [])), 
                       len(graph_result.get('edges', [])))
            
        except Exception as e:
            logger.error("Failed to compute Step D: %s", e, exc_info=True)
    
    def set_compute_params(self, **params):
        """Set parameters needed to compute steps C and D."""
        self.compute_params.update(params)
    
    def _zoom_in(self, event):
        """Zoom in by a factor of 1.5."""
        if self.ax is None:
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xcenter = (xlim[0] + xlim[1]) / 2
        ycenter = (ylim[0] + ylim[1]) / 2
        xrange = (xlim[1] - xlim[0]) / 1.5
        yrange = (ylim[1] - ylim[0]) / 1.5
        self.ax.set_xlim(xcenter - xrange/2, xcenter + xrange/2)
        self.ax.set_ylim(ycenter - yrange/2, ycenter + yrange/2)
        self.fig.canvas.draw()
    
    def _zoom_out(self, event):
        """Zoom out by a factor of 1.5."""
        if self.ax is None:
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xcenter = (xlim[0] + xlim[1]) / 2
        ycenter = (ylim[0] + ylim[1]) / 2
        xrange = (xlim[1] - xlim[0]) * 1.5
        yrange = (ylim[1] - ylim[0]) * 1.5
        self.ax.set_xlim(xcenter - xrange/2, xcenter + xrange/2)
        self.ax.set_ylim(ycenter - yrange/2, ycenter + yrange/2)
        self.fig.canvas.draw()
    
    def _zoom_to_fit(self, event):
        """Zoom to fit the original bounds of the current step."""
        if self.ax is None:
            return
        step_names = ['A', 'B', 'C', 'C2', 'D']
        step_name = step_names[self.current_step]
        bounds = self.original_bounds.get(step_name)
        if bounds:
            margin = max(bounds['xmax'] - bounds['xmin'],
                        bounds['ymax'] - bounds['ymin']) * 0.1
            self.ax.set_xlim(bounds['xmin'] - margin, bounds['xmax'] + margin)
            self.ax.set_ylim(bounds['ymin'] - margin, bounds['ymax'] + margin)
            self.fig.canvas.draw()
    
    def _zoom_to_area(self, eclick, erelease):
        """Zoom to the selected rectangle area."""
        if self.ax is None:
            return
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        # Add small margin
        xmargin = (xmax - xmin) * 0.05
        ymargin = (ymax - ymin) * 0.05
        self.ax.set_xlim(xmin - xmargin, xmax + xmargin)
        self.ax.set_ylim(ymin - ymargin, ymax + ymargin)
        # Disable selector after zooming
        if self.rect_selector is not None:
            self.rect_selector.set_active(False)
        self.fig.canvas.draw()
    
    def _enable_zoom_to_area(self, event):
        """Enable rectangle selection for zoom to area."""
        if self.ax is None:
            return
        # Disable previous selector if any
        if self.rect_selector is not None:
            self.rect_selector.set_active(False)
        # Create new rectangle selector (non-interactive, zooms on release)
        self.rect_selector = RectangleSelector(
            self.ax, self._zoom_to_area,
            useblit=True, button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=False
        )
        self.fig.canvas.draw()
    
    def show(self, block: bool = True):
        """Show the interactive visualization window."""
        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Create navigation buttons
        ax_prev = plt.axes([0.1, 0.02, 0.12, 0.04])
        ax_next = plt.axes([0.78, 0.02, 0.12, 0.04])
        
        btn_prev = Button(ax_prev, '← Previous')
        btn_next = Button(ax_next, 'Next →')
        
        btn_prev.on_clicked(self._prev_step)
        btn_next.on_clicked(self._next_step)
        
        # Create zoom buttons
        ax_zoom_in = plt.axes([0.24, 0.02, 0.08, 0.04])
        ax_zoom_out = plt.axes([0.34, 0.02, 0.08, 0.04])
        ax_zoom_fit = plt.axes([0.44, 0.02, 0.08, 0.04])
        ax_zoom_area = plt.axes([0.54, 0.02, 0.10, 0.04])
        
        btn_zoom_in = Button(ax_zoom_in, 'Zoom In')
        btn_zoom_out = Button(ax_zoom_out, 'Zoom Out')
        btn_zoom_fit = Button(ax_zoom_fit, 'Zoom Fit')
        btn_zoom_area = Button(ax_zoom_area, 'Zoom Area')
        
        btn_zoom_in.on_clicked(self._zoom_in)
        btn_zoom_out.on_clicked(self._zoom_out)
        btn_zoom_fit.on_clicked(self._zoom_to_fit)
        btn_zoom_area.on_clicked(self._enable_zoom_to_area)
        
        # Find first enabled step
        step_names = ['A', 'B', 'C', 'C2', 'D']
        for i, step_name in enumerate(step_names):
            if self.enabled_steps.get(step_name, False):
                self.current_step = i
                break
        
        # Initial display
        self._update_display()
        
        # Show window
        try:
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space for buttons
        except Exception:
            pass  # Ignore layout warnings
        if block:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(0.1)
    
    def close(self):
        """Close the visualization window."""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def open_manual_centerline_mode(self, save_path: Optional[str] = None):
        """
        Open Manual Centerline Mode for the selected polygon.
        
        Args:
            save_path: Path to save/load centerlines JSON. If None, will be auto-generated
                      from DXF filename as "#FILENAME_manual_centerlines.json"
        
        Returns:
            ManualCenterlineMode instance
        """
        try:
            from .manual_centerline_mode import ManualCenterlineMode
        except ImportError:
            logger.error("Manual centerline mode not available")
            return None
        
        # Get polygon from step B (selected channels) or step C
        polygon = None
        
        if 'B' in self.data and self.data['B'].get('selected_polygons'):
            # Use first selected polygon
            from shapely.geometry import Polygon as ShapelyPolygon
            poly_data = self.data['B']['selected_polygons'][0]
            coords = poly_data['polygon']['coordinates'][0]
            polygon = ShapelyPolygon(coords)
        elif 'C' in self.data and self.data['C'].get('polygon'):
            polygon = self.data['C']['polygon']
        
        if polygon is None:
            logger.error("No polygon available for manual centerline mode")
            return None
        
        # Try to get DXF result if available
        dxf_result = None
        if 'A' in self.data:
            dxf_result = self.data['A']
        
        # save_path will be auto-generated from DXF filename if None
        mode = ManualCenterlineMode(polygon, save_path=save_path, dxf_result=dxf_result)
        return mode

