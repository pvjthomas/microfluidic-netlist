"""Export debug images for intermediate steps C and D."""

from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Polygon, Point, LineString
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import networkx as nx
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_bounds_from_geometry(
    polygons: Optional[List[Dict[str, Any]]] = None,
    polygon: Optional[Polygon] = None,  # Shapely polygon
    skeleton_graph: Optional[nx.Graph] = None,
    nodes: Optional[List[Dict[str, Any]]] = None,
    edges: Optional[List[Dict[str, Any]]] = None,
    padding: float = 1000.0  # Increased padding for debug images (was 100.0)
) -> Tuple[float, float, float, float]:
    """
    Compute bounding box for all geometry with padding.
    
    Returns:
        Tuple of (xmin, ymin, xmax, ymax)
    """
    all_x = []
    all_y = []
    
    # Collect points from Shapely polygon
    if polygon is not None:
        # Get exterior ring
        exterior_coords = list(polygon.exterior.coords)
        for coord in exterior_coords:
            all_x.append(coord[0])
            all_y.append(coord[1])
        # Get interior rings (holes)
        for interior in polygon.interiors:
            for coord in interior.coords:
                all_x.append(coord[0])
                all_y.append(coord[1])
    
    # Collect points from polygons (dict format)
    if polygons:
        for poly_data in polygons:
            coords = poly_data['polygon']['coordinates'][0]
            for coord in coords:
                all_x.append(coord[0])
                all_y.append(coord[1])
    
    # Collect points from skeleton graph
    if skeleton_graph:
        for node_id in skeleton_graph.nodes():
            xy = skeleton_graph.nodes[node_id]['xy']
            all_x.append(xy[0])
            all_y.append(xy[1])
    
    # Collect points from nodes
    if nodes:
        for node in nodes:
            all_x.append(node['xy'][0])
            all_y.append(node['xy'][1])
    
    # Collect points from edge centerlines
    if edges:
        for edge in edges:
            coords = edge['centerline']['coordinates']
            for coord in coords:
                all_x.append(coord[0])
                all_y.append(coord[1])
    
    if not all_x:
        return (0, 0, 1000, 1000)
    
    xmin = min(all_x) - padding
    ymin = min(all_y) - padding
    xmax = max(all_x) + padding
    ymax = max(all_y) + padding
    
    return (xmin, ymin, xmax, ymax)


def world_to_image(
    x: float, y: float,
    xmin: float, ymin: float,
    um_per_px: float
) -> Tuple[int, int]:
    """Convert world coordinates to image pixel coordinates."""
    px = int((x - xmin) / um_per_px)
    py = int((y - ymin) / um_per_px)
    return (px, py)


def clamp_bbox(bbox: List[int], img_width: int, img_height: int, outline_width: int = 0) -> List[int]:
    """Clamp bounding box coordinates to image boundaries, accounting for outline width."""
    x0, y0, x1, y1 = bbox
    # Account for outline width that extends beyond bbox
    margin = outline_width // 2 if outline_width > 0 else 0
    x0 = max(margin, min(x0, img_width - 1 - margin))
    y0 = max(margin, min(y0, img_height - 1 - margin))
    x1 = max(margin, min(x1, img_width - 1 - margin))
    y1 = max(margin, min(y1, img_height - 1 - margin))
    return [x0, y0, x1, y1]


def clamp_coord(x: int, y: int, img_width: int, img_height: int) -> Tuple[int, int]:
    """Clamp coordinates to image boundaries."""
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    return (x, y)


def export_debug_image(
    step_name: str,
    output_path: str,
    polygons: Optional[List[Dict[str, Any]]] = None,
    polygon: Optional[Polygon] = None,  # Shapely polygon
    skeleton_graph: Optional[nx.Graph] = None,
    contracted_graph: Optional[nx.Graph] = None,
    junction_clusters: Optional[Dict[int, List[int]]] = None,
    endpoint_clusters: Optional[Dict[int, List[int]]] = None,
    nodes: Optional[List[Dict[str, Any]]] = None,
    edges: Optional[List[Dict[str, Any]]] = None,
    cluster_rep_map: Optional[Dict[int, int]] = None,
    image_width: int = 2000,
    image_height: Optional[int] = None,
) -> None:
    """
    Export debug image for an intermediate step.
    
    Args:
        step_name: Name of the step (e.g., "C_skeleton", "C2_contracted", "D_nodes", "D_edges")
        output_path: Output file path
        polygons: Optional list of polygon dicts
        polygon: Optional Shapely polygon (with holes)
        skeleton_graph: Optional skeleton graph
        contracted_graph: Optional contracted skeleton graph
        junction_clusters: Optional dict mapping cluster_id to list of node IDs
        endpoint_clusters: Optional dict mapping cluster_id to list of node IDs
        nodes: Optional list of node dicts
        edges: Optional list of edge dicts
        cluster_rep_map: Optional dict mapping original node_id to representative node_id
        image_width: Target image width in pixels
        image_height: Target image height (auto if None)
    """
    logger.info("Exporting debug image for step '%s' to: %s", step_name, output_path)
    
    # Determine which graph to use for bounds
    active_graph = skeleton_graph or contracted_graph
    
    # Compute bounds with extra padding for debug images
    xmin, ymin, xmax, ymax = compute_bounds_from_geometry(
        polygons=polygons,
        polygon=polygon,
        skeleton_graph=active_graph,
        nodes=nodes,
        edges=edges,
        padding=1000.0  # Extra padding for markers and text
    )
    width_um = xmax - xmin
    height_um = ymax - ymin
    
    # Add additional whitespace buffer around the image (increased for better visibility)
    buffer_um = 2000.0  # Increased from 1000.0 to add more whitespace
    width_um += 2 * buffer_um
    height_um += 2 * buffer_um
    xmin -= buffer_um
    ymin -= buffer_um
    
    # Calculate um_per_px to fit desired width (accounting for buffer)
    um_per_px = width_um / image_width
    
    # Calculate image height if not specified (accounting for buffer)
    if image_height is None:
        image_height = int(height_um / um_per_px)
    
    # Ensure minimum image dimensions to accommodate large markers and text
    min_width_for_markers = 400  # Space for 4x markers (~200px radius) + text
    min_height_for_markers = 400
    image_width = max(image_width, min_width_for_markers)
    image_height = max(image_height, min_height_for_markers)
    
    logger.debug("Bounds: (%.1f, %.1f) to (%.1f, %.1f), size: %.1f × %.1f µm",
                xmin, ymin, xmax, ymax, width_um, height_um)
    logger.debug("Image size: %d × %d pixels, um_per_px: %.3f", image_width, image_height, um_per_px)
    
    # Create image
    img = Image.new('RGB', (image_width, image_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Try to load a font (4x larger: 10 -> 40)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
    
    # Draw Shapely polygon with holes (if provided)
    if polygon is not None:
        # Draw exterior ring with fill
        exterior_coords = list(polygon.exterior.coords)
        if len(exterior_coords) >= 3:
            # Exclude last duplicate coordinate (Shapely polygons close the ring)
            img_coords_exterior = [
                clamp_coord(*world_to_image(coord[0], coord[1], xmin, ymin, um_per_px), image_width, image_height)
                for coord in exterior_coords[:-1]
            ]
            # Draw exterior polygon with fill
            draw.polygon(img_coords_exterior, fill=(240, 240, 255), outline=(200, 200, 220))
        
        # Draw interior rings (holes) by filling them with background color to create voids
        for interior in polygon.interiors:
            interior_coords = list(interior.coords)
            if len(interior_coords) >= 3:
                # Exclude last duplicate coordinate
                img_coords_interior = [
                    clamp_coord(*world_to_image(coord[0], coord[1], xmin, ymin, um_per_px), image_width, image_height)
                    for coord in interior_coords[:-1]
                ]
                # Draw hole in white (background color) to create void
                draw.polygon(img_coords_interior, fill=(255, 255, 255), outline=(200, 200, 220))
    
    # Draw polygons (if provided) - dict format
    if polygons:
        for poly_data in polygons:
            coords = poly_data['polygon']['coordinates'][0]
            if len(coords) < 3:
                continue
            
            img_coords = [
                clamp_coord(*world_to_image(coord[0], coord[1], xmin, ymin, um_per_px), image_width, image_height)
                for coord in coords
            ]
            draw.polygon(img_coords, fill=(240, 240, 255), outline=(200, 200, 220))
    
    # Draw skeleton graph edges (light blue)
    graph_to_draw = contracted_graph if contracted_graph else skeleton_graph
    if graph_to_draw:
        for u, v in graph_to_draw.edges():
            u_xy = graph_to_draw.nodes[u]['xy']
            v_xy = graph_to_draw.nodes[v]['xy']
            
            u_px, u_py = clamp_coord(*world_to_image(u_xy[0], u_xy[1], xmin, ymin, um_per_px), image_width, image_height)
            v_px, v_py = clamp_coord(*world_to_image(v_xy[0], v_xy[1], xmin, ymin, um_per_px), image_width, image_height)
            
            draw.line([(u_px, u_py), (v_px, v_py)], fill=(200, 200, 255), width=1)
    
    # Draw junction clusters (if provided) - 4x larger markers
    if junction_clusters and graph_to_draw:
        for cluster_id, cluster_nodes in junction_clusters.items():
            for node_id in cluster_nodes:
                if node_id in graph_to_draw:
                    xy = graph_to_draw.nodes[node_id]['xy']
                    x, y = clamp_coord(*world_to_image(xy[0], xy[1], xmin, ymin, um_per_px), image_width, image_height)
                    radius = max(8, int(12 / um_per_px))  # 4x: 3 -> 12
                    bbox = clamp_bbox([x - radius, y - radius, x + radius, y + radius], image_width, image_height, outline_width=1)
                    # Draw in red for junctions
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid bbox
                        draw.ellipse(bbox, fill=(255, 100, 100), outline=(200, 0, 0))
    
    # Draw endpoint clusters (if provided) - 4x larger markers
    if endpoint_clusters and graph_to_draw:
        for cluster_id, cluster_nodes in endpoint_clusters.items():
            for node_id in cluster_nodes:
                if node_id in graph_to_draw:
                    xy = graph_to_draw.nodes[node_id]['xy']
                    x, y = clamp_coord(*world_to_image(xy[0], xy[1], xmin, ymin, um_per_px), image_width, image_height)
                    radius = max(8, int(12 / um_per_px))  # 4x: 3 -> 12
                    bbox = clamp_bbox([x - radius, y - radius, x + radius, y + radius], image_width, image_height, outline_width=1)
                    # Draw in green for endpoints
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid bbox
                        draw.ellipse(bbox, fill=(100, 255, 100), outline=(0, 200, 0))
    
    # Draw endpoints (degree-1 nodes) from contracted_graph for C2_contracted
    if step_name == "C2_contracted" and contracted_graph:
        endpoint_nodes = [n for n in contracted_graph.nodes() if contracted_graph.degree(n) == 1]
        for node_id in endpoint_nodes:
            xy = contracted_graph.nodes[node_id]['xy']
            x, y = clamp_coord(*world_to_image(xy[0], xy[1], xmin, ymin, um_per_px), image_width, image_height)
            radius = max(12, int(20 / um_per_px))  # Larger radius for visibility
            bbox = clamp_bbox([x - radius, y - radius, x + radius, y + radius], image_width, image_height, outline_width=2)
            # Draw in green for endpoints
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid bbox
                draw.ellipse(bbox, fill=(100, 255, 100), outline=(0, 200, 0), width=2)
    
    # Draw cluster representatives (if cluster_rep_map provided) - 4x larger markers
    if cluster_rep_map and graph_to_draw:
        for original_id, rep_id in cluster_rep_map.items():
            if rep_id in graph_to_draw:
                xy = graph_to_draw.nodes[rep_id]['xy']
                x, y = clamp_coord(*world_to_image(xy[0], xy[1], xmin, ymin, um_per_px), image_width, image_height)
                radius = max(12, int(20 / um_per_px))  # 4x: 5 -> 20
                bbox = clamp_bbox([x - radius, y - radius, x + radius, y + radius], image_width, image_height, outline_width=8)
                # Draw larger circle for representatives
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid bbox
                    draw.ellipse(bbox, fill=(255, 200, 0), outline=(200, 150, 0), width=8)  # 4x: 2 -> 8
    
    # Draw final nodes (if provided) - 4x larger markers and text
    if nodes:
        node_radius = max(12, int(20 / um_per_px))  # 4x: 5 -> 20
        for node in nodes:
            x, y = clamp_coord(*world_to_image(node['xy'][0], node['xy'][1], xmin, ymin, um_per_px), image_width, image_height)
            
            # Color by kind
            if node.get('kind') == 'junction':
                color = (255, 0, 0)  # Red
                outline = (200, 0, 0)
            elif node.get('kind') == 'endpoint':
                color = (0, 255, 0)  # Green
                outline = (0, 200, 0)
            else:
                color = (100, 100, 255)  # Blue
                outline = (0, 0, 200)
            
            bbox = clamp_bbox([x - node_radius, y - node_radius, x + node_radius, y + node_radius], image_width, image_height, outline_width=8)
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid bbox
                draw.ellipse(bbox, fill=color, outline=outline, width=8)  # 4x: 2 -> 8
            
            # Draw node label - 4x offset, clamp to image bounds with margin for text
            # Estimate text size: ~40px font means roughly 40px per character
            label = node.get('id', '?')
            estimated_text_width = len(label) * 40
            text_x = max(0, min(x + node_radius + 8, image_width - estimated_text_width - 1))  # 4x: 2 -> 8
            text_y = max(40, min(y - 20, image_height - 40 - 1))  # 4x: 5 -> 20, margin for font height
            try:
                draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
            except Exception:
                # If text drawing fails, skip it
                pass
    
    # Draw final edges (if provided)
    if edges:
        for edge in edges:
            coords = edge['centerline']['coordinates']
            if len(coords) < 2:
                continue
            
            img_coords = [
                clamp_coord(*world_to_image(coord[0], coord[1], xmin, ymin, um_per_px), image_width, image_height)
                for coord in coords
            ]
            
            # Draw centerline in blue
            draw.line(img_coords, fill=(0, 100, 200), width=2)
            
            # Draw edge label at midpoint - 4x larger text, clamp to image bounds
            if len(coords) >= 2:
                midpoint_idx = len(coords) // 2
                mid_coord = coords[midpoint_idx]
                mx, my = clamp_coord(*world_to_image(mid_coord[0], mid_coord[1], xmin, ymin, um_per_px), image_width, image_height)
                label = edge.get('id', '?')
                # Clamp text position to ensure it's within image with margin for text
                estimated_text_width = len(label) * 40
                text_x = max(0, min(mx, image_width - estimated_text_width - 1))
                text_y = max(40, min(my, image_height - 40 - 1))
                try:
                    draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
                except Exception:
                    # If text drawing fails, skip it
                    pass
    
    # Add title text - 4x larger, ensure within bounds with margin
    title = f"Step {step_name}"
    estimated_title_width = len(title) * 40
    title_x = max(0, min(40, image_width - estimated_title_width - 1))
    title_y = max(0, min(40, image_height - 40 - 1))
    try:
        draw.text((title_x, title_y), title, fill=(0, 0, 0), font=font)  # 4x: 10 -> 40
    except Exception:
        # If title drawing fails, skip it
        pass
    
    # Save image
    img.save(output_path, 'PNG')
    logger.info("Debug image saved: %s (%d × %d pixels)", output_path, image_width, image_height)

