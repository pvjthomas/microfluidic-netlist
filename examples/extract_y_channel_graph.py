#!/usr/bin/env python3
"""Extract a network graph from examples/y_channel_scale.dxf with logging + sanity checks."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, LineString, Point, MultiPoint, GeometryCollection
import numpy as np

# ---------------------------------------------------------------------
# Import setup (dev fallback)
# ---------------------------------------------------------------------
# Preferred: install package via `pip install -e core/`
# Fallback: add ./core to sys.path for local dev runs.
CORE_DIR = Path(__file__).resolve().parent.parent / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from microfluidic_ir.dxf_loader import load_dxf  # noqa: E402
from microfluidic_ir.graph_extract import extract_graph_from_polygons  # noqa: E402
from microfluidic_ir.export_overlay import export_overlay_png  # noqa: E402
from microfluidic_ir.graph_extract import find_boundary_intersections, compute_local_normal  # noqa: E402


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
LOG_FILE = Path(__file__).with_suffix(".log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s: %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info("Logging to: %s", LOG_FILE)


@contextmanager
def log_step(name: str):
    """Log step entry/exit + duration."""
    logger.info("→ %s", name)
    start = time.time()
    try:
        yield
    finally:
        logger.info("← %s (%.2fs)", name, time.time() - start)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def poly_bounds(poly_entry: dict[str, Any]) -> tuple[float, float, float]:
    b = poly_entry["bounds"]
    w = float(b["xmax"] - b["xmin"])
    h = float(b["ymax"] - b["ymin"])
    return w, h, max(w, h)


def polyline_length(coords: list[list[float]]) -> float:
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        total += (dx * dx + dy * dy) ** 0.5
    return total


def compute_revised_skeleton(
    centerline_coords: list[list[float]],
    polygon: Polygon,
    node_u: tuple[float, float],
    node_v: tuple[float, float],
    channel_width: float
) -> list[tuple[float, float]]:
    """
    Compute revised skeleton line that follows midpoint of nearest walls.
    
    Near nodes (within 3x channel width), smoothly transitions to node position.
    
    Args:
        centerline_coords: Original centerline coordinates
        polygon: Polygon to find wall intersections with
        node_u: Start node position (x, y)
        node_v: End node position (x, y)
        channel_width: Channel width for transition distance calculation
        
    Returns:
        Revised skeleton coordinates
    """
    if len(centerline_coords) < 2:
        return [(c[0], c[1]) for c in centerline_coords]
    
    # Convert to tuples for compatibility
    coords = [(c[0], c[1]) for c in centerline_coords]
    
    # Compute transition distance (3x channel width)
    transition_dist = 3.0 * channel_width
    
    # Compute distances along centerline from each node
    centerline = LineString(coords)
    total_length = centerline.length
    
    revised_coords = []
    
    for i, coord in enumerate(coords):
        point = Point(coord)
        
        # Compute distance from this point to each node along the centerline
        # Distance from start
        dist_from_start = centerline.project(point)
        dist_from_end = total_length - dist_from_start
        
        # Check if we're within transition distance of either node
        min_dist_to_node = min(dist_from_start, dist_from_end)
        
        if min_dist_to_node < transition_dist:
            # Smoothly blend between midpoint-based position and node position
            # Use a smooth transition (sigmoid-like)
            if dist_from_start < dist_from_end:
                # Closer to start node
                t = dist_from_start / transition_dist  # 0 at node, 1 at transition_dist
                target_node = Point(node_u[0], node_u[1])
            else:
                # Closer to end node
                t = dist_from_end / transition_dist
                target_node = Point(node_v[0], node_v[1])
            
            # Smooth transition function (smoothstep)
            smooth_t = t * t * (3.0 - 2.0 * t)
            
            # Compute midpoint-based position
            normal = compute_local_normal(coords, i)
            left_pt, right_pt = find_boundary_intersections(point, normal, polygon, search_distance=10000.0)
            
            if left_pt and right_pt:
                midpoint = Point(
                    (left_pt.x + right_pt.x) / 2.0,
                    (left_pt.y + right_pt.y) / 2.0
                )
                # Blend between midpoint and node
                blended_x = midpoint.x * smooth_t + target_node.x * (1.0 - smooth_t)
                blended_y = midpoint.y * smooth_t + target_node.y * (1.0 - smooth_t)
                revised_coords.append((blended_x, blended_y))
            else:
                # Fallback: blend between original point and node
                blended_x = point.x * smooth_t + target_node.x * (1.0 - smooth_t)
                blended_y = point.y * smooth_t + target_node.y * (1.0 - smooth_t)
                revised_coords.append((blended_x, blended_y))
        else:
            # Far from nodes: use midpoint of wall intersections
            normal = compute_local_normal(coords, i)
            left_pt, right_pt = find_boundary_intersections(point, normal, polygon, search_distance=10000.0)
            
            if left_pt and right_pt:
                midpoint = Point(
                    (left_pt.x + right_pt.x) / 2.0,
                    (left_pt.y + right_pt.y) / 2.0
                )
                revised_coords.append((midpoint.x, midpoint.y))
            else:
                # Fallback: keep original point
                revised_coords.append(coord)
    
    return revised_coords


def compute_revised_skeleton_with_junctions(
    centerline_coords: list[list[float]],
    polygon: Polygon,
    node_u: tuple[float, float],
    node_v: tuple[float, float],
    channel_width: float,
    transition_factor_u: float = 3.0,
    transition_factor_v: float = 3.0
) -> list[tuple[float, float]]:
    """
    Compute revised skeleton line that follows midpoint of nearest walls.
    
    Near nodes, smoothly transitions to node position with configurable transition distance.
    
    Args:
        centerline_coords: Original centerline coordinates
        polygon: Polygon to find wall intersections with
        node_u: Start node position (x, y)
        node_v: End node position (x, y)
        channel_width: Channel width for transition distance calculation
        transition_factor_u: Multiplier for transition distance at start node (default: 3.0)
        transition_factor_v: Multiplier for transition distance at end node (default: 3.0)
        
    Returns:
        Revised skeleton coordinates
    """
    if len(centerline_coords) < 2:
        return [(c[0], c[1]) for c in centerline_coords]
    
    # Convert to tuples for compatibility
    coords = [(c[0], c[1]) for c in centerline_coords]
    
    # Compute transition distances (different for each node)
    transition_dist_u = transition_factor_u * channel_width
    transition_dist_v = transition_factor_v * channel_width
    
    # Compute distances along centerline from each node
    centerline = LineString(coords)
    total_length = centerline.length
    
    revised_coords = []
    
    for i, coord in enumerate(coords):
        point = Point(coord)
        
        # Compute distance from this point to each node along the centerline
        dist_from_start = centerline.project(point)
        dist_from_end = total_length - dist_from_start
        
        # Check which node we're closer to and use appropriate transition distance
        if dist_from_start < dist_from_end:
            # Closer to start node
            transition_dist = transition_dist_u
            dist_to_node = dist_from_start
            target_node = Point(node_u[0], node_u[1])
        else:
            # Closer to end node
            transition_dist = transition_dist_v
            dist_to_node = dist_from_end
            target_node = Point(node_v[0], node_v[1])
        
        # Check if we're within transition distance
        if dist_to_node < transition_dist:
            # Smoothly blend between midpoint-based position and node position
            t = dist_to_node / transition_dist  # 0 at node, 1 at transition_dist
            
            # Smooth transition function (smoothstep)
            smooth_t = t * t * (3.0 - 2.0 * t)
            
            # Compute midpoint-based position
            normal = compute_local_normal(coords, i)
            left_pt, right_pt = find_boundary_intersections(point, normal, polygon, search_distance=10000.0)
            
            if left_pt and right_pt:
                midpoint = Point(
                    (left_pt.x + right_pt.x) / 2.0,
                    (left_pt.y + right_pt.y) / 2.0
                )
                # Blend between midpoint and node
                blended_x = midpoint.x * smooth_t + target_node.x * (1.0 - smooth_t)
                blended_y = midpoint.y * smooth_t + target_node.y * (1.0 - smooth_t)
                revised_coords.append((blended_x, blended_y))
            else:
                # Fallback: blend between original point and node
                blended_x = point.x * smooth_t + target_node.x * (1.0 - smooth_t)
                blended_y = point.y * smooth_t + target_node.y * (1.0 - smooth_t)
                revised_coords.append((blended_x, blended_y))
        else:
            # Far from nodes: use midpoint of wall intersections
            normal = compute_local_normal(coords, i)
            left_pt, right_pt = find_boundary_intersections(point, normal, polygon, search_distance=10000.0)
            
            if left_pt and right_pt:
                midpoint = Point(
                    (left_pt.x + right_pt.x) / 2.0,
                    (left_pt.y + right_pt.y) / 2.0
                )
                revised_coords.append((midpoint.x, midpoint.y))
            else:
                # Fallback: keep original point
                revised_coords.append(coord)
    
    return revised_coords


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    logger.info("=" * 70)
    logger.info("Extracting Network Graph from y_channel_scale.dxf")
    logger.info("=" * 70)

    dxf_path = Path(__file__).resolve().parent / "y_channel_scale.dxf"
    if not dxf_path.exists():
        logger.error("DXF file not found: %s", dxf_path)
        return 1

    # 1) Load DXF
    try:
        with log_step("DXF load"):
            result = load_dxf(str(dxf_path), snap_close_tol=2.0)
        logger.info("Loaded: %d polygon(s), %d circle(s)", len(result["polygons"]), len(result["circles"]))
    except Exception:
        logger.exception("Error loading DXF")
        return 1

    if not result["polygons"]:
        logger.error("No polygons found in DXF")
        return 1

    # 2) Pick parameters (largest polygon by area is usually the channel)
    polys = result["polygons"]
    # If your loader provides 'area', use it; otherwise fall back to first polygon.
    polys_sorted = sorted(polys, key=lambda p: float(p.get("area", 0.0)), reverse=True)
    primary = polys_sorted[0]

    w, h, max_dim = poly_bounds(primary)
    logger.info("Polygon size: %.1f × %.1f µm (max: %.1f)", w, h, max_dim)

    # Auto-tune resolution and simplify_tolerance based on channel width
    logger.info("Parameters: um_per_px=auto (will be tuned from channel width), simplify_tolerance=auto (0.5 * um_per_px)")
    logger.info("Cross-section defaults: height=50.0 µm, kind=rectangular")

    # 3) Extract graph with port detection
    circles = result.get("circles", [])
    logger.info("Found %d circle(s) for port detection", len(circles))
    
    try:
        with log_step("Graph extraction"):
            graph_result = extract_graph_from_polygons(
                polys_sorted,
                um_per_px=None,  # Auto-tune based on channel width
                simplify_tolerance=None,  # Auto-computed as 0.5 * um_per_px
                auto_tune_resolution=True,
                circles=circles if circles else None,
                port_snap_distance=50.0,  # Snap ports within 50 µm
                detect_polyline_circles=False,  # Don't detect circle-like polylines for now
            )
    except Exception:
        logger.exception("Error extracting graph")
        return 1

    nodes = graph_result.get("nodes", [])
    edges = graph_result.get("edges", [])
    ports = graph_result.get("ports", [])
    logger.info("Graph: %d nodes, %d edges, %d ports", len(nodes), len(edges), len(ports))

    # Node summary
    kinds: dict[str, int] = {}
    for n in nodes:
        kinds[n.get("kind", "unknown")] = kinds.get(n.get("kind", "unknown"), 0) + 1
    for kind in sorted(kinds):
        logger.info("  %s: %d", kind, kinds[kind])

    # Node details (brief)
    for n in nodes:
        x, y = n.get("xy", [None, None])
        logger.info("  %s: %s at (%.1f, %.1f), degree=%s", n.get("id"), n.get("kind"), x, y, n.get("degree", "N/A"))

    # Edge details (with measurements)
    for e in edges:
        length = e.get("length", 0.0)
        width_profile = e.get("width_profile", {})
        w_kind = width_profile.get("kind", "unknown")
        w_median = width_profile.get("w_median", 0.0)
        w_min = width_profile.get("w_min", 0.0)
        w_max = width_profile.get("w_max", 0.0)
        num_samples = len(width_profile.get("samples", []))
        logger.info(
            "  %s: %s → %s, length=%.1f µm, width: %s (median=%.1f, min=%.1f, max=%.1f µm, %d samples)",
            e["id"], e["u"], e["v"], length, w_kind, w_median, w_min, w_max, num_samples
        )

    # Validate references
    node_ids = {n["id"] for n in nodes}
    bad_edges = [e for e in edges if e["u"] not in node_ids or e["v"] not in node_ids]
    if bad_edges:
        logger.warning("Found %d edges with invalid node references", len(bad_edges))
    else:
        logger.info("All edges connect existing nodes")

    # Check near-zero length edges
    zero = []
    for e in edges:
        length = e.get("length", 0.0)
        if length < 1.0:
            zero.append(e["id"])
    if zero:
        logger.warning("Found %d near-zero-length edges: %s", len(zero), zero)
    else:
        logger.info("No near-zero-length edges")

    # Port details
    if ports:
        logger.info("Ports:")
        for port in ports:
            marker = port.get("marker", {})
            center = marker.get("center", [None, None])
            radius = marker.get("radius", None)
            node_id = port.get("node_id")
            logger.info(
                "  %s: center=(%.1f, %.1f), radius=%.1f µm, attached to %s",
                port.get("port_id"), center[0], center[1], radius, node_id or "none"
            )
    else:
        logger.info("No ports detected")
    
    # Save JSON
    out_path = Path(__file__).resolve().parent / "y_channel_graph.json"
    with open(out_path, "w") as f:
        json.dump({"nodes": nodes, "edges": edges, "ports": ports}, f, indent=2)

    logger.info("Graph saved to: %s", out_path)
    
    # Export overlay image
    try:
        with log_step("Export overlay"):
            overlay_path = Path(__file__).resolve().parent / "y_channel_overlay.png"
            export_overlay_png(
                polygons=polys_sorted,
                nodes=nodes,
                edges=edges,
                ports=ports,
                output_path=str(overlay_path),
                image_width=2000
            )
        logger.info("Overlay saved to: %s", overlay_path)
    except Exception:
        logger.exception("Error exporting overlay")
        # Don't fail the script if overlay export fails
    
    # Export edge crops at full scale
    try:
        with log_step("Export edge crops"):
            crop_output_dir = Path(__file__).resolve().parent / "edge_crops"
            crop_output_dir.mkdir(exist_ok=True)
            
            # Convert polys_sorted to Shapely polygons for containment checks
            shapely_polys = []
            for poly_data in polys_sorted:
                coords = poly_data['polygon']['coordinates'][0]
                poly = Polygon(coords)
                if poly.is_valid:
                    shapely_polys.append((poly, poly_data))
            
            for edge in edges:
                edge_id = edge['id']
                centerline_coords = edge['centerline']['coordinates']
                
                if len(centerline_coords) < 2:
                    logger.warning("Skipping %s: insufficient centerline points", edge_id)
                    continue
                
                # Create LineString from centerline
                centerline = LineString(centerline_coords)
                
                # Find polygon(s) that contain or intersect this edge
                # Check which polygon contains the midpoint of the edge
                midpoint = centerline.interpolate(0.5, normalized=True)
                matching_poly = None
                matching_poly_data = None
                
                for poly, poly_data in shapely_polys:
                    if poly.contains(midpoint) or poly.buffer(1.0).contains(midpoint):
                        matching_poly = poly
                        matching_poly_data = poly_data
                        break
                
                # If no exact match, use the largest polygon (likely the main channel)
                if matching_poly is None and shapely_polys:
                    matching_poly, matching_poly_data = shapely_polys[0]
                    logger.debug("Using largest polygon for %s", edge_id)
                
                if matching_poly is None:
                    logger.warning("No polygon found for %s, skipping", edge_id)
                    continue
                
                # Calculate bounding box with padding
                padding = 50.0  # 50 µm padding
                
                # Get bounds from both centerline and polygon
                centerline_bounds = centerline.bounds  # (minx, miny, maxx, maxy)
                poly_bbox = matching_poly.bounds
                
                xmin = min(centerline_bounds[0], poly_bbox[0]) - padding
                ymin = min(centerline_bounds[1], poly_bbox[1]) - padding
                xmax = max(centerline_bounds[2], poly_bbox[2]) + padding
                ymax = max(centerline_bounds[3], poly_bbox[3]) + padding
                
                width_um = xmax - xmin
                height_um = ymax - ymin
                
                # Full scale: 1 pixel = 1 µm
                um_per_px = 1.0
                img_width = int(width_um / um_per_px)
                img_height = int(height_um / um_per_px)
                
                # Create image
                img = Image.new('RGB', (img_width, img_height), (255, 255, 255))
                draw = ImageDraw.Draw(img)
                
                def world_to_px(x: float, y: float) -> tuple[int, int]:
                    """Convert world coordinates to pixel coordinates."""
                    px = int((x - xmin) / um_per_px)
                    py = int((y - ymin) / um_per_px)
                    return (px, py)
                
                # Draw polygon outline
                poly_coords = matching_poly_data['polygon']['coordinates'][0]
                img_poly_coords = [world_to_px(coord[0], coord[1]) for coord in poly_coords]
                if len(img_poly_coords) >= 3:
                    # Draw filled polygon (light gray)
                    draw.polygon(img_poly_coords, fill=(240, 240, 255), outline=(100, 100, 150))
                
                # Get node positions for this edge
                node_u_id = edge['u']
                node_v_id = edge['v']
                node_u_pos = None
                node_v_pos = None
                for node in nodes:
                    if node['id'] == node_u_id:
                        node_u_pos = (node['xy'][0], node['xy'][1])
                    if node['id'] == node_v_id:
                        node_v_pos = (node['xy'][0], node['xy'][1])
                
                if node_u_pos is None or node_v_pos is None:
                    logger.warning("Could not find node positions for %s", edge_id)
                    node_u_pos = (centerline_coords[0][0], centerline_coords[0][1])
                    node_v_pos = (centerline_coords[-1][0], centerline_coords[-1][1])
                
                # Get channel width from edge width profile
                width_profile = edge.get('width_profile', {})
                channel_width = width_profile.get('w_median', 100.0)  # Default to 100 µm if not available
                
                # Compute revised skeleton line
                try:
                    revised_coords = compute_revised_skeleton(
                        centerline_coords,
                        matching_poly,
                        node_u_pos,
                        node_v_pos,
                        channel_width
                    )
                except Exception as e:
                    logger.warning("Failed to compute revised skeleton for %s: %s", edge_id, e)
                    revised_coords = [(c[0], c[1]) for c in centerline_coords]
                
                # Draw original skeleton line (centerline) in red
                img_centerline_coords = [world_to_px(coord[0], coord[1]) for coord in centerline_coords]
                if len(img_centerline_coords) >= 2:
                    draw.line(img_centerline_coords, fill=(255, 0, 0), width=2)
                
                # Draw revised skeleton line in blue
                img_revised_coords = [world_to_px(coord[0], coord[1]) for coord in revised_coords]
                if len(img_revised_coords) >= 2:
                    draw.line(img_revised_coords, fill=(0, 0, 255), width=2)
                
                # Save crop
                crop_path = crop_output_dir / f"{edge_id}_crop.png"
                img.save(crop_path, 'PNG')
                logger.info("Saved crop for %s: %s (%d × %d pixels, %.1f × %.1f µm)",
                           edge_id, crop_path, img_width, img_height, width_um, height_um)
            
            logger.info("Edge crops saved to: %s", crop_output_dir)
    except Exception:
        logger.exception("Error exporting edge crops")
        # Don't fail the script if crop export fails
    
    # Export full-scale image with complete revised skeleton
    try:
        with log_step("Export full-scale revised skeleton"):
            # Compute revised skeleton for all edges
            revised_skeletons = {}
            shapely_polys = []
            for poly_data in polys_sorted:
                coords = poly_data['polygon']['coordinates'][0]
                poly = Polygon(coords)
                if poly.is_valid:
                    shapely_polys.append((poly, poly_data))
            
            # Use the main polygon (largest)
            main_poly = shapely_polys[0][0] if shapely_polys else None
            
            if main_poly is None:
                logger.warning("No valid polygon found for full-scale skeleton export")
            else:
                # Build a map of nodes to edges for intersection prevention
                node_to_edges_map = {}
                for edge in edges:
                    node_u_id = edge['u']
                    node_v_id = edge['v']
                    edge_id = edge['id']
                    if node_u_id not in node_to_edges_map:
                        node_to_edges_map[node_u_id] = []
                    if node_v_id not in node_to_edges_map:
                        node_to_edges_map[node_v_id] = []
                    node_to_edges_map[node_u_id].append(edge_id)
                    node_to_edges_map[node_v_id].append(edge_id)
                
                # Compute revised skeleton for each edge
                for edge in edges:
                    edge_id = edge['id']
                    centerline_coords = edge['centerline']['coordinates']
                    
                    if len(centerline_coords) < 2:
                        continue
                    
                    # Get node positions
                    node_u_id = edge['u']
                    node_v_id = edge['v']
                    node_u_pos = None
                    node_v_pos = None
                    for node in nodes:
                        if node['id'] == node_u_id:
                            node_u_pos = (node['xy'][0], node['xy'][1])
                        if node['id'] == node_v_id:
                            node_v_pos = (node['xy'][0], node['xy'][1])
                    
                    if node_u_pos is None or node_v_pos is None:
                        continue
                    
                    # Get channel width
                    width_profile = edge.get('width_profile', {})
                    channel_width = width_profile.get('w_median', 100.0)
                    
                    # Check if nodes are junctions (have multiple edges)
                    # Use larger transition zone for junctions to prevent crossings
                    node_u_is_junction = len(node_to_edges_map.get(node_u_id, [])) > 1
                    node_v_is_junction = len(node_to_edges_map.get(node_v_id, [])) > 1
                    
                    # Use 5x channel width transition for junctions, 3x for endpoints
                    transition_factor_u = 5.0 if node_u_is_junction else 3.0
                    transition_factor_v = 5.0 if node_v_is_junction else 3.0
                    
                    # Compute revised skeleton with junction-aware transition
                    try:
                        revised_coords = compute_revised_skeleton_with_junctions(
                            centerline_coords,
                            main_poly,
                            node_u_pos,
                            node_v_pos,
                            channel_width,
                            transition_factor_u,
                            transition_factor_v
                        )
                        revised_skeletons[edge_id] = revised_coords
                    except Exception as e:
                        logger.warning("Failed to compute revised skeleton for %s: %s", edge_id, e)
                
                # Check for intersections and adjust if needed
                # Build a list of LineStrings for intersection checking
                revised_lines = {}
                for edge_id, coords in revised_skeletons.items():
                    if len(coords) >= 2:
                        revised_lines[edge_id] = LineString(coords)
                
                # Check for self-intersections within each edge
                for edge_id, line in revised_lines.items():
                    if not line.is_simple:
                        logger.warning("Edge %s has self-intersections, simplifying", edge_id)
                        # Simplify the line to remove self-intersections
                        simplified = line.simplify(1.0, preserve_topology=True)
                        if simplified.is_simple and len(simplified.coords) >= 2:
                            revised_skeletons[edge_id] = list(simplified.coords)
                            revised_lines[edge_id] = simplified
                
                # Check for intersections between different edges and fix them
                intersecting_pairs = []
                edge_ids = list(revised_lines.keys())
                
                # Build a map of edges connected to each node
                node_to_edges = {}
                for edge in edges:
                    node_u = edge['u']
                    node_v = edge['v']
                    edge_id = edge['id']
                    if edge_id in revised_skeletons:
                        if node_u not in node_to_edges:
                            node_to_edges[node_u] = []
                        if node_v not in node_to_edges:
                            node_to_edges[node_v] = []
                        node_to_edges[node_u].append(edge_id)
                        node_to_edges[node_v].append(edge_id)
                
                for i, edge_id1 in enumerate(edge_ids):
                    for edge_id2 in edge_ids[i+1:]:
                        line1 = revised_lines[edge_id1]
                        line2 = revised_lines[edge_id2]
                        if line1.intersects(line2):
                            intersection = line1.intersection(line2)
                            
                            # Check if edges share a common node (allowed to meet at node)
                            edge1 = next((e for e in edges if e['id'] == edge_id1), None)
                            edge2 = next((e for e in edges if e['id'] == edge_id2), None)
                            share_node = False
                            if edge1 and edge2:
                                share_node = (edge1['u'] == edge2['u'] or edge1['u'] == edge2['v'] or
                                            edge1['v'] == edge2['u'] or edge1['v'] == edge2['v'])
                            
                            # Check if intersection is at a shared node
                            if share_node:
                                # Find the shared node position
                                shared_node_id = None
                                if edge1['u'] == edge2['u'] or edge1['u'] == edge2['v']:
                                    shared_node_id = edge1['u']
                                elif edge1['v'] == edge2['u'] or edge1['v'] == edge2['v']:
                                    shared_node_id = edge1['v']
                                
                                if shared_node_id:
                                    shared_node = next((n for n in nodes if n['id'] == shared_node_id), None)
                                    if shared_node:
                                        node_point = Point(shared_node['xy'][0], shared_node['xy'][1])
                                        # Check distance for Point or first point of MultiPoint
                                        if isinstance(intersection, Point):
                                            if intersection.distance(node_point) < 10.0:  # Within 10 µm of node
                                                continue  # This is just meeting at the node, not a crossing
                                        elif isinstance(intersection, MultiPoint):
                                            # Check if all intersection points are near the node
                                            all_near_node = all(
                                                Point(p.x, p.y).distance(node_point) < 10.0 
                                                for p in intersection.geoms
                                            )
                                            if all_near_node:
                                                continue  # All intersections are at the node
                            
                            # Check if intersection is very close to line endpoints (within 5 µm)
                            coords1 = revised_skeletons[edge_id1]
                            coords2 = revised_skeletons[edge_id2]
                            if len(coords1) >= 2 and len(coords2) >= 2:
                                start1 = Point(coords1[0][0], coords1[0][1])
                                end1 = Point(coords1[-1][0], coords1[-1][1])
                                start2 = Point(coords2[0][0], coords2[0][1])
                                end2 = Point(coords2[-1][0], coords2[-1][1])
                                
                                endpoints = [start1, end1, start2, end2]
                                
                                if isinstance(intersection, Point):
                                    min_dist_to_endpoints = min(
                                        intersection.distance(ep) for ep in endpoints
                                    )
                                    if min_dist_to_endpoints < 5.0:  # Within 5 µm of an endpoint
                                        continue  # Likely just meeting at a node
                                elif isinstance(intersection, MultiPoint):
                                    # Check if all intersection points are near endpoints
                                    all_near_endpoints = all(
                                        min(p.distance(ep) for ep in endpoints) < 5.0
                                        for p in intersection.geoms
                                    )
                                    if all_near_endpoints:
                                        continue  # All intersections are near endpoints
                            
                            # This is a real crossing
                            intersecting_pairs.append((edge_id1, edge_id2))
                            if isinstance(intersection, Point):
                                logger.warning("Edges %s and %s intersect at (%.1f, %.1f)", 
                                             edge_id1, edge_id2, intersection.x, intersection.y)
                            else:
                                logger.warning("Edges %s and %s intersect (geometry type: %s)", 
                                             edge_id1, edge_id2, type(intersection).__name__)
                            
                            # Try to fix by adjusting one of the edges
                            # For now, we'll use a simple approach: if edges share a node, 
                            # ensure they stay closer to the original centerline near the junction
                            if share_node and edge1 and edge2:
                                # Find which node they share
                                shared_node_id = None
                                if edge1['u'] == edge2['u'] or edge1['u'] == edge2['v']:
                                    shared_node_id = edge1['u']
                                elif edge1['v'] == edge2['u'] or edge1['v'] == edge2['v']:
                                    shared_node_id = edge1['v']
                                
                                if shared_node_id:
                                    # Increase transition distance near shared nodes to prevent crossing
                                    # This will be handled in the next iteration by using a larger transition zone
                                    logger.info("Edges %s and %s share node %s, will use larger transition zone",
                                              edge_id1, edge_id2, shared_node_id)
                
                # Compute bounds for full-scale image
                all_coords = []
                for coords in revised_skeletons.values():
                    all_coords.extend(coords)
                
                # Also include polygon bounds
                if shapely_polys:
                    poly_bbox = shapely_polys[0][0].bounds
                    all_coords.extend([
                        (poly_bbox[0], poly_bbox[1]),
                        (poly_bbox[2], poly_bbox[3])
                    ])
                
                if all_coords:
                    padding = 50.0  # 50 µm padding
                    x_coords = [c[0] for c in all_coords]
                    y_coords = [c[1] for c in all_coords]
                    xmin = min(x_coords) - padding
                    ymin = min(y_coords) - padding
                    xmax = max(x_coords) + padding
                    ymax = max(y_coords) + padding
                    
                    width_um = xmax - xmin
                    height_um = ymax - ymin
                    
                    # Full scale: 1 pixel = 1 µm
                    um_per_px = 1.0
                    img_width = int(width_um / um_per_px)
                    img_height = int(height_um / um_per_px)
                    
                    # Create image
                    img = Image.new('RGB', (img_width, img_height), (255, 255, 255))
                    draw = ImageDraw.Draw(img)
                    
                    def world_to_px(x: float, y: float) -> tuple[int, int]:
                        px = int((x - xmin) / um_per_px)
                        py = int((y - ymin) / um_per_px)
                        return (px, py)
                    
                    # Draw polygon outline(s)
                    for poly, poly_data in shapely_polys:
                        poly_coords = poly_data['polygon']['coordinates'][0]
                        img_poly_coords = [world_to_px(coord[0], coord[1]) for coord in poly_coords]
                        if len(img_poly_coords) >= 3:
                            draw.polygon(img_poly_coords, fill=(240, 240, 255), outline=(100, 100, 150))
                    
                    # Draw revised skeleton lines
                    for edge_id, coords in revised_skeletons.items():
                        if len(coords) >= 2:
                            img_coords = [world_to_px(coord[0], coord[1]) for coord in coords]
                            draw.line(img_coords, fill=(0, 0, 255), width=2)
                    
                    # Draw nodes
                    node_radius = 5  # 5 pixels = 5 µm
                    for node in nodes:
                        x, y = world_to_px(node['xy'][0], node['xy'][1])
                        bbox = [x - node_radius, y - node_radius, x + node_radius, y + node_radius]
                        draw.ellipse(bbox, fill=(255, 0, 0), outline=(0, 0, 0))
                    
                    # Save image
                    skeleton_path = Path(__file__).resolve().parent / "revised_skeleton_fullscale.png"
                    img.save(skeleton_path, 'PNG')
                    logger.info("Full-scale revised skeleton saved to: %s (%d × %d pixels, %.1f × %.1f µm)",
                               skeleton_path, img_width, img_height, width_um, height_um)
                    if intersecting_pairs:
                        logger.warning("Found %d intersecting edge pairs (see log for details)", len(intersecting_pairs))
    except Exception:
        logger.exception("Error exporting full-scale revised skeleton")
        # Don't fail the script if export fails
    
    logger.info("=" * 70)
    logger.info("Network graph extraction completed successfully")
    logger.info("Full log saved to: %s", LOG_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
