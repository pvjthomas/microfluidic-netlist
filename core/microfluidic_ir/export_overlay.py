"""Export tagged overlay image (PNG/SVG)."""

from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Polygon, Point, LineString
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_bounds(
    polygons: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    ports: Optional[List[Dict[str, Any]]] = None,
    padding: float = 100.0
) -> Tuple[float, float, float, float]:
    """
    Compute bounding box for all geometry with padding.
    
    Returns:
        Tuple of (xmin, ymin, xmax, ymax)
    """
    all_x = []
    all_y = []
    
    # Collect points from polygons
    for poly_data in polygons:
        coords = poly_data['polygon']['coordinates'][0]
        for coord in coords:
            all_x.append(coord[0])
            all_y.append(coord[1])
    
    # Collect points from nodes
    for node in nodes:
        all_x.append(node['xy'][0])
        all_y.append(node['xy'][1])
    
    # Collect points from edge centerlines
    for edge in edges:
        coords = edge['centerline']['coordinates']
        for coord in coords:
            all_x.append(coord[0])
            all_y.append(coord[1])
    
    # Collect points from ports
    if ports:
        for port in ports:
            center = port['marker']['center']
            radius = port['marker']['radius']
            all_x.extend([center[0] - radius, center[0] + radius])
            all_y.extend([center[1] - radius, center[1] + radius])
    
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


def export_overlay_png(
    polygons: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    ports: Optional[List[Dict[str, Any]]] = None,
    output_path: str = "overlay.png",
    image_width: int = 2000,
    image_height: Optional[int] = None,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    polygon_fill_color: Tuple[int, int, int] = (240, 240, 255),
    polygon_outline_color: Tuple[int, int, int] = (200, 200, 220),
    centerline_color: Tuple[int, int, int] = (0, 100, 200),
    node_color: Tuple[int, int, int] = (255, 0, 0),
    port_color: Tuple[int, int, int] = (0, 255, 0),
    label_color: Tuple[int, int, int] = (0, 0, 0),
    font_size: int = 8  # Reduced by 30%: 12 * 0.7 = 8.4 -> 8
) -> None:
    """
    Export tagged overlay image as PNG.
    
    Args:
        polygons: List of polygon dicts with 'polygon' key containing GeoJSON
        nodes: List of node dicts
        edges: List of edge dicts
        ports: Optional list of port dicts
        output_path: Output file path
        image_width: Target image width in pixels
        image_height: Target image height (auto if None)
        background_color: Background RGB color
        polygon_fill_color: Polygon fill RGB color (faint)
        polygon_outline_color: Polygon outline RGB color
        centerline_color: Centerline RGB color
        node_color: Node marker RGB color
        port_color: Port marker RGB color
        label_color: Label text RGB color
        font_size: Font size for labels
    """
    logger.info("Exporting overlay PNG to: %s", output_path)
    
    # Compute bounds
    xmin, ymin, xmax, ymax = compute_bounds(polygons, nodes, edges, ports)
    width_um = xmax - xmin
    height_um = ymax - ymin
    
    # Calculate um_per_px to fit desired width
    um_per_px = width_um / image_width
    
    # Calculate image height if not specified
    if image_height is None:
        image_height = int(height_um / um_per_px)
    
    logger.debug("Bounds: (%.1f, %.1f) to (%.1f, %.1f), size: %.1f × %.1f µm",
                xmin, ymin, xmax, ymax, width_um, height_um)
    logger.debug("Image size: %d × %d pixels, um_per_px: %.3f", image_width, image_height, um_per_px)
    
    # Create image
    img = Image.new('RGB', (image_width, image_height), background_color)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Draw polygons (faint fill)
    for poly_data in polygons:
        coords = poly_data['polygon']['coordinates'][0]
        if len(coords) < 3:
            continue
        
        # Convert to image coordinates
        img_coords = [
            world_to_image(coord[0], coord[1], xmin, ymin, um_per_px)
            for coord in coords
        ]
        
        # Draw polygon fill
        draw.polygon(img_coords, fill=polygon_fill_color, outline=polygon_outline_color)
    
    # Draw centerlines
    for edge in edges:
        coords = edge['centerline']['coordinates']
        if len(coords) < 2:
            continue
        
        # Convert to image coordinates
        img_coords = [
            world_to_image(coord[0], coord[1], xmin, ymin, um_per_px)
            for coord in coords
        ]
        
        # Draw centerline
        draw.line(img_coords, fill=centerline_color, width=2)
    
    # Helper function to check if two text bounding boxes overlap
    def text_bboxes_overlap(bbox1, bbox2, padding=3):
        """Check if two text bounding boxes overlap with padding."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        return not (x1_max + padding < x2_min - padding or 
                   x2_max + padding < x1_min - padding or
                   y1_max + padding < y2_min - padding or
                   y2_max + padding < y1_min - padding)
    
    # Helper function to adjust text position to avoid overlaps
    def adjust_text_position(text_x, text_y, text_width, text_height, existing_bboxes, image_width, image_height, padding=3):
        """Adjust text position to avoid overlaps with existing text."""
        bbox = (text_x, text_y, text_x + text_width, text_y + text_height)
        overlaps = any(text_bboxes_overlap(bbox, existing_bbox, padding) for existing_bbox in existing_bboxes)
        
        if not overlaps:
            return text_x, text_y, bbox
        
        # Try positions around the original (spiral search)
        offsets = [
            (0, -text_height - padding),  # Above
            (text_width + padding, 0),    # Right
            (0, text_height + padding),    # Below
            (-text_width - padding, 0),   # Left
        ]
        
        for offset_x, offset_y in offsets:
            new_x = text_x + offset_x
            new_y = text_y + offset_y
            new_x = max(0, min(new_x, image_width - text_width))
            new_y = max(0, min(new_y, image_height - text_height))
            
            bbox = (new_x, new_y, new_x + text_width, new_y + text_height)
            overlaps = any(text_bboxes_overlap(bbox, existing_bbox, padding) for existing_bbox in existing_bboxes)
            if not overlaps:
                return new_x, new_y, bbox
        
        return text_x, text_y, bbox
    
    # Track text bounding boxes to prevent overlaps
    text_bboxes = []
    
    # Draw nodes
    node_radius = max(3, int(5 / um_per_px))  # ~5 µm radius
    for node in nodes:
        x, y = world_to_image(node['xy'][0], node['xy'][1], xmin, ymin, um_per_px)
        
        # Draw node circle
        bbox = [x - node_radius, y - node_radius, x + node_radius, y + node_radius]
        draw.ellipse(bbox, fill=node_color, outline=(0, 0, 0))
        
        # Draw node label with overlap prevention
        label = node['id']
        char_width = font_size * 0.6
        text_width = int(len(label) * char_width)
        text_height = font_size
        
        # Initial position
        text_x = x + node_radius + 2  # Reduced offset
        text_y = y - text_height // 2
        
        # Adjust position to avoid overlaps
        text_x, text_y, text_bbox = adjust_text_position(
            text_x, text_y, text_width, text_height,
            text_bboxes, image_width, image_height, padding=3
        )
        text_bboxes.append(text_bbox)
        
        draw.text((text_x, text_y), label, fill=label_color, font=font)
    
    # Draw edge labels at midpoints
    for edge in edges:
        coords = edge['centerline']['coordinates']
        if len(coords) < 2:
            continue
        
        # Use LineString to find exact midpoint
        from shapely.geometry import LineString
        centerline = LineString(coords)
        total_length = centerline.length
        
        if total_length == 0:
            continue
        
        # Get midpoint by interpolating at 50% of length
        midpoint_point = centerline.interpolate(0.5, normalized=True)
        mx = midpoint_point.x
        my = midpoint_point.y
        
        # Draw edge label at midpoint (no offset - place directly on centerline)
        label = edge['id']
        text_x, text_y = world_to_image(mx, my, xmin, ymin, um_per_px)
        
        # Center the text on the midpoint
        # Get approximate text size (rough estimate)
        char_width = font_size * 0.6
        text_width = int(len(label) * char_width)
        text_height = font_size
        text_x -= int(text_width / 2)
        text_y -= int(text_height / 2)
        
        # Clamp to image bounds
        text_x = max(0, min(text_x, image_width - text_width))
        text_y = max(0, min(text_y, image_height - text_height))
        
        # Adjust position to avoid overlaps
        text_x, text_y, text_bbox = adjust_text_position(
            text_x, text_y, text_width, text_height,
            text_bboxes, image_width, image_height, padding=3
        )
        text_bboxes.append(text_bbox)
        
        draw.text((text_x, text_y), label, fill=label_color, font=font)
    
    # Draw ports
    if ports:
        for port in ports:
            marker = port['marker']
            center = marker['center']
            radius = marker['radius']
            
            # Convert to image coordinates
            cx, cy = world_to_image(center[0], center[1], xmin, ymin, um_per_px)
            radius_px = int(radius / um_per_px)
            
            # Draw port circle
            bbox = [cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px]
            draw.ellipse(bbox, outline=port_color, width=2)
            
            # Draw port label with overlap prevention
            label = port.get('label', port['port_id'])
            if not label:
                label = port['port_id']
            
            char_width = font_size * 0.6
            text_width = int(len(label) * char_width)
            text_height = font_size
            
            # Initial position
            text_x = cx + radius_px + 2  # Reduced offset
            text_y = cy - text_height // 2
            
            # Adjust position to avoid overlaps
            text_x, text_y, text_bbox = adjust_text_position(
                text_x, text_y, text_width, text_height,
                text_bboxes, image_width, image_height, padding=3
            )
            text_bboxes.append(text_bbox)
            
            draw.text((text_x, text_y), label, fill=label_color, font=font)
    
    # Save image
    img.save(output_path, 'PNG')
    logger.info("Overlay PNG saved: %s (%d × %d pixels)", output_path, image_width, image_height)
