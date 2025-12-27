"""Geometry utilities and normalization."""

import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Tuple, Dict, Any
from shapely.geometry import Polygon
from .skeleton import rasterize_polygon


def local_half_width_from_mask(mask: np.ndarray, pixel_size: float) -> np.ndarray:
    """
    Compute distance to nearest boundary for each interior pixel.
    Output is in the same units as pixel_size.
    """
    # distance_transform_edt computes distance from nonzero pixels to nearest zero pixel
    # mask: True inside polygon -> interior distances to boundary (mask False)
    dist_px = distance_transform_edt(mask)
    dist = dist_px * pixel_size
    return dist


def polygon_to_mask(
    poly: Polygon,
    pixel_size: float,
    pad: float = 0.0
) -> Tuple[np.ndarray, Dict[str, float], Tuple[float, float]]:
    """
    Convert a polygon to a binary mask.
    
    Args:
        poly: Shapely polygon
        pixel_size: Size of each pixel in world units (same as um_per_px)
        pad: Padding around polygon in world units
        
    Returns:
        Tuple of (mask, transform, origin) where:
        - mask: (H, W) bool array, True inside polygon
        - transform: Dict with transform parameters (origin_x, origin_y, um_per_px, etc.)
        - origin: (origin_x, origin_y) tuple
    """
    mask, transform = rasterize_polygon(poly, um_per_px=pixel_size, padding=pad)
    origin = (transform['origin_x'], transform['origin_y'])
    return mask, transform, origin


def local_scale_field(
    poly: Polygon,
    pixel_size: float,
    pad: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Compute local scale field for a polygon.
    
    Args:
        poly: Shapely polygon
        pixel_size: Size of each pixel in world units
        pad: Padding around polygon in world units
        
    Returns:
        Tuple of (dist, width, mask, transform) where:
        - dist: (H, W) float, local half-width (radius) in world units
        - width: (H, W) float, approx local channel width in world units
        - mask: (H, W) bool, inside polygon
        - transform: Dict with transform parameters (origin_x, origin_y, um_per_px, etc.)
    """
    mask, transform, _ = polygon_to_mask(poly, pixel_size=pixel_size, pad=pad)
    dist = local_half_width_from_mask(mask, pixel_size=pixel_size)
    width = 2.0 * dist
    return dist, width, mask, transform


def get_distance_at_coordinate(
    x: float,
    y: float,
    dist_field: np.ndarray,
    transform: Dict[str, float]
) -> float:
    """
    Get the distance (local half-width) value at a specific coordinate.
    
    Args:
        x: X coordinate in world units
        y: Y coordinate in world units
        dist_field: (H, W) distance field array (from local_scale_field)
        transform: Transform dict with origin_x, origin_y, um_per_px
    
    Returns:
        Distance value at the coordinate, or 0.0 if coordinate is outside the field
    """
    # Convert world coordinates to pixel coordinates
    # Note: pixel_to_coords uses: x = origin_x + (col * um_per_px), y = origin_y + (row * um_per_px)
    # So: col = (x - origin_x) / um_per_px, row = (y - origin_y) / um_per_px
    col = (x - transform['origin_x']) / transform['um_per_px']
    row = (y - transform['origin_y']) / transform['um_per_px']
    
    # Convert to integer indices (note: row corresponds to array first dimension)
    row_idx = int(round(row))
    col_idx = int(round(col))
    
    # Check bounds
    if row_idx < 0 or row_idx >= dist_field.shape[0] or col_idx < 0 or col_idx >= dist_field.shape[1]:
        return 0.0
    
    return float(dist_field[row_idx, col_idx])

