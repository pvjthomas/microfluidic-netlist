"""Measure edge length and width profile."""

from typing import Dict, Any, List, Tuple
from shapely.geometry import Point, LineString, Polygon
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_edge_length(centerline_coords: List[List[float]]) -> float:
    """
    Compute edge length from centerline coordinates.
    
    Args:
        centerline_coords: List of [x, y] coordinate pairs
        
    Returns:
        Total length along centerline
    """
    if len(centerline_coords) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(centerline_coords) - 1):
        p1 = centerline_coords[i]
        p2 = centerline_coords[i + 1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        segment_length = np.sqrt(dx * dx + dy * dy)
        total_length += segment_length
    
    return total_length


def sample_width_along_centerline(
    polygon: Polygon,
    centerline_coords: List[List[float]],
    width_sample_step: float = 10.0,
    neighborhood_size: float = 5.0,
    ignore_node_neighborhood: float = 50.0
) -> List[Tuple[float, float]]:
    """
    Sample width along centerline at regular intervals.
    
    Width is estimated as 2 * distance from point to polygon boundary,
    assuming the centerline is medial. Uses median of small neighborhood
    for robustness. Ignores samples near nodes (start/end of edge).
    
    Args:
        polygon: Shapely polygon representing the channel region
        centerline_coords: List of [x, y] coordinate pairs along centerline
        width_sample_step: Distance between samples along centerline (in same units as coords)
        neighborhood_size: Size of neighborhood for median filtering (in same units)
        ignore_node_neighborhood: Distance from start/end to ignore (in same units, default: 50.0)
        
    Returns:
        List of (distance_along_edge, width) tuples
    """
    if len(centerline_coords) < 2:
        return []
    
    # Create LineString for easier distance calculations
    centerline = LineString(centerline_coords)
    total_length = centerline.length
    
    if total_length == 0:
        return []
    
    # Sample points along centerline, but skip node neighborhoods
    samples = []
    s = ignore_node_neighborhood  # Start after node neighborhood
    
    while s <= total_length - ignore_node_neighborhood:
        # Interpolate point at distance s along centerline
        point = centerline.interpolate(s)
        
        # Compute width at this point using neighborhood median for robustness
        width = estimate_width_at_point(polygon, point, neighborhood_size)
        
        samples.append((s, width))
        
        # Move to next sample point
        s += width_sample_step
    
    # If edge is too short, sample at midpoint
    if total_length < 2 * ignore_node_neighborhood and len(samples) == 0:
        midpoint = total_length / 2.0
        point = centerline.interpolate(midpoint)
        width = estimate_width_at_point(polygon, point, neighborhood_size)
        samples.append((midpoint, width))
    
    return samples


def estimate_width_at_point(
    polygon: Polygon,
    point: Point,
    neighborhood_size: float = 5.0
) -> float:
    """
    Estimate local channel width at a point.
    
    Uses median of distances to boundary in a small neighborhood for robustness.
    
    Args:
        polygon: Shapely polygon
        point: Point on centerline
        neighborhood_size: Radius of neighborhood to sample (in same units as coords)
        
    Returns:
        Estimated width (2 * median distance to boundary)
    """
    # Sample a few points in a small neighborhood around the center point
    num_samples = 5
    distances = []
    
    # Sample along perpendicular direction (approximate)
    # For robustness, sample in a small circle around the point
    for angle in np.linspace(0, 2 * np.pi, num_samples, endpoint=False):
        offset_x = neighborhood_size * 0.5 * np.cos(angle)
        offset_y = neighborhood_size * 0.5 * np.sin(angle)
        sample_point = Point(point.x + offset_x, point.y + offset_y)
        
        # Only consider points that are inside the polygon
        if polygon.contains(sample_point) or polygon.boundary.distance(sample_point) < 1e-6:
            dist = polygon.boundary.distance(sample_point)
            distances.append(dist)
    
    # Also include the center point
    if polygon.contains(point) or polygon.boundary.distance(point) < 1e-6:
        dist_center = polygon.boundary.distance(point)
        distances.append(dist_center)
    
    if not distances:
        # Fallback: use distance at center point even if slightly outside
        dist = polygon.boundary.distance(point)
        distances.append(dist)
    
    # Use median for robustness against outliers
    median_dist = np.median(distances)
    
    # Width is approximately 2 * distance to boundary (for medial axis)
    width = 2.0 * median_dist
    
    return max(width, 0.0)  # Ensure non-negative


def compute_width_profile(
    polygon: Polygon,
    centerline_coords: List[List[float]],
    width_sample_step: float = 10.0,
    neighborhood_size: float = 5.0,
    ignore_node_neighborhood: float = 50.0
) -> Dict[str, Any]:
    """
    Compute complete width profile for an edge.
    
    Args:
        polygon: Shapely polygon representing the channel region
        centerline_coords: List of [x, y] coordinate pairs along centerline
        width_sample_step: Distance between samples along centerline
        neighborhood_size: Size of neighborhood for median filtering
        
    Returns:
        Width profile dictionary with:
        - kind: "sampled" (will be classified later)
        - samples: List of [s, width] pairs
        - w_median: Median width
        - w_min: Minimum width
        - w_max: Maximum width
    """
    # Sample width along centerline
    samples = sample_width_along_centerline(
        polygon,
        centerline_coords,
        width_sample_step=width_sample_step,
        neighborhood_size=neighborhood_size,
        ignore_node_neighborhood=ignore_node_neighborhood
    )
    
    if not samples:
        # Fallback: single sample at midpoint
        centerline = LineString(centerline_coords)
        midpoint = centerline.interpolate(0.5, normalized=True)
        width = estimate_width_at_point(polygon, midpoint, neighborhood_size)
        samples = [(centerline.length / 2.0, width)]
    
    # Extract widths
    widths = [w for s, w in samples]
    
    # Compute statistics
    w_median = float(np.median(widths))
    w_min = float(np.min(widths))
    w_max = float(np.max(widths))
    
    # Convert samples to list format: [[s0, w0], [s1, w1], ...]
    samples_list = [[float(s), float(w)] for s, w in samples]
    
    return {
        'kind': 'sampled',  # Will be classified later
        'samples': samples_list,
        'w_median': w_median,
        'w_min': w_min,
        'w_max': w_max,
        'w0': samples_list[0][1] if samples_list else w_median,  # Width at start
        'w1': samples_list[-1][1] if samples_list else w_median  # Width at end
    }


def measure_edge(
    polygon: Polygon,
    centerline_coords: List[List[float]],
    width_sample_step: float = 10.0,
    neighborhood_size: float = 5.0,
    ignore_node_neighborhood: float = 50.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Measure edge length and width profile.
    
    Args:
        polygon: Shapely polygon representing the channel region
        centerline_coords: List of [x, y] coordinate pairs along centerline
        width_sample_step: Distance between samples along centerline
        neighborhood_size: Size of neighborhood for median filtering
        
    Returns:
        Tuple of (length, width_profile_dict)
    """
    length = compute_edge_length(centerline_coords)
    width_profile = compute_width_profile(
        polygon,
        centerline_coords,
        width_sample_step=width_sample_step,
        neighborhood_size=neighborhood_size,
        ignore_node_neighborhood=ignore_node_neighborhood
    )
    
    # Smooth width profile using moving average (simple smoothing)
    if len(width_profile.get('samples', [])) > 3:
        samples = width_profile['samples']
        window_size = min(3, len(samples) // 2)
        if window_size >= 1:
            smoothed_samples = []
            for i, (s, w) in enumerate(samples):
                # Simple moving average
                start_idx = max(0, i - window_size)
                end_idx = min(len(samples), i + window_size + 1)
                window_values = [samples[j][1] for j in range(start_idx, end_idx)]
                smoothed_w = sum(window_values) / len(window_values)
                smoothed_samples.append([s, smoothed_w])
            width_profile['samples'] = smoothed_samples
            # Recompute statistics from smoothed values
            widths = [w for s, w in smoothed_samples]
            width_profile['w_median'] = float(np.median(widths))
            width_profile['w_min'] = float(np.min(widths))
            width_profile['w_max'] = float(np.max(widths))
            width_profile['w0'] = smoothed_samples[0][1] if smoothed_samples else width_profile['w_median']
            width_profile['w1'] = smoothed_samples[-1][1] if smoothed_samples else width_profile['w_median']
    
    return length, width_profile
