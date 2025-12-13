"""Curve fitting for arcs, circles, etc."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.optimize import least_squares
import logging

logger = logging.getLogger(__name__)


def fit_circle_to_points(points: List[List[float]]) -> Tuple[Optional[Dict[str, float]], float]:
    """
    Fit a circle to a set of points using least squares.
    
    Args:
        points: List of [x, y] coordinate pairs
        
    Returns:
        Tuple of (circle_dict, rms_error) where circle_dict contains:
        - center: [cx, cy]
        - radius: r
        Or (None, inf) if fitting fails or points are invalid.
        
        RMS error is computed as sqrt(mean((r_actual - r_fitted)^2))
    """
    if len(points) < 3:
        return None, float('inf')
    
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Initial guess: use geometric center and mean radius
    cx_init = np.mean(x)
    cy_init = np.mean(y)
    r_init = np.mean(np.sqrt((x - cx_init)**2 + (y - cy_init)**2))
    
    def circle_residuals(params):
        """Residual function: distance from point to circle minus radius."""
        cx, cy, r = params
        distances = np.sqrt((x - cx)**2 + (y - cy)**2)
        return distances - r
    
    try:
        # Fit circle using least squares
        result = least_squares(
            circle_residuals,
            [cx_init, cy_init, r_init],
            method='lm'  # Levenberg-Marquardt
        )
        
        if not result.success:
            return None, float('inf')
        
        cx_fit, cy_fit, r_fit = result.x
        
        # Compute RMS error
        distances = np.sqrt((x - cx_fit)**2 + (y - cy_fit)**2)
        residuals = distances - r_fit
        rms_error = np.sqrt(np.mean(residuals**2))
        
        # Check for invalid results
        if r_fit <= 0 or not np.isfinite(r_fit):
            return None, float('inf')
        
        return {
            'center': [float(cx_fit), float(cy_fit)],
            'radius': float(r_fit)
        }, float(rms_error)
        
    except Exception as e:
        logger.debug("Circle fitting failed: %s", e)
        return None, float('inf')


def detect_circle_like_polyline(
    polyline_coords: List[List[float]],
    min_points: int = 8,
    rms_tolerance: float = 1.0
) -> Optional[Dict[str, Any]]:
    """
    Detect if a polyline is circle-like.
    
    Args:
        polyline_coords: List of [x, y] coordinate pairs (should be closed or nearly closed)
        min_points: Minimum number of points to consider (default: 8)
        rms_tolerance: Maximum RMS error to consider as circle (default: 1.0 µm)
        
    Returns:
        Circle dict with 'center', 'radius', and 'rms_error' if detected, else None
    """
    if len(polyline_coords) < min_points:
        return None
    
    # Remove duplicate last point if polyline is closed
    coords = polyline_coords
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    
    if len(coords) < min_points:
        return None
    
    # Fit circle
    circle, rms_error = fit_circle_to_points(coords)
    
    if circle is None:
        return None
    
    if rms_error > rms_tolerance:
        return None
    
    return {
        'center': circle['center'],
        'radius': circle['radius'],
        'rms_error': rms_error
    }
