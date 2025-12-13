"""Classify edges as uniform, taper_linear, or sampled."""

from typing import Dict, Any, Literal
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def classify_width_profile(
    width_profile: Dict[str, Any],
    uniform_tol: float = 0.05,
    taper_r2_threshold: float = 0.95
) -> Dict[str, Any]:
    """
    Classify width profile as uniform, taper_linear, or sampled.
    
    Rules (from spec):
    - uniform: if (w_max - w_min) / w_median <= uniform_tol
    - taper_linear: else if width vs s fits a line with R² > threshold
    - sampled: otherwise
    
    Args:
        width_profile: Width profile dictionary with samples, w_median, w_min, w_max
        uniform_tol: Tolerance for uniform classification (default: 0.05 = 5%)
        taper_r2_threshold: Minimum R² for linear taper classification (default: 0.95)
        
    Returns:
        Updated width_profile dict with 'kind' set appropriately
    """
    w_median = width_profile.get('w_median', 0.0)
    w_min = width_profile.get('w_min', 0.0)
    w_max = width_profile.get('w_max', 0.0)
    samples = width_profile.get('samples', [])
    
    if w_median <= 0 or not samples:
        logger.warning("Invalid width profile for classification")
        width_profile['kind'] = 'sampled'
        return width_profile
    
    # Check for uniform: (w_max - w_min) / w_median <= uniform_tol
    width_range = w_max - w_min
    relative_variation = width_range / w_median if w_median > 0 else float('inf')
    
    if relative_variation <= uniform_tol:
        width_profile['kind'] = 'uniform'
        logger.debug("Classified as uniform: relative_variation=%.4f <= %.4f", 
                    relative_variation, uniform_tol)
        return width_profile
    
    # Check for linear taper: fit line to width vs distance_along_edge
    # Extract s (distance) and w (width) from samples
    if len(samples) < 2:
        width_profile['kind'] = 'sampled'
        return width_profile
    
    s_values = [s[0] for s in samples]
    w_values = [s[1] for s in samples]
    
    # Fit linear regression: w = a * s + b
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(s_values, w_values)
        r_squared = r_value ** 2
        
        if r_squared >= taper_r2_threshold:
            width_profile['kind'] = 'taper_linear'
            width_profile['w0'] = intercept  # w at s=0
            width_profile['w1'] = slope * s_values[-1] + intercept  # w at s=length
            logger.debug("Classified as taper_linear: R²=%.4f >= %.4f, slope=%.4f", 
                        r_squared, taper_r2_threshold, slope)
            return width_profile
    except Exception as e:
        logger.warning("Linear fit failed: %s", e)
        # Fall through to sampled
    
    # Default: sampled
    width_profile['kind'] = 'sampled'
    logger.debug("Classified as sampled: relative_variation=%.4f > %.4f, or R² < %.4f",
                relative_variation, uniform_tol, taper_r2_threshold)
    return width_profile
