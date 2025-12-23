"""Skeletonization of channel polygons."""

import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
import networkx as nx
from skimage.morphology import thin, medial_axis, binary_closing, binary_opening, disk
from skimage import measure
from scipy import ndimage
from typing import List, Tuple, Dict, Any, Optional
import logging
import time
import itertools
from pathlib import Path
from PIL import Image, ImageDraw
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

logger = logging.getLogger(__name__)


def rasterize_polygon(
    polygon: Polygon,
    um_per_px: float = 20.0,
    padding: float = 10.0,
    max_pixels: int = 25_000_000
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Rasterize a Shapely polygon to a binary image using PIL ImageDraw.
    
    Args:
        polygon: Shapely polygon to rasterize
        um_per_px: Microns per pixel (lower = more resolution)
        padding: Padding around polygon in original units (microns)
        max_pixels: Maximum allowed total pixels in raster (default: 25,000,000)
    
    Returns:
        Tuple of (binary_image, transform_dict) where transform_dict contains:
        - origin_x, origin_y: Top-left corner in original coordinates
        - um_per_px: Resolution (microns per pixel)
    """
    logger.debug("rasterize_polygon: entry, um_per_px=%.3f, padding=%.3f, max_pixels=%d", 
                 um_per_px, padding, max_pixels)
    start_raster = time.time()
    
    # Get bounds and add padding
    try:
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        logger.debug("rasterize_polygon: polygon bounds: (%.3f, %.3f, %.3f, %.3f)", 
                     bounds[0], bounds[1], bounds[2], bounds[3])
        width_um = bounds[2] - bounds[0]
        height_um = bounds[3] - bounds[1]
        logger.debug("rasterize_polygon: polygon size: %.3f × %.3f µm", width_um, height_um)
    except Exception as e:
        logger.error("rasterize_polygon: failed to get polygon bounds: %s", e, exc_info=True)
        raise
    
    # Add padding
    padded_minx = bounds[0] - padding
    padded_miny = bounds[1] - padding
    padded_width_um = width_um + 2 * padding
    padded_height_um = height_um + 2 * padding
    logger.debug("rasterize_polygon: padded size: %.3f × %.3f µm", padded_width_um, padded_height_um)
    
    # Calculate image dimensions
    img_width = int(np.ceil(padded_width_um / um_per_px))
    img_height = int(np.ceil(padded_height_um / um_per_px))
    logger.debug("rasterize_polygon: calculated image dimensions: %d × %d pixels", 
                 img_width, img_height)
    
    # Guardrail: check raster size limits
    max_dimension = 8000
    
    total_pixels = img_width * img_height
    logger.debug("rasterize_polygon: total pixels: %d", total_pixels)
    if img_width > max_dimension or img_height > max_dimension:
        # Calculate recommended um_per_px based on max dimension constraint
        # For dimension constraint: um_per_px >= max(width, height) / max_dimension
        recommended_um_per_px_dim = max(padded_width_um, padded_height_um) / max_dimension
        # Also consider pixel count constraint
        recommended_um_per_px_pixels = np.sqrt((padded_width_um * padded_height_um) / max_pixels)
        recommended_um_per_px = max(recommended_um_per_px_dim, recommended_um_per_px_pixels)
        raise ValueError(
            f"Raster size {img_width}×{img_height} exceeds maximum dimension {max_dimension}. "
            f"Polygon size: {width_um:.1f}×{height_um:.1f} µm, um_per_px: {um_per_px:.2f}. "
            f"Recommended um_per_px >= {recommended_um_per_px:.2f} (rough guide: "
            f"sqrt((width×height)/max_pixels) = {recommended_um_per_px_pixels:.2f})."
        )
    
    if total_pixels > max_pixels:
        # Calculate recommended um_per_px: um_per_px >= sqrt((width*height)/max_pixels)
        recommended_um_per_px = np.sqrt((padded_width_um * padded_height_um) / max_pixels)
        raise ValueError(
            f"Raster size {img_width}×{img_height} = {total_pixels} pixels exceeds maximum {max_pixels}. "
            f"Polygon size: {width_um:.1f}×{height_um:.1f} µm, um_per_px: {um_per_px:.2f}. "
            f"Recommended um_per_px >= {recommended_um_per_px:.2f} (rough guide: "
            f"sqrt((width×height)/max_pixels))."
        )
    
    logger.info(
        f"Rasterizing: polygon {width_um:.1f}×{height_um:.1f} µm, "
        f"um_per_px={um_per_px:.2f}, image shape ({img_height}, {img_width}), {total_pixels} pixels"
    )
    
    # Create PIL image (black = 0, white = 1)
    logger.debug("rasterize_polygon: creating PIL image %d × %d", img_width, img_height)
    try:
        img = Image.new('1', (img_width, img_height), 0)
        draw = ImageDraw.Draw(img)
        logger.debug("rasterize_polygon: PIL image created successfully")
    except Exception as e:
        logger.error("rasterize_polygon: failed to create PIL image: %s", e, exc_info=True)
        raise
    
    # Convert polygon coordinates to PIL image coordinates
    # PIL uses: (0,0) at top-left, x increases right, y increases down
    # pixel_to_coords uses: y = origin_y + (row * um_per_px), so row 0 maps to origin_y (minimum y)
    # This means our image has y increasing upward (row increases = y increases in coords)
    # Conversion: image_x = (shapely_x - padded_minx) / um_per_px
    #            image_y = (shapely_y - padded_miny) / um_per_px (same direction)
    
    def shapely_to_image_coords(shapely_x: float, shapely_y: float) -> Tuple[int, int]:
        """Convert Shapely coordinates to PIL image coordinates.
        
        Note: pixel_to_coords uses y = origin_y + (row * um_per_px), so row 0 
        corresponds to minimum y. PIL ImageDraw expects coordinates with y increasing 
        downward. We draw with y increasing upward (no flip here), then flip the final 
        image with np.flipud to match the transform semantics.
        """
        img_x = int((shapely_x - padded_minx) / um_per_px)
        # No Y-flip here - we'll flip the final image with np.flipud
        img_y = int((shapely_y - padded_miny) / um_per_px)
        return (img_x, img_y)
    
    # Fill exterior with 1 (white)
    logger.debug("rasterize_polygon: drawing exterior polygon with %d points", 
                 len(polygon.exterior.coords) - 1)
    try:
        exterior_coords = [shapely_to_image_coords(x, y) for x, y in polygon.exterior.coords[:-1]]  # Exclude last duplicate
        logger.debug("rasterize_polygon: exterior converted to %d image coordinates", len(exterior_coords))
        if len(exterior_coords) >= 3:  # Need at least 3 points for a polygon
            draw.polygon(exterior_coords, fill=1, outline=1)
            logger.debug("rasterize_polygon: exterior polygon drawn")
        else:
            logger.warning("rasterize_polygon: exterior has only %d points, skipping", len(exterior_coords))
    except Exception as e:
        logger.error("rasterize_polygon: failed to draw exterior: %s", e, exc_info=True)
        raise
    
    # Clear each interior ring (hole) with 0 (black)
    logger.debug("rasterize_polygon: processing %d interior rings (holes)", len(polygon.interiors))
    for i, interior in enumerate(polygon.interiors):
        try:
            interior_coords = [shapely_to_image_coords(x, y) for x, y in interior.coords[:-1]]  # Exclude last duplicate
            logger.debug("rasterize_polygon: interior %d converted to %d image coordinates", 
                        i, len(interior_coords))
            if len(interior_coords) >= 3:
                draw.polygon(interior_coords, fill=0, outline=0)
                logger.debug("rasterize_polygon: interior %d drawn", i)
            else:
                logger.warning("rasterize_polygon: interior %d has only %d points, skipping", 
                              i, len(interior_coords))
        except Exception as e:
            logger.error("rasterize_polygon: failed to draw interior %d: %s", i, e, exc_info=True)
            raise
    
    # Convert PIL image to numpy array and flip vertically to match transform semantics
    # (pixel_to_coords expects row 0 = min y, but PIL draws with y=0 at top)
    logger.debug("rasterize_polygon: converting PIL image to numpy array")
    try:
        binary_img = np.array(img, dtype=bool)
        logger.debug("rasterize_polygon: numpy array created, shape: %s, dtype: %s", 
                     binary_img.shape, binary_img.dtype)
        binary_img = np.flipud(binary_img)  # Flip vertically to match transform semantics
        logger.debug("rasterize_polygon: image flipped vertically")
    except Exception as e:
        logger.error("rasterize_polygon: failed to convert PIL to numpy: %s", e, exc_info=True)
        raise
    
    raster_time = time.time() - start_raster
    num_pixels = np.sum(binary_img)
    logger.info(f"  Raster fill: {raster_time:.2f}s")
    logger.info(f"Rasterized: {num_pixels} filled pixels ({100.0 * num_pixels / total_pixels:.1f}%) in {raster_time:.2f}s total")
    logger.debug("rasterize_polygon: filled pixel count: %d / %d (%.2f%%)", 
                 num_pixels, total_pixels, 100.0 * num_pixels / total_pixels if total_pixels > 0 else 0)
    
    # Save binary image if in DEBUG mode
    if logger.isEnabledFor(logging.DEBUG):
        try:
            debug_output_dir = Path("debug_output")
            debug_output_dir.mkdir(exist_ok=True)
            output_path = debug_output_dir / f"rasterized_binary_{int(time.time()*1000)}.png"
            # Convert boolean array to uint8 for saving (0=black, 255=white)
            img_to_save = Image.fromarray((binary_img.astype(np.uint8) * 255), mode='L')
            img_to_save.save(output_path)
            logger.debug("rasterize_polygon: saved rasterized image to %s", output_path)
        except Exception as e:
            logger.warning("rasterize_polygon: failed to save rasterized image: %s", e)
    
    transform = {
        'origin_x': padded_minx,
        'origin_y': padded_miny,
        'um_per_px': um_per_px,
        'img_width': img_width,
        'img_height': img_height
    }
    logger.debug("rasterize_polygon: transform: origin=(%.3f, %.3f), um_per_px=%.3f, size=%dx%d",
                 transform['origin_x'], transform['origin_y'], transform['um_per_px'],
                 transform['img_width'], transform['img_height'])
    logger.debug("rasterize_polygon: exit, total time: %.2fs", raster_time)
    
    return binary_img, transform


def estimate_channel_width_and_resolution(
    polygon: Polygon,
    coarse_um_per_px: float = 30.0,
    target_pixels_per_width: float = 15.0,
    width_percentile: float = 7.5,
    min_um_per_px: float = 1.0,
    max_um_per_px: float = 100.0
) -> Tuple[float, float]:
    """
    Estimate channel width and auto-tune raster resolution.
    
    Uses a coarse distance transform to estimate local channel widths,
    then selects um_per_px so the narrowest channel has approximately
    target_pixels_per_width pixels across it, with a hard minimum of 10 pixels.
    
    Args:
        polygon: Shapely polygon to analyze
        coarse_um_per_px: Resolution for coarse rasterization (default: 30 µm)
        target_pixels_per_width: Target pixels across narrowest channel (default: 15.0)
        width_percentile: Percentile of widths to use (lower = more conservative, default: 7.5)
        min_um_per_px: Minimum allowed resolution (default: 1.0 µm/pixel)
        max_um_per_px: Maximum allowed resolution (default: 100.0 µm/pixel)
    
    Returns:
        Tuple of (estimated_narrowest_width_um, recommended_um_per_px)
    """
    logger.debug("estimate_channel_width_and_resolution: entry, coarse_um_per_px=%.2f",
                 coarse_um_per_px)
    start = time.time()
    
    # Rasterize polygon coarsely
    try:
        binary_img, transform = rasterize_polygon(polygon, um_per_px=coarse_um_per_px, padding=10.0)
        logger.debug("estimate_channel_width_and_resolution: coarse raster shape: %s", binary_img.shape)
    except Exception as e:
        logger.warning("estimate_channel_width_and_resolution: coarse rasterization failed: %s, using fallback", e)
        # Fallback: estimate from polygon bounds
        bounds = polygon.bounds
        width_um = bounds[2] - bounds[0]
        height_um = bounds[3] - bounds[1]
        # Assume narrowest dimension is roughly channel width
        min_dim = min(width_um, height_um)
        estimated_width = min_dim * 0.1  # Conservative estimate
        um_per_px = estimated_width / target_pixels_per_width
        um_per_px = max(min_um_per_px, min(max_um_per_px, um_per_px))
        logger.info("estimate_channel_width_and_resolution: using fallback, estimated_width=%.2f µm, um_per_px=%.2f",
                   estimated_width, um_per_px)
        return estimated_width, um_per_px
    
    # Compute distance transform (distance to boundary)
    logger.debug("estimate_channel_width_and_resolution: computing distance transform")
    try:
        # Distance transform on the filled region (distance to nearest boundary)
        dist_transform = ndimage.distance_transform_edt(binary_img)
        logger.debug("estimate_channel_width_and_resolution: distance transform complete, max distance: %.2f pixels",
                     dist_transform.max())
    except Exception as e:
        logger.error("estimate_channel_width_and_resolution: distance transform failed: %s", e, exc_info=True)
        # Fallback
        bounds = polygon.bounds
        width_um = bounds[2] - bounds[0]
        height_um = bounds[3] - bounds[1]
        min_dim = min(width_um, height_um)
        estimated_width = min_dim * 0.1
        um_per_px = estimated_width / target_pixels_per_width
        um_per_px = max(min_um_per_px, min(max_um_per_px, um_per_px))
        return estimated_width, um_per_px
    
    # Estimate local channel width ≈ 2 × distance_to_boundary
    # Only consider interior pixels (distance > 1 pixel to avoid boundary artifacts)
    interior_mask = dist_transform > 1.0
    if np.sum(interior_mask) == 0:
        logger.warning("estimate_channel_width_and_resolution: no interior pixels found, using fallback")
        bounds = polygon.bounds
        width_um = bounds[2] - bounds[0]
        height_um = bounds[3] - bounds[1]
        min_dim = min(width_um, height_um)
        estimated_width = min_dim * 0.1
        um_per_px = estimated_width / target_pixels_per_width
        um_per_px = max(min_um_per_px, min(max_um_per_px, um_per_px))
        return estimated_width, um_per_px
    
    # Local width in pixels: 2 × distance to boundary
    widths_px = 2 * dist_transform[interior_mask]
    widths_um = widths_px * coarse_um_per_px
    
    # Use low percentile to avoid outliers (e.g., junctions, wide regions)
    percentile_value = np.percentile(widths_um, width_percentile)
    estimated_narrowest_width_um = percentile_value
    
    logger.debug("estimate_channel_width_and_resolution: width stats: min=%.2f, %.1f%%=%.2f, max=%.2f µm",
                 widths_um.min(), width_percentile, percentile_value, widths_um.max())
    
    # Choose um_per_px so narrowest channel has target_pixels_per_width pixels
    recommended_um_per_px = estimated_narrowest_width_um / target_pixels_per_width
    
    # Enforce hard minimum of 10 pixels across narrowest channel
    # This ensures skeleton stability even for very narrow channels
    min_pixels_per_width = 10.0
    max_um_per_px_for_min_pixels = estimated_narrowest_width_um / min_pixels_per_width
    recommended_um_per_px = min(recommended_um_per_px, max_um_per_px_for_min_pixels)
    
    # Clamp to reasonable bounds
    recommended_um_per_px = max(min_um_per_px, min(max_um_per_px, recommended_um_per_px))
    
    # Verify final pixel count
    final_pixels_per_width = estimated_narrowest_width_um / recommended_um_per_px
    logger.debug("estimate_channel_width_and_resolution: final resolution gives %.1f pixels across narrowest channel",
                 final_pixels_per_width)
    
    elapsed = time.time() - start
    logger.info("estimate_channel_width_and_resolution: narrowest width=%.2f µm (%.1f%%ile), recommended um_per_px=%.2f (%.1fs)",
               estimated_narrowest_width_um, width_percentile, recommended_um_per_px, elapsed)
    
    return estimated_narrowest_width_um, recommended_um_per_px


def pixel_to_coords(
    pixel: Tuple[int, int],
    transform: Dict[str, float]
) -> Tuple[float, float]:
    """Convert pixel coordinates to original coordinate space."""
    row, col = pixel
    x = transform['origin_x'] + (col * transform['um_per_px'])
    y = transform['origin_y'] + (row * transform['um_per_px'])
    return (x, y)


def coords_to_pixel(
    x: float,
    y: float,
    transform: Dict[str, float]
) -> Tuple[int, int]:
    """Convert original coordinates to pixel coordinates."""
    col = int((x - transform['origin_x']) / transform['um_per_px'])
    row = int((y - transform['origin_y']) / transform['um_per_px'])
    return (row, col)


def skeletonize_polygon(
    polygon: Polygon,
    um_per_px: float,
    L_spur_cutoff: float,
    simplify_tolerance: Optional[float] = None
) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Skeletonize a polygon and convert to a NetworkX graph.
    
    Args:
        polygon: Shapely polygon to skeletonize
        um_per_px: Resolution for rasterization (microns per pixel) (required)
        L_spur_cutoff: Maximum length in microns for spur pruning. Spurs shorter than
                      this that end at junctions (degree >= 3) will be removed.
        simplify_tolerance: Tolerance for simplifying centerlines (in original units).
                           If None, auto-computed as 0.5 * um_per_px.
    
    Returns:
        Tuple of (skeleton_graph, metadata) where:
        - skeleton_graph: NetworkX graph with nodes as (x, y) tuples
        - metadata: Transform and processing info
    """
    # Auto-compute simplify_tolerance from um_per_px if not specified
    # simplify_tolerance should be <= 0.5 * um_per_px to avoid over-simplification
    if simplify_tolerance is None:
        simplify_tolerance = 0.5 * um_per_px
        logger.debug("skeletonize_polygon: auto-computed simplify_tolerance=%.3f from um_per_px=%.3f",
                     simplify_tolerance, um_per_px)
    elif simplify_tolerance > 0.5 * um_per_px:
        logger.warning("skeletonize_polygon: simplify_tolerance=%.3f exceeds 0.5*um_per_px=%.3f, "
                      "may cause over-simplification", simplify_tolerance, 0.5 * um_per_px)
    
    logger.debug("skeletonize_polygon: entry, um_per_px=%.3f, simplify_tolerance=%.3f",
                 um_per_px, simplify_tolerance)
    start_total = time.time()
    
    # Rasterize polygon
    logger.debug("skeletonize_polygon: calling rasterize_polygon")
    try:
        binary_img, transform = rasterize_polygon(polygon, um_per_px=um_per_px)
        logger.debug("skeletonize_polygon: rasterization complete, image shape: %s", binary_img.shape)
    except Exception as e:
        logger.error("skeletonize_polygon: rasterization failed: %s", e, exc_info=True)
        raise
    
    # Crop to tight mask bounds before skeletonize (plus small margin)
    start_crop = time.time()
    logger.debug("skeletonize_polygon: finding bounding box of filled pixels")
    try:
        # Find bounding box of filled pixels
        filled_rows, filled_cols = np.where(binary_img)
        logger.debug("skeletonize_polygon: found %d filled pixels", len(filled_rows))
    except Exception as e:
        logger.error("skeletonize_polygon: failed to find filled pixels: %s", e, exc_info=True)
        raise
    
    if len(filled_rows) == 0:
        # No filled pixels - return empty graph
        logger.info("Empty raster - no filled pixels")
        logger.debug("skeletonize_polygon: returning empty graph")
        return nx.Graph(), transform
    
    # Get tight bounds
    try:
        min_row_raw = int(filled_rows.min())
        max_row_raw = int(filled_rows.max())
        min_col_raw = int(filled_cols.min())
        max_col_raw = int(filled_cols.max())
        logger.debug("skeletonize_polygon: raw bounds: rows [%d, %d], cols [%d, %d]",
                     min_row_raw, max_row_raw, min_col_raw, max_col_raw)
        
        min_row = max(0, min_row_raw - 2)  # 2 pixel margin
        max_row = min(binary_img.shape[0], max_row_raw + 3)  # 3 pixel margin
        min_col = max(0, min_col_raw - 2)  # 2 pixel margin
        max_col = min(binary_img.shape[1], max_col_raw + 3)  # 3 pixel margin
        logger.debug("skeletonize_polygon: adjusted bounds: rows [%d, %d), cols [%d, %d)",
                     min_row, max_row, min_col, max_col)
        
        # Crop the image
        cropped_img = binary_img[min_row:max_row, min_col:max_col]
        logger.debug("skeletonize_polygon: cropped image shape: %s", cropped_img.shape)
    except Exception as e:
        logger.error("skeletonize_polygon: failed to crop image: %s", e, exc_info=True)
        raise
    
    # Adjust transform to account for cropping
    # pixel_to_coords: x = origin_x + (col * um_per_px), y = origin_y + (row * um_per_px)
    # So row offset directly affects y, and col offset directly affects x
    logger.debug("skeletonize_polygon: adjusting transform for crop")
    try:
        col_offset_in_coords = min_col * transform['um_per_px']
        row_offset_in_coords = min_row * transform['um_per_px']
        logger.debug("skeletonize_polygon: offsets: col=%.3f µm, row=%.3f µm",
                     col_offset_in_coords, row_offset_in_coords)
        
        old_origin_x = transform['origin_x']
        old_origin_y = transform['origin_y']
        # Update transform origin
        transform['origin_x'] = transform['origin_x'] + col_offset_in_coords
        transform['origin_y'] = transform['origin_y'] + row_offset_in_coords
        transform['img_width'] = cropped_img.shape[1]
        transform['img_height'] = cropped_img.shape[0]
        logger.debug("skeletonize_polygon: transform origin updated: (%.3f, %.3f) -> (%.3f, %.3f)",
                     old_origin_x, old_origin_y, transform['origin_x'], transform['origin_y'])
    except Exception as e:
        logger.error("skeletonize_polygon: failed to adjust transform: %s", e, exc_info=True)
        raise
    
    crop_time = time.time() - start_crop
    logger.info(f"Cropped to tight bounds: ({cropped_img.shape[0]}, {cropped_img.shape[1]}) in {crop_time:.2f}s")
    
    # Save binary image before skeletonization if in DEBUG mode
    saved_image_path = None
    if logger.isEnabledFor(logging.DEBUG):
        try:
            debug_output_dir = Path("debug_output")
            debug_output_dir.mkdir(exist_ok=True)
            output_path = debug_output_dir / f"binary_before_skeletonize_{int(time.time()*1000)}.png"
            # Convert boolean array to uint8 for saving (0=black, 255=white)
            img_to_save = Image.fromarray((cropped_img.astype(np.uint8) * 255), mode='L')
            img_to_save.save(output_path)
            saved_image_path = output_path
            logger.debug("skeletonize_polygon: saved binary image before skeletonization to %s", output_path)
        except Exception as e:
            logger.warning("skeletonize_polygon: failed to save binary image before skeletonization: %s", e)
    
    # Use the cropped image for skeletonization
    img = cropped_img
    
    # Skeletonize using scikit-image: medial_axis (default) with thin as fallback
    logger.info("Skeletonizing binary image using medial_axis (default), thin as fallback")
    
    # Validate and log image properties before skeletonization
    assert img.ndim == 2, f"Expected 2D image for skeletonize, got {img.ndim}D with shape {img.shape}"
    
    # Log detailed pre-skeletonization diagnostics
    logger.debug("skeletonize_polygon: pre-skeletonize diagnostics: shape=%s, dtype=%s, contiguous=%s",
                 img.shape, img.dtype, img.flags['C_CONTIGUOUS'])
    
    # Check unique values (catch uint8 0/255, floats, noise)
    unique_vals = np.unique(img)
    logger.debug("skeletonize_polygon: pre-skeletonize unique_count=%d, sample=%s%s",
                 len(unique_vals), 
                 unique_vals[:10] if len(unique_vals) <= 10 else str(unique_vals[:10]) + '...',
                 ' (all values)' if len(unique_vals) <= 10 else '')
    
    # Force true binary bool + contiguous for optimal performance
    img_binary = np.ascontiguousarray(img.astype(bool))
    
    # 1. Pre-clean the binary mask before medial axis
    logger.debug("Pre-cleaning binary mask: morphological operations")
    # Morphological closing: scale radius with um_per_px to maintain physical size
    # Target: ~2 µm physical radius for closing
    closing_radius_um = 2.0  # micrometers
    closing_radius_px = max(1, int(closing_radius_um / um_per_px))
    try:
        img_binary = binary_closing(img_binary, disk(closing_radius_px))
        logger.debug("Applied binary closing with radius %d px (%.2f µm physical)", 
                    closing_radius_px, closing_radius_px * um_per_px)
    except Exception as e:
        logger.warning("Binary closing failed: %s", e)
    
    # Optional opening: scale radius with um_per_px to maintain physical size
    # Target: ~1 µm physical radius for opening
    opening_radius_um = 1.0  # micrometers
    opening_radius_px = max(1, int(opening_radius_um / um_per_px))
    try:
        img_binary = binary_opening(img_binary, disk(opening_radius_px))
        logger.debug("Applied binary opening with radius %d px (%.2f µm physical)", 
                    opening_radius_px, opening_radius_px * um_per_px)
    except Exception as e:
        logger.warning("Binary opening failed: %s", e)
    
    filled = int(img_binary.sum())
    total_pixels = img_binary.size
    fill_ratio = filled / total_pixels if total_pixels > 0 else 0.0
    
    logger.info("skeletonize_polygon: pre-skeletonize: shape=%s, dtype=%s (cast to bool), "
                "filled=%d/%d (%.3f%%), contiguous=%s",
                img_binary.shape, img_binary.dtype, filled, total_pixels, 
                fill_ratio * 100, img_binary.flags['C_CONTIGUOUS'])
    
    # Try skeletonization algorithms: medial_axis (default) with thin as fallback
    # Each algorithm has a 60-second timeout
    ALGORITHM_TIMEOUT = 60.0  # seconds
    skeleton_img = None
    skeleton_time = None
    algo_name = None
    
    def run_medial_axis(img):
        """Run medial_axis algorithm."""
        result = medial_axis(img)
        return result[0] if isinstance(result, tuple) else result
    
    def run_thin(img):
        """Run thin algorithm."""
        return thin(img)
    
    # Try medial_axis first (default algorithm)
    logger.info("skeletonize_polygon: trying medial_axis (default algorithm) from %s.%s (timeout: %.1fs)",
                medial_axis.__module__ if hasattr(medial_axis, '__module__') else 'unknown',
                medial_axis.__name__ if hasattr(medial_axis, '__name__') else str(medial_axis),
                ALGORITHM_TIMEOUT)
    start_algo = time.perf_counter()
    try:
        # Run with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_medial_axis, img_binary)
            try:
                algo_img = future.result(timeout=ALGORITHM_TIMEOUT)
                algo_time = time.perf_counter() - start_algo
                skeleton_pixels_count = int(np.sum(algo_img))
                
                logger.info("skeletonize_polygon: medial_axis took %.4f s for %s, produced %d skeleton pixels",
                           algo_time, img_binary.shape, skeleton_pixels_count)
                logger.debug("skeletonize_polygon: medial_axis output shape: %s, dtype: %s",
                            algo_img.shape, algo_img.dtype)
                
                # Make it 1px-wide and break many 2x2 artifacts
                logger.debug("skeletonize_polygon: applying thin() to medial_axis result")
                try:
                    skel = thin(algo_img)
                    algo_img = skel
                    skeleton_pixels_count_after_thin = int(np.sum(algo_img))
                    logger.info("skeletonize_polygon: after thin(), skeleton has %d pixels (was %d)",
                               skeleton_pixels_count_after_thin, skeleton_pixels_count)
                except Exception as e:
                    logger.warning("skeletonize_polygon: thin() failed on medial_axis result: %s", e)
                    # Continue with original medial_axis result
                
                skeleton_img = algo_img
                skeleton_time = algo_time
                algo_name = 'medial_axis'
                
                # Save result image if in DEBUG mode
                if logger.isEnabledFor(logging.DEBUG):
                    try:
                        debug_output_dir = Path("debug_output")
                        debug_output_dir.mkdir(exist_ok=True)
                        output_path = debug_output_dir / f"skeleton_medial_axis_{int(time.time()*1000)}.png"
                        img_to_save = Image.fromarray((algo_img.astype(np.uint8) * 255), mode='L')
                        img_to_save.save(output_path)
                        logger.debug("skeletonize_polygon: saved medial_axis result to %s", output_path)
                    except Exception as e:
                        logger.warning("skeletonize_polygon: failed to save medial_axis result: %s", e)
                        
            except FutureTimeoutError:
                elapsed = time.perf_counter() - start_algo
                logger.error("skeletonize_polygon: medial_axis algorithm timed out after %.1f s (limit: %.1f s)",
                           elapsed, ALGORITHM_TIMEOUT)
                logger.info("skeletonize_polygon: falling back to thin algorithm")
                # Cancel the future (best effort)
                future.cancel()
                
    except Exception as e:
        logger.error("skeletonize_polygon: medial_axis algorithm failed: %s", e, exc_info=True)
        logger.info("skeletonize_polygon: falling back to thin algorithm")
    
    # Fallback to thin if medial_axis failed or timed out
    if skeleton_img is None:
        logger.info("skeletonize_polygon: trying thin (fallback algorithm) from %s.%s (timeout: %.1fs)",
                    thin.__module__ if hasattr(thin, '__module__') else 'unknown',
                    thin.__name__ if hasattr(thin, '__name__') else str(thin),
                    ALGORITHM_TIMEOUT)
        start_algo = time.perf_counter()
        try:
            # Run with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_thin, img_binary)
                try:
                    algo_img = future.result(timeout=ALGORITHM_TIMEOUT)
                    algo_time = time.perf_counter() - start_algo
                    skeleton_pixels_count = int(np.sum(algo_img))
                    
                    logger.info("skeletonize_polygon: thin took %.4f s for %s, produced %d skeleton pixels",
                               algo_time, img_binary.shape, skeleton_pixels_count)
                    logger.debug("skeletonize_polygon: thin output shape: %s, dtype: %s",
                                algo_img.shape, algo_img.dtype)
                    
                    skeleton_img = algo_img
                    skeleton_time = algo_time
                    algo_name = 'thin'
                    
                    # Save result image if in DEBUG mode
                    if logger.isEnabledFor(logging.DEBUG):
                        try:
                            debug_output_dir = Path("debug_output")
                            debug_output_dir.mkdir(exist_ok=True)
                            output_path = debug_output_dir / f"skeleton_thin_{int(time.time()*1000)}.png"
                            img_to_save = Image.fromarray((algo_img.astype(np.uint8) * 255), mode='L')
                            img_to_save.save(output_path)
                            logger.debug("skeletonize_polygon: saved thin result to %s", output_path)
                        except Exception as e:
                            logger.warning("skeletonize_polygon: failed to save thin result: %s", e)
                            
                except FutureTimeoutError:
                    elapsed = time.perf_counter() - start_algo
                    logger.error("skeletonize_polygon: thin algorithm timed out after %.1f s (limit: %.1f s)",
                               elapsed, ALGORITHM_TIMEOUT)
                    # Cancel the future (best effort)
                    future.cancel()
                    raise RuntimeError(
                        f"All skeletonization algorithms timed out or failed. "
                        f"medial_axis and thin both exceeded {ALGORITHM_TIMEOUT}s timeout."
                    )
                    
        except RuntimeError:
            # Re-raise timeout errors
            raise
        except Exception as e:
            logger.error("skeletonize_polygon: thin algorithm failed: %s", e, exc_info=True)
            raise RuntimeError("All skeletonization algorithms failed (medial_axis and thin)")
    
    if skeleton_img is None:
        raise RuntimeError("All skeletonization algorithms failed")
    
    logger.info(f"  Using {algo_name} result: {skeleton_time:.4f}s for {img_binary.shape}")
    
    logger.debug("skeletonize_polygon: skeletonization complete, output shape: %s, dtype: %s",
                 skeleton_img.shape, skeleton_img.dtype)
    
    # Find skeleton pixels (non-zero)
    logger.debug("skeletonize_polygon: finding skeleton pixels")
    start = time.time()
    try:
        skeleton_pixels = np.argwhere(skeleton_img)
        find_time = time.time() - start
        logger.info(f"  Finding skeleton pixels: {find_time:.2f}s")
        logger.debug("skeletonize_polygon: found %d skeleton pixels", len(skeleton_pixels))
    except Exception as e:
        logger.error("skeletonize_polygon: failed to find skeleton pixels: %s", e, exc_info=True)
        raise
    
    if len(skeleton_pixels) == 0:
        # Empty skeleton - return empty graph
        logger.info("Empty skeleton")
        logger.debug("skeletonize_polygon: returning empty graph")
        return nx.Graph(), transform
    
    logger.info(f"Skeleton: {len(skeleton_pixels)} skeleton pixels in {skeleton_time + find_time:.2f}s total")
    
    # Convert skeleton pixels to coordinate space
    # After np.flipud, the image array is flipped vertically, so row indices are inverted.
    # pixel_to_coords assumes row 0 = minimum y (origin_y). After flipud, what was row (height-1)
    # (top/max y) is now row 0, so we need to invert: use (img_height - 1 - row)
    logger.debug("skeletonize_polygon: converting %d skeleton pixels to coordinates",
                 len(skeleton_pixels))
    skeleton_coords = []
    img_height = skeleton_img.shape[0]
    try:
        for i, pixel in enumerate(skeleton_pixels):
            if i % 10000 == 0 and i > 0:
                logger.debug("skeletonize_polygon: converted %d/%d pixels", i, len(skeleton_pixels))
            row, col = pixel[0], pixel[1]
            # Invert row to account for np.flipud
            inverted_row = img_height - 1 - row
            coords = pixel_to_coords((inverted_row, col), transform)
            skeleton_coords.append(coords)
        logger.debug("skeletonize_polygon: coordinate conversion complete")
    except Exception as e:
        logger.error("skeletonize_polygon: failed to convert pixels to coordinates: %s", 
                     e, exc_info=True)
        raise
    
    # Build graph from skeleton pixels
    # Each pixel becomes a node, connected to its 8-connected neighbors
    logger.debug("skeletonize_polygon: building NetworkX graph from skeleton")
    G = nx.Graph()
    
    # Add all skeleton pixels as nodes
    logger.debug("skeletonize_polygon: adding %d nodes to graph", len(skeleton_pixels))
    start = time.time()
    pixel_to_node = {}
    try:
        for i, pixel in enumerate(skeleton_pixels):
            if i % 10000 == 0 and i > 0:
                logger.debug("skeletonize_polygon: added %d/%d nodes", i, len(skeleton_pixels))
            row, col = pixel[0], pixel[1]
            # Invert row to account for np.flipud
            inverted_row = img_height - 1 - row
            coords = pixel_to_coords((inverted_row, col), transform)
            node_id = i
            G.add_node(node_id, xy=coords)
            pixel_to_node[tuple(pixel)] = node_id
        node_time = time.time() - start
        logger.info(f"  Adding nodes: {node_time:.2f}s")
        logger.debug("skeletonize_polygon: added %d nodes", len(G.nodes()))
    except Exception as e:
        logger.error("skeletonize_polygon: failed to add nodes: %s", e, exc_info=True)
        raise
    
    # Connect neighboring pixels (8-connected)
    logger.debug("skeletonize_polygon: adding edges (8-connected neighbors)")
    start = time.time()
    edges_added = 0
    try:
        for idx, pixel in enumerate(skeleton_pixels):
            if idx % 10000 == 0 and idx > 0:
                logger.debug("skeletonize_polygon: processed %d/%d pixels for edges", 
                            idx, len(skeleton_pixels))
            row, col = pixel[0], pixel[1]
            node_id = pixel_to_node[(row, col)]
            
            # Check 8 neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    # If diagonal neighbor, skip it when it would create a triangle via orthogonal pixels
                    if abs(dr) == 1 and abs(dc) == 1:
                        if (row + dr, col) in pixel_to_node or (row, col + dc) in pixel_to_node:
                            continue
                    
                    neighbor = (row + dr, col + dc)
                    if neighbor in pixel_to_node:
                        neighbor_id = pixel_to_node[neighbor]
                        if not G.has_edge(node_id, neighbor_id):
                            # Calculate distance in coordinate space
                            x1, y1 = G.nodes[node_id]['xy']
                            x2, y2 = G.nodes[neighbor_id]['xy']
                            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            G.add_edge(node_id, neighbor_id, weight=dist)
                            edges_added += 1
        edge_time = time.time() - start
        logger.info(f"  Adding edges: {edge_time:.2f}s")
        logger.debug("skeletonize_polygon: added %d edges", edges_added)
    except Exception as e:
        logger.error("skeletonize_polygon: failed to add edges: %s", e, exc_info=True)
        raise
    
    total_time = time.time() - start_total
    logger.info(f"Skeleton graph: {len(G.nodes())} nodes, {len(G.edges())} edges in {node_time + edge_time:.2f}s total")
    
    # Save skeleton image before pruning
    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("skeleton_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"skeleton_before_pruning_{timestamp}.png"
        # Convert boolean array to uint8 for saving (0=black, 255=white)
        img_to_save = Image.fromarray((skeleton_img.astype(np.uint8) * 255), mode='L')
        img_to_save.save(output_path)
        logger.info(f"Saved skeleton before pruning to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to save skeleton before pruning: {e}")
    
    # 2. Skeleton spur pruning: prune short spurs that end at junctions
    logger.debug("Pruning skeleton spurs: L_spur_cutoff = %.1f µm", L_spur_cutoff)
    
    spurs_pruned = 0
    max_iterations = 100  # Safety limit
    for iteration in range(max_iterations):
        # Find all endpoints E = {v | deg(v)=1}
        endpoints = [n for n in G.nodes() if G.degree(n) == 1]
        if not endpoints:
            break  # No more endpoints to process
        
        paths_to_remove = []  # List of (path_nodes, terminal_node) tuples
        
        for endpoint in endpoints:
            # Walk forward through degree-2 nodes
            path = [endpoint]
            current = endpoint
            path_length = 0.0
            terminal_node = None
            
            while True:
                # Get neighbors
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break  # No neighbors, stop
                
                # Find next node (for degree-1, only one neighbor; for degree-2, choose the one that's not prev)
                if len(path) == 1:
                    # First step: only one neighbor for degree-1
                    next_node = neighbors[0]
                else:
                    # Subsequent steps: choose neighbor that's not the previous node
                    prev_node = path[-2]
                    next_node = None
                    for nbr in neighbors:
                        if nbr != prev_node:
                            next_node = nbr
                            break
                    if next_node is None:
                        break
                
                # Add edge weight to path length
                if G.has_edge(current, next_node):
                    edge_weight = G[current][next_node].get('weight', 0.0)
                    path_length += edge_weight
                
                # Check if we've reached a terminal node (deg != 2)
                next_degree = G.degree(next_node)
                if next_degree != 2:
                    terminal_node = next_node
                    break
                
                # Continue walking
                path.append(next_node)
                current = next_node
            
            # If we reached a junction (deg >= 3) and path length <= L_spur_cutoff, mark for removal
            if terminal_node is not None:
                terminal_degree = G.degree(terminal_node)
                if terminal_degree >= 3 and path_length <= L_spur_cutoff and len(path) > 1:
                    paths_to_remove.append((path, terminal_node))
        
        if not paths_to_remove:
            break  # No more spurs to remove
        
        # Remove marked paths
        for path, terminal_node in paths_to_remove:
            # Remove all nodes in the path (keep terminal_node)
            for node in path:
                if node in G:
                    G.remove_node(node)
                    spurs_pruned += 1
        
        logger.debug("Pruning iteration %d: removed %d nodes from %d spurs",
                    iteration + 1, sum(len(p) for p, _ in paths_to_remove), len(paths_to_remove))
    
    logger.info(f"Pruned {spurs_pruned} spur pixels ({iteration+1} iterations, L_spur_cutoff={L_spur_cutoff:.1f} µm)")
    
    # Save skeleton image after pruning
    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("skeleton_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"skeleton_after_pruning_{timestamp}.png"
        
        # Reconstruct skeleton image from remaining graph nodes
        # Use the same dimensions as the original skeleton_img
        skeleton_img_after = np.zeros_like(skeleton_img, dtype=bool)
        
        # Convert each remaining node's coordinates back to pixel coordinates
        for node_id in G.nodes():
            x, y = G.nodes[node_id]['xy']
            # Note: coords_to_pixel returns (row, col), but we need to account for the flip
            # The original skeleton_img was flipped with np.flipud, so row 0 in the image
            # corresponds to max_y in coordinates. We need to invert the row.
            row, col = coords_to_pixel(x, y, transform)
            # Invert row to match the flipped image coordinate system
            img_height = skeleton_img.shape[0]
            inverted_row = img_height - 1 - row
            
            # Check bounds and set pixel
            if 0 <= inverted_row < skeleton_img.shape[0] and 0 <= col < skeleton_img.shape[1]:
                skeleton_img_after[inverted_row, col] = True
        
        # Convert boolean array to uint8 for saving (0=black, 255=white)
        img_to_save = Image.fromarray((skeleton_img_after.astype(np.uint8) * 255), mode='L')
        img_to_save.save(output_path)
        logger.info(f"Saved skeleton after pruning to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to save skeleton after pruning: {e}")
    
    logger.debug("skeletonize_polygon: complete, total time: %.2fs", total_time)
    
    # Return raw pixel graph (conversion to vector graph should be done by caller if needed)
    return G, transform


def extract_skeleton_paths(
    skeleton_graph: nx.Graph,
    simplify_tolerance: Optional[float] = None
) -> List[List[Tuple[float, float]]]:
    """
    Extract centerline paths from skeleton graph using edge-walk decomposition.
    
    Enumerates every edge exactly once by:
    1. Identifying "key nodes" where degree != 2 (endpoints + junctions)
    2. For each key node, for each neighbor, walk forward along degree-2 nodes
       until reaching another key node, recording that polyline and marking edges visited
    3. Handle any remaining unvisited edges (pure cycles where all nodes have degree==2)
       by picking any unvisited edge and walking the cycle until returning to start
    
    Args:
        skeleton_graph: NetworkX graph from skeletonization
        simplify_tolerance: Tolerance for simplifying paths (in original units).
                           If None, paths are not simplified.
    
    Returns:
        List of paths, where each path is a list of (x, y) coordinates
    """
    logger.debug("extract_skeleton_paths: entry, graph size: %d nodes, %d edges, simplify_tolerance=%.3f",
                 len(skeleton_graph), len(skeleton_graph.edges()), simplify_tolerance)
    
    if len(skeleton_graph) == 0:
        logger.debug("extract_skeleton_paths: empty graph, returning empty list")
        return []
    
    paths = []
    visited_edges = set()
    
    # Helper to make edge tuple (sorted for consistency)
    def edge_tuple(u, v):
        return tuple(sorted([u, v]))
    
    # Helper to walk along degree-2 nodes from a starting node
    def walk_path(start_node, first_neighbor, visited_set):
        """Walk from start_node through first_neighbor along degree-2 nodes until reaching a key node."""
        path_nodes = [start_node, first_neighbor]
        current = first_neighbor
        prev = start_node
        
        # Walk forward along degree-2 nodes
        while skeleton_graph.degree(current) == 2:
            # Find the next node (the neighbor that's not prev)
            neighbors = list(skeleton_graph.neighbors(current))
            if len(neighbors) != 2:
                break  # Safety check
            next_node = neighbors[1] if neighbors[0] == prev else neighbors[0]
            path_nodes.append(next_node)
            prev = current
            current = next_node
            
            # Check if we've already visited this edge (cycle detected)
            edge = edge_tuple(prev, current)
            if edge in visited_set:
                break
        
        return path_nodes, current
    
    # Step 1: Identify key nodes (degree != 2)
    key_nodes = [n for n in skeleton_graph.nodes() if skeleton_graph.degree(n) != 2]
    logger.debug("extract_skeleton_paths: found %d key nodes (degree != 2)", len(key_nodes))
    
    # Step 2: For each key node, for each neighbor, walk forward until reaching another key node
    for key_node in key_nodes:
        for neighbor in skeleton_graph.neighbors(key_node):
            edge = edge_tuple(key_node, neighbor)
            
            # Skip if we've already visited this edge
            if edge in visited_edges:
                continue
            
            # Walk forward from this neighbor
            path_nodes, end_node = walk_path(key_node, neighbor, visited_edges)
            
            # Mark all edges in this path as visited
            for i in range(len(path_nodes) - 1):
                visited_edges.add(edge_tuple(path_nodes[i], path_nodes[i+1]))
            
            # Convert to coordinates
            path_coords = [skeleton_graph.nodes[n]['xy'] for n in path_nodes]
            paths.append(path_coords)
            logger.debug("extract_skeleton_paths: added path from key node %d to %d (%d nodes)",
                        key_node, end_node, len(path_nodes))
    
    # Step 3: Handle remaining unvisited edges (pure cycles where all nodes have degree==2)
    all_edges = set(edge_tuple(u, v) for u, v in skeleton_graph.edges())
    remaining_edges = all_edges - visited_edges
    
    if remaining_edges:
        logger.debug("extract_skeleton_paths: found %d remaining unvisited edges (pure cycles)", 
                     len(remaining_edges))
        
        while remaining_edges:
            # Pick any unvisited edge
            edge = remaining_edges.pop()
            u, v = edge
            
            # Mark this edge as visited
            visited_edges.add(edge)
            
            # Start walking the cycle from u -> v
            path_nodes = [u, v]
            current = v
            prev = u
            cycle_complete = False
            
            # Walk until we return to the start node
            max_steps = len(skeleton_graph)  # Safety limit
            steps = 0
            while not cycle_complete and steps < max_steps:
                steps += 1
                
                # Find next node (neighbor that's not prev)
                neighbors = list(skeleton_graph.neighbors(current))
                if len(neighbors) != 2:
                    # Shouldn't happen in a pure cycle, but handle gracefully
                    logger.warning("extract_skeleton_paths: cycle node %d has degree %d (expected 2)", 
                                 current, len(neighbors))
                    break
                
                next_node = neighbors[1] if neighbors[0] == prev else neighbors[0]
                edge_to_next = edge_tuple(current, next_node)
                
                # Mark this edge as visited
                visited_edges.add(edge_to_next)
                remaining_edges.discard(edge_to_next)
                
                # Check if we've completed the cycle (returned to start)
                if next_node == u:
                    cycle_complete = True
                    # Close the cycle by adding start node at end
                    path_nodes.append(u)
                    break
                
                path_nodes.append(next_node)
                prev = current
                current = next_node
            
            if cycle_complete:
                # Convert to coordinates (including closing the cycle)
                path_coords = [skeleton_graph.nodes[n]['xy'] for n in path_nodes]
                paths.append(path_coords)
                logger.debug("extract_skeleton_paths: added cycle path with %d nodes", len(path_nodes))
            else:
                logger.warning("extract_skeleton_paths: failed to complete cycle starting from edge (%d, %d)", u, v)
    
    logger.debug("extract_skeleton_paths: extracted %d paths before simplification", len(paths))
    
    # Simplify paths using corner-preserving collinear removal (if tolerance specified)
    if simplify_tolerance is not None and simplify_tolerance > 0:
        logger.debug("extract_skeleton_paths: simplifying %d paths with corner-preserving method", len(paths))
        simplified_paths = []
        try:
            for idx, path in enumerate(paths):
                if idx % 10 == 0 and idx > 0:
                    logger.debug("extract_skeleton_paths: simplified %d/%d paths", idx, len(paths))
                if len(path) < 3:
                    # Too short to simplify, keep as-is
                    simplified_paths.append(path)
                    continue
                try:
                    # Use corner-preserving collinear simplification (angle threshold ≈ 3 degrees)
                    simplified = simplify_path_collinear(path, angle_threshold_deg=3.0)
                    simplified_paths.append(simplified)
                except Exception as e:
                    logger.warning("extract_skeleton_paths: failed to simplify path %d: %s, using original", idx, e)
                    simplified_paths.append(path)
                    continue
        except Exception as e:
            logger.error("extract_skeleton_paths: failed during path simplification: %s", e, exc_info=True)
            raise
        
        result = simplified_paths if simplified_paths else paths
    else:
        logger.debug("extract_skeleton_paths: skipping simplification (tolerance not specified or zero)")
        result = paths
    
    logger.debug("extract_skeleton_paths: exit, returning %d paths, visited %d/%d edges", 
                 len(result), len(visited_edges), len(all_edges))
    return result


def build_vector_centerline_graph(
    skeleton_graph: nx.Graph,
    um_per_px: float,
    simplify_tolerance: Optional[float] = None,
    resample_spacing_um: Optional[float] = None,
    merge_distance_um: Optional[float] = None,
    polygon: Optional[Polygon] = None
) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Convert pixel-skeleton graph to vector centerline graph.
    
    Process:
    1. Extract skeleton paths (polylines) from pixel graph
    2. Simplify each polyline using Douglas-Peucker
    3. Resample at fixed spacing to get evenly spaced nodes
    4. Rebuild NetworkX graph with resampled points
    5. Optionally merge nearby endpoints/junctions
    
    Args:
        skeleton_graph: Pixel-skeleton NetworkX graph (one node per skeleton pixel)
        um_per_px: Resolution (microns per pixel)
        simplify_tolerance: Tolerance for path simplification. If None, uses 0.5 * um_per_px
        resample_spacing_um: Spacing for resampling nodes. If None, uses 1.0 * um_per_px
        merge_distance_um: Distance for merging nearby endpoints/junctions. If None, uses 2.0 * um_per_px
        polygon: Optional polygon for validation (if provided, nodes are constrained to stay inside)
    
    Returns:
        Tuple of (vector_graph, metadata_dict) where:
        - vector_graph: NetworkX graph with resampled nodes and edges
        - metadata_dict: Contains original_paths, simplified_paths, etc.
    """
    from shapely.geometry import LineString, Point
    
    logger.info("build_vector_centerline_graph: converting pixel graph to vector centerline graph")
    
    if len(skeleton_graph) == 0:
        logger.warning("build_vector_centerline_graph: empty skeleton graph")
        return nx.Graph(), {'original_paths': [], 'simplified_paths': []}
    
    # Set default parameters
    if simplify_tolerance is None:
        simplify_tolerance = 0.5 * um_per_px
    if resample_spacing_um is None:
        resample_spacing_um = 1.0 * um_per_px
    if merge_distance_um is None:
        merge_distance_um = 2.0 * um_per_px
    
    logger.debug("build_vector_centerline_graph: parameters: simplify_tolerance=%.3f, "
                 "resample_spacing=%.3f, merge_distance=%.3f",
                 simplify_tolerance, resample_spacing_um, merge_distance_um)
    
    # Step 1: Extract skeleton paths
    logger.debug("build_vector_centerline_graph: extracting skeleton paths")
    original_paths = extract_skeleton_paths(skeleton_graph, simplify_tolerance=None)
    logger.info("build_vector_centerline_graph: extracted %d original paths", len(original_paths))
    
    if not original_paths:
        logger.warning("build_vector_centerline_graph: no paths extracted")
        return nx.Graph(), {'original_paths': [], 'simplified_paths': []}
    
    # Step 2: Simplify each path
    logger.debug("build_vector_centerline_graph: simplifying %d paths", len(original_paths))
    simplified_paths = []
    for idx, path in enumerate(original_paths):
        if len(path) < 2:
            continue
        try:
            line = LineString(path)
            simplified = line.simplify(simplify_tolerance, preserve_topology=False)
            if simplified.geom_type == 'LineString':
                simplified_paths.append(list(simplified.coords))
            elif simplified.geom_type == 'MultiLineString':
                # Take longest segment
                longest = max(simplified.geoms, key=lambda g: g.length)
                simplified_paths.append(list(longest.coords))
            else:
                logger.warning("build_vector_centerline_graph: path %d simplified to unexpected type: %s",
                              idx, simplified.geom_type)
                simplified_paths.append(path)  # Fallback to original
        except Exception as e:
            logger.warning("build_vector_centerline_graph: failed to simplify path %d: %s", idx, e)
            simplified_paths.append(path)  # Fallback to original
    
    logger.info("build_vector_centerline_graph: simplified to %d paths", len(simplified_paths))
    
    # Step 3: Resample each path at fixed spacing
    logger.debug("build_vector_centerline_graph: resampling paths at spacing=%.3f", resample_spacing_um)
    resampled_paths = []
    for path_idx, path_coords in enumerate(simplified_paths):
        if len(path_coords) < 2:
            continue
        
        line = LineString(path_coords)
        length = line.length
        
        if length < resample_spacing_um:
            # Path too short, keep original points
            resampled_paths.append(path_coords)
            continue
        
        # Resample at fixed spacing
        num_samples = max(2, int(np.ceil(length / resample_spacing_um)) + 1)
        distances = np.linspace(0, length, num_samples)
        
        resampled_coords = []
        for dist in distances:
            point = line.interpolate(dist)
            x, y = point.x, point.y
            
            # Optional: Project back onto polygon boundary if provided
            if polygon is not None:
                point_obj = Point(x, y)
                if not polygon.contains(point_obj):
                    # Find nearest point on polygon boundary
                    nearest = polygon.exterior.interpolate(polygon.exterior.project(point_obj))
                    x, y = nearest.x, nearest.y
            
            resampled_coords.append((x, y))
        
        resampled_paths.append(resampled_coords)
    
    logger.info("build_vector_centerline_graph: resampled to %d paths", len(resampled_paths))
    
    # Step 4: Build new NetworkX graph
    logger.debug("build_vector_centerline_graph: building vector graph")
    vector_graph = nx.Graph()
    node_counter = itertools.count()
    node_id_map = {}  # Map (x, y) -> node_id
    
    for path_idx, path_coords in enumerate(resampled_paths):
        if len(path_coords) < 2:
            continue
        
        # Add nodes and edges for this path
        prev_node_id = None
        prev_coord = None
        segment_start_idx = 0
        
        for coord_idx, coord in enumerate(path_coords):
            x, y = coord
            
            # Check if node already exists (for path intersections)
            node_id = node_id_map.get(coord)
            if node_id is None:
                node_id = next(node_counter)
                vector_graph.add_node(node_id, xy=(x, y))
                node_id_map[coord] = node_id
            
            # Add edge to previous node in path
            if prev_node_id is not None:
                # Store polyline geometry on edge - segment from prev_coord to coord
                if not vector_graph.has_edge(prev_node_id, node_id):
                    # Extract segment coordinates between prev and current
                    segment_coords = [prev_coord, coord]
                    
                    vector_graph.add_edge(prev_node_id, node_id, 
                                        polyline=segment_coords,
                                        path_index=path_idx)
            
            prev_node_id = node_id
            prev_coord = coord
    
    logger.info("build_vector_centerline_graph: built vector graph with %d nodes, %d edges",
                len(vector_graph), len(vector_graph.edges()))
    
    # Step 5: Merge nearby endpoints/junctions
    if merge_distance_um > 0:
        logger.debug("build_vector_centerline_graph: merging nearby nodes (distance=%.3f)", 
                     merge_distance_um)
        
        # Identify endpoints and junctions
        degrees = dict(vector_graph.degree())
        endpoints = [n for n in vector_graph.nodes() if degrees.get(n, 0) == 1]
        junctions = [n for n in vector_graph.nodes() if degrees.get(n, 0) >= 3]
        
        # Merge close endpoints
        if len(endpoints) > 1:
            merged_endpoints = {}
            used = set()
            cluster_id = 0
            
            for i, node1 in enumerate(endpoints):
                if node1 in used:
                    continue
                
                cluster = [node1]
                used.add(node1)
                xy1 = vector_graph.nodes[node1]['xy']
                point1 = Point(xy1)
                
                for node2 in endpoints[i+1:]:
                    if node2 in used:
                        continue
                    
                    xy2 = vector_graph.nodes[node2]['xy']
                    point2 = Point(xy2)
                    dist = point1.distance(point2)
                    
                    if dist <= merge_distance_um:
                        cluster.append(node2)
                        used.add(node2)
                
                if len(cluster) > 1:
                    merged_endpoints[cluster_id] = cluster
                    cluster_id += 1
            
            # Merge clusters: replace all nodes in cluster with centroid
            for cluster_nodes in merged_endpoints.values():
                if len(cluster_nodes) < 2:
                    continue
                
                # Compute centroid
                coords = [vector_graph.nodes[n]['xy'] for n in cluster_nodes]
                centroid_x = sum(c[0] for c in coords) / len(coords)
                centroid_y = sum(c[1] for c in coords) / len(coords)
                centroid = (centroid_x, centroid_y)
                
                # Keep first node, replace others
                keep_node = cluster_nodes[0]
                vector_graph.nodes[keep_node]['xy'] = centroid
                
                # Merge edges from other nodes to keep_node
                for node in cluster_nodes[1:]:
                    # Transfer edges
                    neighbors = list(vector_graph.neighbors(node))
                    for neighbor in neighbors:
                        if neighbor not in cluster_nodes:
                            if not vector_graph.has_edge(keep_node, neighbor):
                                edge_data = vector_graph.edges[node, neighbor]
                                vector_graph.add_edge(keep_node, neighbor, **edge_data)
                    vector_graph.remove_node(node)
                    # Remove from node_id_map
                    for coord, nid in list(node_id_map.items()):
                        if nid == node:
                            del node_id_map[coord]
            
            logger.info("build_vector_centerline_graph: merged %d endpoint clusters", 
                       len(merged_endpoints))
        
        # Merge close junctions
        if len(junctions) > 1:
            merged_junctions = {}
            used = set()
            cluster_id = 0
            
            for i, node1 in enumerate(junctions):
                if node1 in used:
                    continue
                
                cluster = [node1]
                used.add(node1)
                xy1 = vector_graph.nodes[node1]['xy']
                point1 = Point(xy1)
                
                for node2 in junctions[i+1:]:
                    if node2 in used:
                        continue
                    
                    xy2 = vector_graph.nodes[node2]['xy']
                    point2 = Point(xy2)
                    dist = point1.distance(point2)
                    
                    if dist <= merge_distance_um:
                        cluster.append(node2)
                        used.add(node2)
                
                if len(cluster) > 1:
                    merged_junctions[cluster_id] = cluster
                    cluster_id += 1
            
            # Merge clusters
            for cluster_nodes in merged_junctions.values():
                if len(cluster_nodes) < 2:
                    continue
                
                # Compute centroid
                coords = [vector_graph.nodes[n]['xy'] for n in cluster_nodes]
                centroid_x = sum(c[0] for c in coords) / len(coords)
                centroid_y = sum(c[1] for c in coords) / len(coords)
                centroid = (centroid_x, centroid_y)
                
                # Keep first node, replace others
                keep_node = cluster_nodes[0]
                vector_graph.nodes[keep_node]['xy'] = centroid
                
                # Merge edges
                for node in cluster_nodes[1:]:
                    neighbors = list(vector_graph.neighbors(node))
                    for neighbor in neighbors:
                        if neighbor not in cluster_nodes:
                            if not vector_graph.has_edge(keep_node, neighbor):
                                edge_data = vector_graph.edges[node, neighbor]
                                vector_graph.add_edge(keep_node, neighbor, **edge_data)
                    vector_graph.remove_node(node)
                    # Remove from node_id_map
                    for coord, nid in list(node_id_map.items()):
                        if nid == node:
                            del node_id_map[coord]
            
            logger.info("build_vector_centerline_graph: merged %d junction clusters",
                       len(merged_junctions))
    
    metadata = {
        'original_paths': original_paths,
        'simplified_paths': simplified_paths,
        'resampled_paths': resampled_paths,
        'simplify_tolerance': simplify_tolerance,
        'resample_spacing_um': resample_spacing_um,
        'merge_distance_um': merge_distance_um
    }
    
    logger.info("build_vector_centerline_graph: final graph has %d nodes, %d edges",
                len(vector_graph), len(vector_graph.edges()))
    
    return vector_graph, metadata


def cluster_junction_pixels(
    skeleton_graph: nx.Graph,
    um_per_px: float
) -> Dict[int, List[int]]:
    """
    Cluster junction pixels into connected components.
    
    Args:
        skeleton_graph: NetworkX graph from skeletonization
        um_per_px: Resolution (microns per pixel)
        
    Returns:
        Dict mapping cluster_id to list of node IDs in that cluster
    """
    # Compute degrees
    degrees = dict(skeleton_graph.degree())
    
    # Identify junction pixels (degree >= 3)
    junction_nodes = [n for n in skeleton_graph.nodes() if degrees.get(n, 0) >= 3]
    
    if not junction_nodes:
        return {}
    
    # Create subgraph with only junction nodes and their connections
    junction_subgraph = skeleton_graph.subgraph(junction_nodes).copy()
    
    # Find connected components
    clusters = {}
    cluster_id = 0
    
    for component in nx.connected_components(junction_subgraph):
        clusters[cluster_id] = list(component)
        cluster_id += 1
    
    logger.debug("cluster_junction_pixels: found %d junction clusters from %d junction pixels",
                len(clusters), len(junction_nodes))
    
    return clusters


def compute_cluster_centroid(
    skeleton_graph: nx.Graph,
    cluster_nodes: List[int]
) -> Tuple[float, float]:
    """
    Compute centroid of a cluster of nodes.
    
    Args:
        skeleton_graph: NetworkX graph
        cluster_nodes: List of node IDs in cluster
        
    Returns:
        Tuple of (x, y) centroid coordinates
    """
    if not cluster_nodes:
        return (0.0, 0.0)
    
    x_coords = [skeleton_graph.nodes[n]['xy'][0] for n in cluster_nodes]
    y_coords = [skeleton_graph.nodes[n]['xy'][1] for n in cluster_nodes]
    
    cx = sum(x_coords) / len(x_coords)
    cy = sum(y_coords) / len(y_coords)
    
    return (cx, cy)


def find_cluster_representative(
    skeleton_graph: nx.Graph,
    cluster_nodes: List[int]
) -> int:
    """
    Find the representative node for a cluster (closest to centroid).
    
    Args:
        skeleton_graph: NetworkX graph
        cluster_nodes: List of node IDs in cluster
        
    Returns:
        Node ID closest to cluster centroid
    """
    if not cluster_nodes:
        raise ValueError("Empty cluster")
    if len(cluster_nodes) == 1:
        return cluster_nodes[0]
    
    centroid = compute_cluster_centroid(skeleton_graph, cluster_nodes)
    centroid_point = Point(centroid)
    
    # Find node closest to centroid
    min_dist = float('inf')
    rep_node = cluster_nodes[0]
    
    for node_id in cluster_nodes:
        node_xy = skeleton_graph.nodes[node_id]['xy']
        node_point = Point(node_xy)
        dist = centroid_point.distance(node_point)
        if dist < min_dist:
            min_dist = dist
            rep_node = node_id
    
    return rep_node


def validate_junction_cluster_arms(
    skeleton_graph: nx.Graph,
    cluster_nodes: List[int]
) -> int:
    """
    Count the number of "arms" (connected components) attached to a junction cluster.
    
    This validates that a cluster is a true junction (arms >= 3) vs. a corner artifact.
    
    Algorithm:
    1. Remove cluster nodes from graph
    2. Find all neighbors of cluster nodes (outside the cluster)
    3. Count how many connected components those neighbors belong to in the reduced graph
    4. That count = number of arms
    
    Args:
        skeleton_graph: NetworkX graph
        cluster_nodes: List of node IDs in the cluster
        
    Returns:
        Number of arms (connected components attached to cluster)
    """
    if not cluster_nodes:
        return 0
    
    cluster_set = set(cluster_nodes)
    
    # Find all neighbors of cluster nodes that are outside the cluster
    external_neighbors = set()
    for node_id in cluster_nodes:
        for neighbor in skeleton_graph.neighbors(node_id):
            if neighbor not in cluster_set:
                external_neighbors.add(neighbor)
    
    if not external_neighbors:
        return 0
    
    # Create graph without cluster nodes
    all_nodes_except_cluster = set(skeleton_graph.nodes()) - cluster_set
    reduced_graph = skeleton_graph.subgraph(all_nodes_except_cluster).copy()
    
    # Count connected components that contain external neighbors
    neighbor_components = set()
    for component in nx.connected_components(reduced_graph):
        if component & external_neighbors:  # If component contains any external neighbor
            neighbor_components.add(frozenset(component))
    
    num_arms = len(neighbor_components)
    return num_arms


def contract_junction_clusters(
    skeleton_graph: nx.Graph,
    junction_clusters: Dict[int, List[int]],
    min_arms: int = 3
) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Contract junction clusters into single representative nodes.
    
    Only contracts clusters that have >= min_arms (validates true junctions).
    Invalid clusters (corners) are left as-is.
    
    Args:
        skeleton_graph: Original NetworkX graph
        junction_clusters: Dict mapping cluster_id to list of node IDs
        min_arms: Minimum number of arms to consider cluster valid (default: 3)
        
    Returns:
        Tuple of (contracted_graph, cluster_rep_map) where:
        - contracted_graph: New graph with clusters contracted to single nodes
        - cluster_rep_map: Dict mapping original node_id -> representative node_id for clusters
    """
    # Create a copy of the graph
    contracted = skeleton_graph.copy()
    cluster_rep_map = {}  # Maps original node_id -> rep node_id
    
    # Find valid clusters (arms >= min_arms) and their representatives
    valid_clusters = {}
    for cluster_id, cluster_nodes in junction_clusters.items():
        num_arms = validate_junction_cluster_arms(skeleton_graph, cluster_nodes)
        if num_arms >= min_arms:
            rep_node = find_cluster_representative(skeleton_graph, cluster_nodes)
            valid_clusters[cluster_id] = (cluster_nodes, rep_node)
            logger.debug("Junction cluster %d: %d nodes, %d arms, rep=%d", 
                        cluster_id, len(cluster_nodes), num_arms, rep_node)
        else:
            logger.debug("Junction cluster %d: %d nodes, %d arms (rejected, < %d arms)", 
                        cluster_id, len(cluster_nodes), num_arms, min_arms)
    
    # Contract each valid cluster
    for cluster_id, (cluster_nodes, rep_node) in valid_clusters.items():
        cluster_set = set(cluster_nodes)
        
        # Update rep node position to centroid
        centroid = compute_cluster_centroid(skeleton_graph, cluster_nodes)
        contracted.nodes[rep_node]['xy'] = list(centroid)
        
        # Map all cluster nodes to rep
        for node_id in cluster_nodes:
            cluster_rep_map[node_id] = rep_node
        
        # For each cluster node, redirect its external neighbors to rep_node
        nodes_to_remove = []
        for node_id in cluster_nodes:
            if node_id == rep_node:
                continue  # Keep rep node
            
            # Get neighbors outside the cluster
            for neighbor in list(contracted.neighbors(node_id)):
                if neighbor not in cluster_set:
                    # Add edge from rep to neighbor (if doesn't exist)
                    if not contracted.has_edge(rep_node, neighbor):
                        # Copy edge data if exists
                        if contracted.has_edge(node_id, neighbor):
                            edge_data = contracted.edges[node_id, neighbor]
                            contracted.add_edge(rep_node, neighbor, **edge_data)
                        else:
                            contracted.add_edge(rep_node, neighbor)
            
            nodes_to_remove.append(node_id)
        
        # Remove cluster nodes (except rep)
        for node_id in nodes_to_remove:
            contracted.remove_node(node_id)
    
    logger.info("Contracted %d/%d junction clusters (min_arms=%d)", 
               len(valid_clusters), len(junction_clusters), min_arms)
    
    return contracted, cluster_rep_map


def simplify_path_collinear(
    path: List[Tuple[float, float]],
    angle_threshold_deg: float = 3.0
) -> List[Tuple[float, float]]:
    """
    Simplify path by removing collinear points (corner-preserving).
    
    Unlike Shapely's simplify(), this preserves right angles and sharp corners
    by only removing points that are nearly collinear with their neighbors.
    
    Args:
        path: List of (x, y) coordinates
        angle_threshold_deg: Maximum angle (in degrees) to consider collinear (default: 3.0)
        
    Returns:
        Simplified path with collinear points removed
    """
    if len(path) < 3:
        return path
    
    import math
    
    angle_threshold_rad = math.radians(angle_threshold_deg)
    
    # Always keep first point
    simplified = [path[0]]
    
    for i in range(1, len(path) - 1):
        # Compute vectors: from prev to current, and from current to next
        prev_x, prev_y = path[i-1]
        curr_x, curr_y = path[i]
        next_x, next_y = path[i+1]
        
        vec1_x = curr_x - prev_x
        vec1_y = curr_y - prev_y
        vec2_x = next_x - curr_x
        vec2_y = next_y - curr_y
        
        # Compute lengths
        len1 = math.sqrt(vec1_x*vec1_x + vec1_y*vec1_y)
        len2 = math.sqrt(vec2_x*vec2_x + vec2_y*vec2_y)
        
        if len1 < 1e-10 or len2 < 1e-10:
            # Degenerate edge, keep the point
            simplified.append(path[i])
            continue
        
        # Normalize vectors
        vec1_x /= len1
        vec1_y /= len1
        vec2_x /= len2
        vec2_y /= len2
        
        # Compute dot product (cosine of angle)
        dot_product = vec1_x * vec2_x + vec1_y * vec2_y
        
        # Clamp to [-1, 1] for numerical stability
        dot_product = max(-1.0, min(1.0, dot_product))
        
        # Compute angle
        angle = math.acos(dot_product)
        turn_angle = math.pi - angle  # Turn angle (0 = straight, pi = 180 deg turn)
        
        # Keep point if turn angle is significant
        if turn_angle > angle_threshold_rad:
            simplified.append(path[i])
    
    # Always keep last point
    simplified.append(path[-1])
    
    return simplified


def merge_close_endpoints(
    skeleton_graph: nx.Graph,
    endpoint_nodes: List[int],
    merge_distance_um: float = 100.0,
    um_per_px: Optional[float] = None,
    forbidden_nodes: Optional[set] = None
) -> Dict[int, List[int]]:
    """
    Merge endpoints that are topologically close (using graph-shortest-path).
    
    Only merges endpoints if:
    - Both have degree==1
    - Graph-shortest-path length is small
    - Path contains only degree-2 nodes (no junction pixels)
    - Not within 2 px of forbidden nodes (junction pixels)
    
    Args:
        skeleton_graph: NetworkX graph
        endpoint_nodes: List of endpoint node IDs
        merge_distance_um: Maximum distance to merge endpoints (micrometers)
        um_per_px: Resolution (microns per pixel) for pixel-based distance checks
        forbidden_nodes: Set of forbidden node IDs (junction pixels) to avoid
        
    Returns:
        Dict mapping merged_id to list of node IDs merged together
    """
    import math
    
    if len(endpoint_nodes) < 2:
        return {i: [n] for i, n in enumerate(endpoint_nodes)}
    
    if forbidden_nodes is None:
        forbidden_nodes = set()
    
    # Compute degrees once
    degrees = dict(skeleton_graph.degree())
    
    # Convert merge distance to graph path length (in pixels)
    # K = ceil(merge_distance_um / um_per_px)
    if um_per_px is not None and um_per_px > 0:
        K = int(math.ceil(merge_distance_um / um_per_px))
    else:
        # Fallback: use large value if um_per_px not provided
        K = 1000
    
    # Build forbidden set with 2 px guard zone
    forbidden_with_guard = set(forbidden_nodes)
    if um_per_px is not None and um_per_px > 0 and forbidden_nodes:
        guard_distance_um = 2.0 * um_per_px
        for forbidden_node in forbidden_nodes:
            if forbidden_node not in skeleton_graph:
                continue
            forbidden_xy = skeleton_graph.nodes[forbidden_node]['xy']
            forbidden_point = Point(forbidden_xy)
            # Add all nodes within 2 px to forbidden set
            for node_id in skeleton_graph.nodes():
                if node_id in forbidden_with_guard:
                    continue
                node_xy = skeleton_graph.nodes[node_id]['xy']
                node_point = Point(node_xy)
                if forbidden_point.distance(node_point) <= guard_distance_um:
                    forbidden_with_guard.add(node_id)
    
    # Build clusters using BFS/union-find over endpoints
    merged = {}
    used = set()
    cluster_id = 0
    
    for i, node1 in enumerate(endpoint_nodes):
        if node1 in used:
            continue
        
        # Guard: only merge nodes where degree==1
        if degrees.get(node1, 0) != 1:
            continue
        
        # Check if node1 is in forbidden zone
        if node1 in forbidden_with_guard:
            continue
        
        cluster = [node1]
        used.add(node1)
        
        # Find all topologically close endpoints using BFS
        for j, node2 in enumerate(endpoint_nodes[i+1:], start=i+1):
            if node2 in used:
                continue
            
            # Guard: only merge nodes where degree==1
            if degrees.get(node2, 0) != 1:
                continue
            
            # Check if node2 is in forbidden zone
            if node2 in forbidden_with_guard:
                continue
            
            # Check graph-shortest-path length
            try:
                path_length = nx.shortest_path_length(skeleton_graph, node1, node2)
            except nx.NetworkXNoPath:
                continue
            
            # Check if path length is within limit
            if path_length > K:
                continue
            
            # Get the actual path and check all intermediate nodes are degree-2
            try:
                path = nx.shortest_path(skeleton_graph, node1, node2)
            except nx.NetworkXNoPath:
                continue
            
            # Check that all intermediate nodes (excluding endpoints) are degree-2
            # and not in forbidden set
            path_valid = True
            for n in path[1:-1]:  # Exclude endpoints
                if degrees.get(n, 0) != 2:
                    path_valid = False
                    break
                if n in forbidden_with_guard:
                    path_valid = False
                    break
            
            if path_valid:
                cluster.append(node2)
                used.add(node2)
        
        merged[cluster_id] = cluster
        cluster_id += 1
    
    logger.debug("merge_close_endpoints: merged %d endpoints into %d clusters (topology-aware)",
                len(endpoint_nodes), len(merged))
    
    return merged
