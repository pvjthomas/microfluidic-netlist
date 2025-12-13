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
        corresponds to minimum y (bottom in standard view, but our origin_y is min_y).
        PIL ImageDraw expects coordinates with y increasing downward, but we'll flip
        the image vertically after drawing to match the transform semantics.
        """
        img_x = int((shapely_x - padded_minx) / um_per_px)
        # Flip y-axis: PIL y increases down, but our transform has y increase up with row
        img_y = int(img_height - 1 - (shapely_y - padded_miny) / um_per_px)
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
    target_pixels_per_width: float = 5.0,
    width_percentile: float = 7.5,
    min_um_per_px: float = 1.0,
    max_um_per_px: float = 100.0
) -> Tuple[float, float]:
    """
    Estimate channel width and auto-tune raster resolution.
    
    Uses a coarse distance transform to estimate local channel widths,
    then selects um_per_px so the narrowest channel has approximately
    target_pixels_per_width pixels across it.
    
    Args:
        polygon: Shapely polygon to analyze
        coarse_um_per_px: Resolution for coarse rasterization (default: 30 µm)
        target_pixels_per_width: Target pixels across narrowest channel (default: 5.0)
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
    
    # Clamp to reasonable bounds
    recommended_um_per_px = max(min_um_per_px, min(max_um_per_px, recommended_um_per_px))
    
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
    um_per_px: Optional[float] = None,
    simplify_tolerance: Optional[float] = None,
    auto_tune_resolution: bool = True,
    target_pixels_per_width: float = 5.0
) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Skeletonize a polygon and convert to a NetworkX graph.
    
    Args:
        polygon: Shapely polygon to skeletonize
        um_per_px: Resolution for rasterization (microns per pixel). If None and 
                   auto_tune_resolution=True, will be auto-tuned based on channel width.
        simplify_tolerance: Tolerance for simplifying centerlines (in original units).
                           If None, auto-computed as 0.5 * um_per_px.
        auto_tune_resolution: If True and um_per_px is None, auto-tune resolution based on channel width
        target_pixels_per_width: Target pixels across narrowest channel for auto-tuning (default: 5.0)
    
    Returns:
        Tuple of (skeleton_graph, metadata) where:
        - skeleton_graph: NetworkX graph with nodes as (x, y) tuples
        - metadata: Transform and processing info
    """
    # Auto-tune resolution if needed
    if um_per_px is None and auto_tune_resolution:
        logger.debug("skeletonize_polygon: auto-tuning resolution")
        estimated_width, um_per_px = estimate_channel_width_and_resolution(
            polygon,
            target_pixels_per_width=target_pixels_per_width
        )
        logger.info("skeletonize_polygon: auto-tuned um_per_px=%.2f (narrowest width=%.2f µm)",
                   um_per_px, estimated_width)
    elif um_per_px is None:
        um_per_px = 20.0  # Default fallback
        logger.warning("skeletonize_polygon: um_per_px not specified and auto_tune_resolution=False, using default %.2f",
                      um_per_px)
    
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
    # Morphological closing (disk radius ~2 px) to remove endcap notches
    closing_radius = 2
    try:
        img_binary = binary_closing(img_binary, disk(closing_radius))
        logger.debug("Applied binary closing with radius %d", closing_radius)
    except Exception as e:
        logger.warning("Binary closing failed: %s", e)
    
    # Optional opening (radius 1 px) to remove pepper noise
    opening_radius = 1
    try:
        img_binary = binary_opening(img_binary, disk(opening_radius))
        logger.debug("Applied binary opening with radius %d", opening_radius)
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
    logger.debug("skeletonize_polygon: converting %d skeleton pixels to coordinates",
                 len(skeleton_pixels))
    skeleton_coords = []
    try:
        for i, pixel in enumerate(skeleton_pixels):
            if i % 10000 == 0 and i > 0:
                logger.debug("skeletonize_polygon: converted %d/%d pixels", i, len(skeleton_pixels))
            coords = pixel_to_coords((pixel[0], pixel[1]), transform)
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
            coords = pixel_to_coords((pixel[0], pixel[1]), transform)
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
    
    # 2. Skeleton spur pruning: iteratively delete degree-1 pixels
    max_spur_length_px = max(5, int(50.0 / um_per_px))  # ~50 µm in pixels, minimum 5
    logger.debug("Pruning skeleton spurs: max length = %d pixels", max_spur_length_px)
    spurs_pruned = 0
    for iteration in range(max_spur_length_px):
        # Find all degree-1 nodes (potential spurs)
        degree1_nodes = [n for n in G.nodes() if G.degree(n) == 1]
        if not degree1_nodes:
            break
        
        # Delete degree-1 nodes (they become isolated and will be removed)
        for node in degree1_nodes:
            G.remove_node(node)
            spurs_pruned += 1
    
    logger.info(f"Pruned {spurs_pruned} spur pixels ({iteration+1} iterations)")
    
    logger.debug("skeletonize_polygon: complete, total time: %.2fs", total_time)
    
    return G, transform


def extract_skeleton_paths(
    skeleton_graph: nx.Graph,
    simplify_tolerance: Optional[float] = None
) -> List[List[Tuple[float, float]]]:
    """
    Extract centerline paths from skeleton graph.
    
    Identifies endpoints (degree 1) and junctions (degree >= 3),
    then extracts paths between them.
    
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
    
    # Identify nodes by degree
    logger.debug("extract_skeleton_paths: identifying endpoints and junctions")
    try:
        endpoints = [n for n in skeleton_graph.nodes() if skeleton_graph.degree(n) == 1]
        junctions = [n for n in skeleton_graph.nodes() if skeleton_graph.degree(n) >= 3]
        logger.debug("extract_skeleton_paths: found %d endpoints, %d junctions", 
                     len(endpoints), len(junctions))
    except Exception as e:
        logger.error("extract_skeleton_paths: failed to identify nodes: %s", e, exc_info=True)
        raise
    
    # If no endpoints/junctions, return single path through all nodes
    if not endpoints and not junctions:
        # Simple path through all connected nodes
        if len(skeleton_graph) > 0:
            nodes = list(skeleton_graph.nodes())
            path = [skeleton_graph.nodes[n]['xy'] for n in nodes]
            return [path] if len(path) > 1 else []
        return []
    
    # Extract paths between endpoints and junctions
    logger.debug("extract_skeleton_paths: extracting paths between endpoints and junctions")
    paths = []
    visited_edges = set()
    
    # Start from each endpoint
    try:
        for idx, start in enumerate(endpoints):
            if idx % 10 == 0 and idx > 0:
                logger.debug("extract_skeleton_paths: processed %d/%d endpoints", idx, len(endpoints))
            if start not in skeleton_graph:
                logger.warning("extract_skeleton_paths: endpoint %d not in graph", start)
                continue
            
            # Find path to nearest junction or endpoint
            targets = junctions + [ep for ep in endpoints if ep != start]
            logger.debug("extract_skeleton_paths: endpoint %d has %d targets", start, len(targets))
            
            path_found = False
            for target in targets:
                if target == start:
                    continue
                
                try:
                    path_nodes = nx.shortest_path(skeleton_graph, start, target)
                    path_coords = [skeleton_graph.nodes[n]['xy'] for n in path_nodes]
                    
                    # Check if we've used these edges
                    path_edges = set()
                    for i in range(len(path_nodes) - 1):
                        edge = tuple(sorted([path_nodes[i], path_nodes[i+1]]))
                        path_edges.add(edge)
                    
                    if not path_edges.intersection(visited_edges):
                        paths.append(path_coords)
                        visited_edges.update(path_edges)
                        logger.debug("extract_skeleton_paths: added path from endpoint %d to %d (%d nodes)",
                                    start, target, len(path_nodes))
                        path_found = True
                        break  # Found a path from this endpoint
                except nx.NetworkXNoPath:
                    continue
                except Exception as e:
                    logger.warning("extract_skeleton_paths: error finding path from %d to %d: %s",
                                  start, target, e)
                    continue
            
            if not path_found:
                logger.debug("extract_skeleton_paths: no path found from endpoint %d", start)
    except Exception as e:
        logger.error("extract_skeleton_paths: failed to extract paths from endpoints: %s", 
                     e, exc_info=True)
        raise
    
    # Also handle paths between junctions
    logger.debug("extract_skeleton_paths: extracting paths between %d junctions", len(junctions))
    try:
        for i, j1 in enumerate(junctions):
            if i % 10 == 0 and i > 0:
                logger.debug("extract_skeleton_paths: processed %d/%d junctions", i, len(junctions))
            for j2 in junctions[i+1:]:
                try:
                    path_nodes = nx.shortest_path(skeleton_graph, j1, j2)
                    path_coords = [skeleton_graph.nodes[n]['xy'] for n in path_nodes]
                    
                    path_edges = set()
                    for k in range(len(path_nodes) - 1):
                        edge = tuple(sorted([path_nodes[k], path_nodes[k+1]]))
                        path_edges.add(edge)
                    
                    if not path_edges.intersection(visited_edges):
                        paths.append(path_coords)
                        visited_edges.update(path_edges)
                        logger.debug("extract_skeleton_paths: added path between junctions %d and %d (%d nodes)",
                                    j1, j2, len(path_nodes))
                except nx.NetworkXNoPath:
                    continue
                except Exception as e:
                    logger.warning("extract_skeleton_paths: error finding path between junctions %d and %d: %s",
                                  j1, j2, e)
                    continue
    except Exception as e:
        logger.error("extract_skeleton_paths: failed to extract paths between junctions: %s", 
                     e, exc_info=True)
        raise
    
    logger.debug("extract_skeleton_paths: extracted %d paths before simplification", len(paths))
    
    # Simplify paths using Douglas-Peucker-like approach (if tolerance specified)
    if simplify_tolerance is not None and simplify_tolerance > 0:
        logger.debug("extract_skeleton_paths: simplifying %d paths with tolerance=%.3f", len(paths), simplify_tolerance)
        from shapely.geometry import LineString
        simplified_paths = []
        try:
            for idx, path in enumerate(paths):
                if idx % 10 == 0 and idx > 0:
                    logger.debug("extract_skeleton_paths: simplified %d/%d paths", idx, len(paths))
                if len(path) < 2:
                    logger.debug("extract_skeleton_paths: skipping path %d (too short: %d points)", idx, len(path))
                    continue
                try:
                    line = LineString(path)
                    simplified = line.simplify(simplify_tolerance, preserve_topology=False)
                    if simplified.geom_type == 'LineString':
                        simplified_paths.append(list(simplified.coords))
                    else:
                        # MultiLineString - take longest segment
                        if hasattr(simplified, 'geoms'):
                            longest = max(simplified.geoms, key=lambda g: g.length)
                            simplified_paths.append(list(longest.coords))
                        else:
                            logger.warning("extract_skeleton_paths: path %d simplified to unexpected type: %s",
                                          idx, simplified.geom_type)
                except Exception as e:
                    logger.warning("extract_skeleton_paths: failed to simplify path %d: %s", idx, e)
                    continue
        except Exception as e:
            logger.error("extract_skeleton_paths: failed during path simplification: %s", e, exc_info=True)
            raise
        
        result = simplified_paths if simplified_paths else paths
    else:
        logger.debug("extract_skeleton_paths: skipping simplification (tolerance not specified or zero)")
        result = paths
    logger.debug("extract_skeleton_paths: exit, returning %d paths", len(result))
    return result


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


def merge_close_endpoints(
    skeleton_graph: nx.Graph,
    endpoint_nodes: List[int],
    merge_distance_um: float = 100.0
) -> Dict[int, List[int]]:
    """
    Merge endpoints that are spatially close.
    
    Args:
        skeleton_graph: NetworkX graph
        endpoint_nodes: List of endpoint node IDs
        merge_distance_um: Maximum distance to merge endpoints (micrometers)
        
    Returns:
        Dict mapping merged_id to list of node IDs merged together
    """
    if len(endpoint_nodes) < 2:
        return {i: [n] for i, n in enumerate(endpoint_nodes)}
    
    # Build distance matrix and merge close endpoints
    merged = {}
    used = set()
    cluster_id = 0
    
    for i, node1 in enumerate(endpoint_nodes):
        if node1 in used:
            continue
        
        cluster = [node1]
        used.add(node1)
        
        # Find all close endpoints
        xy1 = skeleton_graph.nodes[node1]['xy']
        point1 = Point(xy1)
        
        for j, node2 in enumerate(endpoint_nodes[i+1:], start=i+1):
            if node2 in used:
                continue
            
            xy2 = skeleton_graph.nodes[node2]['xy']
            point2 = Point(xy2)
            dist = point1.distance(point2)
            
            if dist <= merge_distance_um:
                cluster.append(node2)
                used.add(node2)
        
        merged[cluster_id] = cluster
        cluster_id += 1
    
    logger.debug("merge_close_endpoints: merged %d endpoints into %d clusters",
                len(endpoint_nodes), len(merged))
    
    return merged
