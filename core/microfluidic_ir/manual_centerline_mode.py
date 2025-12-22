"""Manual Centerline Mode: semi-automated centerline extraction via boundary selection."""

from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MPLPolygon
from matplotlib.widgets import Button
from shapely.geometry import Polygon, Point, LineString, MultiLineString
from shapely.ops import unary_union
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BoundarySegment:
    """A boundary segment with stable ID."""
    
    def __init__(self, segment_id: str, coords: List[Tuple[float, float]], 
                 line_string: LineString):
        self.segment_id = segment_id
        self.coords = coords
        self.line_string = line_string
        self.midpoint = line_string.interpolate(0.5, normalized=True)
    
    def distance_to_point(self, point: Point) -> float:
        """Distance from point to this segment."""
        return self.line_string.distance(point)


class ManualCenterlineMode:
    """Interactive manual centerline extraction mode."""
    
    def __init__(self, polygon: Polygon, save_path: Optional[str] = None,
                 dxf_result: Optional[Dict[str, Any]] = None):
        """
        Initialize manual centerline mode.
        
        Args:
            polygon: Channel polygon to extract centerlines from
            save_path: Path to save/load centerlines JSON. If None, will be auto-generated
                       from DXF filename as "#FILENAME_manual_centerlines.json"
            dxf_result: Optional DXF load result dict. If provided, will use DXF edges
                       as boundary segments when available.
        """
        self.polygon = polygon
        self.dxf_result = dxf_result
        
        # Auto-generate save_path from DXF filename if not provided
        if save_path is None:
            if dxf_result and 'source_filename' in dxf_result:
                # Extract filename without extension
                source_filename = Path(dxf_result['source_filename'])
                filename_base = source_filename.stem  # filename without extension
                save_path = f"{filename_base}_manual_centerlines.json"
            else:
                save_path = "manual_centerlines.json"  # Fallback default
        
        self.save_path = Path(save_path)
        self.boundary_segments: List[BoundarySegment] = []
        self.manual_centerlines: List[Dict[str, Any]] = []
        self.selected_segments: List[Optional[str]] = [None, None]  # [seg1_id, seg2_id]
        
        self.fig = None
        self.ax = None
        self.done_button = None  # Done button widget
        self.delete_all_button = None  # Delete all button widget
        self.pairs_text = None  # Text widget showing pairs list
        self.pairs_ax = None  # Axes for pairs list
        self.segment_patches = {}  # Map segment_id -> matplotlib patch
        self.centerline_lines = []  # List of matplotlib line objects
        
        # Load existing centerlines
        self.load_centerlines()
        
        # Extract boundary segments
        self._extract_boundary_segments()
    
    def _extract_boundary_segments(self):
        """Extract boundary segments from polygon exterior or DXF edges if available."""
        # If DXF result is provided, try to use DXF edges as segments
        if self.dxf_result and 'polygons' in self.dxf_result:
            # Find polygon that matches our polygon (by checking if they overlap significantly)
            matching_poly_data = None
            for poly_data in self.dxf_result['polygons']:
                coords = poly_data['polygon']['coordinates'][0]
                dxf_poly = Polygon(coords)
                # Check if polygons overlap significantly
                intersection = self.polygon.intersection(dxf_poly)
                if intersection.area > 0.5 * min(self.polygon.area, dxf_poly.area):
                    matching_poly_data = poly_data
                    break
            
            if matching_poly_data:
                # Use DXF polygon coordinates to create segments
                # This preserves the original DXF edge structure
                coords = matching_poly_data['polygon']['coordinates'][0]
                exterior_coords = coords[:-1] if coords[0] == coords[-1] else coords
                
                for i in range(len(exterior_coords)):
                    start = exterior_coords[i]
                    end = exterior_coords[(i + 1) % len(exterior_coords)]
                    
                    segment_id = f"B{i}"
                    line_string = LineString([start, end])
                    
                    segment = BoundarySegment(segment_id, [start, end], line_string)
                    self.boundary_segments.append(segment)
                
                logger.info(f"Extracted {len(self.boundary_segments)} boundary segments from DXF edges")
                return
        
        # Fallback: Extract from polygon exterior
        exterior_coords = list(self.polygon.exterior.coords[:-1])  # Remove duplicate last point
        
        # Create segments from consecutive coordinate pairs
        for i in range(len(exterior_coords)):
            start = exterior_coords[i]
            end = exterior_coords[(i + 1) % len(exterior_coords)]
            
            segment_id = f"B{i}"
            line_string = LineString([start, end])
            
            segment = BoundarySegment(segment_id, [start, end], line_string)
            self.boundary_segments.append(segment)
        
        logger.info(f"Extracted {len(self.boundary_segments)} boundary segments from polygon exterior")
    
    def load_centerlines(self):
        """Load existing centerlines from JSON file."""
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                    self.manual_centerlines = data.get('centerlines', [])
                logger.info(f"Loaded {len(self.manual_centerlines)} existing centerlines")
            except Exception as e:
                logger.warning(f"Failed to load centerlines: {e}")
                self.manual_centerlines = []
        else:
            self.manual_centerlines = []
    
    def save_centerlines(self):
        """Save centerlines to JSON file."""
        data = {
            'centerlines': self.manual_centerlines,
            'polygon_bounds': {
                'xmin': self.polygon.bounds[0],
                'ymin': self.polygon.bounds[1],
                'xmax': self.polygon.bounds[2],
                'ymax': self.polygon.bounds[3]
            }
        }
        
        try:
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.manual_centerlines)} centerlines to {self.save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save centerlines: {e}")
            return False
    
    def _estimate_text_box_size(self, text: str, fontsize: int = 8) -> Tuple[float, float]:
        """
        Estimate text bounding box size in data coordinates (mm).
        
        Args:
            text: Text string
            fontsize: Font size in points
        
        Returns:
            Tuple of (width, height) in data coordinates (mm)
        """
        # Get current axis limits or use polygon bounds (convert to mm)
        if self.ax is not None:
            try:
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                width_data = xlim[1] - xlim[0]
                height_data = ylim[1] - ylim[0]
            except:
                bounds = self.polygon.bounds
                width_data = (bounds[2] - bounds[0]) / 1000.0  # Convert to mm
                height_data = (bounds[3] - bounds[1]) / 1000.0  # Convert to mm
        else:
            bounds = self.polygon.bounds
            width_data = (bounds[2] - bounds[0]) / 1000.0  # Convert to mm
            height_data = (bounds[3] - bounds[1]) / 1000.0  # Convert to mm
        
        # Estimate figure size in inches (default is 16x10 from show())
        fig_width_inches = 16.0
        fig_height_inches = 10.0
        
        # Character width estimation: fontsize * 0.6 points per character (typical)
        # Character height: fontsize * 1.2 points (includes line spacing)
        # Padding: 0.3 * fontsize on each side (as specified in bbox)
        char_width_pts = fontsize * 0.6
        char_height_pts = fontsize * 1.2
        padding_pts = fontsize * 0.3 * 2  # padding on both sides
        
        # Convert points to inches (1 point = 1/72 inch)
        text_width_inches = (char_width_pts * len(text) + padding_pts) / 72.0
        text_height_inches = (char_height_pts + padding_pts) / 72.0
        
        # Convert to data coordinates
        # data_coord = (inches / fig_inches) * data_range
        text_width = (text_width_inches / fig_width_inches) * width_data
        text_height = (text_height_inches / fig_height_inches) * height_data
        
        # Add a small safety margin (10%)
        text_width *= 1.1
        text_height *= 1.1
        
        return (text_width, text_height)
    
    def _boxes_overlap(self, x1: float, y1: float, w1: float, h1: float,
                       x2: float, y2: float, w2: float, h2: float) -> bool:
        """Check if two axis-aligned bounding boxes overlap."""
        # Calculate box bounds
        x1_min, x1_max = x1 - w1/2, x1 + w1/2
        y1_min, y1_max = y1 - h1/2, y1 + h1/2
        x2_min, x2_max = x2 - w2/2, x2 + w2/2
        y2_min, y2_max = y2 - h2/2, y2 + h2/2
        
        # Check for overlap
        return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)
    
    def _adjust_label_positions(self, label_positions: List[Tuple[float, float, str]], 
                                fontsize: int = 8) -> List[Tuple[float, float, str]]:
        """
        Adjust label positions to avoid bounding box overlaps.
        
        Uses iterative force-based layout to separate overlapping labels.
        
        Args:
            label_positions: List of (x, y, segment_id) tuples
            fontsize: Font size for text (used to estimate box size)
        
        Returns:
            List of adjusted (x, y, segment_id) tuples
        """
        if len(label_positions) <= 1:
            return label_positions
        
        adjusted = list(label_positions)
        max_iterations = 50
        min_separation = 1.3  # Minimum separation factor (boxes should be 30% apart)
        damping = 0.8  # Damping factor to prevent oscillation
        
        # Get bounds for clamping (convert from micrometers to mm)
        bounds = self.polygon.bounds
        margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.3  # Increased margin
        xmin = (bounds[0] - margin) / 1000.0  # Convert to mm
        xmax = (bounds[2] + margin) / 1000.0  # Convert to mm
        ymin = (bounds[1] - margin) / 1000.0  # Convert to mm
        ymax = (bounds[3] + margin) / 1000.0  # Convert to mm
        
        for iteration in range(max_iterations):
            forces = [(0.0, 0.0) for _ in adjusted]  # (fx, fy) for each label
            overlaps_found = False
            
            # Calculate repulsion forces between overlapping labels
            for i in range(len(adjusted)):
                x1, y1, seg_id1 = adjusted[i]
                w1, h1 = self._estimate_text_box_size(seg_id1, fontsize)
                
                for j in range(i + 1, len(adjusted)):
                    x2, y2, seg_id2 = adjusted[j]
                    w2, h2 = self._estimate_text_box_size(seg_id2, fontsize)
                    
                    if self._boxes_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
                        overlaps_found = True
                        
                        # Calculate direction vector
                        dx = x2 - x1
                        dy = y2 - y1
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        if dist < 1e-6:
                            # Labels are at same position, move in random direction
                            angle = np.random.uniform(0, 2 * np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        
                        # Calculate required separation distance
                        # Use the maximum of width and height to ensure no overlap in any direction
                        required_sep_x = (w1 + w2) / 2 * min_separation
                        required_sep_y = (h1 + h2) / 2 * min_separation
                        required_sep = max(required_sep_x, required_sep_y)
                        
                        # Current separation
                        current_sep = max(dist, 1e-6)
                        
                        # Calculate repulsion force (stronger when closer)
                        force_magnitude = (required_sep - current_sep) / required_sep
                        if force_magnitude > 0:
                            # Apply force in opposite directions
                            fx = dx * force_magnitude
                            fy = dy * force_magnitude
                            
                            # Accumulate forces
                            fx1, fy1 = forces[i]
                            fx2, fy2 = forces[j]
                            forces[i] = (fx1 - fx, fy1 - fy)  # Push label i away
                            forces[j] = (fx2 + fx, fy2 + fy)  # Push label j away
            
            if not overlaps_found:
                # No overlaps found, we're done
                break
            
            # Apply forces with damping
            for i in range(len(adjusted)):
                x, y, seg_id = adjusted[i]
                fx, fy = forces[i]
                
                # Scale force by damping and a step size
                step_size = 0.5 * (1.0 - iteration / max_iterations) + 0.1  # Decrease over time
                new_x = x + fx * step_size * damping
                new_y = y + fy * step_size * damping
                
                # Clamp to bounds
                new_x = max(xmin, min(xmax, new_x))
                new_y = max(ymin, min(ymax, new_y))
                
                adjusted[i] = (new_x, new_y, seg_id)
        
        # Final verification pass: check for any remaining overlaps and fix them aggressively
        for _ in range(10):  # Up to 10 more passes
            fixed_any = False
            for i in range(len(adjusted)):
                x1, y1, seg_id1 = adjusted[i]
                w1, h1 = self._estimate_text_box_size(seg_id1, fontsize)
                
                for j in range(i + 1, len(adjusted)):
                    x2, y2, seg_id2 = adjusted[j]
                    w2, h2 = self._estimate_text_box_size(seg_id2, fontsize)
                    
                    if self._boxes_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
                        # Still overlapping, move apart more aggressively
                        dx = x2 - x1
                        dy = y2 - y1
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        if dist < 1e-6:
                            angle = np.random.uniform(0, 2 * np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        
                        required_sep_x = (w1 + w2) / 2 * min_separation
                        required_sep_y = (h1 + h2) / 2 * min_separation
                        required_sep = max(required_sep_x, required_sep_y)
                        
                        move_distance = (required_sep - dist) / 2
                        if move_distance > 0:
                            new_x1 = x1 - dx * move_distance
                            new_y1 = y1 - dy * move_distance
                            new_x2 = x2 + dx * move_distance
                            new_y2 = y2 + dy * move_distance
                            
                            new_x1 = max(xmin, min(xmax, new_x1))
                            new_y1 = max(ymin, min(ymax, new_y1))
                            new_x2 = max(xmin, min(xmax, new_x2))
                            new_y2 = max(ymin, min(ymax, new_y2))
                            
                            adjusted[i] = (new_x1, new_y1, seg_id1)
                            adjusted[j] = (new_x2, new_y2, seg_id2)
                            fixed_any = True
            
            if not fixed_any:
                break
        
        return adjusted
    
    def _find_segment_at_point(self, point: Point, tolerance: Optional[float] = None) -> Optional[str]:
        """Find boundary segment closest to point within tolerance."""
        # Adaptive tolerance based on polygon size
        if tolerance is None:
            bounds = self.polygon.bounds
            max_dim = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
            tolerance = max_dim * 0.01  # 1% of max dimension, but at least 5 units
            tolerance = max(5.0, tolerance)
        
        min_dist = float('inf')
        closest_segment_id = None
        
        for segment in self.boundary_segments:
            dist = segment.distance_to_point(point)
            if dist < min_dist and dist <= tolerance:
                min_dist = dist
                closest_segment_id = segment.segment_id
        
        return closest_segment_id
    
    def _compute_centerline(self, seg1_id: str, seg2_id: str, 
                           spacing: float = 10.0) -> Optional[LineString]:
        """
        Compute centerline between two boundary segments.
        
        Uses paired boundary sampling: sample along boundary1, cast rays in normal direction
        toward boundary2, find intersection point, take midpoint. Fit polyline through midpoints.
        """
        # Get segments
        seg1 = next((s for s in self.boundary_segments if s.segment_id == seg1_id), None)
        seg2 = next((s for s in self.boundary_segments if s.segment_id == seg2_id), None)
        
        if not seg1 or not seg2:
            return None
        
        # Sample along seg1
        num_samples = max(3, int(seg1.line_string.length / spacing))
        sample_points = []
        failed_intersections = 0
        max_failures = num_samples * 0.5  # Allow up to 50% failures before fallback
        
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0.5
            point_on_seg1 = seg1.line_string.interpolate(t, normalized=True)
            
            # Compute normal direction at this point on seg1
            # Get tangent direction along seg1
            if t < 0.01:
                # Near start, use forward direction
                next_point = seg1.line_string.interpolate(min(0.1, 1.0), normalized=True)
                tangent = np.array([next_point.x - point_on_seg1.x, 
                                   next_point.y - point_on_seg1.y])
            elif t > 0.99:
                # Near end, use backward direction
                prev_point = seg1.line_string.interpolate(max(0.9, 0.0), normalized=True)
                tangent = np.array([point_on_seg1.x - prev_point.x,
                                   point_on_seg1.y - prev_point.y])
            else:
                # Use points on either side
                next_point = seg1.line_string.interpolate(min(t + 0.05, 1.0), normalized=True)
                prev_point = seg1.line_string.interpolate(max(t - 0.05, 0.0), normalized=True)
                tangent = np.array([next_point.x - prev_point.x,
                                   next_point.y - prev_point.y])
            
            # Normalize tangent
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-6:
                # Degenerate case, use direction to closest point on seg2
                closest_point_on_seg2 = seg2.line_string.interpolate(
                    seg2.line_string.project(point_on_seg1)
                )
                direction = np.array([
                    closest_point_on_seg2.x - point_on_seg1.x,
                    closest_point_on_seg2.y - point_on_seg1.y
                ])
                direction_norm = np.linalg.norm(direction)
                if direction_norm < 1e-6:
                    failed_intersections += 1
                    continue
                direction = direction / direction_norm
            else:
                tangent = tangent / tangent_norm
                # Compute normal (perpendicular to tangent, pointing toward seg2)
                # Normal = rotate tangent by 90 degrees
                normal = np.array([-tangent[1], tangent[0]])
                
                # Determine which direction of normal points toward seg2
                # Test both directions and pick the one that gets closer to seg2
                test_point1 = Point(point_on_seg1.x + normal[0] * 10, 
                                   point_on_seg1.y + normal[1] * 10)
                test_point2 = Point(point_on_seg1.x - normal[0] * 10,
                                   point_on_seg1.y - normal[1] * 10)
                
                dist1 = test_point1.distance(seg2.line_string)
                dist2 = test_point2.distance(seg2.line_string)
                
                # Use the direction that gets closer to seg2
                if dist1 < dist2:
                    direction = normal
                else:
                    direction = -normal
            
            # Cast ray from point_on_seg1 in normal direction
            ray_length = max(
                self.polygon.bounds[2] - self.polygon.bounds[0],
                self.polygon.bounds[3] - self.polygon.bounds[1]
            ) * 2  # Use polygon diagonal as max ray length
            ray_end = Point(
                point_on_seg1.x + direction[0] * ray_length,
                point_on_seg1.y + direction[1] * ray_length
            )
            ray = LineString([point_on_seg1, ray_end])
            
            # Find intersection with seg2
            intersection = ray.intersection(seg2.line_string)
            
            if not intersection.is_empty:
                if isinstance(intersection, Point):
                    midpoint = Point(
                        (point_on_seg1.x + intersection.x) / 2,
                        (point_on_seg1.y + intersection.y) / 2
                    )
                    # Check if midpoint is inside polygon
                    if self.polygon.contains(midpoint) or self.polygon.touches(midpoint):
                        sample_points.append((midpoint.x, midpoint.y))
                    else:
                        failed_intersections += 1
                elif isinstance(intersection, LineString) and len(intersection.coords) > 0:
                    # Line intersection, use first point
                    first_point = Point(intersection.coords[0])
                    midpoint = Point(
                        (point_on_seg1.x + first_point.x) / 2,
                        (point_on_seg1.y + first_point.y) / 2
                    )
                    if self.polygon.contains(midpoint) or self.polygon.touches(midpoint):
                        sample_points.append((midpoint.x, midpoint.y))
                    else:
                        failed_intersections += 1
                else:
                    failed_intersections += 1
            else:
                failed_intersections += 1
        
        # If too many intersections failed, use fallback
        if len(sample_points) < 2 or failed_intersections > max_failures:
            logger.warning(f"Ray casting failed ({failed_intersections}/{num_samples} failures), "
                          f"using skeletonization fallback")
            return self._compute_centerline_skeleton_fallback(seg1, seg2)
        
        # Fit polyline through sample points
        centerline = LineString(sample_points)
        
        # Simplify slightly
        simplified = centerline.simplify(tolerance=spacing * 0.5, preserve_topology=False)
        
        return simplified if simplified.geom_type == 'LineString' else centerline
    
    def _compute_centerline_skeleton_fallback(self, seg1: BoundarySegment, 
                                             seg2: BoundarySegment) -> Optional[LineString]:
        """Fallback: skeletonize corridor region between two segments."""
        try:
            from .skeleton import rasterize_polygon, pixel_to_coords
            from skimage.morphology import skeletonize
            from shapely.geometry import box
            
            # Create a corridor polygon between the two segments
            # Use bounding box of both segments with some padding
            seg1_bounds = seg1.line_string.bounds
            seg2_bounds = seg2.line_string.bounds
            
            padding = max(
                self.polygon.bounds[2] - self.polygon.bounds[0],
                self.polygon.bounds[3] - self.polygon.bounds[1]
            ) * 0.1  # 10% of max dimension
            
            xmin = min(seg1_bounds[0], seg2_bounds[0]) - padding
            ymin = min(seg1_bounds[1], seg2_bounds[1]) - padding
            xmax = max(seg1_bounds[2], seg2_bounds[2]) + padding
            ymax = max(seg1_bounds[3], seg2_bounds[3]) + padding
            
            # Clip polygon to corridor region
            corridor_box = box(xmin, ymin, xmax, ymax)
            clipped_poly = self.polygon.intersection(corridor_box)
            
            if clipped_poly.is_empty:
                return None
            
            # Ensure it's a Polygon
            if clipped_poly.geom_type == 'Polygon':
                corridor_poly = clipped_poly
            elif clipped_poly.geom_type == 'MultiPolygon':
                corridor_poly = max(clipped_poly.geoms, key=lambda g: g.area)
            else:
                return None
            
            # Skeletonize - use um_per_px parameter (rasterize_polygon expects um_per_px)
            # Convert to microns per pixel (assuming coordinates are in microns)
            um_per_px = 10.0  # 10 microns per pixel
            binary_img, transform = rasterize_polygon(corridor_poly, um_per_px=um_per_px)
            skeleton_img = skeletonize(binary_img)
            
            # Extract skeleton path
            skeleton_pixels = np.argwhere(skeleton_img)
            if len(skeleton_pixels) == 0:
                return None
            
            # Convert to coordinates and find path from seg1 to seg2
            skeleton_coords = []
            for pixel in skeleton_pixels:
                coords = pixel_to_coords((pixel[0], pixel[1]), transform)
                skeleton_coords.append(coords)
            
            # Find closest skeleton points to segment midpoints
            seg1_point = Point(seg1.midpoint)
            seg2_point = Point(seg2.midpoint)
            
            min_dist1 = float('inf')
            min_dist2 = float('inf')
            closest_idx1 = 0
            closest_idx2 = 0
            
            for i, coord in enumerate(skeleton_coords):
                p = Point(coord)
                d1 = p.distance(seg1_point)
                d2 = p.distance(seg2_point)
                if d1 < min_dist1:
                    min_dist1 = d1
                    closest_idx1 = i
                if d2 < min_dist2:
                    min_dist2 = d2
                    closest_idx2 = i
            
            # Extract path between these points (simplified - just take direct path)
            if closest_idx1 < closest_idx2:
                path_coords = skeleton_coords[closest_idx1:closest_idx2+1]
            else:
                path_coords = skeleton_coords[closest_idx2:closest_idx1+1]
            
            if len(path_coords) < 2:
                return None
            
            return LineString(path_coords)
            
        except Exception as e:
            logger.error(f"Skeleton fallback failed: {e}", exc_info=True)
            return None
    
    def _on_click(self, event):
        """Handle mouse click to select boundary segments."""
        if event.inaxes != self.ax:
            return
        
        if event.button != 1:  # Left click only
            return
        
        # event.xdata and event.ydata are already in mm (from axis limits)
        # Convert back to micrometers for segment distance calculation
        point_mm = Point(event.xdata, event.ydata)
        point_um = Point(event.xdata * 1000.0, event.ydata * 1000.0)
        segment_id = self._find_segment_at_point(point_um)
        
        if segment_id:
            # Toggle selection
            if self.selected_segments[0] == segment_id:
                # Deselect first
                self.selected_segments[0] = None
            elif self.selected_segments[1] == segment_id:
                # Deselect second
                self.selected_segments[1] = None
            elif self.selected_segments[0] is None:
                # Select as first
                self.selected_segments[0] = segment_id
            elif self.selected_segments[1] is None:
                # Select as second
                self.selected_segments[1] = segment_id
            else:
                # Both selected, replace first
                self.selected_segments[0] = segment_id
            
            self._update_display()
    
    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'enter':
            # Compute centerline
            if self.selected_segments[0] and self.selected_segments[1]:
                self._add_centerline(self.selected_segments[0], self.selected_segments[1])
                # Clear selection after adding
                self.selected_segments = [None, None]
                self._update_display()
        elif event.key == 'backspace':
            # Undo last centerline
            self._undo_last_centerline()
        elif event.key == 'escape':
            # Reset selection
            self.selected_segments = [None, None]
            self._update_display()
    
    def _add_centerline(self, seg1_id: str, seg2_id: str):
        """Add a new centerline between two segments, or remove if it already exists."""
        # Check for duplicates - if exists, remove it (toggle behavior)
        for i, record in enumerate(self.manual_centerlines):
            boundary_ids = record['boundary_ids']
            # Check both orders: (seg1, seg2) and (seg2, seg1)
            if (boundary_ids[0] == seg1_id and boundary_ids[1] == seg2_id) or \
               (boundary_ids[0] == seg2_id and boundary_ids[1] == seg1_id):
                # Pair exists, remove it
                removed = self.manual_centerlines.pop(i)
                self.save_centerlines()
                logger.info(f"Removed existing centerline: {removed['boundary_ids']}")
                self._update_display()
                return
        
        # Pair doesn't exist, add it
        centerline = self._compute_centerline(seg1_id, seg2_id)
        
        if centerline is None:
            logger.warning(f"Failed to compute centerline between {seg1_id} and {seg2_id}")
            return
        
        # Add to manual centerlines
        record = {
            'boundary_ids': [seg1_id, seg2_id],
            'centerline': {
                'type': 'LineString',
                'coordinates': [[float(x), float(y)] for x, y in centerline.coords]
            }
        }
        
        self.manual_centerlines.append(record)
        self.save_centerlines()
        logger.info(f"Added centerline between {seg1_id} and {seg2_id}")
    
    def _undo_last_centerline(self):
        """Remove the last added centerline."""
        if self.manual_centerlines:
            removed = self.manual_centerlines.pop()
            self.save_centerlines()
            logger.info(f"Removed centerline: {removed['boundary_ids']}")
            self._update_display()
    
    def _update_display(self):
        """Update the visualization display."""
        self.ax.clear()
        
        # Draw polygon fill (convert from micrometers to mm)
        poly_coords_mm = [(x / 1000.0, y / 1000.0) for x, y in self.polygon.exterior.coords]
        poly_patch = mpatches.Polygon(
            poly_coords_mm,
            fill=True, edgecolor='gray', facecolor='lightblue', 
            alpha=0.3, linewidth=1
        )
        self.ax.add_patch(poly_patch)
        
        # Draw boundary segments (convert from micrometers to mm)
        self.segment_patches = {}
        label_positions = []  # Store (x, y, segment_id) for overlap detection
        
        for segment in self.boundary_segments:
            is_selected = segment.segment_id in self.selected_segments
            
            color = 'red' if is_selected else 'black'
            linewidth = 3 if is_selected else 1
            alpha = 1.0 if is_selected else 0.7
            
            # Convert coordinates to mm
            x1_mm = segment.coords[0][0] / 1000.0
            y1_mm = segment.coords[0][1] / 1000.0
            x2_mm = segment.coords[1][0] / 1000.0
            y2_mm = segment.coords[1][1] / 1000.0
            
            line, = self.ax.plot(
                [x1_mm, x2_mm],
                [y1_mm, y2_mm],
                color=color, linewidth=linewidth, alpha=alpha,
                picker=True, pickradius=5
            )
            self.segment_patches[segment.segment_id] = line
            
            # Calculate initial label position (midpoint of segment, in mm)
            mid_x = (x1_mm + x2_mm) / 2
            mid_y = (y1_mm + y2_mm) / 2
            label_positions.append((mid_x, mid_y, segment.segment_id))
        
        # Adjust label positions to avoid overlaps (need to set bounds first for accurate box size estimation)
        # Set bounds temporarily for box size estimation (convert to mm)
        bounds = self.polygon.bounds
        margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        # Convert to mm for display
        xlim_mm = [(bounds[0] - margin) / 1000.0, (bounds[2] + margin) / 1000.0]
        ylim_mm = [(bounds[1] - margin) / 1000.0, (bounds[3] + margin) / 1000.0]
        self.ax.set_xlim(xlim_mm)
        self.ax.set_ylim(ylim_mm)
        
        # Adjust label positions to avoid overlaps (already in mm)
        adjusted_positions = self._adjust_label_positions(label_positions, fontsize=8)
        
        # Draw labels with adjusted positions (already in mm)
        for x, y, segment_id in adjusted_positions:
            self.ax.text(x, y, segment_id, fontsize=8,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Draw existing manual centerlines (convert from micrometers to mm)
        self.centerline_lines = []
        for i, record in enumerate(self.manual_centerlines):
            coords = record['centerline']['coordinates']
            # Convert from micrometers to mm
            xs = [c[0] / 1000.0 for c in coords]
            ys = [c[1] / 1000.0 for c in coords]
            line, = self.ax.plot(xs, ys, 'g-', linewidth=2, alpha=0.8, label=f'Centerline {i+1}')
            self.centerline_lines.append(line)
        
        # Set bounds (already converted to mm above)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_title('Manual Centerline Mode - Click 2 boundary segments, press Enter', 
                         fontsize=12, fontweight='bold')
        
        # Status overlay
        selected_text = f"Selected: {self.selected_segments[0] or 'None'}, {self.selected_segments[1] or 'None'}"
        count_text = f"Centerlines: {len(self.manual_centerlines)}"
        save_text = f"Save: {self.save_path}"
        
        status_text = f"{selected_text} | {count_text} | {save_text}"
        self.ax.text(0.02, 0.98, status_text, transform=self.ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Instructions
        instructions = "Click 2 segments → Enter | Backspace: Undo | Escape: Reset"
        self.ax.text(0.5, 0.02, instructions, transform=self.ax.transAxes,
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Update pairs list and delete all button
        self._update_pairs_list()
        
        self.fig.canvas.draw()
    
    def _update_pairs_list(self):
        """Update the pairs list display and delete all button."""
        # Remove old text
        if self.pairs_text:
            try:
                self.pairs_text.remove()
            except:
                pass
            self.pairs_text = None
        
        # Don't remove delete_all_button here - it's a persistent UI element
        # Only recreate it if it doesn't exist
        if self.delete_all_button is None and self.fig:
            ax_delete_all = plt.axes([0.70, 0.02, 0.12, 0.04])
            self.delete_all_button = Button(ax_delete_all, 'Delete All', 
                                           color='darkred', hovercolor='red')
            self.delete_all_button.on_clicked(self._delete_all_centerlines)
        
        # Remove old axes if it exists
        if self.pairs_ax:
            try:
                self.pairs_ax.remove()
            except:
                pass
        
        # Create pairs list area on the right side
        if self.fig:
            self.pairs_ax = self.fig.add_axes([0.70, 0.15, 0.28, 0.70])  # Right side, below title
        else:
            self.pairs_ax = plt.axes([0.70, 0.15, 0.28, 0.70])  # Fallback
        self.pairs_ax.axis('off')
        
        if len(self.manual_centerlines) > 0:
            # Build pairs list text
            pairs_lines = ["Line Pairs:", "=" * 25]
            for i, record in enumerate(self.manual_centerlines):
                boundary_ids = record['boundary_ids']
                pairs_lines.append(f"{i+1}. {boundary_ids[0]} ↔ {boundary_ids[1]}")
            
            pairs_text = "\n".join(pairs_lines)
            self.pairs_text = self.pairs_ax.text(0.05, 0.95, pairs_text, 
                                           transform=self.pairs_ax.transAxes,
                                           fontsize=9, verticalalignment='top',
                                           family='monospace',
                                           bbox=dict(boxstyle='round', 
                                                    facecolor='lightgray', 
                                                    alpha=0.8, edgecolor='black'))
        else:
            # Show empty state
            pairs_text = "Line Pairs:\n" + "=" * 25 + "\n(No pairs yet)"
            self.pairs_text = self.pairs_ax.text(0.05, 0.95, pairs_text, 
                                           transform=self.pairs_ax.transAxes,
                                           fontsize=9, verticalalignment='top',
                                           family='monospace',
                                           bbox=dict(boxstyle='round', 
                                                    facecolor='lightgray', 
                                                    alpha=0.8, edgecolor='black'))
    
    def _delete_all_centerlines(self, event):
        """Delete all centerlines."""
        if self.manual_centerlines:
            count = len(self.manual_centerlines)
            self.manual_centerlines.clear()
            self.save_centerlines()
            logger.info(f"Deleted all {count} centerlines")
            self._update_display()
    
    def _on_done(self, event):
        """Handle Done button click - close the window."""
        logger.info("Done button clicked - closing manual centerline mode")
        self.close()
    
    def show(self, block: bool = True):
        """Show the interactive manual centerline mode window."""
        self.fig, self.ax = plt.subplots(figsize=(16, 10))  # Wider to accommodate pairs list
        
        # Create Done button first (right side)
        ax_done = plt.axes([0.83, 0.02, 0.12, 0.04])
        self.done_button = Button(ax_done, 'Done', color='lightgreen', hovercolor='green')
        self.done_button.on_clicked(self._on_done)
        
        # Create Delete All button (left of Done, with gap)
        # Done button: x=0.83, width=0.12, so ends at 0.95
        # Delete All: x=0.70, width=0.12, so ends at 0.82
        # Gap between them: 0.83 - 0.82 = 0.01 (1% of figure width)
        ax_delete_all = plt.axes([0.70, 0.02, 0.12, 0.04])
        self.delete_all_button = Button(ax_delete_all, 'Delete All', 
                                       color='darkred', hovercolor='red')
        self.delete_all_button.on_clicked(self._delete_all_centerlines)
        
        # Adjust main plot area to leave room for pairs list
        self.ax.set_position([0.08, 0.08, 0.60, 0.85])  # Left side, leave right for pairs
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        # Initial display
        self._update_display()
        
        # Show window
        try:
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        except Exception:
            pass
        
        if block:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(0.1)
    
    def close(self):
        """Close the visualization window."""
        # Clean up buttons
        if self.delete_all_button:
            try:
                self.delete_all_button.disconnect_events()
                self.delete_all_button.ax.remove()
            except:
                pass
            self.delete_all_button = None
        
        if self.pairs_text:
            try:
                self.pairs_text.remove()
            except:
                pass
            self.pairs_text = None
        
        if self.pairs_ax:
            try:
                self.pairs_ax.remove()
            except:
                pass
            self.pairs_ax = None
        
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.done_button = None
    
    def get_centerlines(self) -> List[Dict[str, Any]]:
        """Get all manual centerlines."""
        return self.manual_centerlines.copy()

