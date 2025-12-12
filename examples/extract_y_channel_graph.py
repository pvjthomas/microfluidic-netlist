#!/usr/bin/env python3
"""Extract network graph from y-channel.dxf with optimized parameters."""

import sys
import logging
import time
from pathlib import Path
from contextlib import contextmanager

# Add core directory to path for imports (fallback if package not installed)
# Note: Package should be installed via: pip install -e core/
# This path manipulation is a fallback for development
core_dir = Path(__file__).parent.parent / "core"
if str(core_dir) not in sys.path:
    sys.path.insert(0, str(core_dir))

# These imports work at runtime (package installed in editable mode)
# IDE warnings here are false positives - static analysis doesn't see runtime path setup
from microfluidic_ir.dxf_loader import load_dxf  # noqa: E402
from microfluidic_ir.graph_extract import extract_graph_from_polygons  # noqa: E402
import json

# Setup logging to both console and file
log_file = Path(__file__).parent / "extract_y_channel_graph.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")


@contextmanager
def log_step(step_name):
    """Context manager for logging step entry/exit with timing."""
    logger.info(f"→ {step_name}")
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"← {step_name} ({elapsed:.2f}s)")


def extract_y_channel_graph():
    """Extract network graph from y-channel.dxf."""
    
    logger.info("=" * 70)
    logger.info("Extracting Network Graph from y-channel.dxf")
    logger.info("=" * 70)
    
    dxf_path = Path(__file__).parent / "y-channel.dxf"
    
    if not dxf_path.exists():
        logger.error(f"DXF file not found at {dxf_path}")
        return None
    
    # Step 1: Load DXF
    try:
        with log_step("DXF load"):
            result = load_dxf(str(dxf_path), snap_close_tol=2.0)
        logger.info(f"Loaded: {len(result['polygons'])} polygon(s), {len(result['circles'])} circle(s)")
    except Exception as e:
        logger.error(f"Error loading DXF: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    if not result['polygons']:
        logger.error("No polygons found in DXF")
        return None
    
    # Step 2: Determine optimal parameters based on polygon size
    poly = result['polygons'][0]
    bounds = poly['bounds']
    width = bounds['xmax'] - bounds['xmin']
    height = bounds['ymax'] - bounds['ymin']
    max_dim = max(width, height)
    
    logger.info(f"Polygon size: {width:.1f} × {height:.1f} (max: {max_dim:.1f})")
    
    # For very large coordinates, use very low resolution to avoid memory issues
    if max_dim > 1000000:
        px_per_unit = 0.001  # Very low resolution for huge coordinates
        simplify_tolerance = 5000.0
    elif max_dim > 100000:
        px_per_unit = 0.01
        simplify_tolerance = 500.0
    elif max_dim > 10000:
        px_per_unit = 0.1
        simplify_tolerance = 100.0
    else:
        px_per_unit = 10.0
        simplify_tolerance = 1.0
    
    logger.info(f"Parameters: px_per_unit={px_per_unit}, simplify_tolerance={simplify_tolerance}")
    
    # Step 3: Extract graph
    try:
        with log_step("Graph extraction"):
            graph_result = extract_graph_from_polygons(
                result['polygons'],
                px_per_unit=px_per_unit,
                simplify_tolerance=simplify_tolerance
            )
        
        nodes = graph_result['nodes']
        edges = graph_result['edges']
        
        logger.info(f"Graph: {len(nodes)} nodes, {len(edges)} edges")
        
        # Analyze nodes by kind
        node_kinds = {}
        for node in nodes:
            kind = node['kind']
            node_kinds[kind] = node_kinds.get(kind, 0) + 1
        
        for kind, count in sorted(node_kinds.items()):
            logger.info(f"  {kind}: {count}")
        
        # Show node details
        for node in nodes:
            logger.info(f"  {node['id']}: {node['kind']} at ({node['xy'][0]:.1f}, {node['xy'][1]:.1f}), "
                       f"degree={node.get('degree', 'N/A')}")
        
        # Show edge details
        for edge in edges:
            centerline = edge['centerline']['coordinates']
            # Calculate length
            length = 0.0
            for i in range(len(centerline) - 1):
                dx = centerline[i+1][0] - centerline[i][0]
                dy = centerline[i+1][1] - centerline[i][1]
                length += (dx*dx + dy*dy)**0.5
            
            logger.info(f"  {edge['id']}: {edge['u']} → {edge['v']}, "
                       f"length={length:.1f}, points={len(centerline)}")
        
        # Validate graph
        node_ids = {node['id'] for node in nodes}
        edge_errors = [e for e in edges if e['u'] not in node_ids or e['v'] not in node_ids]
        
        if edge_errors:
            logger.warning(f"Found {len(edge_errors)} edges with invalid node references")
        else:
            logger.info("All edges connect existing nodes")
        
        # Check: No zero-length edges
        zero_length = []
        for edge in edges:
            centerline = edge['centerline']['coordinates']
            if len(centerline) < 2:
                zero_length.append(edge['id'])
            else:
                total_length = sum(
                    ((centerline[i+1][0] - centerline[i][0])**2 + 
                     (centerline[i+1][1] - centerline[i][1])**2)**0.5
                    for i in range(len(centerline) - 1)
                )
                if total_length < 1.0:
                    zero_length.append(edge['id'])
        
        if zero_length:
            logger.warning(f"Found {len(zero_length)} zero-length edges: {zero_length}")
        else:
            logger.info("No zero-length edges")
        
        # Save result to JSON (without skeleton_graph)
        output_path = Path(__file__).parent / "y_channel_graph.json"
        serializable_result = {
            'nodes': nodes,
            'edges': edges
        }
        with open(output_path, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        
        logger.info(f"Graph saved to: {output_path}")
        logger.info("=" * 70)
        logger.info("Network graph extraction completed successfully")
        logger.info(f"Full log saved to: {log_file}")
        
        return graph_result
        
    except Exception as e:
        logger.error(f"Error extracting graph: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = extract_y_channel_graph()
    exit(0 if result else 1)

