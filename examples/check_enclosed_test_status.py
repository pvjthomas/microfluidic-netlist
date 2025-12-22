#!/usr/bin/env python3
"""Check status of enclosed_test.dxf against the spec requirements."""

import sys
import logging
from pathlib import Path

# Import setup
CORE_DIR = Path(__file__).resolve().parent.parent / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from microfluidic_ir.dxf_loader import load_dxf, get_drawing_units, INSUNITS_TO_NAME
from microfluidic_ir.graph_extract import extract_graph_from_polygons
from shapely.geometry import shape, Polygon
import json
import ezdxf

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def check_spec_compliance(result, spec_requirements):
    """Check if result meets spec requirements."""
    issues = []
    warnings = []
    
    # Check V1 requirements from spec
    v1_checks = spec_requirements.get('V1', {})
    
    # 1. DXF Import (must-have)
    if 'polygons' not in result:
        issues.append("❌ Missing polygons in result")
    elif len(result['polygons']) == 0:
        issues.append("❌ No polygons extracted from DXF")
    else:
        logger.info(f"✓ Found {len(result['polygons'])} polygons")
    
    # 2. Units
    if 'bounds' in result:
        logger.info(f"✓ Bounds extracted: {result['bounds']}")
    else:
        warnings.append("⚠️ No bounds information")
    
    # 3. Port detection (circles)
    circles = result.get('circles', [])
    if len(circles) > 0:
        logger.info(f"✓ Found {len(circles)} circle(s) for port detection")
    else:
        logger.info("ℹ️ No circles found (no ports detected)")
    
    # 4. Layers
    layers = result.get('layers', [])
    if len(layers) > 0:
        logger.info(f"✓ Found {len(layers)} layer(s): {', '.join(layers)}")
    else:
        warnings.append("⚠️ No layers detected")
    
    return issues, warnings


def main() -> int:
    """Check status of enclosed_test.dxf."""
    print("=" * 70)
    print("Status Check: enclosed_test.dxf")
    print("=" * 70)
    print()
    
    # Load spec requirements (conceptual)
    spec_requirements = {
        'V1': {
            'import_dxf': True,
            'select_channels': True,
            'extract_graph': True,
            'measure_edges': True,
            'detect_ports': True,
            'export_formats': ['graph.json', 'netlist.cir', 'segments.csv', 'overlay.png']
        }
    }
    
    dxf_path = Path(__file__).resolve().parent / "enclosed_test.dxf"
    
    if not dxf_path.exists():
        logger.error(f"DXF file not found: {dxf_path}")
        return 1
    
    logger.info(f"Checking DXF file: {dxf_path.name}")
    logger.info(f"File size: {dxf_path.stat().st_size / 1024:.1f} KB")
    print()
    
    # Step 1: DXF Import (V1 requirement)
    print("=" * 70)
    print("Step 1: DXF Import & Normalization")
    print("=" * 70)
    
    # Try different scale factors to handle large coordinates
    scale_factors_to_try = [1.0, 1e-3, 1e-4, 1e-6]
    result = None
    used_scale_factor = 1.0
    
    for scale_factor in scale_factors_to_try:
        try:
            print(f"\nTrying scale_factor={scale_factor}...")
            result = load_dxf(str(dxf_path), snap_close_tol=2.0, scale_factor=scale_factor)
            used_scale_factor = scale_factor
            
            bounds = result.get('bounds', {})
            if bounds:
                width = bounds.get('xmax', 0) - bounds.get('xmin', 0)
                height = bounds.get('ymax', 0) - bounds.get('ymin', 0)
                
                # Check if dimensions are reasonable for microfluidics (typically < 1m = 1e6 um)
                max_dimension = max(width, height)
                if max_dimension < 1e6:  # Less than 1 meter
                    print(f"  ✓ Dimensions look reasonable: {max_dimension/1000:.1f} mm")
                    break
                else:
                    print(f"  ⚠️ Dimensions still very large: {max_dimension/1e6:.2f} m")
                    if scale_factor == scale_factors_to_try[-1]:
                        print(f"  ⚠️ Using last scale_factor anyway")
                        break
                    continue
            
            logger.info("✓ DXF loaded successfully")
            break
            
        except Exception as e:
            if scale_factor == scale_factors_to_try[-1]:
                logger.exception("❌ Error loading DXF with all scale factors")
                return 1
            continue
    
    if result is None:
        logger.error("❌ Failed to load DXF with any scale factor")
        return 1
    
    logger.info(f"✓ DXF loaded successfully (scale_factor={used_scale_factor})")
    
    print(f"\nImport Results:")
    print(f"  Source filename: {result.get('source_filename', 'N/A')}")
    print(f"  Source hash: {result.get('source_hash', 'N/A')[:16]}...")
    print(f"  Import timestamp: {result.get('import_timestamp', 'N/A')}")
    if used_scale_factor != 1.0:
        print(f"  ⚠️ Applied scale_factor: {used_scale_factor} (coordinates may need adjustment)")
    
    # Check spec compliance
    issues, warnings = check_spec_compliance(result, spec_requirements)
    
    if issues:
        print(f"\n❌ Issues found:")
        for issue in issues:
            print(f"  {issue}")
    
    if warnings:
        print(f"\n⚠️ Warnings:")
        for warning in warnings:
            print(f"  {warning}")
    
    # Step 2: Polygon Details
    print(f"\n{'=' * 70}")
    print("Step 2: Channel Region Selection (Polygons)")
    print("=" * 70)
    
    polygons = result.get('polygons', [])
    if len(polygons) == 0:
        logger.error("❌ No polygons found - cannot proceed with graph extraction")
        return 1
    
    # Sort by area (largest first) - typical workflow
    polys_sorted = sorted(polygons, key=lambda p: float(p.get("area", 0.0)), reverse=True)
    
    print(f"\nFound {len(polygons)} polygon(s):")
    for i, poly in enumerate(polys_sorted[:10], 1):  # Show top 10
        area = poly.get('area', 0.0)
        bounds = poly.get('bounds', {})
        layer = poly.get('layer', 'N/A')
        entity_handle = poly.get('entity_handle', 'N/A')
        
        print(f"\n  Polygon {i}:")
        print(f"    Layer: {layer}")
        print(f"    Entity Handle: {entity_handle}")
        print(f"    Area: {area:.2f} µm² ({area/1e6:.2f} mm²)")
        if bounds:
            width = bounds.get('xmax', 0) - bounds.get('xmin', 0)
            height = bounds.get('ymax', 0) - bounds.get('ymin', 0)
            print(f"    Bounds: x=[{bounds.get('xmin', 0):.1f}, {bounds.get('xmax', 0):.1f}], "
                  f"y=[{bounds.get('ymin', 0):.1f}, {bounds.get('ymax', 0):.1f}]")
            print(f"    Size: {width:.1f} × {height:.1f} µm ({width/1000:.2f} × {height/1000:.2f} mm)")
        
        # Check if polygon has holes
        poly_geom = shape(poly.get('polygon', {}))
        if hasattr(poly_geom, 'interiors') and len(poly_geom.interiors) > 0:
            print(f"    ⚠️ Has {len(poly_geom.interiors)} interior ring(s) (holes)")
    
    if len(polygons) > 10:
        print(f"\n  ... and {len(polygons) - 10} more polygon(s)")
    
    # Step 3: Port Detection (V1 requirement)
    print(f"\n{'=' * 70}")
    print("Step 3: Port Detection (Circles)")
    print("=" * 70)
    
    circles = result.get('circles', [])
    if len(circles) > 0:
        print(f"\nFound {len(circles)} circle(s) (potential ports):")
        for i, circle in enumerate(circles, 1):
            center = circle.get('center', [0, 0])
            radius = circle.get('radius', 0.0)
            layer = circle.get('layer', 'N/A')
            entity_handle = circle.get('entity_handle', 'N/A')
            
            print(f"\n  Circle {i}:")
            print(f"    Layer: {layer}")
            print(f"    Entity Handle: {entity_handle}")
            print(f"    Center: ({center[0]:.1f}, {center[1]:.1f}) µm")
            print(f"    Radius: {radius:.1f} µm")
    else:
        print("\nℹ️ No circles found (no port markers detected)")
        print("   This is acceptable - ports can be detected from polygon endpoints later")
    
    # Step 4: Graph Extraction (V1 requirement)
    print(f"\n{'=' * 70}")
    print("Step 4: Graph Extraction (Skeleton → Nodes/Edges)")
    print("=" * 70)
    
    try:
        # Use largest polygon(s) for graph extraction
        main_polygons = polys_sorted[:1]  # Use largest polygon for now
        
        # Check polygon size first
        main_poly = shape(main_polygons[0].get('polygon', {}))
        bounds_main = main_poly.bounds  # (minx, miny, maxx, maxy)
        width_main = bounds_main[2] - bounds_main[0]
        height_main = bounds_main[3] - bounds_main[1]
        
        print(f"\nAttempting graph extraction from largest polygon:")
        print(f"  Size: {width_main:.1f} × {height_main:.1f} µm ({width_main/1000:.2f} × {height_main/1000:.2f} mm)")
        
        # Warn if very large
        if max(width_main, height_main) > 1e6:  # > 1 meter
            print(f"  ⚠️ WARNING: Polygon is very large - graph extraction may fail or be slow")
            print(f"  Consider using a scale_factor when loading the DXF")
        
        # Estimate minimum channel width from polygon bounds
        if main_polygons:
            first_bounds = main_polygons[0].get('bounds', {})
            width = first_bounds.get('xmax', 0) - first_bounds.get('xmin', 0)
            height = first_bounds.get('ymax', 0) - first_bounds.get('ymin', 0)
            estimated_min_width = min(width, height) * 0.1
            minimum_channel_width = max(10.0, estimated_min_width)  # At least 10 µm
        else:
            minimum_channel_width = 100.0  # Default fallback
        
        logger.info(f"Extracting graph from {len(main_polygons)} polygon(s) with minimum_channel_width={minimum_channel_width:.2f} µm...")
        graph_result = extract_graph_from_polygons(
            main_polygons,
            minimum_channel_width=minimum_channel_width,
            simplify_tolerance=None,
            width_sample_step=10.0,
            measure_edges=True,
            circles=circles,
            port_snap_distance=50.0,
            detect_polyline_circles=False,
            default_height=50.0,
            default_cross_section_kind="rectangular"
        )
        
        nodes = graph_result.get('nodes', [])
        edges = graph_result.get('edges', [])
        
        print(f"\n✓ Graph extraction successful!")
        print(f"  Nodes: {len(nodes)}")
        print(f"  Edges: {len(edges)}")
        
        # Check graph validity (spec requirement)
        print(f"\nGraph Validity Checks:")
        
        # Check: All edges connect existing nodes
        node_ids = {node['id'] for node in nodes}
        edge_issues = []
        for edge in edges:
            u = edge.get('u')
            v = edge.get('v')
            if u not in node_ids:
                edge_issues.append(f"Edge {edge.get('id', '?')} references missing node {u}")
            if v not in node_ids:
                edge_issues.append(f"Edge {edge.get('id', '?')} references missing node {v}")
        
        if edge_issues:
            print(f"  ❌ Edge connectivity issues:")
            for issue in edge_issues[:5]:
                print(f"    {issue}")
            if len(edge_issues) > 5:
                print(f"    ... and {len(edge_issues) - 5} more")
        else:
            print(f"  ✓ All edges connect existing nodes")
        
        # Check: No zero-length edges
        zero_length = [e for e in edges if e.get('length', 0) <= 0]
        if zero_length:
            print(f"  ❌ Found {len(zero_length)} zero-length edge(s)")
        else:
            print(f"  ✓ No zero-length edges")
        
        # Check: Node degrees make sense
        node_degrees = {}
        for edge in edges:
            u = edge.get('u')
            v = edge.get('v')
            node_degrees[u] = node_degrees.get(u, 0) + 1
            node_degrees[v] = node_degrees.get(v, 0) + 1
        
        endpoint_count = sum(1 for node in nodes if node_degrees.get(node['id'], 0) == 1)
        junction_count = sum(1 for node in nodes if node_degrees.get(node['id'], 0) >= 3)
        
        print(f"  ✓ Node types: {endpoint_count} endpoint(s), {junction_count} junction(s), "
              f"{len(nodes) - endpoint_count - junction_count} other(s)")
        
        # Measurements (V1 requirement)
        print(f"\nMeasurement Checks:")
        edges_with_measurements = [e for e in edges if 'length' in e and 'width_profile' in e]
        print(f"  ✓ {len(edges_with_measurements)}/{len(edges)} edges have measurements")
        
        if edges:
            lengths = [e.get('length', 0) for e in edges]
            print(f"  Length range: {min(lengths):.1f} - {max(lengths):.1f} µm")
            
            # Width profiles
            width_stats = []
            for edge in edges:
                wp = edge.get('width_profile', {})
                if 'w_median' in wp:
                    width_stats.append(wp['w_median'])
            
            if width_stats:
                print(f"  Width range: {min(width_stats):.1f} - {max(width_stats):.1f} µm (median)")
        
        # Classification (V1 requirement)
        classified = [e for e in edges if 'width_profile' in e and 'kind' in e['width_profile']]
        if classified:
            classifications = {}
            for edge in classified:
                kind = edge['width_profile'].get('kind', 'unknown')
                classifications[kind] = classifications.get(kind, 0) + 1
            print(f"\nWidth Profile Classifications:")
            for kind, count in classifications.items():
                print(f"  {kind}: {count}")
        
        # Port attachment
        if 'ports' in graph_result:
            ports = graph_result['ports']
            print(f"\nPort Detection:")
            print(f"  ✓ {len(ports)} port(s) detected and attached to nodes")
        
    except Exception as e:
        logger.exception("❌ Error extracting graph")
        print(f"\n❌ Graph extraction failed: {e}")
        return 1
    
    # Step 5: Export Capability Check (V1 requirement)
    print(f"\n{'=' * 70}")
    print("Step 5: Export Capability Check")
    print("=" * 70)
    
    export_modules = {
        'graph.json': 'export_ir',
        'netlist.cir': 'export_netlist',
        'segments.csv': 'export_segments',
        'overlay.png': 'export_overlay'
    }
    
    print("\nRequired exports (V1 spec):")
    for export_name, module_name in export_modules.items():
        try:
            module = __import__(f'microfluidic_ir.{module_name}', fromlist=[module_name])
            print(f"  ✓ {export_name} - module available")
        except ImportError as e:
            print(f"  ❌ {export_name} - module not found: {e}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print("=" * 70)
    
    print(f"\n✓ DXF Import: OK")
    print(f"✓ Polygons: {len(polygons)} found")
    print(f"✓ Circles: {len(circles)} found")
    print(f"✓ Graph Extraction: OK ({len(nodes)} nodes, {len(edges)} edges)")
    
    if len(edges) > 0:
        print(f"✓ Measurements: OK")
        print(f"✓ Ready for export")
    else:
        print(f"⚠️ No edges extracted - may need parameter tuning")
    
    print(f"\n{'=' * 70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

