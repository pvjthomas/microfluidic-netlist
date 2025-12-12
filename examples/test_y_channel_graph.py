#!/usr/bin/env python3
"""Test finding interior and extracting network graph from y-channel.dxf."""

from pathlib import Path
from microfluidic_ir.dxf_loader import load_dxf
from microfluidic_ir.graph_extract import extract_graph_from_polygons
from shapely.geometry import Point, Polygon
import json


def test_y_channel_graph_extraction():
    """Test finding interior and extracting graph from y-channel.dxf per spec."""
    
    print("=" * 70)
    print("Testing y-channel.dxf: Find Interior & Extract Network Graph")
    print("=" * 70)
    
    dxf_path = Path(__file__).parent / "y-channel.dxf"
    
    if not dxf_path.exists():
        print(f"Error: DXF file not found at {dxf_path}")
        return False
    
    # Step 1: Load DXF
    print("\n" + "="*70)
    print("Step 1: Load DXF")
    print("="*70)
    
    try:
        result = load_dxf(str(dxf_path), snap_close_tol=2.0)
        print(f"✓ Successfully loaded DXF")
        print(f"  Layers: {result['layers']}")
        print(f"  Polygons: {len(result['polygons'])}")
        print(f"  Circles: {len(result['circles'])}")
    except Exception as e:
        print(f"✗ Error loading DXF: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Find interior (channel region selection)
    print("\n" + "="*70)
    print("Step 2: Find Interior (Channel Region Selection)")
    print("="*70)
    
    # Method A: Layer selection (if layer name is known)
    print("\nMethod A: Layer Selection")
    print("-" * 70)
    
    # Check what layers we have
    for layer in result['layers']:
        layer_polygons = [p for p in result['polygons'] if p['layer'] == layer]
        print(f"  Layer '{layer}': {len(layer_polygons)} polygon(s)")
        for i, poly in enumerate(layer_polygons, 1):
            print(f"    Polygon {i}: area={poly['area']:.2f}, "
                  f"bounds=[{poly['bounds']['xmin']:.1f}, {poly['bounds']['xmax']:.1f}, "
                  f"{poly['bounds']['ymin']:.1f}, {poly['bounds']['ymax']:.1f}]")
    
    # Select all polygons (for y-channel, likely just one polygon on layer '0')
    selected_polygons = result['polygons']
    print(f"\n  Selected {len(selected_polygons)} polygon(s) for processing")
    
    # Method B: Click-to-select interior (test with a point inside)
    print("\nMethod B: Click-to-Select Interior")
    print("-" * 70)
    
    # Find a point inside the polygon (use centroid)
    if selected_polygons:
        test_poly = selected_polygons[0]
        coords = test_poly['polygon']['coordinates'][0]
        poly_geom = Polygon(coords)
        centroid = poly_geom.centroid
        
        print(f"  Testing click at polygon centroid: ({centroid.x:.1f}, {centroid.y:.1f})")
        
        # Verify it's inside
        if poly_geom.contains(centroid):
            print(f"  ✓ Point is inside polygon")
        else:
            print(f"  ⚠ Point may be on boundary")
        
        # Test a few more points
        test_points = [
            (0, 0, "Origin"),
            (centroid.x, centroid.y, "Centroid"),
            (centroid.x + 1000, centroid.y, "Offset from centroid"),
        ]
        
        for x, y, desc in test_points:
            click_point = Point(x, y)
            found = []
            for i, poly_data in enumerate(result['polygons']):
                coords = poly_data['polygon']['coordinates'][0]
                poly_geom = Polygon(coords)
                if poly_geom.contains(click_point) or poly_geom.touches(click_point):
                    found.append((i, poly_data))
            
            if found:
                print(f"    Click at ({x:.1f}, {y:.1f}) - {desc}: ✓ Found {len(found)} polygon(s)")
            else:
                print(f"    Click at ({x:.1f}, {y:.1f}) - {desc}: ✗ Not found")
    
    # Step 3: Extract network graph (skeleton → graph)
    print("\n" + "="*70)
    print("Step 3: Extract Network Graph (Skeleton → Graph)")
    print("="*70)
    
    if not selected_polygons:
        print("✗ No polygons selected, cannot extract graph")
        return False
    
    try:
        # Extract graph with reasonable resolution
        # Note: y-channel.dxf has very large coordinates (millions), so we need lower px_per_unit
        # or we'll run out of memory. Let's use a lower resolution.
        px_per_unit = 0.1  # Lower resolution for large coordinates
        simplify_tolerance = 100.0  # Larger tolerance for large coordinates
        
        print(f"\nExtracting graph with:")
        print(f"  px_per_unit: {px_per_unit} (lower for large coordinates)")
        print(f"  simplify_tolerance: {simplify_tolerance}")
        
        graph_result = extract_graph_from_polygons(
            selected_polygons,
            px_per_unit=px_per_unit,
            simplify_tolerance=simplify_tolerance
        )
        
        print(f"\n✓ Graph extraction complete")
        
        # Analyze results
        nodes = graph_result['nodes']
        edges = graph_result['edges']
        
        print(f"\nGraph Statistics:")
        print(f"  Nodes: {len(nodes)}")
        print(f"  Edges: {len(edges)}")
        
        # Analyze nodes by kind
        node_kinds = {}
        for node in nodes:
            kind = node['kind']
            node_kinds[kind] = node_kinds.get(kind, 0) + 1
        
        print(f"\n  Node breakdown:")
        for kind, count in node_kinds.items():
            print(f"    {kind}: {count}")
        
        # Show node details
        print(f"\n  Node details:")
        for node in nodes[:10]:  # Show first 10
            print(f"    {node['id']}: {node['kind']} at ({node['xy'][0]:.1f}, {node['xy'][1]:.1f}), "
                  f"degree={node.get('degree', 'N/A')}")
        if len(nodes) > 10:
            print(f"    ... and {len(nodes) - 10} more nodes")
        
        # Show edge details
        print(f"\n  Edge details:")
        for edge in edges[:10]:  # Show first 10
            centerline = edge['centerline']['coordinates']
            # Calculate approximate length
            length = 0.0
            for i in range(len(centerline) - 1):
                dx = centerline[i+1][0] - centerline[i][0]
                dy = centerline[i+1][1] - centerline[i][1]
                length += (dx*dx + dy*dy)**0.5
            
            print(f"    {edge['id']}: {edge['u']} → {edge['v']}, "
                  f"length≈{length:.1f}, points={len(centerline)}")
        if len(edges) > 10:
            print(f"    ... and {len(edges) - 10} more edges")
        
        # Validate graph structure (per spec section "Acceptance tests")
        print(f"\n" + "="*70)
        print("Graph Validation (per spec)")
        print("="*70)
        
        # Check: All edges connect existing nodes
        node_ids = {node['id'] for node in nodes}
        edge_errors = []
        for edge in edges:
            if edge['u'] not in node_ids:
                edge_errors.append(f"Edge {edge['id']}: node {edge['u']} not found")
            if edge['v'] not in node_ids:
                edge_errors.append(f"Edge {edge['id']}: node {edge['v']} not found")
        
        if edge_errors:
            print(f"✗ Graph validation errors:")
            for err in edge_errors[:5]:
                print(f"  {err}")
            if len(edge_errors) > 5:
                print(f"  ... and {len(edge_errors) - 5} more errors")
        else:
            print(f"✓ All edges connect existing nodes")
        
        # Check: No zero-length edges
        zero_length_edges = []
        for edge in edges:
            centerline = edge['centerline']['coordinates']
            if len(centerline) < 2:
                zero_length_edges.append(edge['id'])
            else:
                total_length = 0.0
                for i in range(len(centerline) - 1):
                    dx = centerline[i+1][0] - centerline[i][0]
                    dy = centerline[i+1][1] - centerline[i][1]
                    total_length += (dx*dx + dy*dy)**0.5
                if total_length < 1.0:  # Very small threshold
                    zero_length_edges.append(edge['id'])
        
        if zero_length_edges:
            print(f"⚠ Found {len(zero_length_edges)} potentially zero-length edges: {zero_length_edges[:5]}")
        else:
            print(f"✓ No zero-length edges detected")
        
        # Check: Node degrees make sense
        degree_issues = []
        for node in nodes:
            kind = node['kind']
            degree = node.get('degree', 0)
            if kind == 'endpoint' and degree != 1:
                degree_issues.append(f"{node['id']}: endpoint with degree {degree} (expected 1)")
            elif kind == 'junction' and degree < 3:
                degree_issues.append(f"{node['id']}: junction with degree {degree} (expected >= 3)")
        
        if degree_issues:
            print(f"⚠ Node degree issues:")
            for issue in degree_issues[:5]:
                print(f"  {issue}")
        else:
            print(f"✓ Node degrees are consistent with node kinds")
        
        print(f"\n" + "="*70)
        print("✅ Graph extraction test completed successfully")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"✗ Error extracting graph: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_y_channel_graph_extraction()
    exit(0 if success else 1)


