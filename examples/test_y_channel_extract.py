#!/usr/bin/env python3
"""Extract network graph from y-channel.dxf per spec requirements."""

from pathlib import Path
from microfluidic_ir.dxf_loader import load_dxf
from microfluidic_ir.graph_extract import extract_graph_from_polygons
from shapely.geometry import Point, Polygon
import json


def extract_y_channel_graph():
    """Extract network graph from y-channel.dxf following spec section C."""
    
    print("=" * 70)
    print("Extracting Network Graph from y-channel.dxf")
    print("Per spec: Section C - Skeleton → graph extraction")
    print("=" * 70)
    
    dxf_path = Path(__file__).parent / "y-channel.dxf"
    
    if not dxf_path.exists():
        print(f"Error: DXF file not found at {dxf_path}")
        return None
    
    # Step 1: Load DXF (Section A: DXF import & normalization)
    print("\n[Step 1] Loading DXF file...")
    try:
        result = load_dxf(str(dxf_path), snap_close_tol=2.0)
        print(f"✓ Loaded: {len(result['polygons'])} polygon(s), {len(result['circles'])} circle(s)")
        print(f"  Layers: {result['layers']}")
    except Exception as e:
        print(f"✗ Error loading DXF: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 2: Channel region selection (Section B)
    print("\n[Step 2] Channel region selection...")
    
    # For y-channel.dxf, select all polygons (typically one polygon on layer '0')
    selected_polygons = result['polygons']
    
    if not selected_polygons:
        print("✗ No polygons found in DXF")
        return None
    
    print(f"✓ Selected {len(selected_polygons)} polygon(s) for processing")
    
    # Show polygon details
    for i, poly in enumerate(selected_polygons, 1):
        print(f"  Polygon {i}:")
        print(f"    Layer: {poly['layer']}")
        print(f"    Area: {poly['area']:.2f}")
        print(f"    Bounds: x=[{poly['bounds']['xmin']:.1f}, {poly['bounds']['xmax']:.1f}], "
              f"y=[{poly['bounds']['ymin']:.1f}, {poly['bounds']['ymax']:.1f}]")
    
    # Step 3: Extract network graph (Section C: Skeleton → graph extraction)
    print("\n[Step 3] Extracting network graph (skeletonization)...")
    print("  Per spec: Rasterize → Skeletonize → Convert to graph")
    
    # Determine appropriate minimum channel width based on polygon size
    if selected_polygons:
        bounds = selected_polygons[0]['bounds']
        width = bounds['xmax'] - bounds['xmin']
        height = bounds['ymax'] - bounds['ymin']
        max_dim = max(width, height)
        
        # Estimate minimum channel width (use 10% of smaller dimension, with reasonable bounds)
        min_dim = min(width, height)
        estimated_min_width = min_dim * 0.1
        minimum_channel_width = max(10.0, min(estimated_min_width, 1000.0))  # Between 10 and 1000 µm
        
        # um_per_px will be calculated as ceil(minimum_channel_width / 10)
        simplify_tolerance = None  # Will be auto-computed
        
        print(f"  Polygon size: {max_dim:.1f} units")
        print(f"  Estimated minimum channel width: {minimum_channel_width:.1f} µm")
        print(f"  um_per_px will be: {__import__('math').ceil(minimum_channel_width / 10.0):.1f}")
    
    try:
        graph_result = extract_graph_from_polygons(
            selected_polygons,
            minimum_channel_width=minimum_channel_width,
            simplify_tolerance=simplify_tolerance
        )
        
        nodes = graph_result['nodes']
        edges = graph_result['edges']
        
        print(f"\n✓ Graph extraction complete!")
        print(f"\nGraph Statistics:")
        print(f"  Nodes: {len(nodes)}")
        print(f"  Edges: {len(edges)}")
        
        # Analyze nodes by kind (per spec: nodes at junctions + endpoints + ports)
        node_kinds = {}
        for node in nodes:
            kind = node['kind']
            node_kinds[kind] = node_kinds.get(kind, 0) + 1
        
        print(f"\n  Node breakdown:")
        for kind, count in sorted(node_kinds.items()):
            print(f"    {kind}: {count}")
        
        # Show node details
        print(f"\n  Nodes:")
        for node in nodes:
            print(f"    {node['id']}: {node['kind']} at ({node['xy'][0]:.1f}, {node['xy'][1]:.1f}), "
                  f"degree={node.get('degree', 'N/A')}")
        
        # Show edge details
        print(f"\n  Edges:")
        for edge in edges:
            centerline = edge['centerline']['coordinates']
            # Calculate length
            length = 0.0
            for i in range(len(centerline) - 1):
                dx = centerline[i+1][0] - centerline[i][0]
                dy = centerline[i+1][1] - centerline[i][1]
                length += (dx*dx + dy*dy)**0.5
            
            print(f"    {edge['id']}: {edge['u']} → {edge['v']}, "
                  f"length={length:.1f}, points={len(centerline)}")
        
        # Validate graph (per spec "Acceptance tests")
        print(f"\n" + "="*70)
        print("Graph Validation (per spec acceptance tests)")
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
            for err in edge_errors:
                print(f"  {err}")
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
                if total_length < 1.0:
                    zero_length_edges.append(edge['id'])
        
        if zero_length_edges:
            print(f"⚠ Found {len(zero_length_edges)} zero-length edges: {zero_length_edges}")
        else:
            print(f"✓ No zero-length edges")
        
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
            for issue in degree_issues:
                print(f"  {issue}")
        else:
            print(f"✓ Node degrees are consistent with node kinds")
        
        print(f"\n" + "="*70)
        print("✅ Network graph extraction completed successfully")
        print("="*70)
        
        return graph_result
        
    except Exception as e:
        print(f"✗ Error extracting graph: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = extract_y_channel_graph()
    exit(0 if result else 1)

