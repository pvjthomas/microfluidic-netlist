#!/usr/bin/env python3
"""Test skeletonization and graph extraction."""

from pathlib import Path
from microfluidic_ir.dxf_loader import load_dxf
from microfluidic_ir.graph_extract import extract_graph_from_polygons
import json

def test_skeletonization():
    """Test skeletonization on Y-channel."""
    dxf_path = Path(__file__).parent / "y_channel_100um_4mmlegs.dxf"
    
    print("Loading DXF...")
    result = load_dxf(str(dxf_path), snap_close_tol=2.0)
    
    # Get CHANNEL polygon
    channel_polygons = [p for p in result['polygons'] if p['layer'] == 'CHANNEL']
    
    if not channel_polygons:
        print("No CHANNEL polygons found!")
        return
    
    print(f"\nFound {len(channel_polygons)} channel polygon(s)")
    print(f"Extracting graph with minimum_channel_width=100.0 µm...")
    
    # Extract graph
    # um_per_px will be calculated as ceil(100 / 3) = 34
    graph_result = extract_graph_from_polygons(
        channel_polygons,
        minimum_channel_width=100.0,
        simplify_tolerance=1.0
    )
    
    print(f"\n✓ Graph extraction complete")
    print(f"\nNodes found: {len(graph_result['nodes'])}")
    for node in graph_result['nodes']:
        print(f"  {node['id']}: {node['kind']} at ({node['xy'][0]:.1f}, {node['xy'][1]:.1f}), degree={node['degree']}")
    
    print(f"\nEdges found: {len(graph_result['edges'])}")
    for edge in graph_result['edges']:
        centerline = edge['centerline']['coordinates']
        length = sum(
            ((centerline[i+1][0] - centerline[i][0])**2 + 
             (centerline[i+1][1] - centerline[i][1])**2)**0.5
            for i in range(len(centerline) - 1)
        )
        print(f"  {edge['id']}: {edge['u']} -> {edge['v']}, length={length:.1f} µm, points={len(centerline)}")
    
    print(f"\n✓ Test complete")

if __name__ == "__main__":
    test_skeletonization()

