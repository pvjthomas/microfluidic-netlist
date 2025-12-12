#!/usr/bin/env python3
"""Test script to load and display DXF file contents."""

from pathlib import Path
from microfluidic_ir.dxf_loader import load_dxf
import json

def main():
    dxf_path = Path(__file__).parent / "y_channel_100um_4mmlegs.dxf"
    
    if not dxf_path.exists():
        print(f"Error: DXF file not found at {dxf_path}")
        return
    
    print(f"Loading DXF: {dxf_path}")
    print("-" * 60)
    
    try:
        result = load_dxf(str(dxf_path), snap_close_tol=2.0)
        
        print(f"✓ Successfully loaded DXF")
        print(f"\nLayers found: {len(result['layers'])}")
        for layer in result['layers']:
            print(f"  - {layer}")
        
        print(f"\nPolygons found: {len(result['polygons'])}")
        for i, poly in enumerate(result['polygons'], 1):
            print(f"  Polygon {i}:")
            print(f"    Layer: {poly['layer']}")
            print(f"    Entity Handle: {poly['entity_handle']}")
            print(f"    Area: {poly['area']:.2f} µm²")
            print(f"    Bounds: x=[{poly['bounds']['xmin']:.1f}, {poly['bounds']['xmax']:.1f}], "
                  f"y=[{poly['bounds']['ymin']:.1f}, {poly['bounds']['ymax']:.1f}]")
        
        print(f"\nCircles found: {len(result['circles'])}")
        for i, circle in enumerate(result['circles'], 1):
            print(f"  Circle {i}:")
            print(f"    Layer: {circle['layer']}")
            print(f"    Entity Handle: {circle['entity_handle']}")
            print(f"    Center: ({circle['center'][0]:.1f}, {circle['center'][1]:.1f})")
            print(f"    Radius: {circle['radius']:.1f} µm")
        
        print(f"\nOverall bounds:")
        bounds = result['bounds']
        print(f"  x: [{bounds['xmin']:.1f}, {bounds['xmax']:.1f}]")
        print(f"  y: [{bounds['ymin']:.1f}, {bounds['ymax']:.1f}]")
        print(f"  Width: {bounds['xmax'] - bounds['xmin']:.1f} µm")
        print(f"  Length: {bounds['ymax'] - bounds['ymin']:.1f} µm")
        
        print(f"\nSource hash: {result['source_hash'][:16]}...")
        print(f"Import timestamp: {result['import_timestamp']}")
        
    except Exception as e:
        print(f"✗ Error loading DXF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

