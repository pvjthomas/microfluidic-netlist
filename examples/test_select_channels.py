#!/usr/bin/env python3
"""Test script to test channel selection functionality."""

from pathlib import Path
from microfluidic_ir.dxf_loader import load_dxf
import json
import tempfile
import shutil

def test_channel_selection():
    """Test both layer and click selection modes."""
    dxf_path = Path(__file__).parent / "y_channel_100um_4mmlegs.dxf"
    
    # Load DXF
    print("Loading DXF...")
    result = load_dxf(str(dxf_path), snap_close_tol=2.0)
    
    # Simulate session storage
    session_dir = Path(tempfile.mkdtemp(prefix="test_session_"))
    try:
        # Save import result
        result_path = session_dir / "import_result.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✓ Import result saved to session")
        print(f"  Layers available: {result['layers']}")
        print(f"  Polygons available: {len(result['polygons'])}")
        
        # Test layer selection
        print("\n" + "="*60)
        print("Testing LAYER selection mode")
        print("="*60)
        
        selected_layers = ["CHANNEL"]
        selected_polygons = [
            (i, poly) for i, poly in enumerate(result['polygons'])
            if poly['layer'] in selected_layers
        ]
        
        print(f"Selected layers: {selected_layers}")
        print(f"Found {len(selected_polygons)} matching polygons:")
        for idx, (i, poly) in enumerate(selected_polygons, 1):
            region_id = f"R{i+1}"
            print(f"  Region {region_id}:")
            print(f"    Layer: {poly['layer']}")
            print(f"    Area: {poly['area']:.2f} µm²")
            print(f"    Bounds: x=[{poly['bounds']['xmin']:.1f}, {poly['bounds']['xmax']:.1f}], "
                  f"y=[{poly['bounds']['ymin']:.1f}, {poly['bounds']['ymax']:.1f}]")
        
        # Test click selection
        print("\n" + "="*60)
        print("Testing CLICK selection mode")
        print("="*60)
        
        from shapely.geometry import Point, Polygon
        
        # Test point inside the channel (near junction)
        test_points = [
            (0, 0, "Junction center"),
            (1000, 1000, "Upper right arm"),
            (0, -2000, "Lower stem"),
            (-2000, 1000, "Upper left arm"),
            (5000, 5000, "Outside (should not match)"),
        ]
        
        for x, y, desc in test_points:
            click_point = Point(x, y)
            found_polygons = []
            
            for i, poly_data in enumerate(result['polygons']):
                coords = poly_data['polygon']['coordinates'][0]
                poly_geom = Polygon(coords)
                
                if poly_geom.contains(click_point) or poly_geom.touches(click_point):
                    found_polygons.append((i, poly_data))
            
            print(f"\n  Click at ({x}, {y}) - {desc}:")
            if found_polygons:
                for idx, (i, poly) in enumerate(found_polygons):
                    region_id = f"R{i+1}"
                    print(f"    ✓ Found region {region_id} (layer: {poly['layer']})")
            else:
                print(f"    ✗ No polygon found")
        
        print("\n" + "="*60)
        print("✓ Channel selection tests completed")
        print("="*60)
        
    finally:
        # Cleanup
        shutil.rmtree(session_dir, ignore_errors=True)

if __name__ == "__main__":
    test_channel_selection()


