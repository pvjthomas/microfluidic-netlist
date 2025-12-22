#!/usr/bin/env python3
"""Test Manual Centerline Mode."""

from pathlib import Path
from microfluidic_ir.dxf_loader import load_dxf
from microfluidic_ir.graph_extract import extract_graph_from_polygons
from microfluidic_ir.manual_centerline_mode import ManualCenterlineMode
from shapely.geometry import Polygon
import sys

def test_manual_centerline_mode():
    """Test manual centerline mode on enclosed_test.dxf."""
    
    # Use enclosed_test.dxf or command line argument
    if len(sys.argv) > 1:
        dxf_path = Path(sys.argv[1])
    else:
        dxf_path = Path(__file__).parent / "enclosed_test.dxf"
    
    if not dxf_path.exists():
        print(f"Error: DXF file not found at {dxf_path}")
        return
    
    print("=" * 70)
    print("Testing Manual Centerline Mode")
    print("=" * 70)
    print(f"DXF file: {dxf_path}")
    print("\nInstructions:")
    print("  1. Click on a boundary segment (highlighted in red)")
    print("  2. Click on another boundary segment")
    print("  3. Press Enter to compute and save centerline")
    print("  4. Press Backspace to undo last centerline")
    print("  5. Press Escape to reset selection")
    print("  6. Close window when done\n")
    
    # Load DXF
    print("[Step 1] Loading DXF...")
    dxf_result = load_dxf(str(dxf_path), snap_close_tol=2.0)
    print(f"✓ Loaded: {len(dxf_result['polygons'])} polygon(s)")
    
    # Get polygon
    if not dxf_result['polygons']:
        print("✗ No polygons found")
        return
    
    poly_data = dxf_result['polygons'][0]
    coords = poly_data['polygon']['coordinates'][0]
    polygon = Polygon(coords)
    
    print(f"✓ Polygon area: {polygon.area:.2f}")
    print(f"  Bounds: {polygon.bounds}")
    
    # Create manual centerline mode (save_path will be auto-generated from DXF filename)
    print(f"\n[Step 2] Opening Manual Centerline Mode...")
    
    # Auto-generate save path from DXF filename
    # Pattern: "#FILENAME_manual_centerlines.json"
    if 'source_filename' in dxf_result:
        source_filename = Path(dxf_result['source_filename'])
        filename_base = source_filename.stem  # filename without extension
        save_path = Path(__file__).parent / f"{filename_base}_manual_centerlines.json"
    else:
        save_path = Path(__file__).parent / "manual_centerlines.json"
    
    print(f"  Save path: {save_path}")
    
    # Pass DXF result to use DXF edges as segments when available
    mode = ManualCenterlineMode(polygon, save_path=str(save_path), dxf_result=dxf_result)
    
    print(f"✓ Extracted {len(mode.boundary_segments)} boundary segments")
    print(f"✓ Loaded {len(mode.manual_centerlines)} existing centerlines")
    
    print("\n[Step 3] Opening interactive window...")
    print("  Follow the instructions in the window.")
    print("  Close the window when finished.\n")
    
    # Show interactive window
    mode.show(block=True)
    
    # Report results
    print("\n" + "=" * 70)
    print("Manual Centerline Mode Complete")
    print("=" * 70)
    print(f"Total centerlines saved: {len(mode.manual_centerlines)}")
    
    for i, record in enumerate(mode.manual_centerlines, 1):
        boundary_ids = record['boundary_ids']
        coords = record['centerline']['coordinates']
        print(f"  Centerline {i}: {boundary_ids[0]} → {boundary_ids[1]} ({len(coords)} points)")
    
    print(f"\nSaved to: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    test_manual_centerline_mode()

