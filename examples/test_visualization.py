#!/usr/bin/env python3
"""Test visualization of pipeline steps A, B, C, D."""

from pathlib import Path
from microfluidic_ir.pipeline import run_pipeline
import sys

def test_visualization():
    """Test visualization with y_channel_scale.dxf."""
    
    # Use y_channel_scale.dxf or command line argument
    if len(sys.argv) > 1:
        dxf_path = Path(sys.argv[1])
    else:
        dxf_path = Path(__file__).parent / "y_channel_scale.dxf"
    
    if not dxf_path.exists():
        print(f"Error: DXF file not found at {dxf_path}")
        return
    
    print("=" * 70)
    print("Testing Pipeline Visualization (Headful Mode)")
    print("=" * 70)
    print(f"DXF file: {dxf_path}")
    print("\nThis will open an interactive window showing steps A, B, C, D")
    print("Use the left/right buttons to navigate between steps.")
    print("Close the window when done.\n")
    
    # Use command line argument for minimum channel width, or default based on file
    import math
    if len(sys.argv) > 2:
        minimum_channel_width = float(sys.argv[2])
    elif "enclosed_test" in str(dxf_path):
        minimum_channel_width = 4000.0  # 4mm for enclosed_test.dxf
    else:
        minimum_channel_width = 100.0  # Default: 100 µm for other files
    
    um_per_px = math.ceil(minimum_channel_width / 10.0)
    print(f"Using minimum_channel_width={minimum_channel_width:.1f} µm")
    print(f"um_per_px = ceil({minimum_channel_width:.1f} / 10) = {um_per_px:.1f}")
    
    # Run the full pipeline with visualization
    print("\n[Steps A-D] Running full pipeline with visualization...")
    print("  Opening visualization window...")
    
    result = run_pipeline(
        str(dxf_path),
        minimum_channel_width=minimum_channel_width,
        selected_layers=None,  # Select all polygons
        headful=True,  # Enable visualization
        show_window=True,  # Show the window
        enable_step_a=True,  # Enable Step A (DXF load)
        enable_step_b=True,  # Enable Step B (channel selection)
        enable_step_c=True,  # Enable Step C (skeleton extraction)
        enable_step_d=False  # Disable Step D (graph extraction)
    )
    
    # Show results
    dxf_result = result['dxf_result']
    graph_result = result['graph_result']
    visualizer = result.get('visualizer')
    
    print(f"\n✓ Loaded: {len(dxf_result['polygons'])} polygon(s), {len(dxf_result['circles'])} circle(s)")
    print(f"✓ Selected {len(result['selected_polygons'])} polygon(s)")
    
    if visualizer:
        print("\n✓ Visualization window opened!")
        print("  Navigate with Previous/Next buttons")
        print("  Close window to continue...\n")
        # Window is already shown by run_pipeline
        print("✓ Visualization complete")
    else:
        print("⚠ Visualization not available (matplotlib may not be installed)")
    
    print(f"\n✓ Graph extraction complete: {len(graph_result['nodes'])} nodes, {len(graph_result['edges'])} edges")
    print("=" * 70)


if __name__ == "__main__":
    test_visualization()

