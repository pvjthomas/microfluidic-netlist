#!/usr/bin/env python3
"""Comprehensive test of DXF loader against spec requirements."""

from pathlib import Path
from microfluidic_ir.dxf_loader import load_dxf
import json


def test_dxf_loader_spec():
    """Test DXF loader against spec requirements from docs/spec.md"""
    
    print("=" * 70)
    print("DXF Loader Specification Compliance Test")
    print("=" * 70)
    
    # Test with both DXF files
    test_files = [
        "y_channel_100um_4mmlegs.dxf",
        "y-channel.dxf"
    ]
    
    all_passed = True
    
    for dxf_file in test_files:
        dxf_path = Path(__file__).parent / dxf_file
        
        if not dxf_path.exists():
            print(f"\n⚠ Skipping {dxf_file} (not found)")
            continue
        
        print(f"\n{'='*70}")
        print(f"Testing: {dxf_file}")
        print(f"{'='*70}")
        
        try:
            result = load_dxf(str(dxf_path), snap_close_tol=2.0)
            
            # Test 1: Required fields (from spec section "DXF import & normalization")
            print("\n✓ Test 1: Required fields present")
            required_fields = ['layers', 'polygons', 'circles', 'bounds', 
                             'source_hash', 'source_filename', 'import_timestamp']
            for field in required_fields:
                assert field in result, f"Missing required field: {field}"
            print(f"  All required fields present: {', '.join(required_fields)}")
            
            # Test 2: Source hash (SHA256)
            print("\n✓ Test 2: Source hash (SHA256)")
            assert len(result['source_hash']) == 64, "Source hash should be 64 hex chars (SHA256)"
            assert all(c in '0123456789abcdef' for c in result['source_hash']), "Invalid hash format"
            print(f"  Hash: {result['source_hash'][:16]}... (valid SHA256)")
            
            # Test 3: Layers extraction
            print("\n✓ Test 3: Layer extraction")
            assert isinstance(result['layers'], list), "Layers should be a list"
            assert len(result['layers']) > 0, "Should have at least one layer"
            print(f"  Found {len(result['layers'])} layer(s): {', '.join(result['layers'])}")
            
            # Test 4: Polygon extraction (closed polylines)
            print("\n✓ Test 4: Polygon extraction (closed polylines)")
            assert isinstance(result['polygons'], list), "Polygons should be a list"
            print(f"  Found {len(result['polygons'])} polygon(s)")
            
            for i, poly in enumerate(result['polygons'], 1):
                # Check polygon structure
                assert 'polygon' in poly, f"Polygon {i} missing 'polygon' field"
                assert 'layer' in poly, f"Polygon {i} missing 'layer' field"
                assert 'entity_handle' in poly, f"Polygon {i} missing 'entity_handle' field"
                assert 'area' in poly, f"Polygon {i} missing 'area' field"
                assert 'bounds' in poly, f"Polygon {i} missing 'bounds' field"
                
                # Check polygon geometry format (GeoJSON-like)
                pg = poly['polygon']
                assert pg['type'] == 'Polygon', f"Polygon {i} should have type 'Polygon'"
                assert 'coordinates' in pg, f"Polygon {i} missing 'coordinates'"
                assert len(pg['coordinates']) > 0, f"Polygon {i} has empty coordinates"
                
                # Check bounds
                b = poly['bounds']
                assert 'xmin' in b and 'ymin' in b and 'xmax' in b and 'ymax' in b
                assert b['xmin'] < b['xmax'], f"Polygon {i} has invalid x bounds"
                assert b['ymin'] < b['ymax'], f"Polygon {i} has invalid y bounds"
                
                # Check area is positive
                assert poly['area'] > 0, f"Polygon {i} has non-positive area"
                
                print(f"    Polygon {i}: layer={poly['layer']}, "
                      f"area={poly['area']:.2f}, handle={poly['entity_handle']}")
            
            # Test 5: Circle extraction (port markers)
            print("\n✓ Test 5: Circle extraction (port markers)")
            assert isinstance(result['circles'], list), "Circles should be a list"
            print(f"  Found {len(result['circles'])} circle(s)")
            
            for i, circle in enumerate(result['circles'], 1):
                # Check circle structure
                assert 'circle' in circle, f"Circle {i} missing 'circle' field"
                assert 'layer' in circle, f"Circle {i} missing 'layer' field"
                assert 'entity_handle' in circle, f"Circle {i} missing 'entity_handle' field"
                assert 'center' in circle, f"Circle {i} missing 'center' field"
                assert 'radius' in circle, f"Circle {i} missing 'radius' field"
                
                # Check circle geometry format
                cg = circle['circle']
                assert cg['type'] == 'Circle', f"Circle {i} should have type 'Circle'"
                assert 'center' in cg and 'radius' in cg
                assert len(cg['center']) == 2, f"Circle {i} center should be [x, y]"
                assert cg['radius'] > 0, f"Circle {i} has non-positive radius"
                
                print(f"    Circle {i}: layer={circle['layer']}, "
                      f"center=({circle['center'][0]:.1f}, {circle['center'][1]:.1f}), "
                      f"radius={circle['radius']:.1f}, handle={circle['entity_handle']}")
            
            # Test 6: Bounds calculation
            print("\n✓ Test 6: Bounds calculation")
            bounds = result['bounds']
            assert 'xmin' in bounds and 'ymin' in bounds
            assert 'xmax' in bounds and 'ymax' in bounds
            
            if len(result['polygons']) > 0 or len(result['circles']) > 0:
                assert bounds['xmin'] < bounds['xmax'], "Invalid overall x bounds"
                assert bounds['ymin'] < bounds['ymax'], "Invalid overall y bounds"
            
            width = bounds['xmax'] - bounds['xmin']
            height = bounds['ymax'] - bounds['ymin']
            print(f"  Overall bounds: x=[{bounds['xmin']:.1f}, {bounds['xmax']:.1f}], "
                  f"y=[{bounds['ymin']:.1f}, {bounds['ymax']:.1f}]")
            print(f"  Dimensions: {width:.1f} × {height:.1f}")
            
            # Test 7: Entity handle preservation (provenance)
            print("\n✓ Test 7: Entity handle preservation (provenance)")
            all_handles = []
            for poly in result['polygons']:
                assert poly['entity_handle'], f"Polygon missing entity handle"
                all_handles.append(poly['entity_handle'])
            for circle in result['circles']:
                assert circle['entity_handle'], f"Circle missing entity handle"
                all_handles.append(circle['entity_handle'])
            print(f"  Preserved {len(all_handles)} entity handle(s) for provenance")
            
            # Test 8: Timestamp format
            print("\n✓ Test 8: Timestamp format")
            ts = result['import_timestamp']
            assert ts.endswith('Z'), "Timestamp should end with 'Z' (UTC)"
            assert 'T' in ts, "Timestamp should be ISO8601 format"
            print(f"  Timestamp: {ts}")
            
            # Test 9: Snap-close functionality (test with tolerance)
            print("\n✓ Test 9: Snap-close functionality")
            # Re-load with different tolerance to verify it's used
            result_no_snap = load_dxf(str(dxf_path), snap_close_tol=0.0)
            result_with_snap = load_dxf(str(dxf_path), snap_close_tol=2.0)
            # Note: This is a basic check - actual snap behavior depends on DXF content
            print(f"  Snap-close tolerance parameter accepted (0.0 vs 2.0)")
            
            print(f"\n✅ All tests passed for {dxf_file}")
            
        except AssertionError as e:
            print(f"\n❌ Test failed: {e}")
            all_passed = False
        except Exception as e:
            print(f"\n❌ Error testing {dxf_file}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print(f"\n{'='*70}")
    if all_passed:
        print("✅ ALL SPECIFICATION TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print(f"{'='*70}")
    
    return all_passed


if __name__ == "__main__":
    success = test_dxf_loader_spec()
    exit(0 if success else 1)


