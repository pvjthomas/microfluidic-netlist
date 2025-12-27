"""Export segments CSV according to spec."""

from typing import List, Dict, Any
import csv
import logging

logger = logging.getLogger(__name__)


def export_segments_csv(
    edges: List[Dict[str, Any]],
    output_path: str,
    default_height: float = 50.0,
    default_cross_section_kind: str = "rectangular"
) -> None:
    """
    Export segments CSV according to spec.
    
    Columns:
    - edge_id, node_u, node_v
    - L (length)
    - W_median, W_min, W_max
    - width_kind (constant/taper_linear/sampled)
    - H (height)
    - cross_section_kind
    - region_id
    - source_handles (optional)
    
    Args:
        edges: List of edge dicts
        output_path: Output CSV file path
        default_height: Default height if not in edge cross_section
        default_cross_section_kind: Default cross-section kind if not in edge
    """
    logger.info("Exporting segments CSV to: %s", output_path)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'edge_id',
            'node_u',
            'node_v',
            'L',
            'W_median',
            'W_min',
            'W_max',
            'width_kind',
            'H',
            'cross_section_kind',
            'region_id',
            'source_handles'
        ])
        
        # Write data rows
        for edge in edges:
            edge_id = edge.get('id', '')
            node_u = edge.get('u', '')
            node_v = edge.get('v', '')
            
            # Length
            length = edge.get('length', 0.0)
            
            # Width profile
            width_profile = edge.get('width_profile', {})
            w_median = width_profile.get('w_median', 0.0)
            w_min = width_profile.get('w_min', 0.0)
            w_max = width_profile.get('w_max', 0.0)
            width_kind = width_profile.get('kind', 'sampled')
            
            # Map width_kind to spec values
            # spec uses "constant" but we use "uniform"
            if width_kind == 'uniform':
                width_kind = 'constant'
            
            # Cross section
            cross_section = edge.get('cross_section', {})
            height = cross_section.get('height', default_height)
            cross_section_kind = cross_section.get('kind', default_cross_section_kind)
            
            # Region ID (may not be present in current implementation)
            region_id = edge.get('region_id', '')
            
            # Source handles (optional)
            source = edge.get('source', {})
            source_handles = source.get('entity_handles', [])
            source_handles_str = ','.join(source_handles) if source_handles else ''
            
            writer.writerow([
                edge_id,
                node_u,
                node_v,
                f"{length:.3f}",
                f"{w_median:.3f}",
                f"{w_min:.3f}",
                f"{w_max:.3f}",
                width_kind,
                f"{height:.3f}",
                cross_section_kind,
                region_id,
                source_handles_str
            ])
    
    logger.info("Segments CSV exported: %d edges", len(edges))




