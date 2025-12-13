"""Apply cross-section and height defaults to edges."""

from typing import List, Dict, Any, Optional, Literal, Union
import logging

logger = logging.getLogger(__name__)


def update_edge_cross_section(
    edge: Dict[str, Any],
    height: Optional[float] = None,
    cross_section_kind: Optional[Literal["rectangular", "trapezoid"]] = None
) -> None:
    """
    Update cross-section for a single edge.
    
    Args:
        edge: Edge dict (modified in place)
        height: New height in micrometers (optional, keeps existing if None)
        cross_section_kind: New cross-section type (optional, keeps existing if None)
    """
    if 'cross_section' not in edge:
        edge['cross_section'] = {
            'kind': 'rectangular',
            'height': 50.0,
            'params': {}
        }
    
    if height is not None:
        edge['cross_section']['height'] = float(height)
    
    if cross_section_kind is not None:
        if cross_section_kind not in ('rectangular', 'trapezoid'):
            logger.warning("Invalid cross_section_kind '%s', using 'rectangular'", cross_section_kind)
            cross_section_kind = 'rectangular'
        edge['cross_section']['kind'] = cross_section_kind


def apply_cross_section_defaults(
    edges: List[Dict[str, Any]],
    default_height: float = 50.0,
    default_cross_section_kind: Literal["rectangular", "trapezoid"] = "rectangular",
    per_edge_overrides: Optional[Dict[str, Dict[str, Any]]] = None
) -> None:
    """
    Apply cross-section and height defaults to edges.
    
    Args:
        edges: List of edge dicts (modified in place)
        default_height: Default height in micrometers (default: 50.0)
        default_cross_section_kind: Default cross-section type (default: "rectangular")
        per_edge_overrides: Optional dict mapping edge_id to override dict with:
            - height: float (optional)
            - cross_section_kind: str (optional)
    """
    per_edge_overrides = per_edge_overrides or {}
    
    for edge in edges:
        edge_id = edge.get('id', '')
        
        # Get overrides for this edge if any
        overrides = per_edge_overrides.get(edge_id, {})
        
        # Determine height and cross-section kind
        height = overrides.get('height', default_height)
        cross_section_kind = overrides.get('cross_section_kind', default_cross_section_kind)
        
        # Validate cross-section kind
        if cross_section_kind not in ('rectangular', 'trapezoid'):
            logger.warning("Invalid cross_section_kind '%s' for edge %s, using 'rectangular'", 
                          cross_section_kind, edge_id)
            cross_section_kind = 'rectangular'
        
        # Create cross_section dict
        cross_section = {
            'kind': cross_section_kind,
            'height': float(height),
            'params': {}
        }
        
        # For trapezoid, params could include angle, etc. (future enhancement)
        if cross_section_kind == 'trapezoid':
            # Could add angle, top_width, bottom_width, etc. in params
            # For now, just empty params
            pass
        
        edge['cross_section'] = cross_section
        
        if overrides:
            logger.debug("Applied overrides to edge %s: height=%.1f, kind=%s",
                        edge_id, height, cross_section_kind)

