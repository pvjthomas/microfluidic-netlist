"""Pydantic models for the canonical IR schema."""

from pydantic import BaseModel
from typing import Literal, Optional, List, Dict, Any


class Provenance(BaseModel):
    """Provenance metadata for the IR."""
    source_filename: str
    source_sha256: str
    import_timestamp: str
    units: Literal["um", "mm", "inch"]
    tolerances: Dict[str, float]


class Region(BaseModel):
    """Channel region from DXF."""
    region_id: str
    source: Dict[str, Any]
    polygon: Dict[str, Any]  # GeoJSON-like


class Node(BaseModel):
    """Graph node (junction, endpoint, or port)."""
    id: str
    xy: List[float]
    kind: Literal["junction", "endpoint", "port"]
    label: str
    port: Optional[Dict[str, Any]] = None
    distance: Optional[float] = None  # Local half-width (distance to boundary) at node location


class Edge(BaseModel):
    """Graph edge (channel segment)."""
    id: str
    u: str
    v: str
    region_id: str
    centerline: Dict[str, Any]  # GeoJSON-like
    length: float
    width_profile: Dict[str, Any]
    cross_section: Dict[str, Any]
    fit_geometry: Optional[Dict[str, Any]] = None
    source: Dict[str, Any]


class Port(BaseModel):
    """Port marker."""
    port_id: str
    node_id: str
    marker: Dict[str, Any]
    source: Dict[str, Any]
    label: str
    role: Literal["inlet", "outlet", "internal", "unknown"]


class GraphIR(BaseModel):
    """Canonical IR for microfluidic network."""
    version: str
    provenance: Provenance
    channel_regions: List[Region]
    nodes: List[Node]
    edges: List[Edge]
    ports: List[Port]


# Pixel-level graph schemas
class PixelNode(BaseModel):
    """Node in raw pixel graph G_px."""
    radius: float  # Local half-width (distance to boundary) from distance field
    width: float  # 2 * radius
    x: Optional[float] = None  # World x coordinate (optional)
    y: Optional[float] = None  # World y coordinate (optional)


class PixelEdge(BaseModel):
    """Edge in raw pixel graph G_px."""
    length: float  # Length in world units
    cost: Optional[float] = None  # Optional path cost


# Branch-level schemas
class BranchEdge(BaseModel):
    """Edge in reduced branch graph G_br (represents a branch between terminals)."""
    length: float
    mean_width: float
    min_width: float
    max_width: float
    width_std: float
    slenderness: float  # length / mean_width
    polyline: List[List[float]]  # List of [x, y] coordinates
    u_terminal_type: Literal["endpoint", "junction"]  # Type of terminal at u
    v_terminal_type: Literal["endpoint", "junction"]  # Type of terminal at v

