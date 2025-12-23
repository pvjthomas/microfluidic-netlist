# Step D: Final Graph Extraction - Detailed Specification v1

## Overview

Step D is the final graph extraction phase that converts the pixel-based skeleton graph from Step C into a vector-based network graph with measured nodes and edges. It performs node clustering, edge reconstruction, centerline refinement, measurement, and graph cleanup to produce the canonical graph IR.

**Input**: 
- Skeleton graph (NetworkX graph of skeleton pixels) from Step C
- Polygon (Shapely Polygon) representing the channel region
- Parameters: `minimum_channel_width`, `um_per_px`, `width_sample_step`, etc.

**Output**: 
- Nodes list (junctions and endpoints with coordinates)
- Edges list (channel segments with centerlines, length, width profiles)
- Ports list (optional, from circle detection)
- Cross-section information applied

---

## Algorithm Phases

### Phase 1: Node Clustering and Identification

#### 1.1 Junction Pixel Clustering

**Purpose**: Identify junction regions where multiple channel branches meet.

**Algorithm**:
1. Identify all skeleton pixels with degree ≥ 3 (junction pixels)
2. Cluster connected junction pixels using spatial connectivity
3. Compute centroid for each junction cluster
4. Create one "junction" node per cluster

**Details**:
- Uses `cluster_junction_pixels()` function
- Clustering distance threshold: based on `um_per_px` (typically ~1-2 pixels)
- Junction clusters become nodes with `kind='junction'`
- Store cluster node IDs for later edge reconstruction

**Output**: Dictionary mapping cluster_id → list of skeleton node IDs

#### 1.2 Endpoint Pixel Identification and Merging

**Purpose**: Identify and merge close endpoint pixels to avoid creating spurious nodes.

**Algorithm**:
1. Identify all skeleton pixels with degree == 1 (endpoint pixels)
2. Use topology-aware merging to cluster close endpoints:
   - Compute merge distance: `min(2.0 * um_per_px, 0.2 * minimum_channel_width) * endpoint_merge_distance_factor`
   - This keeps merging at pixel-scale and prevents giant merge distances
3. Merge endpoints that are:
   - Within merge distance
   - Not separated by junction pixels (forbidden set)
   - Topologically connected (no path through a junction between them)
4. Create one "endpoint" node per merged cluster

**Details**:
- Uses `merge_close_endpoints()` function
- Forbidden set: all junction pixel IDs (prevents merging across junctions)
- Merged endpoints become nodes with `kind='endpoint'`
- Initial endpoint position: centroid of cluster, then snapped to polygon boundary

**Output**: Dictionary mapping cluster_id → list of skeleton node IDs

#### 1.3 Node Creation

**Algorithm**:
1. For each junction cluster:
   - Compute centroid using `compute_cluster_centroid()`
   - Create node with ID `N{i}` (sequential numbering)
   - Set `kind='junction'`, store `cluster_nodes` for reference
2. For each endpoint cluster:
   - Compute centroid (or use single node position if cluster size == 1)
   - Snap to polygon boundary using `snap_endpoint_to_boundary()`
   - Create node with ID `N{i}` (sequential numbering)
   - Set `kind='endpoint'`, store `cluster_nodes` for reference

**Node Schema**:
```python
{
    'id': 'N1',                    # Unique identifier (N1, N2, ...)
    'xy': [x, y],                  # Coordinates in micrometers
    'kind': 'junction' | 'endpoint',
    'degree': int,                 # Number of skeleton pixels in cluster
    'cluster_nodes': [...]         # List of skeleton node IDs (for debugging)
}
```

---

### Phase 2: Edge Reconstruction

#### 2.1 Edge Building Algorithm

**Purpose**: Reconstruct edges by finding paths between nodes, collapsing degree-2 chains.

**Algorithm** (BFS-based, O(n×m) complexity):
1. Create mappings:
   - `true_node_to_skeleton_nodes`: Map node ID → list of skeleton nodes
   - `skeleton_node_to_true_node`: Reverse mapping for fast lookup
2. For each true node `node1`:
   - For each skeleton node `start_skeleton_node` in node1's cluster:
     - Perform BFS walk through skeleton graph:
       - Start at `start_skeleton_node`
       - Explore neighbors that are:
         - Unassigned skeleton nodes (degree-2 chains)
         - Nodes belonging to other true nodes (targets)
       - Stop when reaching a node belonging to a different true node `node2`
     - Validate path:
       - All intermediate nodes must be degree-2 (collapsing chains)
       - Path must not cross another true node
     - If valid, create edge from `node1` to `node2`
     - Mark edge as visited (undirected: sorted tuple of node IDs)

**Edge Validation Rules**:
- Path intermediate nodes must all have degree == 2 (collapsing degree-2 chains into edges)
- Path must not contain skeleton nodes belonging to a third true node
- Edge must be unique (no duplicate edges between same node pair)

**Optimization**:
- BFS terminates when target node found (don't explore further)
- Use reverse mapping for O(1) node lookup
- Only explore one edge per start_skeleton_node per node pair

#### 2.2 Centerline Extraction and Refinement

**For each valid edge path**:

1. **Initial Path Extraction**:
   - Extract skeleton pixel coordinates along path: `path_coords_raw`
   - These are pixel-scale coordinates from skeleton graph

2. **Centerline Refinement** (if path length > 50 µm):
   - Use `refine_centerline_with_boundary()` to refine centerline:
     - Input: raw skeleton coordinates, polygon
     - Iterative refinement (5 iterations + 2 final iterations = 7 total)
     - Process:
       1. Resample to target spacing: `max(5.0, 1.0 * um_per_px)`
       2. Smooth using Savitzky-Golay filter
       3. Iterative midpoint refinement: project points to midpoint of boundary intersections
       4. Use stabilized normals for robust boundary intersection
       5. Final refinement passes with stabilized normals
     - Validation: If refined length > 2× original length, use original (avoid artifacts)
   - If refinement fails or path is short (≤50 µm), use skeleton coordinates directly

3. **Node Position Update**:
   - Update `node1['xy']` to match refined centerline start point
   - Update `node2['xy']` to match refined centerline end point
   - This ensures nodes are positioned on refined centerline endpoints

**Edge Schema**:
```python
{
    'id': 'E1',                    # Unique identifier (E1, E2, ...)
    'u': 'N1',                     # Source node ID
    'v': 'N2',                     # Target node ID
    'centerline': {
        'type': 'LineString',
        'coordinates': [[x1, y1], [x2, y2], ...]  # Refined centerline coordinates
    }
    # Note: length and width_profile added in Phase 4
}
```

---

### Phase 3: Graph Cleanup

#### 3.1 Junction Merging

**Purpose**: Remove spurious junctions that are too close together (artifacts along straight channels).

**Algorithm**:
1. For each pair of junction nodes:
   - Check if there's a direct edge between them
   - If edge exists and length < `junction_merge_threshold`:
     - Threshold: `3.0 * um_per_px` (3 pixels in world units)
     - Mark for merging
2. Merge junctions:
   - Keep first node, remove second
   - Average positions: `(node1_xy + node2_xy) / 2`
   - Redirect all edges from removed node to kept node
   - Remove self-loops (edges where u == v after merge)
   - Update node list

**Rationale**: Extra junctions along straight segments are skeletonization artifacts and should be collapsed.

#### 3.2 Junction Re-centering

**Purpose**: Optimize junction positions for symmetric branch intersections.

**Algorithm**:
1. For each junction node:
   - Find all incident edges
   - Compute tangent direction for each edge at the junction
   - Fit best intersection point using `re_center_junctions()`:
     - Estimate tangents from edge centerlines near junction
     - Compute weighted average of intersection points
     - Weight by edge length or use geometric mean
   - Update junction coordinates

**Timing**: Performed AFTER centerline refinement but BEFORE measurement, so junction positions are accurate for width profile calculations.

---

### Phase 4: Edge Measurement

#### 4.1 Length Measurement

**Algorithm**:
- For each edge, compute length from centerline coordinates:
  - Sum of Euclidean distances between consecutive centerline points
  - Formula: `length = Σ√[(x[i+1]-x[i])² + (y[i+1]-y[i])²]`
- Store as `edge['length']` in micrometers

#### 4.2 Width Profile Measurement

**Algorithm** (for each edge):
1. **Sampling**:
   - Sample points along centerline at intervals of `width_sample_step`
   - Default: `width_sample_step = minimum_channel_width / 3`
   - Skip samples within `ignore_node_neighborhood` (default 50 µm) of start/end
   - If edge is too short, sample at midpoint only

2. **Width Estimation at Each Sample Point**:
   - For each sample point:
     - Use `estimate_width_at_point()`:
       - Sample 5 points in small neighborhood (radius = `neighborhood_size`, default 5 µm)
       - For each neighborhood point:
         - Compute distance to polygon boundary: `polygon.boundary.distance(point)`
         - Only consider points inside polygon
       - Compute median distance across neighborhood
       - Width = 2 × median_distance (assuming medial axis property)
   - Store as `(distance_along_edge, width)` tuples

3. **Statistics Computation**:
   - `w_median`: Median width across all samples
   - `w_min`: Minimum width
   - `w_max`: Maximum width
   - `w0`: Width at start (first sample)
   - `w1`: Width at end (last sample)

4. **Width Profile Classification**:
   - Use `classify_width_profile()`:
     - **uniform**: If `(w_max - w_min) / w_median <= uniform_tol` (default 0.05 = 5%)
     - **taper_linear**: Else if width vs. distance fits linear with R² > 0.95
     - **sampled**: Otherwise (leave as sampled data)
   - Store classification as `width_profile['kind']`

**Width Profile Schema**:
```python
{
    'kind': 'uniform' | 'taper_linear' | 'sampled',
    'samples': [[s0, w0], [s1, w1], ...],  # Distance along edge (µm), width (µm)
    'w_median': float,                      # Median width (µm)
    'w_min': float,                         # Minimum width (µm)
    'w_max': float,                         # Maximum width (µm)
    'w0': float,                            # Width at start (µm)
    'w1': float                             # Width at end (µm)
}
```

#### 4.3 Performance Optimization

**Scaling with Polygon Size**:
- `width_sample_step` scales with `minimum_channel_width / 3` to reduce sample count for large channels
- For very large polygons (e.g., 500+ km), this prevents millions of expensive boundary distance calculations

**Short Edge Handling**:
- Edges ≤ 50 µm use skeleton coordinates directly (skip refinement)
- Reduces refinement iterations for short edges (7 → 0 iterations)

---

### Phase 5: Graph Pruning

#### 5.1 Short Leaf Edge Pruning

**Purpose**: Remove spurious short branches at channel endpoints (skeletonization artifacts).

**Algorithm** (iterative):
1. Compute node degrees (number of incident edges per node)
2. Identify leaf edges (edges where one endpoint has degree == 1)
3. For each leaf edge:
   - Compute minimum length threshold: `L_min = max(2.5 × local_width, 30.0 µm)`
   - `local_width` = median width of the edge
   - If edge length < `L_min`, mark for removal
4. Remove marked edges
5. Recompute degrees and repeat until no more edges to remove (max 10 iterations)

**Rationale**: Short branches at endpoints are often skeleton artifacts from boundary roughness or slight channel width variations. The 2.5× width factor ensures we only remove edges shorter than 2-3 channel widths.

#### 5.2 Pitchfork Branch Collapse

**Purpose**: Remove short side branches from junctions (pitchfork patterns).

**Algorithm**:
1. For each junction node (degree ≥ 3):
   - Find incident edges
   - Sort by length (longest = main edge, others = leaf edges)
   - Check if all leaf edges are:
     - Shorter than 50% of main edge length, AND
     - Shorter than 150 µm absolute
   - If at least 2 leaf edges meet criteria, mark them as pitchfork branches
2. Remove marked pitchfork edges
3. Reclassify nodes:
   - If junction becomes degree 1 after removal, reclassify as endpoint

**Rationale**: Short side branches from junctions are often skeleton artifacts. Only remove if there are multiple short branches (pitchfork pattern), indicating a spurious junction.

---

### Phase 6: Final Node Cleanup

#### 6.1 Endpoint Reclassification

**Algorithm**:
- After all pruning, recompute node degrees
- Reclassify any node with degree == 1 as `kind='endpoint'` (even if originally classified as junction)

#### 6.2 Endpoint Boundary Snapping

**Purpose**: Ensure all endpoints are precisely on the polygon boundary (not just near it).

**Algorithm** (for each endpoint node):
1. Find incident edge(s) to get tangent direction
2. Use `snap_endpoint_to_boundary()`:
   - Follow edge tangent direction from endpoint
   - Find intersection with polygon boundary
   - Position endpoint at intersection point
3. Update endpoint coordinates
4. Extend incident edge centerlines to new endpoint position:
   - Update first coordinate if endpoint is at edge start (u)
   - Update last coordinate if endpoint is at edge end (v)

**Rationale**: After centerline refinement, endpoints may be slightly off the boundary. This ensures they're precisely on the boundary for accurate width measurements and visualization.

---

### Phase 7: Cross-Section Application

**Algorithm**:
1. Apply default cross-section to all edges:
   - `default_height`: Default height in micrometers (default: 50.0 µm)
   - `default_cross_section_kind`: 'rectangular' or 'trapezoid' (default: 'rectangular')
2. Apply per-edge overrides if provided:
   - `per_edge_overrides`: Dict mapping `edge_id` → `{'height': ..., 'cross_section_kind': ...}`
3. Store in `edge['cross_section']`

**Cross-Section Schema**:
```python
{
    'kind': 'rectangular' | 'trapezoid',
    'height': float,                # Height in micrometers
    'params': {}                    # Additional parameters for trapezoid (future)
}
```

---

## Parameters

### Required Parameters

- `polygon`: Shapely Polygon - Channel region polygon
- `minimum_channel_width`: float - Minimum channel width in micrometers (used for sampling and thresholds)
- `um_per_px`: float - Resolution from skeletonization (microns per pixel)

### Optional Parameters

- `simplify_tolerance`: float | None - Path simplification tolerance. If None, computed as `simplify_tolerance_factor * minimum_channel_width` (default factor: 0.5)
- `width_sample_step`: float | None - Distance between width samples along centerline. If None, computed as `minimum_channel_width / 3`
- `measure_edges`: bool - If True, measure length and width profile (default: True)
- `default_height`: float - Default channel height in micrometers (default: 50.0)
- `default_cross_section_kind`: str - 'rectangular' or 'trapezoid' (default: 'rectangular')
- `per_edge_overrides`: Dict[str, Dict] | None - Per-edge height/cross-section overrides
- `simplify_tolerance_factor`: float - Factor for computing simplify_tolerance (default: 0.5)
- `endpoint_merge_distance_factor`: float - Factor for endpoint merge distance (default: 1.0)

### Derived Parameters

- `endpoint_merge_distance`: `min(2.0 * um_per_px, 0.2 * minimum_channel_width) * endpoint_merge_distance_factor`
- `junction_merge_threshold`: `3.0 * um_per_px`
- `leaf_edge_min_length`: `max(2.5 * local_width, 30.0 µm)` (per edge, based on edge width)

---

## Output Data Structures

### Nodes

```python
{
    'id': 'N1',                    # Unique identifier
    'xy': [x, y],                  # Coordinates in micrometers
    'kind': 'junction' | 'endpoint',
    'degree': int,                 # Number of skeleton pixels (for junctions) or 1 (for endpoints)
    'cluster_nodes': [...]         # List of skeleton node IDs (optional, for debugging)
}
```

### Edges

```python
{
    'id': 'E1',                    # Unique identifier
    'u': 'N1',                     # Source node ID
    'v': 'N2',                     # Target node ID
    'centerline': {
        'type': 'LineString',
        'coordinates': [[x1, y1], [x2, y2], ...]  # Refined centerline coordinates
    },
    'length': float,               # Edge length in micrometers
    'width_profile': {
        'kind': 'uniform' | 'taper_linear' | 'sampled',
        'samples': [[s, w], ...],  # Distance along edge (µm), width (µm)
        'w_median': float,
        'w_min': float,
        'w_max': float,
        'w0': float,
        'w1': float
    },
    'cross_section': {
        'kind': 'rectangular' | 'trapezoid',
        'height': float,           # Height in micrometers
        'params': {}
    }
}
```

### Return Value

```python
{
    'nodes': List[NodeDict],
    'edges': List[EdgeDict],
    'skeleton_graph': nx.Graph,    # Original skeleton graph (for debugging)
    'c2_data': {                   # Intermediate data from Phase 1 (for debugging)
        'skeleton_graph': nx.Graph,
        'junction_clusters': Dict,
        'endpoint_clusters': Dict,
        'forbidden_nodes': Set,
        'polygon': Polygon
    }
}
```

---

## Performance Characteristics

### Complexity

- **Node clustering**: O(P) where P = number of skeleton pixels
- **Edge building**: O(N × M) where N = number of nodes, M = average skeleton nodes per node
  - Improved from O(N² × M²) using BFS instead of all-pairs shortest path
- **Measurement**: O(E × S × B) where:
  - E = number of edges
  - S = samples per edge (proportional to edge_length / width_sample_step)
  - B = boundary distance computation cost (O(boundary_segments) per point)

### Optimizations

1. **Scalable width sampling**: `width_sample_step = minimum_channel_width / 3` reduces sample count for large channels
2. **BFS edge building**: O(N×M) instead of O(N²×M²)
3. **Conditional refinement**: Skip refinement for edges ≤ 50 µm
4. **Reduced refinement iterations**: 7 iterations (5 + 2) instead of 13 (10 + 3)

### Performance Targets

- Small channels (100 µm width, 1-5 edges): < 1 second
- Medium channels (1000 µm width, 10-20 edges): < 5 seconds
- Large channels (4 mm width, 100+ km polygon): < 10 seconds (with optimized sampling)

---

## Error Handling

### Validation

1. **Empty skeleton graph**: Return empty nodes/edges list
2. **Refinement failures**: Fall back to skeleton coordinates with warning
3. **Measurement failures**: Set default values (length=0, empty width_profile) with warning
4. **Invalid paths**: Skip invalid paths, continue with next candidate

### Logging

- INFO level: Major phase completions, node/edge counts, timing
- DEBUG level: Detailed per-edge timing, node movements, validation details
- WARNING level: Refinement failures, measurement failures, unexpected conditions

---

## Dependencies

### External Libraries

- `networkx`: Graph data structures and algorithms
- `shapely`: Geometric operations (Polygon, Point, LineString)
- `numpy`: Numerical computations
- `scipy`: Interpolation, signal processing (for refinement)

### Internal Modules

- `skeleton.py`: `cluster_junction_pixels()`, `merge_close_endpoints()`, `compute_cluster_centroid()`
- `measure.py`: `measure_edge()`, `estimate_width_at_point()`
- `classify.py`: `classify_width_profile()`
- `cross_section.py`: `apply_cross_section_defaults()`
- `graph_extract.py`: `refine_centerline_with_boundary()`, `re_center_junctions()`, `snap_endpoint_to_boundary()`

---

## Acceptance Criteria

### Graph Validity

1. All edges connect existing nodes (edge['u'] and edge['v'] exist in nodes list)
2. No zero-length edges (length > 0 for all measured edges)
3. Node degrees match edge connectivity:
   - Endpoint nodes: degree == 1 (except after pitchfork removal)
   - Junction nodes: degree >= 3
4. No duplicate edges (each node pair has at most one edge)
5. All nodes have valid coordinates (finite numbers)

### Measurement Accuracy

1. Edge lengths are positive and within expected range
2. Width profiles have valid statistics (w_min ≤ w_median ≤ w_max)
3. Width profile samples are in ascending order by distance along edge
4. Uniform edges are classified correctly (within tolerance)
5. Linear taper edges have R² > 0.95

### Geometric Correctness

1. All endpoints are on polygon boundary (within tolerance)
2. Centerlines are within polygon (or very close, accounting for refinement)
3. Junction positions are reasonable (within junction cluster region)
4. Refined centerlines don't increase length dramatically (< 2× original)

---

## Future Enhancements (v1.1+)

1. **Adaptive width sampling**: Vary sampling density based on local width variation
2. **Better refinement**: Use distance transform for faster boundary distance calculations
3. **Parallel edge measurement**: Measure multiple edges in parallel
4. **Manual centerline integration**: Support manual centerlines from Step B.5
5. **Multi-polygon handling**: Better union and graph extraction for multiple disjoint regions
6. **Edge splitting**: Split edges at width discontinuities or user-specified points

---

## References

- Main pipeline spec: `docs/spec.md`
- Step C spec: Skeleton extraction (not yet documented)
- Implementation: `core/microfluidic_ir/graph_extract.py`

