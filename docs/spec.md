Microfluidic DXF to Network IR + Netlist + Tagged Overlay
Purpose

Convert 2D microfluidic mask designs (DXF) into a canonical graph IR (nodes/edges + geometry + dimensions), generate a SPICE-like resistive netlist, and export a tagged image overlay showing the labeled network on top of the design. Future goal: use the same IR to align and compare microscopy images vs CAD.

Primary user pain

DXF measurement of each subsection’s L, W, H is tedious. The tool should segment the design into sections automatically and only ask the user for what CAD cannot provide (height, cross-section model, port semantics).

Scope
V1 (must-have)

Import DXF (typical input: filled closed polylines; sometimes not closed).

User selects channel geometry via:

layer selection OR

click-to-select interior (if polygons exist)

Extract network graph:

nodes at junctions + endpoints + ports

edges are channel segments between nodes

Measure each edge:

length L from centerline

width profile W(s) sampled along centerline

classify edge: uniform or taper_linear (heuristic)

Ask user for:

height model H (global default, optionally per-edge override)

cross-section type (rectangular or trapezoid) (global default, optionally per-edge override)

Detect ports:

circles (native DXF circles or circle-like polylines)

allow user assignment: inlet/outlet/other

Export:

design.graph.json (canonical IR)

design.netlist.cir (SPICE-like resistive netlist)

design.segments.csv (table)

design.overlay.png (tagged image overlay with labels)

V1.1 (nice-to-have)

“Snap close” helper: if polyline endpoints are within tolerance, auto-close (optional toggle).

Support DXF bulge arcs (true arcs in LWPOLYLINE segments).

Better curve fitting for circle ports exported as polylines.

V2 (future)

Microscopy image upload + segmentation + skeleton

CAD↔image registration (guided + refined)

Compare measured widths/lengths by edge ID and report deltas

Architecture
Monorepo layout
microfluidic-netlist/
  core/                    # Python library (geometry + graph + exports)
    microfluidic_ir/
      __init__.py
      dxf_loader.py
      geometry.py
      curve_fit.py
      skeleton.py
      graph_extract.py
      measure.py
      classify.py
      export_ir.py
      export_netlist.py
      export_overlay.py
      schemas.py
  backend/                  # FastAPI (local server)
    app.py
    routes/
      import_dxf.py
      select_region.py
      extract_graph.py
      ports.py
      export_bundle.py
    storage.py              # temp workspace per session
  frontend/                 # React/TS (local UI)
    src/
      pages/
      components/
      api/
      state/
  examples/
  docs/

Execution model (local, not a service)

Users run backend locally (FastAPI) and frontend locally (Vite/React).

Backend does all heavy geometry computation and returns:

preview geometry for rendering

computed IR + overlays + exports

Canonical Data Model (IR)
Units

Store explicit units in IR: units: "um" | "mm" | "inch"

All geometry values in those units.

IR JSON schema (conceptual)
{
  "version": "0.1",
  "provenance": {
    "source_filename": "design.dxf",
    "source_sha256": "...",
    "import_timestamp": "ISO8601",
    "units": "um",
    "tolerances": {
      "close_snap_tol": 2.0,
      "circle_fit_rms_tol": 1.0,
      "width_sample_step": 10.0
    }
  },
  "channel_regions": [
    {
      "region_id": "R1",
      "source": {"layers": ["CHANNEL"], "entity_handles": ["AB12", "AB13"]},
      "polygon": {"type": "Polygon", "coordinates": [[[x,y], ...]]}
    }
  ],
  "nodes": [
    {
      "id": "N1",
      "xy": [x, y],
      "kind": "junction|endpoint|port",
      "label": "N1",
      "port": {
        "port_id": "P1",
        "role": "inlet|outlet|internal|unknown"
      }
    }
  ],
  "edges": [
    {
      "id": "E1",
      "u": "N1",
      "v": "N2",
      "region_id": "R1",
      "centerline": {"type": "LineString", "coordinates": [[x,y], ...]},
      "length": 1234.5,
      "width_profile": {
        "kind": "constant|taper_linear|sampled",
        "w0": 80.0,
        "w1": 80.0,
        "samples": [[0.0, 80.0], [100.0, 81.0]]
      },
      "cross_section": {
        "kind": "rectangular|trapezoid",
        "height": 50.0,
        "params": {}
      },
      "fit_geometry": {
        "kind": "none|arc|circle|ellipse|spline",
        "error_rms": 0.3
      },
      "source": {"entity_handles": ["AB12"]}
    }
  ],
  "ports": [
    {
      "port_id": "P1",
      "node_id": "N7",
      "marker": {"kind": "circle", "center": [x,y], "radius": r},
      "source": {"entity_handles": ["C123"]},
      "label": "IN1",
      "role": "inlet"
    }
  ]
}

Notes

Edges hold enough information for regeneration and matching later (centerline + width profile + provenance).

width_profile.samples uses distance-along-edge s in the same units as geometry.

Core algorithms (V1)
A) DXF import & normalization

Parse using ezdxf.

Extract candidate entities:

closed polylines (LWPOLYLINE/POLYLINE) as polygons

HATCH regions as polygons (optional v1.1)

circles (CIRCLE) for ports

Normalize into Shapely geometry.

If polyline is nearly closed and snap-close enabled:

close if distance(first,last) <= close_snap_tol.

B) Channel region selection

Two modes:

Layer select: user chooses one or more layers to treat as channel regions.

Click select: user clicks point p; backend finds polygon containing p; selects that region (and optionally connected adjacent regions).

Output:

selected_region_ids

C) Skeleton → graph extraction (polygon-based)

Input: one or more channel polygons.

Approach (implementation choice for V1):

Rasterize polygon to binary image at chosen resolution (e.g., px_per_unit)

Skeletonize (e.g., Zhang-Suen / medial axis via skimage)

Convert skeleton pixels into a graph:

nodes: endpoints + junction pixels

edges: pixel paths between nodes

Convert pixel paths back to coordinate space (polyline simplification)

Optionally smooth centerlines (simplify tolerance)

(Polygon skeleton is the simplest reliable v1 even if it’s “approximate”; your overlay + manual correction can handle edge cases.)

D) Edge segmentation rules

Build a graph where nodes are:

degree 1 (endpoints)

degree >= 3 (junctions)

plus “port nodes” snapped to nearest endpoint/junction if within tolerance

Every maximal path between two nodes becomes an edge E#.

E) Measure length & width profile

Length:

sum of centerline segment distances.

Width:

sample along centerline every width_sample_step:

at each sample point, estimate local width as 2 * distance(point, polygon_boundary) if centerline is medial-ish.

robustify: take median across small neighborhood.

Store as sampled W(s) and also compute:

w_median, w_min, w_max.

Classification:

uniform if (w_max - w_min) / w_median <= uniform_tol (e.g. 2–5%)

else taper_linear if width vs s fits a line with high R² (e.g. > 0.95)

else sampled (leave as sampled)

F) Port detection

Native DXF circles become port markers.

Also detect “circle-like polylines” (common export artifact):

if a polyline has many points and circle-fit RMS error < tol → treat as circle marker.

Candidate port marker gets attached to nearest skeleton endpoint/junction within snap distance.

User assigns role and label in UI.

G) Cross-section & height

Default: global height and cross_section.kind set by user.

Allow per-edge overrides later (v1 optional, v1.1 likely).

Outputs
1) Tagged overlay image (must-have)

Generate overlay.png with:

Base: rendered channel regions (faint fill) + centerlines

Node labels N# at node points

Edge labels E# near midpoint of edge

Port labels IN1/OUT1 near port marker

Also export overlay.svg if easy (optional).

2) Segments CSV

Columns:

edge_id, node_u, node_v

L

W_median, W_min, W_max

width_kind (constant/taper_linear/sampled)

H

cross_section_kind

region_id

source_handles (optional)

3) Netlist (resistive)

Simple SPICE-like format:

Comments include units

One element per edge:

R_E1 N1 N2 R=<computed_or_param> L=<...> W=<...> H=<...> CS=<...>
For V1 you can either:

output a param-only netlist (no physics):

R_E1 N1 N2 L=... W=... H=...

or include a placeholder resistance model:

R = k * L / (W*H) (clearly marked as placeholder)

(Decide v1: I recommend param-only unless you already have a trusted model.)

Backend API (FastAPI)
Session model

Backend stores uploaded DXF and intermediate results in a temp workspace keyed by session_id.

Endpoints
POST /api/import-dxf

Input: multipart file upload
Returns:

session_id

detected layers list

preview geometry bounds

candidate channel polygons count

candidate port markers count

POST /api/select-channels

Input:

{"session_id":"...","mode":"layer","layers":["CHANNEL"]}


or

{"session_id":"...","mode":"click","point":[x,y]}


Returns:

selected region IDs

lightweight geometry for frontend rendering (polygons + bounds)

POST /api/extract-graph

Input:

{
  "session_id":"...",
  "tolerances": {...},
  "width_sample_step": 10.0
}


Returns:

nodes + edges (without H/cross_section yet)

suggested port attachments

POST /api/set-defaults

Input:

{
  "session_id":"...",
  "defaults":{
    "height": 50.0,
    "cross_section_kind":"rectangular"
  }
}

POST /api/assign-ports

Input:

{
  "session_id":"...",
  "assignments":[
    {"port_id":"P1","role":"inlet","label":"IN1"},
    {"port_id":"P2","role":"outlet","label":"OUT1"}
  ]
}

POST /api/export

Returns a zip containing:

design.graph.json

design.segments.csv

design.netlist.cir

design.overlay.png (and svg if included)

Frontend UI (React/TS)
Screens
Screen 1: Import

Upload DXF

Show layers list and counts

“Select channels by layer” dropdown (multi-select)

“Or click inside channel” toggle

Screen 2: Select channels

Render DXF channel polygons (or all geometry faint)

If click mode: user clicks; selected polygons highlight

“Continue” triggers graph extraction

Screen 3: Label & parameters

Render centerlines + nodes + edges

Show table listing edges (E#, L, W stats, class)

Global defaults: Height H + cross-section kind

Ports panel: list port markers with role + label

Screen 4: Export

Show final tagged overlay preview

Download zip button

Acceptance tests (minimum)

Use examples/ DXFs and assert:

Graph validity

All edges connect existing nodes

No zero-length edges

Node degrees make sense (junction degree >= 3, endpoints degree 1 unless merged)

Measurements sanity

L is positive and within expected range

W_median within expected range for known example

uniform edges classified as uniform within tolerance

Port detection

Example with circle markers: detect correct count, attach to nearest endpoints

Overlay output

overlay.png exists

labels for at least first N nodes/edges are present (basic pixel/text existence checks or snapshot tests if you want)

Cursor implementation prompts (copy/paste)
Prompt A — scaffold repo

“Create a monorepo with core/ (Python package), backend/ (FastAPI), and frontend/ (React+Vite+TS). Add a Makefile with targets dev, backend, frontend, fmt, test. Use pyproject.toml with dependencies: ezdxf, shapely, numpy, networkx, fastapi, uvicorn, pydantic, pillow. Frontend uses a canvas/SVG renderer.”

Prompt B — implement IR schemas

“Implement core/microfluidic_ir/schemas.py with Pydantic models for IR: Provenance, Region, Node, Edge, Port, and GraphIR. Ensure JSON serialization matches the spec fields.”

Prompt C — DXF loader

“Implement dxf_loader.py that loads DXF with ezdxf, extracts: closed polylines as polygons, circles as port markers, and returns a normalized geometry object with layers and entity handles. Add optional snap-close if endpoints within tolerance.”

Prompt D — skeleton + graph extraction v1

“Implement skeleton.py that rasterizes a shapely polygon to a binary image, skeletonizes it, converts skeleton pixels into a networkx graph, and outputs nodes+centerline polylines in original coordinates. Keep it deterministic and parameterized by resolution.”

Prompt E — measurement + classification

“Implement measure.py to compute edge length and width_profile sampled along edge centerline using distance to polygon boundary. Implement classify.py to tag constant/taper_linear/sampled.”

Prompt F — overlay export

“Implement export_overlay.py to draw polygons, centerlines, nodes, edge labels, and port labels to a PNG. Use Pillow. Ensure label placement doesn’t overlap too badly (simple offsets are fine).”

Prompt G — FastAPI endpoints

“Implement FastAPI backend with session storage in temp directories. Endpoints: import-dxf, select-channels, extract-graph, set-defaults, assign-ports, export. Export returns a zip.”

Implementation notes / guardrails

Always preserve provenance (DXF hash, tolerances).

Never discard raw geometry; store it or at least store entity handles so you can debug.

Prefer “approx but stable” over “perfect but fragile” in V1 (especially skeleton extraction).

Keep the IR the source of truth. Netlist and overlay are views.

If you want, I can also provide:

a concrete design.graph.json example

a first-pass resistance model placeholder you can safely label as such

a small synthetic DXF generator (so you can unit test without hand-made CAD files)