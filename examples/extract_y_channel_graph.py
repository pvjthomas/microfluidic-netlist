#!/usr/bin/env python3
"""Extract a network graph from examples/y_channel_scale.dxf with logging + sanity checks."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
# Import setup (dev fallback)
# ---------------------------------------------------------------------
# Preferred: install package via `pip install -e core/`
# Fallback: add ./core to sys.path for local dev runs.
CORE_DIR = Path(__file__).resolve().parent.parent / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from microfluidic_ir.dxf_loader import load_dxf  # noqa: E402
from microfluidic_ir.graph_extract import extract_graph_from_polygons  # noqa: E402
from microfluidic_ir.export_overlay import export_overlay_png  # noqa: E402


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
LOG_FILE = Path(__file__).with_suffix(".log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s: %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info("Logging to: %s", LOG_FILE)


@contextmanager
def log_step(name: str):
    """Log step entry/exit + duration."""
    logger.info("→ %s", name)
    start = time.time()
    try:
        yield
    finally:
        logger.info("← %s (%.2fs)", name, time.time() - start)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def poly_bounds(poly_entry: dict[str, Any]) -> tuple[float, float, float]:
    b = poly_entry["bounds"]
    w = float(b["xmax"] - b["xmin"])
    h = float(b["ymax"] - b["ymin"])
    return w, h, max(w, h)


def polyline_length(coords: list[list[float]]) -> float:
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        total += (dx * dx + dy * dy) ** 0.5
    return total


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    logger.info("=" * 70)
    logger.info("Extracting Network Graph from y_channel_scale.dxf")
    logger.info("=" * 70)

    dxf_path = Path(__file__).resolve().parent / "y_channel_scale.dxf"
    if not dxf_path.exists():
        logger.error("DXF file not found: %s", dxf_path)
        return 1

    # 1) Load DXF
    try:
        with log_step("DXF load"):
            result = load_dxf(str(dxf_path), snap_close_tol=2.0)
        logger.info("Loaded: %d polygon(s), %d circle(s)", len(result["polygons"]), len(result["circles"]))
    except Exception:
        logger.exception("Error loading DXF")
        return 1

    if not result["polygons"]:
        logger.error("No polygons found in DXF")
        return 1

    # 2) Pick parameters (largest polygon by area is usually the channel)
    polys = result["polygons"]
    # If your loader provides 'area', use it; otherwise fall back to first polygon.
    polys_sorted = sorted(polys, key=lambda p: float(p.get("area", 0.0)), reverse=True)
    primary = polys_sorted[0]

    w, h, max_dim = poly_bounds(primary)
    logger.info("Polygon size: %.1f × %.1f µm (max: %.1f)", w, h, max_dim)

    # Auto-tune resolution and simplify_tolerance based on channel width
    logger.info("Parameters: um_per_px=auto (will be tuned from channel width), simplify_tolerance=auto (0.5 * um_per_px)")
    logger.info("Cross-section defaults: height=50.0 µm, kind=rectangular")

    # 3) Extract graph with port detection
    circles = result.get("circles", [])
    logger.info("Found %d circle(s) for port detection", len(circles))
    
    try:
        with log_step("Graph extraction"):
            graph_result = extract_graph_from_polygons(
                polys_sorted,
                um_per_px=None,  # Auto-tune based on channel width
                simplify_tolerance=None,  # Auto-computed as 0.5 * um_per_px
                auto_tune_resolution=True,
                circles=circles if circles else None,
                port_snap_distance=50.0,  # Snap ports within 50 µm
                detect_polyline_circles=False,  # Don't detect circle-like polylines for now
            )
    except Exception:
        logger.exception("Error extracting graph")
        return 1

    nodes = graph_result.get("nodes", [])
    edges = graph_result.get("edges", [])
    ports = graph_result.get("ports", [])
    logger.info("Graph: %d nodes, %d edges, %d ports", len(nodes), len(edges), len(ports))

    # Node summary
    kinds: dict[str, int] = {}
    for n in nodes:
        kinds[n.get("kind", "unknown")] = kinds.get(n.get("kind", "unknown"), 0) + 1
    for kind in sorted(kinds):
        logger.info("  %s: %d", kind, kinds[kind])

    # Node details (brief)
    for n in nodes:
        x, y = n.get("xy", [None, None])
        logger.info("  %s: %s at (%.1f, %.1f), degree=%s", n.get("id"), n.get("kind"), x, y, n.get("degree", "N/A"))

    # Edge details (with measurements)
    for e in edges:
        length = e.get("length", 0.0)
        width_profile = e.get("width_profile", {})
        w_kind = width_profile.get("kind", "unknown")
        w_median = width_profile.get("w_median", 0.0)
        w_min = width_profile.get("w_min", 0.0)
        w_max = width_profile.get("w_max", 0.0)
        num_samples = len(width_profile.get("samples", []))
        logger.info(
            "  %s: %s → %s, length=%.1f µm, width: %s (median=%.1f, min=%.1f, max=%.1f µm, %d samples)",
            e["id"], e["u"], e["v"], length, w_kind, w_median, w_min, w_max, num_samples
        )

    # Validate references
    node_ids = {n["id"] for n in nodes}
    bad_edges = [e for e in edges if e["u"] not in node_ids or e["v"] not in node_ids]
    if bad_edges:
        logger.warning("Found %d edges with invalid node references", len(bad_edges))
    else:
        logger.info("All edges connect existing nodes")

    # Check near-zero length edges
    zero = []
    for e in edges:
        length = e.get("length", 0.0)
        if length < 1.0:
            zero.append(e["id"])
    if zero:
        logger.warning("Found %d near-zero-length edges: %s", len(zero), zero)
    else:
        logger.info("No near-zero-length edges")

    # Port details
    if ports:
        logger.info("Ports:")
        for port in ports:
            marker = port.get("marker", {})
            center = marker.get("center", [None, None])
            radius = marker.get("radius", None)
            node_id = port.get("node_id")
            logger.info(
                "  %s: center=(%.1f, %.1f), radius=%.1f µm, attached to %s",
                port.get("port_id"), center[0], center[1], radius, node_id or "none"
            )
    else:
        logger.info("No ports detected")
    
    # Save JSON
    out_path = Path(__file__).resolve().parent / "y_channel_graph.json"
    with open(out_path, "w") as f:
        json.dump({"nodes": nodes, "edges": edges, "ports": ports}, f, indent=2)

    logger.info("Graph saved to: %s", out_path)
    
    # Export overlay image
    try:
        with log_step("Export overlay"):
            overlay_path = Path(__file__).resolve().parent / "y_channel_overlay.png"
            export_overlay_png(
                polygons=polys_sorted,
                nodes=nodes,
                edges=edges,
                ports=ports,
                output_path=str(overlay_path),
                image_width=2000
            )
        logger.info("Overlay saved to: %s", overlay_path)
    except Exception:
        logger.exception("Error exporting overlay")
        # Don't fail the script if overlay export fails
    logger.info("=" * 70)
    logger.info("Network graph extraction completed successfully")
    logger.info("Full log saved to: %s", LOG_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
