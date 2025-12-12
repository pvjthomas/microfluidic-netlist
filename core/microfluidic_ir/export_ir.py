"""Export canonical IR to JSON."""

from .schemas import GraphIR
import json


def export_graph_ir(graph_ir: GraphIR, output_path: str) -> None:
    """Export GraphIR to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(graph_ir.model_dump(), f, indent=2)

