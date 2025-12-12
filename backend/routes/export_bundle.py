"""Export bundle endpoint."""

from fastapi import APIRouter, Response
from pydantic import BaseModel
import zipfile
import io

router = APIRouter()


class ExportRequest(BaseModel):
    session_id: str


@router.post("/export")
async def export_bundle(req: ExportRequest):
    """Export zip containing graph.json, segments.csv, netlist.cir, overlay.png."""
    # TODO: Generate all exports and bundle into zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Placeholder files
        zip_file.writestr("design.graph.json", "{}")
        zip_file.writestr("design.segments.csv", "edge_id,node_u,node_v\n")
        zip_file.writestr("design.netlist.cir", "* Placeholder netlist\n")
        zip_file.writestr("design.overlay.png", b"")
    
    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.read(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=design_export.zip"}
    )

