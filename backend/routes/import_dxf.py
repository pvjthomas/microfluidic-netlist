"""DXF import endpoint."""

import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from microfluidic_ir.dxf_loader import load_dxf
from storage import get_session_dir

router = APIRouter()


class ImportResponse(BaseModel):
    session_id: str
    layers: list[str]
    bounds: dict
    candidate_polygons_count: int
    candidate_ports_count: int


@router.post("/import-dxf", response_model=ImportResponse)
async def import_dxf(file: UploadFile = File(...)):
    """Import DXF file and return session info."""
    # Generate session ID
    session_id = str(uuid.uuid4())
    session_dir = get_session_dir(session_id)
    
    # Save uploaded file
    dxf_path = session_dir / "design.dxf"
    with open(dxf_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Load DXF
    try:
        result = load_dxf(str(dxf_path), snap_close_tol=2.0)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load DXF: {e}")
    
    # Store result in session (save as JSON for later use)
    import json
    result_path = session_dir / "import_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    
    return ImportResponse(
        session_id=session_id,
        layers=result["layers"],
        bounds=result["bounds"],
        candidate_polygons_count=len(result["polygons"]),
        candidate_ports_count=len(result["circles"]),
    )
