"""Graph extraction endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ExtractGraphRequest(BaseModel):
    session_id: str
    tolerances: dict | None = None
    width_sample_step: float = 10.0


class ExtractGraphResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]
    suggested_port_attachments: list[dict]


@router.post("/extract-graph", response_model=ExtractGraphResponse)
async def extract_graph(req: ExtractGraphRequest):
    """Extract network graph from selected regions."""
    # TODO: Implement graph extraction
    return ExtractGraphResponse(
        nodes=[],
        edges=[],
        suggested_port_attachments=[],
    )

