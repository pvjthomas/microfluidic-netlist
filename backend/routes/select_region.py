"""Channel region selection endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal

router = APIRouter()


class SelectChannelsRequest(BaseModel):
    session_id: str
    mode: Literal["layer", "click"]
    layers: list[str] | None = None
    point: list[float] | None = None


class SelectChannelsResponse(BaseModel):
    selected_region_ids: list[str]
    geometry: dict  # Lightweight geometry for frontend


@router.post("/select-channels", response_model=SelectChannelsResponse)
async def select_channels(req: SelectChannelsRequest):
    """Select channel regions by layer or click."""
    # TODO: Implement region selection
    return SelectChannelsResponse(
        selected_region_ids=[],
        geometry={},
    )

