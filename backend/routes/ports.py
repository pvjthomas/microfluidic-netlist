"""Port assignment endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal

router = APIRouter()


class PortAssignment(BaseModel):
    port_id: str
    role: Literal["inlet", "outlet", "internal", "unknown"]
    label: str


class AssignPortsRequest(BaseModel):
    session_id: str
    assignments: list[PortAssignment]


class SetDefaultsRequest(BaseModel):
    session_id: str
    defaults: dict  # height, cross_section_kind


@router.post("/set-defaults")
async def set_defaults(req: SetDefaultsRequest):
    """Set global defaults for height and cross-section."""
    # TODO: Store defaults in session
    return {"status": "ok"}


@router.post("/assign-ports")
async def assign_ports(req: AssignPortsRequest):
    """Assign roles and labels to ports."""
    # TODO: Store port assignments in session
    return {"status": "ok"}

