"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import import_dxf, select_region, extract_graph, ports, export_bundle

app = FastAPI(title="Microfluidic DXF Converter API", version="0.1.0")

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(import_dxf.router, prefix="/api", tags=["import"])
app.include_router(select_region.router, prefix="/api", tags=["channels"])
app.include_router(extract_graph.router, prefix="/api", tags=["graph"])
app.include_router(ports.router, prefix="/api", tags=["ports"])
app.include_router(export_bundle.router, prefix="/api", tags=["export"])


@app.get("/")
async def root():
    return {"message": "Microfluidic DXF Converter API", "version": "0.1.0"}


@app.get("/health")
async def health():
    return {"status": "ok"}

