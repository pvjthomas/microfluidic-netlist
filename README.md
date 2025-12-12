# Microfluidic DXF to Network IR Converter

Convert 2D microfluidic mask designs (DXF) into a canonical graph IR, generate SPICE-like netlists, and export tagged overlays.

## Setup

1. Install Python 3.12+ and Node.js
2. Install dependencies:
   ```bash
   make install
   ```

## Development

Run both backend and frontend:
```bash
make dev
```

Or run separately:
```bash
make backend   # FastAPI on http://localhost:8000
make frontend  # Vite on http://localhost:5173
```

## Project Structure

- `core/` - Python library (geometry + graph + exports)
- `backend/` - FastAPI server
- `frontend/` - React/TypeScript UI
- `examples/` - Sample DXF files
- `docs/` - Documentation

