.PHONY: dev backend frontend fmt test install clean

# Install all dependencies
install:
	cd core && pip install -e .
	cd backend && pip install -e .
	cd frontend && npm install

# Run both backend and frontend
dev: backend frontend

# Run backend only
backend:
	cd backend && uvicorn app:app --reload --port 8000

# Run frontend only
frontend:
	cd frontend && npm run dev

# Format code
fmt:
	cd core && black . && isort .
	cd backend && black . && isort .
	cd frontend && npm run fmt || true

# Run tests
test:
	cd core && pytest
	cd backend && pytest
	cd frontend && npm test || true

# Clean temporary files
clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	cd frontend && rm -rf node_modules dist || true

