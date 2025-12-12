"""Session storage for temporary workspace per session."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional

# Global session storage
_sessions: Dict[str, Path] = {}


def get_session_dir(session_id: str) -> Path:
    """Get or create temp directory for session."""
    if session_id not in _sessions:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"microfluidic_{session_id}_"))
        _sessions[session_id] = temp_dir
    return _sessions[session_id]


def cleanup_session(session_id: str) -> None:
    """Clean up session directory."""
    if session_id in _sessions:
        shutil.rmtree(_sessions[session_id], ignore_errors=True)
        del _sessions[session_id]

