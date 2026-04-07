"""Entry point for IndiFact-Agentic-RAG Streamlit app."""
import os
import sys
from pathlib import Path

# Add the workspace root to Python path
workspace_root = Path(__file__).parent
sys.path.insert(0, str(workspace_root))

# Keep model libraries quieter in local app logs.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Streamlit moved its CLI module in newer releases.
try:
    from streamlit.web import cli as stcli
except ImportError:  # pragma: no cover - compatibility with older Streamlit versions
    import streamlit.cli as stcli

sys.argv = ["streamlit", "run", str(workspace_root / "src" / "ui" / "app.py"), "--logger.level=error"]
sys.exit(stcli.main())
