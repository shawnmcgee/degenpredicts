"""Put the repo root on sys.path so `import cfb` / `import ncaab` work under pytest.

pytest only auto-inserts the rootdir when it finds a conftest.py there, and CI invokes it as
`pytest tests/...` from the repo root. Without this, collection fails with ModuleNotFoundError.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
