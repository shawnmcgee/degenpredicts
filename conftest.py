"""Put the repo root on sys.path so `import cfb` works when pytest runs from anywhere."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
