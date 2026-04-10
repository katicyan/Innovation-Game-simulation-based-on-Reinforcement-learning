import sys
from pathlib import Path

# Ensure project-root modules are importable during pytest collection.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
