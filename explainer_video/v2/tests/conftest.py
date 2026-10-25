"""Make the v2 tooling importable from the test session."""
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))
