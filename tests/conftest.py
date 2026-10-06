import sys
from pathlib import Path

# The package next to the tests, whichever directory pytest starts from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
