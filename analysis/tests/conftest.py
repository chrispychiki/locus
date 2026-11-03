import sys
from pathlib import Path

# The event builders, db builders, and the fixtures both suites run under live once, in the
# evidence suite's support module; this suite depends on them and reaches them by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evidence" / "tests"))

from _analysis_support import (
    shipped_card_roster,  # noqa: F401 — autouse fixture
    spend_isolation,  # noqa: F401 — autouse fixture
)
from _support import (
    contained,  # noqa: F401 — autouse fixture, registered by import
    distilled_db,  # noqa: F401
    two_visitor_db,  # noqa: F401
)
