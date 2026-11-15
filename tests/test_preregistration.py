"""The pre-registration must not change after it was hashed."""
import hashlib
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_preregistration_hash_unchanged():
    recorded = (DOCS / "PREREGISTRATION.sha256").read_text().split()[0]
    actual = hashlib.sha256((DOCS / "PREREGISTRATION.md").read_bytes()).hexdigest()
    assert actual == recorded, "docs/PREREGISTRATION.md was edited after it was hashed; log changes in DEVIATIONS.md"
