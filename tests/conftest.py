"""Optional heavy dependencies are skipped at the individual test/module boundary."""
from __future__ import annotations

import pytest


@pytest.fixture
def x265_library():
    pytest.importorskip("numpy")
    from prism.codec.x265_bridge import X265Library, X265Unavailable
    try:
        return X265Library()
    except X265Unavailable as exc:
        pytest.skip(str(exc))
