from __future__ import annotations

from pathlib import Path

import pytest

from core.storage.db import Database


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")
