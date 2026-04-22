from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from commit0.harness.constants import RepoInstance, SimpleInstance


@pytest.fixture
def sample_repo_instance() -> RepoInstance:
    return RepoInstance(
        instance_id="test/repo",
        repo="test-repo",
        base_commit="abc123",
        reference_commit="def456",
        setup={
            "python": "3.12",
            "packages": "requirements.txt",
            "install": "pip install -e .",
        },
        test={"test_cmd": "pytest", "test_file.py": "def test_example(): pass"},
        src_dir="src",
    )


@pytest.fixture
def sample_simple_instance() -> SimpleInstance:
    return SimpleInstance(
        instance_id="simple/1",
        prompt="Write hello",
        canonical_solution="print('hello')",
        test="assert True",
    )


@pytest.fixture
def mock_logger() -> MagicMock:
    logger = MagicMock()
    logger.log_file = Path("/tmp/test.log")
    return logger
