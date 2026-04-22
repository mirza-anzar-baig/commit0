from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field

from commit0.harness.constants import (
    DOCKERFILES_DIR,
    RepoInstance,
    TestStatus,
)

__all__ = [
    "RustRepoInstance",
    "RUST_VERSION",
    "RUST_STUB_MARKER",
    "RUST_SPLIT",
    "CARGO_NEXTEST_VERSION",
    "RUN_RUST_TESTS_LOG_DIR",
    "RUST_TEST_IDS_DIR",
    "DOCKERFILES_DIR",
    "TestStatus",
]

# Rust toolchain version
RUST_VERSION = "stable"

# Marker used to identify stub functions in Rust source
RUST_STUB_MARKER = 'todo!("STUB")'

# Repo split mapping for Rust repos
RUST_SPLIT: Dict[str, list[str]] = {
    "all": [
        "Rust-commit0/opentelemetry-rust",
    ],
}

# cargo-nextest version for test execution
CARGO_NEXTEST_VERSION = "0.9.96"

# Log directory for Rust test runs
RUN_RUST_TESTS_LOG_DIR = Path("logs/rust_tests")

# Directory containing per-repo Rust test IDs
RUST_TEST_IDS_DIR = Path(__file__).parent.parent / "data" / "rust_test_ids"


class RustRepoInstance(RepoInstance):
    """Repo instance with Rust-specific metadata."""

    edition: str = "2021"
    features: List[str] = Field(default_factory=list)
    workspace: bool = False
