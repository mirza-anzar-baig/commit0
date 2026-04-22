"""Rust-specific Spec subclass for the commit0 harness pipeline."""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass
from typing import Optional, Union

from commit0.harness.spec import Spec
from commit0.harness.constants import (
    ABSOLUTE_REPO_DIR,
    RELATIVE_REPO_DIR,
    RepoInstance,
    SimpleInstance,
)
from commit0.harness.constants_rust import RustRepoInstance, RUST_VERSION
from commit0.harness.dockerfiles.__init__rust import (
    get_dockerfile_rust_base,
    get_dockerfile_rust_repo,
)

logger = logging.getLogger(__name__)


@dataclass
class Commit0RustSpec(Spec):
    """Spec subclass for Rust repositories evaluated via cargo-nextest."""

    def _get_rust_version(self) -> str:
        setup = self._get_setup_dict()
        return str(setup.get("rust", RUST_VERSION))

    @property
    def base_image_key(self) -> str:
        rust_version = self._get_rust_version()
        return f"commit0.base.rust.{rust_version}:latest"

    @property
    def base_dockerfile(self) -> str:
        return get_dockerfile_rust_base(self._get_rust_version())

    @property
    def repo_dockerfile(self) -> str:
        specs = self._get_setup_dict()
        features = None
        if isinstance(self.instance, RustRepoInstance):
            features = self.instance.features or None
        elif isinstance(self.instance, dict):
            features = specs.get("features")
        return get_dockerfile_rust_repo(
            base_image=self.base_image_key,
            pre_install=specs.get("pre_install"),
            packages=specs.get("packages"),
            install_cmd=specs.get("install"),
            features=features if features else None,
        )

    def get_container_name(self, run_id: Optional[str] = None) -> str:
        repo = self.repo.split("/")[-1]
        if not run_id:
            return f"commit0.rust.eval.{repo}"
        return f"commit0.rust.eval.{repo}.{run_id}".lower()

    def make_repo_script_list(self) -> list[str]:
        """Set up a Rust repo: clone, fetch commits, reset, warm cargo cache."""
        repo = self.instance["repo"]
        env_setup_commit = self.instance["reference_commit"]
        base_commit = self.instance["base_commit"]

        setup_commands = [
            f"git clone --depth 1 -o origin https://github.com/{repo} {self.repo_directory}",
            f"chmod -R 777 {self.repo_directory}",
            f"cd {self.repo_directory}",
            f"git fetch --depth 1 origin {env_setup_commit} {base_commit}",
            f"git reset --hard {env_setup_commit}",
            "git submodule update --init --recursive 2>/dev/null || true",
            "git remote remove origin",
            f"git reset --hard {base_commit}",
            # Warm cargo dependency cache
            "cargo fetch 2>/dev/null || true",
        ]
        return setup_commands

    def make_eval_script_list(self) -> list[str]:
        """Run Rust tests via cargo-nextest."""
        diff_path = "/patch.diff" if self.absolute else "../patch.diff"
        eval_script_list = [
            f"cd {self.repo_directory}",
            f"git reset --hard {self.instance['base_commit']}",
            f"git apply --allow-empty -v {diff_path}",
            "git status",
            "cargo nextest run --message-format libtest-json {test_ids} > test_output.txt 2>&1",
            "echo $? > cargo_test_exit_code.txt",
        ]
        return eval_script_list


def make_rust_spec(
    instance: Union[RepoInstance, RustRepoInstance],
    absolute: bool = True,
) -> Commit0RustSpec:
    """Factory: create a Commit0RustSpec from a repo instance."""
    repo_directory = ABSOLUTE_REPO_DIR if absolute else RELATIVE_REPO_DIR
    return Commit0RustSpec(
        repo=instance["instance_id"],
        repo_directory=repo_directory,
        instance=instance,
        absolute=absolute,
    )


__all__ = ["Commit0RustSpec", "make_rust_spec"]
