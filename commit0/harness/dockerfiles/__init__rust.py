from __future__ import annotations

import logging
from typing import List, Optional

from commit0.harness.constants import DOCKERFILES_DIR

_logger = logging.getLogger(__name__)


def get_dockerfile_rust_base(rust_version: str) -> str:
    """Read the Dockerfile.rust template and substitute the version placeholder.

    Args:
        rust_version: Rust toolchain version string (e.g. ``"1.78.0"``).

    Returns:
        Dockerfile content with ``RUST_VERSION`` replaced by *rust_version*.

    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    template_path = DOCKERFILES_DIR / "Dockerfile.rust"
    if not template_path.exists():
        raise FileNotFoundError(f"Base Dockerfile template not found: {template_path}")
    content = template_path.read_text()
    return content.replace("RUST_VERSION", rust_version)


def get_dockerfile_rust_repo(
    base_image: str,
    pre_install: Optional[List[str]] = None,
    packages: Optional[str] = None,
    install_cmd: Optional[str] = None,
    features: Optional[List[str]] = None,
) -> str:
    """Generate a repo-specific Dockerfile for a Rust project.

    Args:
        base_image: Base Docker image tag (e.g. ``"commit0/rust-base:1.78"``).
        pre_install: Optional list of shell commands to run before the build.
            ``apt-get install`` / ``apt install`` commands are collected and
            batched into a single ``RUN`` layer.
        packages: Optional path to a requirements file to copy and process.
        install_cmd: Optional custom build/install command.
        features: Optional list of Cargo feature names; stored in
            ``CARGO_FEATURES`` environment variable for later use.

    Returns:
        Complete Dockerfile content as a string.
    """
    lines = [
        f"FROM {base_image}",
        "",
        'ARG http_proxy=""',
        'ARG https_proxy=""',
        'ARG HTTP_PROXY=""',
        'ARG HTTPS_PROXY=""',
        'ARG no_proxy="localhost,127.0.0.1,::1"',
        'ARG NO_PROXY="localhost,127.0.0.1,::1"',
        "",
        "COPY ./setup.sh /root/",
        "RUN chmod +x /root/setup.sh && /bin/bash /root/setup.sh",
        "",
        "WORKDIR /testbed/",
        "",
    ]

    apt_packages: list[str] = []
    if pre_install:
        for cmd in pre_install:
            if cmd.startswith("apt-get install") or cmd.startswith("apt install"):
                pkgs = cmd.split("install", 1)[1].replace("-y", "").strip().split()
                apt_packages.extend(p for p in pkgs if not p.startswith("-"))
            else:
                lines.append(f"RUN {cmd}")

    if apt_packages:
        pkg_str = " \\\n    ".join(sorted(set(apt_packages)))
        lines.append(
            f"RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
            f"    {pkg_str} \\\n"
            f"    && rm -rf /var/lib/apt/lists/*"
        )
        lines.append("")

    if packages:
        lines.append(f"COPY {packages} /testbed/{packages}")
        lines.append("")

    if features:
        features_str = ",".join(features)
        lines.append(f'ENV CARGO_FEATURES="{features_str}"')
        lines.append("")

    if install_cmd:
        lines.append(f"RUN {install_cmd}")
        lines.append("")

    # Dependency manifest for debugging
    lines.append(
        "RUN cargo --version > /testbed/.dep-manifest.txt"
        " && rustc --version >> /testbed/.dep-manifest.txt"
    )
    lines.append("")

    lines.append("WORKDIR /testbed/")
    lines.append("")

    return "\n".join(lines)


__all__: list[str] = [
    "get_dockerfile_rust_base",
    "get_dockerfile_rust_repo",
]
