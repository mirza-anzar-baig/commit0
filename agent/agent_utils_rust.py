import logging
import os
import re

from commit0.harness.constants_rust import RUST_STUB_MARKER

logger = logging.getLogger(__name__)

_EXCLUDED_DIRS = {"tests", "benches", "examples", "target", ".git"}

# group(1) = full signature, group(2) = fn name; handles pub/async/unsafe/const/generics/return type
_FN_PATTERN = re.compile(
    r"((?:pub(?:\s*\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:const\s+)?"
    r"fn\s+(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)(?:\s*->\s*[^{]+?)?\s*)\{",
    re.DOTALL,
)


def find_rust_files_to_edit(src_dir: str) -> list[str]:
    """Walk *src_dir* and collect ``.rs`` files, excluding non-source paths.

    Excluded directories: ``tests``, ``benches``, ``examples``, ``target``, ``.git``.
    Excluded files: ``build.rs`` at any level.

    Returns absolute paths, sorted.
    """
    rs_files: list[str] = []

    for dirpath, dirnames, filenames in os.walk(src_dir):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIRS]

        for fname in filenames:
            if not fname.endswith(".rs"):
                continue
            if fname == "build.rs":
                continue
            rs_files.append(os.path.normpath(os.path.join(dirpath, fname)))

    rs_files.sort()
    return rs_files


def get_target_edit_files_rust(src_dir: str) -> list[str]:
    """Return the subset of ``.rs`` files that contain the stub marker.

    The stub marker is :data:`commit0.harness.constants_rust.RUST_STUB_MARKER`
    (``todo!("STUB")``).
    """
    all_files = find_rust_files_to_edit(src_dir)
    target_files: list[str] = []

    for file_path in all_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            if RUST_STUB_MARKER in content:
                target_files.append(file_path)
        except OSError as exc:
            logger.warning("Could not read %s: %s", file_path, exc)

    return target_files


def extract_rust_function_stubs(file_path: str) -> list[dict]:
    """Find functions whose body contains the stub marker.

    Returns a list of dicts, each with:
      - ``name``  : function name (str)
      - ``line``  : 1-based line number of the ``fn`` keyword (int)
      - ``signature``: full text from qualifiers through the opening ``{`` (str)
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
    except OSError as exc:
        logger.warning("Could not read %s: %s", file_path, exc)
        return []

    stubs: list[dict] = []

    for match in _FN_PATTERN.finditer(content):
        fn_name = match.group(2)
        signature = match.group(1).strip()
        line_number = content[: match.start()].count("\n") + 1

        depth = 1
        pos = match.end()
        while pos < len(content) and depth > 0:
            ch = content[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            pos += 1

        body = content[match.end() : pos - 1] if depth == 0 else ""

        if RUST_STUB_MARKER in body:
            stubs.append(
                {
                    "name": fn_name,
                    "line": line_number,
                    "signature": signature,
                }
            )

    return stubs


def get_rust_file_dependencies(file_path: str) -> list[str]:
    """Parse ``use`` and ``mod`` statements to determine module dependencies.

    Extracts:
      - ``use crate::...`` imports  (returns the crate-relative module path)
      - ``mod name;`` declarations  (external module references, not inline blocks)

    Returns a deduplicated, sorted list of module path strings.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
    except OSError as exc:
        logger.warning("Could not read %s: %s", file_path, exc)
        return []

    deps: set[str] = set()

    for m in re.finditer(r"use\s+crate::(\S+?)\s*[;{]", content):
        path = m.group(1).rstrip(":").rstrip("{")
        if path:
            deps.add(path)

    for m in re.finditer(r"use\s+super::(\S+?)\s*[;{]", content):
        path = m.group(1).rstrip(":").rstrip("{")
        if path:
            deps.add(f"super::{path}")

    for m in re.finditer(r"mod\s+(\w+)\s*;", content):
        deps.add(m.group(1))

    return sorted(deps)
