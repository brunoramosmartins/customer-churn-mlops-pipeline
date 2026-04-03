"""Path helpers so committed YAML/JSON use repo-relative POSIX paths (CI, Docker, clones)."""

from __future__ import annotations

from pathlib import Path


def path_relative_to_repo(root: Path, path: Path) -> str:
    """Return ``path`` relative to ``root`` with forward slashes, or absolute if outside repo."""
    root_r = root.resolve()
    path_r = path.resolve()
    try:
        return path_r.relative_to(root_r).as_posix()
    except ValueError:
        return str(path_r)
