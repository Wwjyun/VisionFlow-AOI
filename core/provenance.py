from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def build_provenance() -> dict[str, Any]:
    """Return process-stable application provenance without spawning Git per image."""
    commit, dirty, source = _cached_build_provenance()
    return {"commit": commit, "dirty": dirty, "source": source}


@lru_cache(maxsize=1)
def _cached_build_provenance() -> tuple[str, bool | None, str]:
    """Resolve build identity once; a running executable cannot change its own build."""
    env_commit = os.environ.get("VISIONFLOW_BUILD_COMMIT", "").strip()
    if env_commit:
        return env_commit, _env_bool("VISIONFLOW_BUILD_DIRTY"), "environment"
    packaged = _read_packaged_provenance()
    if packaged is not None:
        return str(packaged["commit"]), bool(packaged["dirty"]), str(packaged["source"])
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
        return commit, dirty, "git"
    except (OSError, subprocess.SubprocessError):
        return "unknown", None, "unavailable"


def inspection_provenance(recipe_path: Path, effective_recipe: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "recipe_source_sha256": sha256_bytes(Path(recipe_path).read_bytes()),
        "effective_recipe_sha256": canonical_sha256(effective_recipe),
        "app": build_provenance(),
        "detector_params": {
            str(detector_id): dict(config.get("params", {}))
            for detector_id, config in effective_recipe.get("detectors", {}).items()
        },
    }


def _read_packaged_provenance() -> dict[str, Any] | None:
    roots = [Path(getattr(sys, "_MEIPASS", "")), Path(__file__).resolve().parent.parent]
    for root in roots:
        if not str(root):
            continue
        path = root / "build_provenance.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return {"commit": str(data["commit"]), "dirty": bool(data["dirty"]), "source": "embedded"}
            except (KeyError, TypeError, ValueError, OSError):
                continue
    return None


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, check=True, text=True, encoding="utf-8"
    ).stdout.strip()


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}
