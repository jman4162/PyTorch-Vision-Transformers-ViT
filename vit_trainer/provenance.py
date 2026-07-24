"""Run provenance: what produced a number, on what, from which code.

A reported accuracy is only reproducible if the commit, environment, hardware,
seed, and data split that produced it are recorded alongside it.
"""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union


def git_revision(repo_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Best-effort git commit and working-tree state.

    Returns `{"commit": None}` outside a repository or without git installed —
    the case when the package was installed from PyPI.
    """
    cwd = str(repo_dir) if repo_dir else str(Path(__file__).resolve().parent)

    def run(args):
        return subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()

    try:
        return {
            "commit": run(["git", "rev-parse", "HEAD"]),
            "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(run(["git", "status", "--porcelain"])),
        }
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return {"commit": None, "branch": None, "dirty": None}


def hash_indices(indices: Sequence[int]) -> str:
    """Stable fingerprint of a data split.

    Two runs with the same hash trained on the same examples, whatever the seed
    handling looked like.
    """
    payload = ",".join(str(int(i)) for i in sorted(indices)).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def environment_info() -> Dict[str, Any]:
    """Versions and hardware relevant to reproducing a training run."""
    import torch
    import torchvision

    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "cudnn": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "gpus": [],
    }

    if torch.cuda.is_available():
        info["gpus"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]

    return info


def collect_run_metadata(
    config: Any,
    split_hash: Optional[str] = None,
    results: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble everything needed to attribute a result to a run.

    Args:
        config: TrainingConfig for the run
        split_hash: Fingerprint of the validation indices, from `hash_indices`
        results: Metrics, timings, peak memory
        timestamp: ISO timestamp; generated if omitted

    Returns:
        JSON-serializable run manifest
    """
    from datetime import datetime, timezone

    from . import __version__

    return {
        "vit_trainer_version": __version__,
        "timestamp": timestamp
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_revision(),
        "environment": environment_info(),
        "config": config.to_dict() if hasattr(config, "to_dict") else dict(config),
        "split_hash": split_hash,
        "results": results or {},
    }


def save_run_metadata(metadata: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Write a run manifest as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return path
