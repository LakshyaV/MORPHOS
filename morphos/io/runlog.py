"""Run directories and JSONL metric logging.

JSONL rather than TensorBoard or wandb: ~60 KB per run, greppable, jq-able,
append-only so it survives a crash mid-write, and it needs no server running on
an 8 GB machine that is also doing the training.

Every line is flushed and fsync'd. At one line per 50 steps that costs nothing,
and crash-safety matters more than throughput here.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def _git_info(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"git_sha": sha, "git_dirty": bool(status) if status is not None else None}


class RunLog:
    """Owns a run directory: config, metadata, metrics.jsonl, events.jsonl."""

    def __init__(self, run_dir: str | Path, config: dict, *, resume: bool = False) -> None:
        self.dir = Path(run_dir)
        (self.dir / "ckpt").mkdir(parents=True, exist_ok=True)
        (self.dir / "media").mkdir(parents=True, exist_ok=True)

        mode = "a" if resume else "w"
        self._metrics = open(self.dir / "metrics.jsonl", mode)
        self._events = open(self.dir / "events.jsonl", mode)

        if not resume:
            (self.dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            repo = Path(__file__).resolve().parents[2]
            meta = {
                "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
                "argv": sys.argv,
                **_git_info(repo),
            }
            try:
                import torch

                meta["torch"] = torch.__version__
            except ImportError:
                pass
            (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def _write(self, handle, payload: dict) -> None:
        handle.write(json.dumps(payload, default=float) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def log(self, **fields: Any) -> None:
        """One metrics record per line."""
        self._write(self._metrics, fields)

    def event(self, kind: str, **fields: Any) -> None:
        """Notable occurrences: death resets, gate results, phase transitions."""
        self._write(self._events, {"kind": kind, "time": time.time(), **fields})

    @property
    def ckpt_dir(self) -> Path:
        return self.dir / "ckpt"

    @property
    def media_dir(self) -> Path:
        return self.dir / "media"

    def close(self) -> None:
        for h in (self._metrics, self._events):
            if not h.closed:
                h.close()

    def __enter__(self) -> RunLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_metrics(run_dir: str | Path) -> list[dict]:
    """Load metrics.jsonl back. Tolerates a truncated final line from a hard kill."""
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return rows
