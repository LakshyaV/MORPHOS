"""Checkpoint save/load.

Disk policy matters here: the sample pool is 1024 x 28 x 32 x 32 float32 =
117 MB. Writing that into every numbered checkpoint would be ~10 GB across a
handful of runs on a machine that was at 99% full. So:

    ckpt/last.pt        weights + opt + sched + rng + POOL + step   ~118 MB, overwritten
    ckpt/step_NNNNNN.pt weights + rng + step                        ~110 KB, kept
    ckpt/best.pt        weights + rng + gate metrics                ~110 KB

Only tensors and plain containers are saved -- never `torch.save(model)`, which
pickles the class and breaks on any refactor. torch 2.9 defaults to
`weights_only=True` on load, so the config travels as a JSON *string*.

Writes go to a temporary file then `os.replace`, which is atomic on POSIX: a
crash mid-write leaves the previous checkpoint intact rather than a truncated one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from morphos.seeding import RNG, load_rng_state, rng_state


def save(
    path: str | Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    rng: RNG | None = None,
    pool: Any = None,
    config: dict | None = None,
    target_fingerprint: str | None = None,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "step": step,
        "model": model.state_dict(),
        "torch_version": torch.__version__,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if rng is not None:
        payload["rng"] = rng_state(rng)
    if pool is not None:
        payload["pool"] = pool.state_dict()
    if config is not None:
        payload["config_json"] = json.dumps(config, default=str)
    if target_fingerprint is not None:
        payload["target_fingerprint"] = target_fingerprint
    if extra:
        payload["extra"] = json.dumps(extra, default=str)

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic
    return path


def load(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    rng: RNG | None = None,
    pool: Any = None,
    map_location: str | torch.device = "cpu",
    expect_target_fingerprint: str | None = None,
) -> dict:
    """Restore in place and return the raw payload."""
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)

    if expect_target_fingerprint is not None:
        got = ckpt.get("target_fingerprint")
        if got is not None and got != expect_target_fingerprint:
            raise ValueError(
                f"checkpoint was trained against target {got}, but the current "
                f"target is {expect_target_fingerprint}; refusing to resume"
            )

    if model is not None:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if rng is not None and "rng" in ckpt:
        load_rng_state(rng, ckpt["rng"])
    if pool is not None and "pool" in ckpt:
        pool.load_state_dict(ckpt["pool"])
    return ckpt


def latest(run_dir: str | Path) -> Path | None:
    p = Path(run_dir) / "ckpt" / "last.pt"
    return p if p.exists() else None


def config_from(ckpt: dict) -> dict | None:
    raw = ckpt.get("config_json")
    return json.loads(raw) if raw else None


def load_organism(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[Any, dict, int | None]:
    """Rebuild a trained organism straight from a checkpoint.

    Lives here rather than in a script so evaluate / make_video / probe all share
    one loader -- scripts importing each other is not a package layout.

    -> (model in eval mode, config dict, step)
    """
    from morphos.substrate.nca import SENDER_LAYOUT, NCAOrganism

    raw = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = config_from(raw)
    if cfg is None:
        raise ValueError(f"{path} carries no embedded config")

    model = NCAOrganism(
        layout=SENDER_LAYOUT,
        hidden=cfg["nca"]["hidden"],
        grid=cfg["nca"]["grid"],
        fire_rate=cfg["nca"]["fire_rate"],
        alive_threshold=cfg["nca"]["alive_threshold"],
    ).to(device)
    model.load_state_dict(raw["model"])
    model.eval()
    return model, cfg, raw.get("step")
