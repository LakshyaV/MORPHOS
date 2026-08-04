"""Experiment configuration: frozen dataclasses + YAML with `_base_` inheritance.

Deliberately not Hydra. Hydra changes the working directory (which breaks
relative paths and confuses the ffmpeg subprocess), and its real value is
composition across many orthogonal groups -- this project has four swappable
axes. What is worth keeping from it is `_base_` inheritance and dotted CLI
overrides, which is ~70 lines.

Unknown keys raise. A typo'd `--set` is the single most common config bug, and
silently accepting it means discovering days later that an experiment never had
the setting you thought it did.

torch is deliberately not imported at module scope, so `--help` stays instant.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass(frozen=True)
class TargetCfg:
    shape: str = "disk"  # disk | disk_gradient | tadpole
    radius: float = 10.0
    edge: float = 1.0
    color: tuple[float, float, float] = (0.20, 0.60, 0.86)


@dataclass(frozen=True)
class NCACfg:
    hidden: int = 128
    grid: int = 32
    fire_rate: float = 0.5
    alive_threshold: float = 0.1


@dataclass(frozen=True)
class TrainCfg:
    phase: str = "morph"  # morph | comm
    steps: int = 6000
    batch: int = 32
    lr: float = 2.0e-3
    lr_milestones: tuple[int, ...] = (2000,)
    lr_gamma: float = 0.1
    grad_norm_per_tensor: bool = True

    # Upstream draws ONE rollout length per training step, not per sample.
    rollout_min: int = 64
    rollout_max: int = 96

    # Mandatory on 8 GB: 3300 MB -> 1110 MB at T=64, and memory is then nearly
    # independent of rollout length.
    checkpoint_every: int | None = 8

    pool_size: int = 1024
    pool_from_step: int | None = 1000  # None => Growing regime forever
    damage_from_step: int | None = 1000  # None => never damage

    # Scale by FRACTION, not count: reseed/batch is the loss weight on
    # "grow from a seed" and batch/reseed is the mean pool-entry age. Upstream is
    # 1-of-8 and 3-of-8, so at batch 32 these are 4 and 12.
    reseed_worst: int = 4
    damage_best: int = 12

    log_interval: int = 50
    gate_interval: int = 500
    ckpt_interval: int = 1000
    early_stop_on_gates: bool = True


@dataclass(frozen=True)
class GuardCfg:
    death_threshold: float = 0.1
    death_patience: int = 3
    max_resets: int = 3


@dataclass(frozen=True)
class EvalCfg:
    t_grow: int = 64
    t_persist: int = 256
    g1_rollouts: int = 128
    g2_trials: int = 256
    g2_severity: float = 0.30
    g2_tolerance: float = 0.02
    g2_recover_steps: int = 128


@dataclass(frozen=True)
class Config:
    name: str = "morph"
    seed: int = 0
    device: str = "mps"
    target: TargetCfg = field(default_factory=TargetCfg)
    nca: NCACfg = field(default_factory=NCACfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    guard: GuardCfg = field(default_factory=GuardCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    def __post_init__(self) -> None:
        t, n, tr = self.target, self.nca, self.train
        if t.radius * 2 + 2 >= n.grid:
            raise ValueError(
                f"target radius {t.radius} does not fit in a {n.grid}x{n.grid} grid "
                "with room for the antialiased rim"
            )
        if tr.reseed_worst + tr.damage_best > tr.batch:
            raise ValueError(
                f"reseed_worst({tr.reseed_worst}) + damage_best({tr.damage_best}) "
                f"exceeds batch({tr.batch}); the reseeded and damaged slices would overlap"
            )
        if tr.rollout_min > tr.rollout_max:
            raise ValueError(f"rollout_min {tr.rollout_min} > rollout_max {tr.rollout_max}")
        if tr.damage_from_step is not None and tr.pool_from_step is None:
            raise ValueError("damage requires the pool; set pool_from_step")
        if self.device not in {"cpu", "mps", "cuda"}:
            raise ValueError(f"unknown device {self.device!r}")


# --- YAML loading -------------------------------------------------------------

_SECTIONS = {f.name: f.type for f in fields(Config)}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml_with_base(path: Path, _seen: set[Path] | None = None) -> dict:
    seen = _seen or set()
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular _base_ chain at {path}")
    seen.add(path)

    raw = yaml.safe_load(path.read_text()) or {}
    base_ref = raw.pop("_base_", None)
    if base_ref is None:
        return raw
    base_path = (path.parent / base_ref).resolve()
    return _deep_merge(_read_yaml_with_base(base_path, seen), raw)


def _coerce(value: Any, target_type: Any) -> Any:
    """Coerce a YAML scalar/sequence into the dataclass field's declared type."""
    if value is None:
        return None
    origin = getattr(target_type, "__origin__", None)
    if target_type is tuple or origin is tuple:
        return tuple(value)
    if target_type in (int, float, str, bool):
        return target_type(value)
    return value


@lru_cache(maxsize=None)
def _hints(cls: type) -> dict[str, Any]:
    """Resolved type hints.

    `from __future__ import annotations` makes `dataclasses.field.type` a *string*,
    so `is_dataclass(f.type)` would be False for every nested section and the
    sub-configs would silently arrive as raw dicts. get_type_hints resolves them.
    """
    return get_type_hints(cls)


def _build(cls: type, data: dict, path: str = "") -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        where = path or cls.__name__
        raise KeyError(
            f"unknown config key(s) {sorted(unknown)} in {where}; "
            f"valid keys are {sorted(known)}"
        )
    hints = _hints(cls)
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        v, ann = data[name], hints[name]
        if is_dataclass(ann) and isinstance(v, dict):
            kwargs[name] = _build(ann, v, f"{path}.{name}" if path else name)
        else:
            kwargs[name] = _coerce(v, ann)
    return cls(**kwargs)


def _apply_override(data: dict, spec: str) -> dict:
    """Apply one `a.b=c` override. RHS is parsed as YAML so types come out right."""
    if "=" not in spec:
        raise ValueError(f"malformed --set {spec!r}; expected key.path=value")
    key, _, raw = spec.partition("=")
    value = yaml.safe_load(raw)

    parts = key.strip().split(".")
    node = data
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ValueError(f"--set {spec!r} descends into a non-mapping at {p!r}")
    node[parts[-1]] = value
    return data


def load_config(path: str | Path, *, overrides: Sequence[str] = ()) -> Config:
    data = _read_yaml_with_base(Path(path))
    for spec in overrides:
        data = _apply_override(data, spec)
    return _build(Config, data)


def config_dict(cfg: Config) -> dict:
    return asdict(cfg)


def config_hash(cfg: Config) -> str:
    blob = json.dumps(config_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def run_dir_name(cfg: Config, *, date: str) -> str:
    """e.g. 20260804-morph-a91c3e0f21b8-s0. Date is passed in, never computed here,
    so the function stays pure and testable."""
    return f"{date}-{cfg.name}-{config_hash(cfg)}-s{cfg.seed}"


def with_overrides(cfg: Config, **kw: Any) -> Config:
    return replace(cfg, **kw)
