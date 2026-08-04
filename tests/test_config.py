"""Config loading tests.

The nested-dataclass test matters more than it looks: with
`from __future__ import annotations`, `dataclasses.field.type` is a *string*, so a
naive `is_dataclass(f.type)` check silently leaves every sub-config as a raw dict
and the failure only surfaces deep in training as an AttributeError.
"""

from __future__ import annotations

import pytest

from morphos.config import (
    Config,
    NCACfg,
    TargetCfg,
    config_hash,
    load_config,
    run_dir_name,
)

CONFIGS = "configs"


def test_base_config_loads_with_nested_dataclasses():
    cfg = load_config(f"{CONFIGS}/base.yaml")
    assert isinstance(cfg, Config)
    assert isinstance(cfg.target, TargetCfg), "nested section did not become a dataclass"
    assert isinstance(cfg.nca, NCACfg)
    assert cfg.nca.grid == 32
    assert cfg.target.radius == 10.0
    assert cfg.train.batch == 32
    assert cfg.device == "mps"


def test_tuple_fields_are_tuples_not_lists():
    cfg = load_config(f"{CONFIGS}/base.yaml")
    assert isinstance(cfg.train.lr_milestones, tuple)
    assert cfg.train.lr_milestones == (2000,)
    assert isinstance(cfg.target.color, tuple)
    assert len(cfg.target.color) == 3


def test_base_inheritance_deep_merges():
    growing = load_config(f"{CONFIGS}/morph_growing.yaml")
    assert growing.name == "morph_growing"
    assert growing.train.pool_from_step is None
    assert growing.train.steps == 2000
    # Values not mentioned in the child come from base, and sibling sections survive.
    assert growing.nca.grid == 32
    assert growing.train.batch == 32
    assert growing.train.lr == 2.0e-3


def test_override_types_are_parsed_as_yaml():
    cfg = load_config(
        f"{CONFIGS}/base.yaml",
        overrides=[
            "train.lr=3e-4",
            "train.pool_from_step=null",
            "train.damage_from_step=null",
            "train.early_stop_on_gates=false",
            "nca.grid=24",
            "target.radius=7.0",
        ],
    )
    assert cfg.train.lr == 3e-4 and isinstance(cfg.train.lr, float)
    assert cfg.train.pool_from_step is None
    assert cfg.train.early_stop_on_gates is False
    assert isinstance(cfg.train.early_stop_on_gates, bool)
    assert cfg.nca.grid == 24 and isinstance(cfg.nca.grid, int)


def test_unknown_key_raises():
    with pytest.raises(KeyError, match="unknown config key"):
        load_config(f"{CONFIGS}/base.yaml", overrides=["train.learning_rate=1e-3"])
    with pytest.raises(KeyError, match="unknown config key"):
        load_config(f"{CONFIGS}/base.yaml", overrides=["nonsense.foo=1"])


def test_malformed_override_raises():
    with pytest.raises(ValueError, match="malformed"):
        load_config(f"{CONFIGS}/base.yaml", overrides=["train.lr"])


def test_validation_catches_impossible_combinations():
    # Target too large for the grid.
    with pytest.raises(ValueError, match="does not fit"):
        load_config(f"{CONFIGS}/base.yaml", overrides=["target.radius=20.0"])

    # Reseed + damage slices would overlap.
    with pytest.raises(ValueError, match="exceeds batch"):
        load_config(f"{CONFIGS}/base.yaml", overrides=["train.batch=8"])

    # Damage without a pool is meaningless.
    with pytest.raises(ValueError, match="damage requires the pool"):
        load_config(f"{CONFIGS}/base.yaml", overrides=["train.pool_from_step=null"])

    with pytest.raises(ValueError, match="unknown device"):
        load_config(f"{CONFIGS}/base.yaml", overrides=["device=tpu"])


def test_hash_and_run_dir_are_stable_and_sensitive():
    a = load_config(f"{CONFIGS}/base.yaml")
    b = load_config(f"{CONFIGS}/base.yaml")
    c = load_config(f"{CONFIGS}/base.yaml", overrides=["train.lr=1e-4"])

    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)

    name = run_dir_name(a, date="20260804")
    assert name.startswith("20260804-base-")
    assert name.endswith("-s0")


def test_morph_config_is_the_full_regime():
    cfg = load_config(f"{CONFIGS}/morph.yaml")
    assert cfg.train.pool_from_step == 1000
    assert cfg.train.damage_from_step == 1000
    assert cfg.train.steps == 6000
    # Upstream fractions preserved at batch 32.
    assert cfg.train.reseed_worst == round(cfg.train.batch / 8)
    assert cfg.train.damage_best == round(3 * cfg.train.batch / 8)
