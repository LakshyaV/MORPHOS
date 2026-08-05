"""Train a MORPHOS organism.

    python scripts/train.py --config configs/morph_growing.yaml
    python scripts/train.py --config configs/morph.yaml --set train.steps=500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from morphos.config import config_dict, load_config, run_dir_name
from morphos.io.runlog import RunLog
from morphos.seeding import seed_everything
from morphos.task.targets import build_target, target_fingerprint
from morphos.train.loop import train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="dotted override, e.g. train.lr=3e-4")
    p.add_argument("--run-dir", default=None, help="defaults to runs/<auto-name>")
    p.add_argument("--gates", action="store_true", help="run G1/G2 during training")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = p.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    seed_everything(cfg.seed)

    device = torch.device(cfg.device)
    target = build_target(
        cfg.target.shape,
        cfg.nca.grid,
        radius=cfg.target.radius,
        edge=cfg.target.edge,
        color=cfg.target.color,
        device=device,
    )

    run_dir = Path(args.run_dir or Path("runs") / run_dir_name(cfg, date=time.strftime("%Y%m%d")))

    gate_fn = None
    if args.gates:
        from morphos.eval.gates import gate_report

        gate_fn = gate_report

    print(f"run dir     : {run_dir}")
    print(f"device      : {device}   target fp: {target_fingerprint(target)}")
    print(f"regime      : pool_from={cfg.train.pool_from_step} damage_from={cfg.train.damage_from_step}")
    print(f"steps       : {cfg.train.steps}  batch {cfg.train.batch}  "
          f"rollout U[{cfg.train.rollout_min},{cfg.train.rollout_max}]")

    with RunLog(run_dir, config_dict(cfg), resume=args.resume is not None) as log:
        st = train(cfg, target, run_dir=run_dir, log=log, gate_fn=gate_fn,
                   resume=args.resume)

    print(f"done at step {st.step} after {st.n_resets} death reset(s)")
    print(f"metrics -> {run_dir}/metrics.jsonl")


if __name__ == "__main__":
    main()
