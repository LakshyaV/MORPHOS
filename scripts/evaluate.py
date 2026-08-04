"""Check the pre-registered gates against a trained checkpoint.

    python scripts/evaluate.py --ckpt runs/<id>/ckpt/last.pt --gates

Exits nonzero if any gate fails, so it can be used in a script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from morphos.eval.gates import check_g1, check_g2
from morphos.substrate.nca import SENDER_LAYOUT, NCAOrganism
from morphos.task.targets import build_target
from morphos.train.checkpoint import config_from


def load_model(ckpt_path: Path, device: torch.device):
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = config_from(raw)
    if cfg is None:
        raise SystemExit(f"{ckpt_path} carries no config")
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--g1-rollouts", type=int, default=128)
    p.add_argument("--g2-trials", type=int, default=256)
    p.add_argument("--skip-g2", action="store_true")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    model, cfg, step = load_model(Path(args.ckpt), device)
    target = build_target(
        cfg["target"]["shape"], cfg["nca"]["grid"],
        radius=cfg["target"]["radius"], edge=cfg["target"]["edge"],
        color=tuple(cfg["target"]["color"]), device=device,
    )

    print(f"checkpoint : {args.ckpt}  (step {step})")
    print(f"target     : {cfg['target']['shape']} r={cfg['target']['radius']} "
          f"on {cfg['nca']['grid']}x{cfg['nca']['grid']}\n")

    results = [
        check_g1(
            model, target, device=device, seed=cfg.get("seed", 0),
            n_rollouts=args.g1_rollouts, batch=32,
            t_grow=cfg["eval"]["t_grow"], t_persist=cfg["eval"]["t_persist"],
        )
    ]
    if not args.skip_g2:
        results.append(
            check_g2(
                model, target, device=device, seed=cfg.get("seed", 0),
                n_trials=args.g2_trials, batch=32,
                severity=cfg["eval"]["g2_severity"],
                tolerance=cfg["eval"]["g2_tolerance"],
                t_mature=cfg["eval"]["t_grow"],
                t_recover=cfg["eval"]["g2_recover_steps"],
            )
        )

    for r in results:
        print(r.report())
        print()

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([r.as_dict() for r in results], indent=2)
        )

    ok = all(r.passed for r in results)
    print("ALL GATES PASS" if ok else "GATES FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
