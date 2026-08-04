"""Render an organism's life from a checkpoint.

    python scripts/make_video.py --ckpt runs/<id>/ckpt/last.pt --out media/growth.mp4
    python scripts/make_video.py --ckpt ... --damage-at 64 --severity 0.3   # grow/cut/regrow
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from morphos.config import Config, load_config
from morphos.eval.metrics import alive_count, body_mask, iou
from morphos.seeding import make_rng
from morphos.substrate.nca import SENDER_LAYOUT, NCAOrganism
from morphos.task.targets import build_target
from morphos.train.checkpoint import config_from, load
from morphos.viz.frames import ascii_preview, side_by_side, to_frames
from morphos.viz.video import write_mp4


def build_from_ckpt(ckpt_path: Path, device: torch.device) -> tuple[NCAOrganism, Config]:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = config_from(raw)
    if cfg_dict is None:
        raise SystemExit(f"{ckpt_path} has no embedded config")
    cfg = Config(**{})  # defaults, then patch the fields we need
    grid = cfg_dict["nca"]["grid"]
    hidden = cfg_dict["nca"]["hidden"]
    model = NCAOrganism(
        layout=SENDER_LAYOUT,
        hidden=hidden,
        grid=grid,
        fire_rate=cfg_dict["nca"]["fire_rate"],
        alive_threshold=cfg_dict["nca"]["alive_threshold"],
    ).to(device)
    model.load_state_dict(raw["model"])
    model.eval()
    return model, cfg_dict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--steps", type=int, default=192)
    p.add_argument("--damage-at", type=int, default=None,
                   help="step at which to cut a disk out of the organism")
    p.add_argument("--severity", type=float, default=0.30)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=12)
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    model, cfg_dict = build_from_ckpt(ckpt_path, device)
    grid = cfg_dict["nca"]["grid"]

    target = build_target(
        cfg_dict["target"]["shape"], grid,
        radius=cfg_dict["target"]["radius"],
        edge=cfg_dict["target"]["edge"],
        color=tuple(cfg_dict["target"]["color"]),
        device=device,
    )

    rng = make_rng(args.seed, device)
    x = model.seed_state(1, device=device)

    frames, ious, alives = [], [], []
    with torch.no_grad():
        for t in range(args.steps):
            if args.damage_at is not None and t == args.damage_at:
                from morphos.damage.ops import kill_fraction

                x, info = kill_fraction(
                    x, args.severity, generator=rng.damage
                )
                print(f"damage at t={t}: killed {info['achieved_f'].item():.1%} of cells")

            frames.append(to_frames(x)[0])
            ious.append(iou(body_mask(x), body_mask(target)).item())
            alives.append(int(alive_count(x).item()))
            x = model.rollout(x, 1, generator=rng.update)

    out = Path(args.out or ckpt_path.parent.parent / "media" / "life.mp4")
    write_mp4(np.stack(frames), out, fps=args.fps)

    print(f"wrote {out}  ({len(frames)} frames)")
    print(f"final IoU {ious[-1]:.3f}   alive {alives[-1]} (target {int(body_mask(target).sum())})")
    marks = [0, min(31, len(ious) - 1), min(63, len(ious) - 1), len(ious) - 1]
    for t in marks:
        print(f"  t={t:>4}  IoU {ious[t]:.3f}  alive {alives[t]}")
    print("\nfinal body:")
    print(ascii_preview(x))


if __name__ == "__main__":
    main()
