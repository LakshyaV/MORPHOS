"""Measure how fast information crosses the organism.

    python scripts/probe_propagation.py --ckpt runs/<id>/ckpt/last.pt

This is the decision that gates Week 2: it says whether T_comm is long enough,
and -- crucially -- distinguishes a signal that is merely slow (fix: raise T_comm)
from one that attenuates (fix: more injection, not more time).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from morphos.eval.probe import ETA2_THRESHOLD, run_probe
from morphos.seeding import make_rng
from morphos.substrate.nca import SENDER_LAYOUT
from morphos.train.checkpoint import load_organism


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--t-comm", type=int, default=64)
    p.add_argument("--t-probe", type=int, default=128)
    p.add_argument("--t-inject", type=int, default=16)
    p.add_argument("--n-noise", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="npz path for the raw eta2 field")
    args = p.parse_args()

    device = torch.device(args.device)
    model, cfg, step = load_organism(Path(args.ckpt), device)

    res = run_probe(
        model, SENDER_LAYOUT,
        rng=make_rng(args.seed, device), device=device,
        t_grow=cfg["eval"]["t_grow"], t_probe=args.t_probe,
        t_inject=args.t_inject, n_noise=args.n_noise,
    )

    print(f"checkpoint : {args.ckpt} (step {step})")
    print(f"body       : max geodesic {res.max_dist} cells from the sensor patch\n")
    print(res.summary(args.t_comm))

    print(f"\nattenuation (max over time of eta2, threshold {ETA2_THRESHOLD}):")
    for d in range(res.max_dist + 1):
        v = res.eta2_max_by_dist[d].item()
        bar = "#" * int(round(v * 40))
        flag = "" if v >= ETA2_THRESHOLD else "   <- below threshold"
        print(f"  d={d:>3}  {v:.3f}  {bar}{flag}")

    print("\nfront position over time:")
    for t in range(0, res.eta2.shape[0], max(1, res.eta2.shape[0] // 12)):
        print(f"  t={t:>4}  front d={res.front[t].item():.0f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            eta2=res.eta2.numpy().astype(np.float32),
            dist=res.dist.numpy().astype(np.float32),
            eta2_by_dist=res.eta2_by_dist.numpy().astype(np.float32),
            front=res.front.numpy().astype(np.float32),
            t_cover=res.t_cover,
            speed=res.speed,
        )
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
