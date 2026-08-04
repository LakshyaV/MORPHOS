"""Held-out evaluation of the broadcast task.

Training-batch numbers are not evidence. This re-grows fresh organisms with a
salted RNG and measures on episodes the model never optimised against.

The headline measurement is per-cell vote accuracy AS A FUNCTION OF GEODESIC
DISTANCE from the sensor patch. That is the direct answer to what the Milestone 2
probe left open: a morphology-only organism relayed an injected signal ~5 cells
against a body radius of 9, so the question is whether per-cell supervision
carries it all the way to the periphery.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from morphos.eval.metrics import body_mask, geodesic_distance, iou
from morphos.eval.probe import sensor_patch_mask
from morphos.seeding import make_rng
from morphos.task.broadcast import make_inject_fn, referent_codes, sample_referents
from morphos.task.readout import (
    VotePool,
    agreement,
    cell_argmax,
    consensus,
    quorum_fraction,
)
from morphos.task.targets import build_target
from morphos.train.checkpoint import load_organism

GATE_SALT = 0xB0CA


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dev = torch.device(args.device)
    model, cfg, step = load_organism(Path(args.ckpt), dev)
    grid = cfg["nca"]["grid"]
    n_ref = cfg["task"]["n_referents"]
    t_inject = cfg["task"]["t_inject"]
    t_grow = cfg["eval"]["t_grow"]

    target = build_target(
        cfg["target"]["shape"], grid, radius=cfg["target"]["radius"],
        edge=cfg["target"]["edge"], color=tuple(cfg["target"]["color"]), device=dev,
    )
    pool = VotePool(channels=model.layout.vote, n_out=n_ref)
    rng = make_rng(args.seed ^ GATE_SALT, dev)
    codes_all = referent_codes(n_ref, model.layout.n_sensor, device=dev)

    accs, agrees, cons, quorums, ious = [], [], [], [], []
    ring_hit: dict[int, list[float]] = {}

    for _ in range(max(1, args.episodes // args.batch)):
        x = model.rollout(
            model.seed_state(args.batch, device=dev), t_grow, generator=rng.update
        )
        ref = sample_referents(args.batch, n_ref, generator=rng.task, device=dev)
        inj = make_inject_fn(codes_all[ref], model.layout, grid, t_inject,
                             patch=cfg["task"]["patch"])
        out = model.rollout(x, t_grow, generator=rng.update, inject_fn=inj)

        alive = model.alive_mask(out)
        dec = pool.decision(out, alive)
        accs.append((dec == ref).float().mean().item())
        agrees.append(agreement(pool, out, alive).mean().item())
        cons.append(consensus(pool, out, alive).mean().item())
        quorums.append(quorum_fraction(pool, out, alive).mean().item())
        ious.append(iou(body_mask(out), body_mask(target)).mean().item())

        # per-ring accuracy: does the PERIPHERY know the referent?
        opinions = cell_argmax(pool, out)
        correct = (opinions == ref.view(-1, 1, 1))
        bm = body_mask(out)
        src = sensor_patch_mask(grid, cfg["task"]["patch"], device=dev)
        for i in range(out.shape[0]):
            d = geodesic_distance(bm[i : i + 1], (src & bm[i : i + 1]))[0, 0]
            for ring in range(int(d[torch.isfinite(d)].max().item()) + 1):
                m = d == ring
                if m.any():
                    ring_hit.setdefault(ring, []).append(correct[i][m].float().mean().item())

    mean = lambda v: sum(v) / len(v)
    print(f"checkpoint : {args.ckpt} (step {step})")
    print(f"held-out   : {args.episodes} episodes, {n_ref} referents, chance {1/n_ref:.3f}\n")
    print(f"  pooled vote accuracy   {mean(accs):.4f}   (need >= 0.90)")
    print(f"  cell agreement         {mean(agrees):.4f}   (need >= 0.80)")
    print(f"  consensus (plurality)  {mean(cons):.4f}")
    print(f"  quorum fraction        {mean(quorums):.4f}   (~0.5 distributed, ~0.02 megaphone)")
    print(f"  morphology IoU         {mean(ious):.4f}   (need >= 0.85)")

    print(f"\n  per-cell accuracy by geodesic distance from the sensor:")
    chance = 1.0 / n_ref
    for ring in sorted(ring_hit):
        v = mean(ring_hit[ring])
        bar = "#" * int(round(v * 40))
        flag = "" if v > chance * 2 else "   <- at chance"
        print(f"    d={ring:>2}  {v:.3f}  {bar}{flag}")

    far = [mean(ring_hit[r]) for r in sorted(ring_hit) if r >= max(ring_hit) - 1]
    passed = (
        mean(accs) >= 0.90 and mean(agrees) >= 0.80
        and mean(ious) >= 0.85 and mean(far) > 2 * chance
    )
    print(f"\n  periphery (outermost 2 rings) {mean(far):.3f} vs chance {chance:.3f}")
    print("\nBROADCAST GATE: " + ("PASS" if passed else "FAIL"))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
