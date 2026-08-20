"""Check gates G3 and G4 against a trained comm checkpoint.

    python scripts/eval_comm.py --ckpt runs/comm/ckpt/last.pt

Held-out episodes: fresh organisms, salted RNG, deterministic argmax channel.
Exits nonzero unless G3 (strong tier) and G4 both pass, so it can gate a script.
Cross-play (the last G4 item) needs a second seed and is reported as pending.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from morphos.eval.comm_gates import collect_episodes, gate_g3, gate_g4
from morphos.train.checkpoint import load_comm_pair


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--permutations", type=int, default=1000)
    p.add_argument("--json-out", default=None)
    p.add_argument("--cross-receiver", default=None, metavar="CKPT",
                   help="take the receiver from another seed's checkpoint: the "
                        "cross-play arm of G4 (a convention, not a universal, "
                        "so accuracy must fall to <= 1/N + 0.10)")
    args = p.parse_args()

    sender, receiver, cfg, step = load_comm_pair(Path(args.ckpt), args.device)
    N, V = cfg["task"]["n_referents"], cfg["task"]["vocab"]

    if args.cross_receiver:
        _, receiver, rcfg, rstep = load_comm_pair(Path(args.cross_receiver), args.device)
        assert (rcfg["task"]["n_referents"], rcfg["task"]["vocab"]) == (N, V)
        bank = collect_episodes(
            sender, receiver, cfg,
            episodes=args.episodes, batch=args.batch, seed=args.seed, device=args.device,
        )
        acc = float((bank.answer_idx == bank.referents).double().mean())
        ok = acc <= 1 / N + 0.10
        print(f"cross-play : sender {args.ckpt} (step {step}) x "
              f"receiver {args.cross_receiver} (step {rstep})")
        print(f"  accuracy {acc:.4f}   (need <= {1 / N + 0.10:.3f})   "
              f"{'PASS' if ok else 'FAIL -- protocol may be a universal'}")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(
                {"mode": "crossplay", "sender_ckpt": str(args.ckpt),
                 "receiver_ckpt": str(args.cross_receiver), "acc": acc,
                 "passed": ok}, indent=2))
        raise SystemExit(0 if ok else 1)

    bank = collect_episodes(
        sender, receiver, cfg,
        episodes=args.episodes, batch=args.batch, seed=args.seed, device=args.device,
    )
    g3 = gate_g3(bank, permutations=args.permutations, seed=args.seed)
    g4 = gate_g4(bank, mi_mr_debiased=g3["mi_debiased"],
                 permutations=args.permutations, seed=args.seed)

    print(f"checkpoint : {args.ckpt} (step {step})")
    print(f"held-out   : {g3['episodes']} episodes, {N} referents, "
          f"{V} symbols, chance {1 / N:.3f}\n")

    print("G3 -- a protocol emerged")
    print(f"  accuracy (argmax channel)  {g3['acc']:.4f}   (need >= 0.90)")
    print(f"  Wilson lower bound         {g3['acc_wilson_lb']:.4f}   (need >= 0.85)")
    print(f"  accuracy (noisy channel)   {g3['acc_noisy_channel']:.4f}   (reported, not gated)")
    print(f"  I(M;R) debiased            {g3['mi_debiased']:.3f} bits (need >= 2.4; "
          f"raw {g3['mi_raw']:.3f}, p={g3['p_value']:.4f})")
    print(f"  NMI                        {g3['nmi']:.3f}")
    print(f"  attribute-wise MI          "
          + "  ".join(f"{v:.3f}" for v in g3["attr_mi"]) + " bits")
    print(f"  V_eff                      {g3['v_eff']:.2f}   (need >= 6; V_used {g3['v_used']})")
    print(f"  H(M)                       {g3['entropy']:.3f} bits (MM {g3['entropy_mm']:.3f})")
    print(f"  strong tier {'PASS' if g3['strong'] else 'FAIL'}   weak tier "
          f"{'PASS' if g3['weak'] else 'FAIL'}\n")

    print("G4 -- the protocol is causally real")
    c = g4["checks"]
    flag = lambda ok: "PASS" if ok else "FAIL"
    print(f"  message removed            {g4['acc_removed']:.4f}   (need <= 0.175)  {flag(c['removed'])}")
    print(f"  marginal-resampled         {g4['acc_resampled']:.4f}   (need <= 0.175)  {flag(c['resampled'])}")
    print(f"  vocabulary permuted        {g4['acc_permuted']:.4f}   (need <= 0.225)  {flag(c['permuted'])}")
    print(f"  CIC                        {g4['cic_bits']:.3f} bits (need >= 1.0)   {flag(c['cic'])}"
          f"   CSE {g4['cse']:.3f}")
    print(f"  DPI: I(A;R) <= I(M;R)+0.1  {g4['mi_answer_referent']:.3f} <= "
          f"{g3['mi_debiased'] + 0.1:.3f}   {flag(c['dpi'])}")
    print(f"  channel purity             {flag(c['purity'])}")
    print(f"  cross-play                 {g4['crossplay']}")

    passed = g3["passed"] and g4["passed"]
    print(f"\nCOMM GATES: {'PASS' if passed else 'FAIL'}"
          + ("" if passed or not g3["weak"] else "   (weak tier holds)"))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"ckpt": str(args.ckpt), "step": step, "seed": args.seed,
             "g3": g3, "g4": g4}, indent=2))
        print(f"json -> {args.json_out}")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
