# The paper

Draft of the MORPHOS workshop paper. `main.tex` is the entry point; each section
is a separate file under `sections/` so they can be written and reviewed
independently.

## This draft has never been compiled

There is no LaTeX toolchain on the development machine — no `pdflatex`,
`xelatex`, `tectonic`, `latexmk`, or `pandoc`, and no MacTeX. **Every `.tex`
file here is unproofed.** Expect the first real compile to surface undefined
references, overfull boxes, and float placement problems that nobody has been
able to see yet.

To make it verifiable, either:

```bash
brew install --cask mactex-no-gui   # ~4 GB, then: make paper
```

or upload the directory to Overleaf, which needs no local install.

```bash
make paper    # latex -> bibtex -> latex -> latex, then greps the log
make check    # lists every claim still waiting on an experiment
make clean
```

## Draft conventions

`\gated{...}` marks a claim whose supporting experiment **has not run yet**. It
renders in red so a placeholder can never be mistaken for a finding. `make check`
lists them; the list must be empty before submission.

Numbers in the text come from committed artifacts only — `docs/gates_m2.json`,
`runs/*/metrics.jsonl`, `runs/milestone2/probe.npz`. Nothing is typed from
memory. If a number has no artifact behind it, it does not go in the paper.

## Blocking work, in priority order

1. **The propagation decoder has no code.** `docs/RESULTS_M2.md` reports a
   per-ring ridge-decoder table (information reaching $d \approx 5$) that exists
   only as prose — no implementation anywhere in the repo, and not recoverable
   from `runs/milestone2/probe.npz`, which stores variance ratios rather than
   per-cell states. The claim is load-bearing: it is the stated basis for the
   whole M3 pivot. Write the decoder into `morphos/eval/probe.py`, re-run against
   `runs/milestone2/ckpt/last.pt`, and confirm the η² values reproduce before
   trusting the new decoding numbers. Until then §5.2 carries η² only.

2. **`readout_dropout` is not wired in.** Configured, implemented in
   `VotePool.__call__`, unit-tested — and never passed by `morphos/train/loop.py`.
   The paper describes it as one of three anti-centralisation mechanisms, so
   either wire it up and retrain, or the description must change.

3. **`grad_ratio` logs NaN.** `morphos/train/loop.py:232` computes it when
   `st.step % 50 == 0`, but `st.step` increments after `train_step` returns, so
   the logged steps never coincide. Every row after step 1 is NaN. This is the
   diagnostic for the morphology-versus-message weighting, so the "~20×" figure
   in the config comment currently has no artifact behind it.

4. **No figures, no plotting code.** `make figures` targets
   `morphos/viz/figures.py`, which does not exist. Four figures are buildable
   from data already in hand: setup schematic, G1/G2 curves, the probe rings, and
   the broadcast learning curve.

5. **Length.** The draft runs ~8k words before figures. Workshop limits are
   typically 4–9 pages, so expect to cut once figures take space — §5 and the
   appendix are the obvious donors.

## The style file

`neurips_2026.sty` was fetched from the community-maintained
[NeurIPS-2026-Latex-Unified](https://github.com/lizhemin15/NeurIPS-2026-Latex-Unified)
mirror on 2026-08-04; the official `media.neurips.cc` paths returned 403/404.
It self-identifies as `2026-01-29 NeurIPS 2026 submission/camera-ready style
file` and carries the standard options plus workshop variants. **Replace it with
the official download before submitting** — a mirror is fine for drafting and
not fine for a camera-ready.

The draft builds with `[preprint]`. At submission:

```latex
\usepackage[dblblindworkshop]{neurips_2026}
\workshoptitle{<exact workshop name>}
```

The style file warns if a workshop option is set without `\workshoptitle`.
