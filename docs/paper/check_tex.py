#!/usr/bin/env python3
"""Static checks for the draft, standing in for a LaTeX compiler.

There is no toolchain on the dev machine (see README.md), so this catches the
mistakes a first compile would otherwise surface: missing \\input targets,
citations with no bib entry, references with no label, and unbalanced braces or
environments. It is not a substitute for compiling -- it cannot see spacing,
float placement, or overfull boxes.

Usage:  python3 check_tex.py
Exit code is nonzero if anything failed.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
FAILURES: list[str] = []
NOTES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)


def strip_comments(text: str) -> str:
    # Remove % comments but keep \% escapes.
    return re.sub(r"(?<!\\)%.*", "", text)


def collect() -> dict[Path, str]:
    files = {}
    main = ROOT / "main.tex"
    if not main.exists():
        fail("main.tex is missing")
        return files
    files[main] = strip_comments(main.read_text())
    for m in re.finditer(r"\\input\{([^}]+)\}", files[main]):
        p = ROOT / (m.group(1) + ".tex")
        if not p.exists():
            fail(f"\\input{{{m.group(1)}}} -> missing file {p.relative_to(ROOT)}")
        else:
            files[p] = strip_comments(p.read_text())
    return files


def check_braces(files: dict[Path, str]) -> None:
    for path, text in files.items():
        depth = 0
        for i, ch in enumerate(text):
            if ch == "{" and (i == 0 or text[i - 1] != "\\"):
                depth += 1
            elif ch == "}" and (i == 0 or text[i - 1] != "\\"):
                depth -= 1
                if depth < 0:
                    fail(f"{path.name}: unbalanced closing brace")
                    break
        if depth > 0:
            fail(f"{path.name}: {depth} unclosed brace(s)")


def check_environments(files: dict[Path, str]) -> None:
    for path, text in files.items():
        stack = []
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
            kind, name = m.group(1), m.group(2)
            if kind == "begin":
                stack.append(name)
            else:
                if not stack:
                    fail(f"{path.name}: \\end{{{name}}} with no matching begin")
                elif stack[-1] != name:
                    fail(f"{path.name}: \\end{{{name}}} closes \\begin{{{stack[-1]}}}")
                    stack.pop()
                else:
                    stack.pop()
        for leftover in stack:
            fail(f"{path.name}: \\begin{{{leftover}}} never closed")


def check_citations(files: dict[Path, str]) -> None:
    bib = ROOT / "refs.bib"
    if not bib.exists():
        fail("refs.bib is missing")
        return
    keys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text()))
    used = set()
    for path, text in files.items():
        for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", text):
            for key in m.group(1).split(","):
                key = key.strip()
                used.add(key)
                if key not in keys:
                    fail(f"{path.name}: \\cite{{{key}}} has no entry in refs.bib")
    unused = keys - used
    if unused:
        NOTES.append(f"bib entries never cited: {', '.join(sorted(unused))}")


def check_refs(files: dict[Path, str]) -> None:
    labels, refs = set(), []
    for path, text in files.items():
        labels |= set(re.findall(r"\\label\{([^}]+)\}", text))
        for m in re.finditer(r"\\(?:eq)?ref\{([^}]+)\}", text):
            refs.append((path.name, m.group(1)))
    for fname, r in refs:
        if r not in labels:
            fail(f"{fname}: \\ref{{{r}}} has no matching \\label")


_KNOWN = set("""
begin end documentclass usepackage newcommand renewcommand newif newtheorem
theoremstyle title author maketitle bibliographystyle bibliography appendix
input label ref eqref cite citet citep section subsection paragraph item
itemsep textbf textit emph texttt textcolor url href hypersetup
frac dfrac tfrac sum prod int max min log exp sqrt leq geq neq approx
times cdot odot wedge vee neg forall exists in subseteq setminus
alpha beta gamma delta epsilon varepsilon theta lambda mu sigma tau phi psi
omega Delta Sigma Omega Gamma Lambda Phi Psi
mathrm mathbb mathcal mathbf mathit text quad qquad hspace vspace
left right big Big bigg Bigg centering small footnotesize scriptsize
toprule midrule bottomrule multirow multicolumn cmidrule
includegraphics caption centering nicefrac
gtrsim lesssim ldots dots cdots to mapsto rightarrow Rightarrow
proof relax par noindent bf it rm sl sc tt
lim sup inf arg det dim ker deg gcd hom
""".split())


def check_macros(files: dict[Path, str]) -> None:
    """Flag control sequences that are neither standard nor defined in main.tex.

    Without a compiler an undefined macro is invisible until the first build.
    False positives are expected -- this is a review list, not a gate.
    """
    main = files.get(ROOT / "main.tex", "")
    defined = set(re.findall(r"\\(?:new|renew)command\{?\\(\w+)", main))
    defined |= set(re.findall(r"\\newif\\if(\w+)", main))
    defined |= set(re.findall(r"\\newtheorem\{(\w+)\}", main))
    defined |= {"showgatedtrue", "showgatedfalse", "ifshowgated", "fi", "else"}

    unknown: dict[str, set[str]] = {}
    for path, text in files.items():
        for m in re.finditer(r"\\([a-zA-Z]+)", text):
            name = m.group(1)
            if name in _KNOWN or name in defined:
                continue
            unknown.setdefault(name, set()).add(path.name)
    if unknown:
        NOTES.append(
            "control sequences to eyeball (may be fine -- standard-package "
            "commands are not all listed): "
            + ", ".join(f"\\{k}" for k in sorted(unknown))
        )


def check_gated(files: dict[Path, str]) -> None:
    n = sum(len(re.findall(r"\\gated\{", t)) for t in files.values())
    NOTES.append(f"{n} \\gated{{}} marker(s) -- unsupported claims still in the draft")


def check_packages(files: dict[Path, str]) -> None:
    """Commands used but whose package may not be loaded."""
    main = files.get(ROOT / "main.tex", "")
    loaded = set(re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", main))
    loaded = {p.strip() for group in loaded for p in group.split(",")}
    needs = {
        "multirow": "multirow",
        "toprule": "booktabs",
        "midrule": "booktabs",
        "bottomrule": "booktabs",
        "includegraphics": "graphicx",
        "textcolor": "xcolor",
        "nicefrac": "nicefrac",
        "url": "url",
    }
    body = "\n".join(files.values())
    for cmd, pkg in needs.items():
        if re.search(r"\\" + cmd + r"\b", body) and pkg not in loaded:
            fail(f"\\{cmd} used but package '{pkg}' is not loaded in main.tex")


def main() -> int:
    files = collect()
    if not files:
        print("FAIL: nothing to check")
        return 1
    check_braces(files)
    check_environments(files)
    check_citations(files)
    check_refs(files)
    check_packages(files)
    check_macros(files)
    check_gated(files)

    print(f"checked {len(files)} file(s): {', '.join(sorted(p.name for p in files))}\n")
    for n in NOTES:
        print(f"  note: {n}")
    if FAILURES:
        print(f"\n{len(FAILURES)} problem(s):")
        for f in FAILURES:
            print(f"  FAIL {f}")
        return 1
    print("\nno structural problems found")
    print("NOTE: this is not a compile. Spacing, floats and overfull boxes are unchecked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
