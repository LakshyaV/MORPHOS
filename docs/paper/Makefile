TEX := pdflatex
BIB := bibtex

.PHONY: paper clean check

# Standard latex-bibtex-latex-latex cycle.
paper:
	@command -v $(TEX) >/dev/null 2>&1 || { \
	  echo "No LaTeX toolchain found. See README.md for install options."; \
	  exit 1; }
	$(TEX) -interaction=nonstopmode -halt-on-error main.tex
	$(BIB) main || true
	$(TEX) -interaction=nonstopmode -halt-on-error main.tex
	$(TEX) -interaction=nonstopmode -halt-on-error main.tex
	@echo "--- undefined references / citations ---"
	@grep -i "undefined" main.log || echo "none"
	@echo "--- overfull boxes ---"
	@grep -c "Overfull" main.log || echo "0"

# Every claim still waiting on an experiment. Must be empty before submission.
check:
	@echo "=== unsupported claims still in the draft ==="
	@grep -rn "gated{\|todocite{" sections/ || echo "none -- draft is fully supported"

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
