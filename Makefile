PY := .venv/bin/python

.PHONY: fast test bench figures clean

# Unit suite: must stay under 30s on CPU. No MPS, no training.
fast:
	$(PY) -m pytest tests -m "not slow and not mps" -q

# Full suite including the tiny end-to-end training run.
test:
	$(PY) -m pytest tests -q

# CPU vs MPS decision table. Rule: whichever wins the 2-organism comm step;
# if within 20%, prefer CPU (determinism is easier there).
bench:
	$(PY) scripts/bench_device.py

figures:
	$(PY) -m morphos.viz.figures --all

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
