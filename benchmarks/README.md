# Benchmarks

Performance benchmarks for ml4gw, using [pytest-benchmark](https://pytest-benchmark.readthedocs.io/).

## Running benchmarks

```bash
# Full suite
uv run pytest benchmarks/

# Single subdirectory
uv run pytest benchmarks/waveforms/

# Single file
uv run pytest benchmarks/transforms/test_qtransform.py
```

The suite auto-detects CUDA and falls back to CPU if unavailable.

## Measuring the impact of a change

Use `--benchmark-save` / `--benchmark-compare` to produce a before/after table.
**Run these benchmarks on GPU hardware.**

```bash
# 1. Save a baseline on the base branch
git checkout main
uv run pytest benchmarks/ --benchmark-save=baseline

# 2. Switch to your feature branch
git checkout my-feature-branch

# 3. Run and compare
uv run pytest benchmarks/ --benchmark-compare=0001_baseline
```

Saved files are written to `.benchmarks/<platform>/NNNN_<name>.json`, where `NNNN` is a zero-padded counter that increments with each save.

Paste the comparison table into your PR description so reviewers can see the delta. 
Target the file(s) that your change affects rather than running the full suite.
