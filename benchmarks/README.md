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

**Run these benchmarks on GPU hardware.**

```bash
# 1. Save a baseline on the base branch
git checkout main
uv run pytest benchmarks/ --benchmark-save=baseline

# 2. Switch to your feature branch
git checkout my-feature-branch

# 3. Save a result for the feature branch
uv run pytest benchmarks/ --benchmark-save=feature

# 4. Compare
uv run python benchmarks/compare_benchmarks.py baseline feature
```

Saved files are written to `.benchmarks/<platform>/NNNN_<name>.json`, where
`NNNN` is a zero-padded counter that increments with each save.
Benchmark files can be specified by substrings rather than full paths;
the script will `rglob` in `.benchmarks/` for any matching `.json` files.

Options for `compare_benchmarks.py`:

```bash
❯ uv run python benchmarks/compare_benchmarks.py --help                          
usage: compare_benchmarks.py [--config CONFIG] [--metric {min,max,mean,median}] [--threshold THRESHOLD] [--sort {true,false}] baseline current

Compare benchmarks between two JSON files from pytest-benchmark.

positional arguments:
  baseline              Path to the baseline JSON file. (required, type: <class 'Path'>)
  current               Path to the current JSON file. (required, type: <class 'Path'>)

options:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one or
                        more keywords separated by comma. The supported flags are: skip_default, skip_null.
  --metric {min,max,mean,median}
                        The metric to compare (min, max, mean, median). (type: Literal['min', 'max', 'mean', 'median'], default: median)
  --threshold THRESHOLD
                        Percentage threshold for highlighting differences. (type: float, default: 5.0)
  --sort {true,false}   Whether to sort the output by percent change in the metric. (type: bool, default: False)
```

Paste the comparison table into your PR description so reviewers can see the
delta. Target the file(s) that your change affects rather than running the full
suite.
