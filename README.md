# slicehound

[![CI](https://img.shields.io/github/actions/workflow/status/sushxnthd/slicehound/ci.yml?branch=main&label=CI)](.github/workflows/ci.yml) [![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](pyproject.toml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[**Live demo →**](https://sushxnthd.github.io/slicehound/) · [Architecture](docs/architecture.md) · [Benchmark](benchmarks/results.json)

`slicehound` searches numerical feature space for compact rules that isolate disproportionately high model error.

```bash
pip install -e .
python examples/make_example.py
slicehound examples/failures.npz --depth 2
```

Example output:

```text
x0 > 0.59 AND x1 <= 0.40
support=173  error_rate=0.76  lift=3.8x
```

## Search

1. Generate quantile thresholds per feature.
2. Score one-predicate slices.
3. Keep a beam of promising masks.
4. Extend them into short conjunctions.
5. Deduplicate equivalent masks and rank by excess error weighted by support.

The result is intentionally human-readable rather than a latent cluster that still needs interpretation.

```python
from slicehound import discover_slices

slices = discover_slices(X, errors, feature_names=columns, max_depth=2)
```

## Benchmark

`python benchmarks/run.py` injects an error hotspot defined by two feature conditions into otherwise low-error data. The regression check requires the search to recover a high-lift conjunction overlapping that hotspot.

## Limitations

Quantile predicates are simple by design. Highly oblique, semantic, or representation-level failure regions need different search spaces.

## Roadmap

- categorical predicates
- holdout significance checks
- fairness-aware minimum support rules
- image/embedding slice adapters

MIT licensed.
