# Contributing to `SeroRef`

Contributions are welcome, and they are greatly appreciated! Every improvement — from bug reports and documentation refinements to Numba kernels, biosynthetic locus archetypes, and reference strain curation — helps advance open bacterial genomics.

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/SeroVerse/SeroRef/issues.

If you are reporting a bug, please include:

- Your operating system name, Python version (`>=3.14`), and Numba/NumPy versions.
- Minimal FASTA inputs or code snippets sufficient to reproduce the issue.
- Expected biological or algorithmic behavior vs. actual output.

### Fix Bugs & Implement Features

Look through the GitHub issues for open bugs or enhancements at https://github.com/SeroVerse/SeroRef/issues. If you are proposing a new feature:

- Explain how it fits into `SeroRef`'s three-layer architecture (`seroref.genes`, `seroref.loci`, `seroref.db`) built on `seroverse.core` and `seroverse.client`.
- Keep hot-path data structures in contiguous Structure-of-Arrays (SoA) NumPy buffers (`BatchedContainer` / `RaggedArrayContainer`) so Numba kernels remain zero-copy and GIL-free.
- Maintain strict dependency boundaries: `seroref` depends only on `seroverse[all]` (`numpy`, `scipy`, `numba`) and must never import `kbank`, `serogen`, or `kaptive`.

## Local Development Setup

`SeroRef` uses [`uv`](https://docs.astral.sh/uv/) for dependency and environment management.

1. Fork and clone the repository:

```bash
git clone git@github.com:YOUR_USERNAME/SeroRef.git
cd SeroRef
```

2. Sync all development, test, and documentation dependency groups:

```bash
uv sync --all-groups
```

3. Create a feature branch for your changes:

```bash
git checkout -b feat/your-feature-name
```

## Quality Checks & Testing

Before opening a pull request, verify that formatting, linting, static type checking, unit tests, and strict documentation builds all pass:

```bash
# 1. Lint and check formatting with Ruff
uv run ruff check .
uv run ruff format --check .

# 2. Run static type checking with ty
uv run ty check src/ tests/

# 3. Run the pytest test suite
uv run pytest -n 0

# 4. Build the Zensical documentation site in strict mode
uv run zensical build --clean --strict
```

## Pull Request Guidelines

1. Include unit tests in `tests/` covering any new algorithms, edge cases, or bug fixes.
2. Document public classes, methods, and dataclasses using Google-style docstrings so `mkdocstrings` renders complete API references.
3. Ensure all CI checks (`ci.yml` and `docs.yml`) pass cleanly.
