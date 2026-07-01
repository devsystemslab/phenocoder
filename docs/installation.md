# Installation

## Install

```bash
# Clone the repository
git clone https://github.com/devsystemslab/phenocoder.git
cd phenocoder

# Install with uv (recommended)
uv pip install .

# Or with pip
pip install .
```

## Requirements

- Python >= 3.10, < 3.13
- TensorFlow >= 2.19.0 (CUDA on Linux, Metal on macOS)
- SpatialData >= 0.5.0
- Additional dependencies listed in `pyproject.toml`

Phenocoder is built on Keras 3 with the TensorFlow backend. On Linux the full CUDA stack is
installed (`tensorflow[and-cuda]`); on macOS GPU acceleration is provided through
`tensorflow-metal`.

## Development install

From a clone of the repository (see above), install in editable mode with the dev extras:

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
ruff format .

# Lint
ruff check .
```

## Building the docs

```bash
uv pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```
