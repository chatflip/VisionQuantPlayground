# template-python

## Requirement

- python >= 3.11
- uv >= 0.3

## Installation

```bash
uv sync
pre-commit install
```

## Usage

### Run

```bash
python src/data/download_food101.py
```

### Lint & Format

```bash
ruff format
ruff check --fix .
mypy .
```

## Author

[chatflip](https://github.com/chatflip)
