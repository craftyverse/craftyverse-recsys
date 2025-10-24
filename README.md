# 🧠 Recsys — Modular Recommender System

A lightweight, extensible **Python recommender system** project scaffold.  
Built to evolve from simple prototypes (CSV + notebooks) to full ML pipelines (training, evaluation, and serving).

---

## 🚀 Quickstart

### 1. Clone & enter the project
```bash
git clone https://github.com/craftyverse/craftyverse-recsys.git
cd recsys
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # (zsh/bash)
# or: source .venv/bin/activate.fish   # on fish shell
```

### 3. Install dependencies (with dev tools)
```bash
pip install -e '.[dev]'
```

If you only want runtime dependencies:
```bash
pip install -e .
```

---

## 🧩 Project Structure

```
recsys/
├── pyproject.toml                 # project metadata & dependencies
├── README.md                      # you are here
├── .gitignore                     # ignored files (data, caches, etc.)
├── .pre-commit-config.yaml        # lint & formatting hooks
├── configs/                       # YAML configs for training, etc.
│   ├── default.yaml
│   └── train/implicit_als.yaml
├── data/                          # local datasets (gitignored)
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/                     # exploratory notebooks
│   └── 00_exploration.ipynb
├── src/
│   └── recsys/
│       ├── __init__.py
│       ├── cli.py                 # CLI entrypoint (hello world)
│       ├── pipelines/
│       │   ├── train.py
│       │   └── infer.py
│       ├── evaluation/
│       │   └── metrics.py
│       ├── data/
│       │   ├── loaders.py
│       │   └── splits.py
│       ├── models/
│       │   ├── base.py
│       │   └── implicit_als.py
│       └── utils/
│           ├── io.py
│           └── logging.py
└── tests/
    ├── test_metrics.py
    └── test_pipeline.py
```

---

## 🧪 Usage

### Run the CLI
```bash
python -m recsys.cli hello
python -m recsys.cli hello Tony
```

### Train a model (placeholder)
```bash
python -m recsys.cli train
```

### Infer (placeholder)
```bash
python -m recsys.cli infer --artifacts-dir artifacts/als_baseline --users 1,2,3 --k 10
```

### Run tests
```bash
pytest -q
```

### Run linters
```bash
ruff check src
mypy src
```

---

## 🧰 Development Setup

### Pre-commit hooks
Install once:
```bash
pre-commit install
```
Now every `git commit` will automatically lint and fix your code.

### Makefile shortcuts
```bash
make setup   # install deps + pre-commit
make lint    # run ruff + mypy
make test    # run pytest
make train   # train pipeline
make infer   # run inference
```

---

## 🧠 Philosophy

This scaffold follows a **modular ML architecture**:
- `src/recsys/data` – load & split data
- `src/recsys/models` – model definitions
- `src/recsys/pipelines` – training/inference orchestration
- `src/recsys/evaluation` – metrics and validation
- `configs/` – reproducible experiment settings (Hydra-style)
- `notebooks/` – exploration only; production logic lives in `src/`

Artifacts (models, logs, outputs) are written to:
```
artifacts/<run_name>/
```

---

## 🧭 Next Steps
1. Add a small sample dataset under `data/processed/interactions.csv`.
2. Implement `train.py` to load the CSV and print summary stats.
3. Build a first recommender (`implicit_als.py`).
4. Add `evaluation/metrics.py` for recall@k / map@k.
5. Serve via FastAPI or connect to your Fastify gateway.

---

## 📦 License
MIT © 2025 — your-org-name
