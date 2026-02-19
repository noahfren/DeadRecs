# AGENTS.md

## Project Overview

DeadRecs is a CLI recommendation engine for Grateful Dead live performances, powered by a GNN trained on community ratings from Headyversion.com. It operates as a four-stage pipeline: scrape, graph construction, GNN training, and recommendation via embedding lookup.

## Using the Docs

The `docs/` directory contains the authoritative specifications for this project. **Read these before making changes.**

Docs are organized into numerically ordered **epics** (e.g., `docs/01-initial-impl/`), each containing their own planning files (requirements, design, implementation plan). Items in the roadmap will eventually become their own epics.

- **`docs/01-initial-impl/`** — The initial implementation epic:
  - `requirements.md` — Functional and non-functional requirements. Consult this to understand what the system should do and its constraints (Python 3.10+, CLI-only, all data local, recommendations under 2 seconds).
  - `design.md` — Architecture, graph schema, GNN design, and data layout. This is the primary reference for how the system works: node types and their properties, edge types, weighted Jaccard similarity, GraphSAGE architecture, training objective, and the recommendation algorithm. Read this before touching `graph.py`, `model.py`, `train.py`, `features.py`, or `recommend.py`.
  - `implementation-plan.md` — Phased build plan (Phases 1-6) with specific deliverables per phase. Shows what files implement what functionality, what tests are expected, and the implementation order. Use this to understand where new code should go.

- **`docs/roadmap.md`** — Planned future features (natural language queries, web app, evaluation benchmarks, additional data sources). Not yet implemented. Items here will become their own numbered epics when work begins.

## Project Structure

```
deadrecs/           # Main package
  cli.py            # Click CLI entry point (scrape, train, recommend, info)
  scraper.py        # Headyversion.com scraper
  graph.py          # NetworkX graph construction
  features.py       # IDF, description embeddings, setlist neighbor edges
  model.py          # PyG HeteroData conversion + GraphSAGE model definition
  train.py          # GNN training loop (link prediction, early stopping)
  recommend.py      # Embedding lookup + KNN recommendations
  utils.py          # Data path constants
data/               # All generated data (not committed)
  raw/              # Scraped JSON from Headyversion
  graph.pickle      # Serialized graph
  model.pt          # Trained GNN weights
  embeddings.pt     # Node embeddings for recommendation
tests/              # pytest tests
  fixtures/         # Sample HTML for scraper tests
```

## Key Conventions

- **CLI framework**: Click. All commands are subcommands of `deadrecs` (entry point in `pyproject.toml`).
- **Graph library**: NetworkX for construction, PyTorch Geometric for GNN training.
- **Node IDs**: `show:{date}`, `song:{headyversion_id}`, `perf:{song_id}:{show_date}`.
- **Testing**: pytest. Run with `pytest` from the project root.
- **Install**: `pip install -e ".[dev]"` for development.

## Development Workflow

1. Read the relevant doc in `docs/` before making changes.
2. Follow existing patterns in the codebase — check how similar functionality is already implemented.
3. Run `pytest` to verify changes don't break existing tests.
4. When adding new pipeline functionality, add corresponding tests (see Phase test sections in `docs/implementation-plan.md`).
