# DeadRecs — Implementation Plan

## Phase 1: Project Setup

### 1.1 Dependencies
- Update `pyproject.toml` with dependencies: `networkx`, `torch`, `torch-geometric`, `sentence-transformers`, `beautifulsoup4`, `requests`, `click`, `numpy`.
- Add dev dependencies: `pytest`, `pytest-cov`.
- Create `deadrecs/__init__.py`.

### 1.2 CLI Skeleton
- Implement `deadrecs/cli.py` using Click with subcommand groups:
  - `deadrecs scrape`
  - `deadrecs train`
  - `deadrecs recommend`
  - `deadrecs info`
- Wire up to `pyproject.toml` entry point (already configured as `deadrecs.cli:main`).
- Each subcommand initially prints a placeholder message.

### 1.3 Data Directory Structure
- Create `data/raw/songs/` directory structure on first run.
- Add utility functions for resolving data paths in `deadrecs/utils.py`.

**Deliverable**: `deadrecs scrape` runs and prints "Scraper not yet implemented." Dependencies install cleanly.

---

## Phase 2: Scraper

### 2.1 Song Index Scraper
- `deadrecs/scraper.py`
- Fetch `https://headyversion.com/search/all/?order=count`.
- Parse HTML to extract all song entries: `{id, name, slug, url}`.
- Save to `data/raw/songs_index.json`.
- Handle pagination if the song index is paginated.

### 2.2 Song Performance Scraper
- For each song in the index, fetch its page: `/song/{id}/grateful-dead/{slug}/`.
- Parse each rated performance: `{votes, date, venue, description, archive_url}`.
- Handle "show more submissions" pagination (inspect whether this is AJAX/additional pages).
- Save per-song to `data/raw/songs/{id}_{slug}.json`.
- Configurable delay between requests (`--delay`, default 1 second).
- Resume support: skip songs already scraped (check if file exists).

### 2.3 Transition Parsing
- Detect combo song entries (containing "→", "->", or ">") in song names.
- Parse into individual song pairs for `TRANSITIONED_TO` edges.
- Store transition relationships in `data/raw/songs_index.json` alongside song metadata.

### 2.4 Tests
- Unit tests for HTML parsing logic using saved fixture HTML.
- Test transition parsing for various arrow formats.

**Deliverable**: `deadrecs scrape` fetches all ~380 songs and their performances. Data cached in `data/raw/`.

---

## Phase 3: Graph Construction

### 3.1 Graph Builder
- `deadrecs/graph.py`
- Load scraped JSON data from `data/raw/`.
- Create NetworkX graph with typed nodes:
  - Show nodes: keyed by date, properties: `{date, venue, type="show"}`.
  - Song nodes: keyed by headyversion ID, properties: `{name, slug, type="song"}`.
  - SongPerformance nodes: keyed by `{song_id}:{date}`, properties: `{votes, description, archive_url, type="performance"}`.
- Create edges: `HAS_PERFORMANCE`, `OF_SONG`, `TRANSITIONED_TO`.
- Handle date deduplication (multiple venue strings for the same date → same Show node).

### 3.2 IDF Computation
- `deadrecs/features.py`
- Compute `df(s)` for each song (number of distinct show dates containing that song).
- Compute `idf(s) = log(N / df(s))` where N = total distinct show dates.
- Store IDF as a property on Song nodes.

### 3.3 Description Embeddings
- `deadrecs/features.py`
- Load `sentence-transformers/all-MiniLM-L6-v2` model (downloaded on first run, ~80 MB).
- Collect all performance descriptions from the graph's `SongPerformance` nodes.
- Batch-encode descriptions (batch size 256, CPU-only) into 384-dim vectors.
- For empty descriptions (~8% of performances), use a zero vector.
- Cache result to `data/description_embeddings.pt` as a dict mapping `perf:{song_id}:{show_date}` → tensor.
- On subsequent runs, load from cache if the file exists and is newer than the raw data.
- Store the embedding vector as a property on each `SongPerformance` node.

### 3.4 Weighted Jaccard + SETLIST_NEIGHBOR Edges
- `deadrecs/features.py`
- For each show, build its IDF-weighted song vector.
- Compute pairwise weighted Jaccard similarity.
  - Optimization: only compare shows that share at least one song (sparse intersection) to avoid O(n^2) over all show pairs.
- Add top-k `SETLIST_NEIGHBOR` edges per show (default k=10).

### 3.5 Serialization
- Save graph to `data/graph.pickle`.
- Print graph statistics: node counts by type, edge counts by type, density.

### 3.6 Tests
- Test graph construction with a small synthetic dataset.
- Test IDF computation.
- Test description embedding generation (mock sentence-transformers model for unit tests).
- Test zero-vector fallback for empty descriptions.
- Test weighted Jaccard similarity with known values.

**Deliverable**: `deadrecs train` (first half) builds and saves the graph. Stats printed to console.

---

## Phase 4: GNN Training

### 4.1 PyG Data Conversion
- `deadrecs/model.py`
- Convert NetworkX graph to PyTorch Geometric `HeteroData`:
  - Map node types to separate feature tensors.
  - Map edge types to separate edge index tensors.
- Node feature engineering:
  - Show: [num_performances, mean_vote, era_onehot (6 dims)] — 8 dims
  - Song: [idf, total_votes, num_performances] — 3 dims
  - SongPerformance: [normalized_votes, rank_within_song, description_embedding (384 dims)] — 386 dims
- The first GraphSAGE layer's per-type linear projection handles the dimension mismatch between node types (projecting each to the shared 128-dim hidden space).

### 4.2 Model Definition
- `deadrecs/model.py`
- Heterogeneous GraphSAGE model using `torch_geometric.nn.HeteroConv` with `SAGEConv` per edge type.
- 2 message-passing layers.
- Hidden dimension: 128. Output embedding dimension: 128.
- ReLU activation + dropout between layers.

### 4.3 Training Loop
- `deadrecs/train.py`
- Self-supervised link prediction objective.
- Edge split: 85% train / 15% validation.
- Negative sampling: random edges of the same type.
- Optimizer: Adam, lr=0.01.
- Training loop with configurable epochs (default 100).
- Early stopping on validation AUC (patience=10).
- Save best model to `data/model.pt`.
- Save embeddings to `data/embeddings.pt` (dict mapping node IDs to embedding vectors).

### 4.4 Tests
- Test PyG conversion produces valid HeteroData.
- Test model forward pass runs without errors on small graph.
- Test training loop completes one epoch on small fixture.

**Deliverable**: `deadrecs train` builds graph, trains GNN, saves model + embeddings. Reports training loss and validation AUC.

---

## Phase 5: Recommendation Engine

### 5.1 Input Parsing
- `deadrecs/recommend.py`
- Parse `--like` arguments:
  - Date pattern (YYYY-MM-DD or partial like 1977-05-08) → Show node lookup.
  - String → fuzzy match against Song names (simple substring match, warn if ambiguous).
- Error handling for unrecognized inputs.

### 5.2 Embedding Lookup + KNN
- `deadrecs/recommend.py`
- Load embeddings from `data/embeddings.pt`.
- For each input, get its embedding vector.
- Average multiple input embeddings into a single query vector.
- Compute cosine similarity against all embeddings of the target type.
- Return top-n results, excluding inputs.

### 5.3 Output Formatting
- Pretty-print recommendations to terminal:
  ```
  #1  1972-08-27 — Veneta, OR (similarity: 0.94)
      Notable: Dark Star (135 votes), Playing in the Band (98 votes)
      Listen: https://archive.org/...

  #2  1977-05-09 — Buffalo, NY (similarity: 0.91)
      ...
  ```
- For song recommendations, show top-rated performances of the song.

### 5.4 Tests
- Test input parsing (dates vs song names).
- Test recommendation output with mock embeddings.

**Deliverable**: `deadrecs recommend --like "1977-05-08" --like "Dark Star" --n 5` returns ranked results.

---

## Phase 6: Info Command + Polish

### 6.1 Info Command
- `deadrecs info "1977-05-08"` — show date, venue, all rated performances with votes.
- `deadrecs info "Dark Star"` — show all rated performances of the song, ranked by votes.
- Pulls data from the serialized graph.

### 6.2 Error Handling + UX
- Friendly error messages when data isn't scraped yet ("Run `deadrecs scrape` first").
- Friendly error messages when model isn't trained yet ("Run `deadrecs train` first").
- Progress bars for scraping (using click.progressbar or similar).
- Progress reporting during training (epoch, loss, AUC).

### 6.3 Integration Test
- End-to-end test with a small fixture dataset: scrape fixture → build graph → train (1 epoch) → recommend.

**Deliverable**: Fully functional CLI. All subcommands working end-to-end.

---

## Implementation Order Summary

| Phase | What                        | Key Files                          |
|-------|-----------------------------|------------------------------------|
| 1     | Project setup + CLI skeleton | `cli.py`, `utils.py`, `pyproject.toml` |
| 2     | Scraper                     | `scraper.py`                       |
| 3     | Graph construction          | `graph.py`, `features.py`         |
| 4     | GNN training                | `model.py`, `train.py`            |
| 5     | Recommendations             | `recommend.py`                     |
| 6     | Info + polish               | `cli.py` updates                   |

Each phase builds on the previous and has a testable deliverable.
