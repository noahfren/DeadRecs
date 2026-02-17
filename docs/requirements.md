# DeadRecs — Requirements

## Overview

DeadRecs is a CLI-based recommendation engine for Grateful Dead live performances. Given shows, songs, or specific performances a user already enjoys, it recommends other shows and songs they're likely to appreciate.

The system uses a Graph Neural Network (GNN) trained on a graph of Grateful Dead performances, built from community-sourced ratings on Headyversion.com.

## Functional Requirements

### FR-1: Data Ingestion (Scraping)

- **FR-1.1**: Scrape the full Headyversion.com song catalog (~380 songs).
- **FR-1.2**: For each song, scrape all rated performances including: vote count, date, venue, description, and archive.org link.
- **FR-1.3**: Handle pagination on song pages (some songs have "show more submissions").
- **FR-1.4**: Parse song transition relationships from combo entries (e.g., "Scarlet Begonias -> Fire On The Mountain" implies a `TRANSITIONED_TO` edge).
- **FR-1.5**: Cache scraped data to disk as structured JSON so re-scraping is not required on every run.
- **FR-1.6**: Respect rate limits — include configurable delay between requests.

### FR-2: Graph Construction

- **FR-2.1**: Build a NetworkX graph from scraped data with three node types: `Show`, `Song`, `SongPerformance`.
- **FR-2.2**: Create edges: `HAS_PERFORMANCE` (Show -> SongPerformance), `OF_SONG` (SongPerformance -> Song), `TRANSITIONED_TO` (Song -> Song).
- **FR-2.3**: Compute IDF weights for songs: `log(total_shows / shows_containing_song)`.
- **FR-2.4**: Compute weighted Jaccard similarity between all show pairs using IDF-weighted song sets.
- **FR-2.5**: Create `SETLIST_NEIGHBOR` edges for the top-k most similar shows per show (k configurable, default 10), with similarity score as edge weight.
- **FR-2.6**: Serialize the constructed graph to disk for reuse.

### FR-3: GNN Training

- **FR-3.1**: Convert the NetworkX graph to a PyTorch Geometric `HeteroData` object (heterogeneous graph with typed nodes and edges).
- **FR-3.2**: Define node features: vote counts, song IDF, performance count per show, and any other meaningful numeric properties.
- **FR-3.3**: Compute sentence embeddings for performance descriptions using a lightweight pretrained model (`sentence-transformers/all-MiniLM-L6-v2`) and include them as `SongPerformance` node features. Cache embeddings to disk for reuse.
- **FR-3.4**: For performances with empty descriptions (~8%), use a zero vector of the same dimensionality as a fallback.
- **FR-3.5**: Train a GNN (GraphSAGE or similar) to produce node embeddings via self-supervised learning (e.g., link prediction as the training objective).
- **FR-3.6**: Save trained model weights and learned embeddings to disk.
- **FR-3.7**: Support retraining when new data is scraped.

### FR-4: Recommendations

- **FR-4.1**: Accept user input as a list of shows (by date) and/or songs (by name).
- **FR-4.2**: Look up the corresponding node embeddings.
- **FR-4.3**: Find the nearest neighbors in embedding space (cosine similarity).
- **FR-4.4**: Return ranked recommendations with metadata: date, venue, notable songs, vote counts, archive.org links.
- **FR-4.5**: Support filtering recommendations by type (shows only, songs only, or both).
- **FR-4.6**: Support configuring the number of recommendations returned (default 10).

### FR-5: CLI Interface

- **FR-5.1**: `deadrecs scrape` — Run the Headyversion scraper. Options: `--delay` (seconds between requests).
- **FR-5.2**: `deadrecs train` — Build graph from cached data and train the GNN. Options: `--epochs`, `--k-neighbors`.
- **FR-5.3**: `deadrecs recommend` — Get recommendations. Options: `--like` (repeatable, accepts dates or song names), `--n` (number of results), `--type` (shows/songs/both).
- **FR-5.4**: `deadrecs info` — Display details about a show or song. Accepts a date or song name.

## Non-Functional Requirements

- **NFR-1**: Pure CLI tool — no web server, no database server, no external services beyond the initial scrape.
- **NFR-2**: All data cached locally in a `data/` directory.
- **NFR-3**: Reasonable scrape time — full scrape should complete in under 30 minutes with polite rate limiting.
- **NFR-4**: Recommendations should return in under 2 seconds (embedding lookup, not retraining).
- **NFR-5**: Python 3.10+.

## Data Source

**Headyversion.com** — a community-driven site for rating live Grateful Dead performances.

- Song index: `https://headyversion.com/search/all/?order=count`
- Song page: `https://headyversion.com/song/{id}/grateful-dead/{slug}/`
- Data per performance: votes, date, venue, description, archive.org link
- ~380 songs, ~2,300 shows, estimated 10,000-30,000 rated performances
