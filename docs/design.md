# DeadRecs — Design

## Architecture Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Scraper    │────>│    Graph     │────>│  GNN Train   │────>│  Recommend   │
│              │     │  Builder     │     │              │     │              │
│ headyversion │     │  NetworkX    │     │  PyG + SAGE  │     │  Embedding   │
│  → JSON      │     │  → pickle    │     │  → .pt model │     │  KNN lookup  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
     scrape               train                train              recommend
```

Each stage reads from the previous stage's output on disk. The CLI subcommands map directly to these stages.

## Graph Schema

### Node Types

#### Show
Represents a single Grateful Dead concert.

| Property      | Type   | Description                       |
|---------------|--------|-----------------------------------|
| `date`        | str    | Concert date (YYYY-MM-DD)         |
| `venue`       | str    | Venue name (parsed from scrape)   |
| `node_id`     | str    | `show:{date}` (e.g., `show:1977-05-08`) |

#### Song
Represents a song in the catalog.

| Property          | Type   | Description                        |
|-------------------|--------|------------------------------------|
| `name`            | str    | Song name                          |
| `headyversion_id` | int   | ID from headyversion URL           |
| `slug`            | str    | URL slug from headyversion         |
| `idf`             | float  | Inverse document frequency weight  |
| `node_id`         | str    | `song:{headyversion_id}`           |

#### SongPerformance
A specific rated performance of a song at a show.

| Property      | Type   | Description                        |
|---------------|--------|------------------------------------|
| `votes`       | int    | Community upvote count             |
| `description` | str    | User-submitted description         |
| `archive_url` | str    | Link to archive.org recording      |
| `node_id`     | str    | `perf:{song_id}:{show_date}`       |

### Edge Types

| Edge                | Source → Target              | Properties           |
|---------------------|------------------------------|----------------------|
| `HAS_PERFORMANCE`   | Show → SongPerformance       | —                    |
| `OF_SONG`           | SongPerformance → Song       | —                    |
| `TRANSITIONED_TO`   | Song → Song                  | Directed (A → B)     |
| `SETLIST_NEIGHBOR`   | Show → Show                  | `weight`: similarity |

### Example Subgraph

```
[Song: Dark Star] <──OF_SONG── [Perf: Dark Star @ 1972-08-27, 135 votes]
                                        │
                                  HAS_PERFORMANCE
                                        │
                               [Show: 1972-08-27, Veneta OR]
                                        │
                                 SETLIST_NEIGHBOR (0.45)
                                        │
                               [Show: 1972-08-21, Berkeley CA]
                                        │
                                  HAS_PERFORMANCE
                                        │
                               [Perf: Dark Star @ 1972-08-21, 80 votes]
                                        │
                                    OF_SONG
                                        │
                               [Song: Dark Star]
```

## Weighted Jaccard Similarity

Used to compute `SETLIST_NEIGHBOR` edges between shows.

### IDF Weighting

For each song `s`:

```
idf(s) = log(N / df(s))
```

Where:
- `N` = total number of shows in the dataset
- `df(s)` = number of shows containing song `s`

Songs performed rarely (e.g., "Dark Star" at ~50 shows) get high IDF. Songs performed at nearly every show (e.g., "Truckin'") get low IDF.

### Similarity Computation

For two shows `A` and `B`, represent each as a set of songs with IDF weights:

```
sim(A, B) = Σ_s min(idf(s) ∈ A, idf(s) ∈ B) / Σ_s max(idf(s) ∈ A, idf(s) ∈ B)
```

Where the sum is over the union of songs in A and B, and a song not in a set contributes weight 0.

### Edge Creation

For each show, create `SETLIST_NEIGHBOR` edges to its top-k most similar shows (default k=10). Edge weight = similarity score.

## GNN Architecture

### Model: GraphSAGE on Heterogeneous Graph

We use a heterogeneous GraphSAGE model that learns separate embeddings for each node type while passing messages across edge types.

```
Input Features          GraphSAGE Layers (×2)        Output Embeddings
─────────────          ──────────────────────        ─────────────────
Show features    ──>   Message passing across   ──>  Show embeddings (d=128)
Song features    ──>   all edge types with      ──>  Song embeddings (d=128)
Perf features    ──>   per-type projections     ──>  Perf embeddings (d=128)
```

### Node Features

| Node Type        | Features                                                        |
|------------------|-----------------------------------------------------------------|
| Show             | Number of rated performances, mean vote of performances, era (one-hot encoded by decade) |
| Song             | IDF weight, total votes across all performances, number of rated performances |
| SongPerformance  | Vote count (normalized), vote rank within song                  |

### Training Objective: Link Prediction (Self-Supervised)

No labeled "good/bad show" data exists, so we use self-supervised link prediction:

1. **Positive edges**: Existing `SETLIST_NEIGHBOR` edges (and optionally `HAS_PERFORMANCE` edges).
2. **Negative edges**: Randomly sampled non-existing edges of the same type.
3. **Loss**: Binary cross-entropy — the model learns to predict whether an edge exists.
4. **Result**: Node embeddings that place structurally similar nodes close together in vector space.

This means shows with similar setlists, highly-rated overlapping songs, and shared neighborhood structure will end up with similar embeddings — exactly what we need for recommendations.

### Training/Validation Split

- 85% of edges for training, 15% held out for validation.
- Monitor validation AUC to detect overfitting.
- Early stopping with patience of 10 epochs.

## Recommendation Engine

### Input Processing

User provides `--like` arguments which can be:
- A date string (e.g., `1977-05-08`) → maps to a Show node
- A song name (e.g., `Dark Star`) → maps to a Song node

### Embedding Lookup

1. Look up the embedding vector for each input node.
2. Average the embeddings to produce a single query vector (handles multiple `--like` inputs).

### Nearest Neighbor Search

1. Compute cosine similarity between the query vector and all node embeddings of the target type (shows, songs, or both).
2. Exclude the input nodes from results.
3. Return top-n results ranked by similarity.

### Output Format

Each recommendation includes:
- Rank and similarity score
- Show date and venue (or song name)
- Notable performances with vote counts
- Archive.org link(s)

## Data Storage Layout

```
data/
├── raw/
│   ├── songs_index.json       # Song catalog: [{id, name, slug, url}, ...]
│   └── songs/
│       ├── 58_dark-star.json  # Per-song performances
│       ├── 207_playin-in-the-band.json
│       └── ...
├── graph.pickle               # Serialized NetworkX graph
├── model.pt                   # Trained GNN weights
└── embeddings.pt              # Node embeddings tensor + ID mapping
```

## Dependencies

| Package              | Purpose                              |
|----------------------|--------------------------------------|
| `networkx`           | Graph construction and storage       |
| `torch`              | Deep learning framework              |
| `torch-geometric`    | GNN layers, data loaders, transforms |
| `beautifulsoup4`     | HTML parsing for scraper             |
| `requests`           | HTTP client for scraper              |
| `click`              | CLI framework                        |
| `numpy`              | Numeric operations                   |
