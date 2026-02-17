# DeadRecs

A CLI recommendation engine for Grateful Dead live performances. Given shows or songs you already enjoy, DeadRecs recommends others you're likely to appreciate.

## How It Works

DeadRecs builds a graph of every community-rated Grateful Dead performance on [Headyversion.com](https://headyversion.com) and trains a Graph Neural Network to learn what makes shows and songs similar.

### Data

The dataset contains ~25,000 rated performances across ~2,100 shows and ~380 songs. Each performance has a community vote count, venue, date, and (for ~92% of entries) a fan-written description like *"Absolutely MASSIVE version. Great jamming"* or *"The meld into Mind Left Body Jam is sublime"*.

### Graph

Performances are modeled as a heterogeneous graph with three node types:

- **Show** — a concert on a given date and venue
- **Song** — a song in the catalog
- **SongPerformance** — a specific performance of a song at a show, with votes and a description

Shows are connected to their performances, performances link to their songs, and shows are linked to similar shows via IDF-weighted Jaccard similarity over their setlists (rare songs count more than warhorses).

### Features

The GNN learns from both structural and textual signals:

- **Numeric features** — vote counts, IDF weights, era encoding, performance ranks
- **Text embeddings** — fan descriptions are encoded into 384-dim vectors using `sentence-transformers/all-MiniLM-L6-v2`, capturing the qualitative character of each performance (energy, style, notable jams)

### Model

A heterogeneous GraphSAGE network is trained via self-supervised link prediction. The resulting node embeddings place structurally and qualitatively similar shows/songs close together in vector space. Recommendations are nearest-neighbor lookups in this space.

## Usage

```
# Scrape performance data from Headyversion
deadrecs scrape

# Build graph and train the GNN
deadrecs train

# Get recommendations
deadrecs recommend --like "1977-05-08" --like "Dark Star" --n 5

# Look up details for a show or song
deadrecs info "1972-08-27"
```

## Installation

```
pip install -e .
```

Requires Python 3.10+.

## License

MIT
