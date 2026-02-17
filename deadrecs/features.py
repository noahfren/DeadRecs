"""Feature computation for the DeadRecs graph.

Computes:
- IDF weights for songs
- Description embeddings for SongPerformance nodes (via sentence-transformers)
- Weighted Jaccard similarity between shows → SETLIST_NEIGHBOR edges
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

import click
import networkx as nx
import numpy as np
import torch

from deadrecs.utils import DESCRIPTION_EMBEDDINGS_PATH

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


# ---------------------------------------------------------------------------
# IDF
# ---------------------------------------------------------------------------


def compute_idf(G: nx.DiGraph) -> dict[str, float]:
    """Compute IDF weights for each song node in the graph.

    idf(s) = log(N / df(s)) where:
    - N = total number of distinct shows
    - df(s) = number of shows containing song s

    Returns a dict mapping song node IDs to IDF values.
    Also sets the 'idf' attribute on each Song node in the graph.
    """
    # Count shows per song by traversing performance edges
    show_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "show"]
    N = len(show_nodes)

    if N == 0:
        return {}

    # Build song -> set of shows from the stored show_songs mapping
    song_show_count: dict[int, int] = defaultdict(int)
    show_songs = G.graph.get("show_songs", {})

    for date, song_ids in show_songs.items():
        for song_id in song_ids:
            song_show_count[song_id] += 1

    idf_values: dict[str, float] = {}
    for node_id, data in G.nodes(data=True):
        if data.get("type") != "song":
            continue
        hv_id = data["headyversion_id"]
        df = song_show_count.get(hv_id, 0)
        if df > 0:
            idf = math.log(N / df)
        else:
            idf = 0.0
        G.nodes[node_id]["idf"] = idf
        idf_values[node_id] = idf

    click.echo(f"Computed IDF for {len(idf_values)} songs (N={N} shows)")
    return idf_values


# ---------------------------------------------------------------------------
# Description Embeddings
# ---------------------------------------------------------------------------


def compute_description_embeddings(
    G: nx.DiGraph,
    batch_size: int = 256,
    force: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute sentence embeddings for all SongPerformance descriptions.

    Uses sentence-transformers/all-MiniLM-L6-v2.  Results are cached to
    data/description_embeddings.pt.

    Returns a dict mapping performance node IDs to 384-dim tensors.
    Also sets the 'embedding' attribute on each SongPerformance node.
    """
    # Check cache
    if not force and DESCRIPTION_EMBEDDINGS_PATH.exists():
        raw_data_mtime = max(
            f.stat().st_mtime
            for f in G.graph.get("_song_files", [DESCRIPTION_EMBEDDINGS_PATH])
        ) if G.graph.get("_song_files") else 0
        cache_mtime = DESCRIPTION_EMBEDDINGS_PATH.stat().st_mtime

        if cache_mtime > raw_data_mtime or raw_data_mtime == 0:
            click.echo("Loading cached description embeddings...")
            cached = torch.load(
                DESCRIPTION_EMBEDDINGS_PATH, map_location="cpu", weights_only=True
            )
            # Apply to graph nodes
            for node_id, emb in cached.items():
                if node_id in G:
                    G.nodes[node_id]["embedding"] = emb
            click.echo(f"Loaded {len(cached)} cached embeddings")
            return cached

    # Collect descriptions
    perf_nodes = []
    descriptions = []
    for node_id, data in G.nodes(data=True):
        if data.get("type") != "performance":
            continue
        perf_nodes.append(node_id)
        descriptions.append(data.get("description", ""))

    click.echo(f"Computing embeddings for {len(perf_nodes)} performances...")

    # Load sentence-transformers model
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Separate non-empty and empty descriptions
    non_empty_indices = [i for i, d in enumerate(descriptions) if d.strip()]
    empty_indices = [i for i, d in enumerate(descriptions) if not d.strip()]
    non_empty_texts = [descriptions[i] for i in non_empty_indices]

    click.echo(
        f"  {len(non_empty_texts)} with descriptions, "
        f"{len(empty_indices)} empty (will use zero vector)"
    )

    # Batch encode non-empty descriptions
    embeddings_array = model.encode(
        non_empty_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Build result dict
    zero_vec = torch.zeros(EMBEDDING_DIM)
    result: dict[str, torch.Tensor] = {}

    for idx, emb_idx in enumerate(non_empty_indices):
        node_id = perf_nodes[emb_idx]
        emb_tensor = torch.from_numpy(embeddings_array[idx]).float()
        result[node_id] = emb_tensor
        G.nodes[node_id]["embedding"] = emb_tensor

    for emb_idx in empty_indices:
        node_id = perf_nodes[emb_idx]
        result[node_id] = zero_vec
        G.nodes[node_id]["embedding"] = zero_vec

    # Cache to disk
    DESCRIPTION_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, DESCRIPTION_EMBEDDINGS_PATH)
    click.echo(f"Cached embeddings to {DESCRIPTION_EMBEDDINGS_PATH}")

    return result


# ---------------------------------------------------------------------------
# Weighted Jaccard Similarity + SETLIST_NEIGHBOR Edges
# ---------------------------------------------------------------------------


def _build_show_idf_vectors(
    G: nx.DiGraph,
) -> tuple[dict[str, dict[int, float]], dict[int, set[str]]]:
    """Build IDF-weighted song vectors for each show.

    Returns:
        show_vectors: dict mapping show date -> {song_id: idf_weight}
        song_to_shows: dict mapping song_id -> set of show dates
    """
    show_songs = G.graph.get("show_songs", {})

    # Collect IDF values
    song_idf: dict[int, float] = {}
    for node_id, data in G.nodes(data=True):
        if data.get("type") == "song":
            song_idf[data["headyversion_id"]] = data.get("idf", 0.0)

    # Build per-show vectors
    show_vectors: dict[str, dict[int, float]] = {}
    song_to_shows: dict[int, set[str]] = defaultdict(set)

    for date, song_ids in show_songs.items():
        vec: dict[int, float] = {}
        for sid in song_ids:
            idf = song_idf.get(sid, 0.0)
            vec[sid] = idf
            song_to_shows[sid].add(date)
        show_vectors[date] = vec

    return show_vectors, song_to_shows


def weighted_jaccard(vec_a: dict[int, float], vec_b: dict[int, float]) -> float:
    """Compute weighted Jaccard similarity between two IDF-weighted song vectors.

    sim(A, B) = sum(min(w_a, w_b)) / sum(max(w_a, w_b))
    over the union of songs, where absent songs have weight 0.
    """
    all_songs = set(vec_a) | set(vec_b)
    if not all_songs:
        return 0.0

    numerator = 0.0
    denominator = 0.0
    for s in all_songs:
        wa = vec_a.get(s, 0.0)
        wb = vec_b.get(s, 0.0)
        numerator += min(wa, wb)
        denominator += max(wa, wb)

    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def add_setlist_neighbor_edges(G: nx.DiGraph, k: int = 10) -> int:
    """Add SETLIST_NEIGHBOR edges between shows based on weighted Jaccard similarity.

    For each show, connects it to its top-k most similar shows.
    Only compares shows that share at least one song (sparse intersection)
    to avoid O(n^2) over all show pairs.

    Returns the number of edges added.
    """
    show_vectors, song_to_shows = _build_show_idf_vectors(G)
    dates = sorted(show_vectors.keys())

    click.echo(f"Computing setlist similarity for {len(dates)} shows (k={k})...")

    edges_added = 0

    for i, date_a in enumerate(dates):
        vec_a = show_vectors[date_a]

        # Only compare against shows that share at least one song
        candidate_dates: set[str] = set()
        for sid in vec_a:
            candidate_dates.update(song_to_shows[sid])
        candidate_dates.discard(date_a)

        if not candidate_dates:
            continue

        # Compute similarities
        similarities: list[tuple[str, float]] = []
        for date_b in candidate_dates:
            sim = weighted_jaccard(vec_a, show_vectors[date_b])
            if sim > 0:
                similarities.append((date_b, sim))

        # Keep top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        for date_b, sim in similarities[:k]:
            show_a = f"show:{date_a}"
            show_b = f"show:{date_b}"
            if not G.has_edge(show_a, show_b):
                G.add_edge(show_a, show_b, type="SETLIST_NEIGHBOR", weight=sim)
                edges_added += 1

        if (i + 1) % 500 == 0:
            click.echo(f"  Processed {i + 1}/{len(dates)} shows...")

    click.echo(f"Added {edges_added} SETLIST_NEIGHBOR edges")
    return edges_added
