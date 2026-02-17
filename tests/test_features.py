"""Tests for feature computation (IDF, embeddings, weighted Jaccard)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest
import torch

from deadrecs.features import (
    EMBEDDING_DIM,
    add_setlist_neighbor_edges,
    compute_description_embeddings,
    compute_idf,
    weighted_jaccard,
    _build_show_idf_vectors,
)


# ---------------------------------------------------------------------------
# Helpers — build small test graphs
# ---------------------------------------------------------------------------


def _make_test_graph() -> nx.DiGraph:
    """Build a small test graph with known structure.

    Shows: A (songs 1,2), B (songs 2,3), C (songs 1,2,3)
    Songs: 1 (df=2), 2 (df=3), 3 (df=2)
    N = 3 shows
    """
    G = nx.DiGraph()

    # Songs
    G.add_node("song:1", type="song", name="Dark Star", headyversion_id=1, slug="dark-star")
    G.add_node("song:2", type="song", name="Truckin'", headyversion_id=2, slug="truckin")
    G.add_node("song:3", type="song", name="Eyes", headyversion_id=3, slug="eyes")

    # Shows
    G.add_node("show:1972-08-27", type="show", date="1972-08-27", venue="Veneta, OR")
    G.add_node("show:1977-05-08", type="show", date="1977-05-08", venue="Ithaca, NY")
    G.add_node("show:1974-02-24", type="show", date="1974-02-24", venue="San Francisco, CA")

    # Performances for show A (1972-08-27): songs 1, 2
    G.add_node("perf:1:1972-08-27", type="performance", votes=135, description="Amazing")
    G.add_edge("show:1972-08-27", "perf:1:1972-08-27", type="HAS_PERFORMANCE")
    G.add_edge("perf:1:1972-08-27", "song:1", type="OF_SONG")

    G.add_node("perf:2:1972-08-27", type="performance", votes=50, description="Classic")
    G.add_edge("show:1972-08-27", "perf:2:1972-08-27", type="HAS_PERFORMANCE")
    G.add_edge("perf:2:1972-08-27", "song:2", type="OF_SONG")

    # Performances for show B (1977-05-08): songs 2, 3
    G.add_node("perf:2:1977-05-08", type="performance", votes=30, description="")
    G.add_edge("show:1977-05-08", "perf:2:1977-05-08", type="HAS_PERFORMANCE")
    G.add_edge("perf:2:1977-05-08", "song:2", type="OF_SONG")

    G.add_node("perf:3:1977-05-08", type="performance", votes=80, description="Great Eyes")
    G.add_edge("show:1977-05-08", "perf:3:1977-05-08", type="HAS_PERFORMANCE")
    G.add_edge("perf:3:1977-05-08", "song:3", type="OF_SONG")

    # Performances for show C (1974-02-24): songs 1, 2, 3
    G.add_node("perf:1:1974-02-24", type="performance", votes=98, description="Wall of sound")
    G.add_edge("show:1974-02-24", "perf:1:1974-02-24", type="HAS_PERFORMANCE")
    G.add_edge("perf:1:1974-02-24", "song:1", type="OF_SONG")

    G.add_node("perf:2:1974-02-24", type="performance", votes=40, description="")
    G.add_edge("show:1974-02-24", "perf:2:1974-02-24", type="HAS_PERFORMANCE")
    G.add_edge("perf:2:1974-02-24", "song:2", type="OF_SONG")

    G.add_node("perf:3:1974-02-24", type="performance", votes=60, description="Smooth")
    G.add_edge("show:1974-02-24", "perf:3:1974-02-24", type="HAS_PERFORMANCE")
    G.add_edge("perf:3:1974-02-24", "song:3", type="OF_SONG")

    # Store show_songs mapping
    G.graph["show_songs"] = {
        "1972-08-27": {1, 2},
        "1977-05-08": {2, 3},
        "1974-02-24": {1, 2, 3},
    }

    return G


# ---------------------------------------------------------------------------
# IDF
# ---------------------------------------------------------------------------


class TestComputeIDF:
    def test_basic_idf(self):
        G = _make_test_graph()
        idf = compute_idf(G)

        N = 3
        # Song 1 appears in 2 shows → idf = log(3/2)
        assert idf["song:1"] == pytest.approx(math.log(N / 2))
        # Song 2 appears in all 3 shows → idf = log(3/3) = 0
        assert idf["song:2"] == pytest.approx(math.log(N / 3))
        # Song 3 appears in 2 shows → idf = log(3/2)
        assert idf["song:3"] == pytest.approx(math.log(N / 2))

    def test_idf_stored_on_nodes(self):
        G = _make_test_graph()
        compute_idf(G)
        assert G.nodes["song:1"]["idf"] == pytest.approx(math.log(3 / 2))

    def test_idf_empty_graph(self):
        G = nx.DiGraph()
        G.graph["show_songs"] = {}
        idf = compute_idf(G)
        assert idf == {}

    def test_idf_song_not_performed(self):
        G = _make_test_graph()
        # Add a song with no performances
        G.add_node("song:99", type="song", name="Unknown", headyversion_id=99, slug="unknown")
        G.graph["show_songs"]["1972-08-27"].add(99)  # hack: pretend it was at 1 show
        idf = compute_idf(G)
        assert idf["song:99"] == pytest.approx(math.log(3 / 1))


# ---------------------------------------------------------------------------
# Weighted Jaccard
# ---------------------------------------------------------------------------


class TestWeightedJaccard:
    def test_identical_vectors(self):
        vec = {1: 0.5, 2: 1.0}
        assert weighted_jaccard(vec, vec) == pytest.approx(1.0)

    def test_disjoint_vectors(self):
        vec_a = {1: 0.5}
        vec_b = {2: 1.0}
        assert weighted_jaccard(vec_a, vec_b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        vec_a = {1: 1.0, 2: 0.5}
        vec_b = {2: 0.5, 3: 1.0}
        # Union = {1, 2, 3}
        # min: 0 + 0.5 + 0 = 0.5
        # max: 1.0 + 0.5 + 1.0 = 2.5
        assert weighted_jaccard(vec_a, vec_b) == pytest.approx(0.5 / 2.5)

    def test_empty_vectors(self):
        assert weighted_jaccard({}, {}) == 0.0

    def test_one_empty_vector(self):
        assert weighted_jaccard({1: 1.0}, {}) == 0.0

    def test_known_values(self):
        """Test with hand-computed values from design doc."""
        # Show A has songs {1: idf=0.405, 2: idf=0.0}
        # Show B has songs {2: idf=0.0, 3: idf=0.405}
        idf_1 = math.log(3 / 2)  # ~0.405
        idf_2 = math.log(3 / 3)  # 0.0
        idf_3 = math.log(3 / 2)  # ~0.405

        vec_a = {1: idf_1, 2: idf_2}  # show A: songs 1, 2
        vec_b = {2: idf_2, 3: idf_3}  # show B: songs 2, 3

        # Union = {1, 2, 3}
        # min: min(0.405, 0) + min(0, 0) + min(0, 0.405) = 0
        # max: max(0.405, 0) + max(0, 0) + max(0, 0.405) = 0.405 + 0 + 0.405 = 0.81
        expected = 0.0 / (idf_1 + 0.0 + idf_3)
        assert weighted_jaccard(vec_a, vec_b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# SETLIST_NEIGHBOR edges
# ---------------------------------------------------------------------------


class TestSetlistNeighborEdges:
    def test_adds_edges(self):
        G = _make_test_graph()
        compute_idf(G)
        count = add_setlist_neighbor_edges(G, k=2)
        sn_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d["type"] == "SETLIST_NEIGHBOR"]
        assert len(sn_edges) > 0
        assert count == len(sn_edges)

    def test_edges_have_weight(self):
        G = _make_test_graph()
        compute_idf(G)
        add_setlist_neighbor_edges(G, k=2)
        for u, v, d in G.edges(data=True):
            if d["type"] == "SETLIST_NEIGHBOR":
                assert "weight" in d
                assert 0 < d["weight"] <= 1.0

    def test_k_limits_edges_per_show(self):
        G = _make_test_graph()
        compute_idf(G)
        add_setlist_neighbor_edges(G, k=1)
        # Each show should have at most k=1 outgoing SETLIST_NEIGHBOR edge
        for node_id, data in G.nodes(data=True):
            if data["type"] == "show":
                out_sn = [
                    v for _, v, d in G.out_edges(node_id, data=True)
                    if d["type"] == "SETLIST_NEIGHBOR"
                ]
                assert len(out_sn) <= 1

    def test_show_c_most_similar_to_a_and_b(self):
        """Show C (songs 1,2,3) shares songs with both A and B."""
        G = _make_test_graph()
        compute_idf(G)
        add_setlist_neighbor_edges(G, k=10)

        # C should be connected to both A and B
        c_neighbors = set()
        for _, v, d in G.out_edges("show:1974-02-24", data=True):
            if d["type"] == "SETLIST_NEIGHBOR":
                c_neighbors.add(v)

        assert "show:1972-08-27" in c_neighbors  # shares songs 1, 2
        assert "show:1977-05-08" in c_neighbors  # shares songs 2, 3


# ---------------------------------------------------------------------------
# Description Embeddings
# ---------------------------------------------------------------------------


class TestDescriptionEmbeddings:
    def test_embedding_with_mock_model(self, tmp_path, monkeypatch):
        """Test embedding computation with a mocked sentence transformer."""
        G = _make_test_graph()

        monkeypatch.setattr(
            "deadrecs.features.DESCRIPTION_EMBEDDINGS_PATH",
            tmp_path / "desc_emb.pt",
        )

        # Count performances
        perf_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "performance"]
        non_empty = [n for n in perf_nodes if G.nodes[n].get("description", "").strip()]
        empty = [n for n in perf_nodes if not G.nodes[n].get("description", "").strip()]

        # Mock the sentence transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(len(non_empty), EMBEDDING_DIM).astype(np.float32)

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            result = compute_description_embeddings(G, force=True)

        assert len(result) == len(perf_nodes)

        # Non-empty descriptions should have non-zero embeddings
        for node_id in non_empty:
            assert result[node_id].shape == (EMBEDDING_DIM,)
            # May or may not be zero depending on random values

        # Empty descriptions should have zero vectors
        for node_id in empty:
            assert result[node_id].shape == (EMBEDDING_DIM,)
            assert torch.all(result[node_id] == 0)

    def test_zero_vector_for_empty_description(self, tmp_path, monkeypatch):
        G = _make_test_graph()
        monkeypatch.setattr(
            "deadrecs.features.DESCRIPTION_EMBEDDINGS_PATH",
            tmp_path / "desc_emb.pt",
        )

        # Perf with empty description: "perf:2:1977-05-08" and "perf:2:1974-02-24"
        perf_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "performance"]
        non_empty = [n for n in perf_nodes if G.nodes[n].get("description", "").strip()]

        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((len(non_empty), EMBEDDING_DIM), dtype=np.float32)

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            result = compute_description_embeddings(G, force=True)

        # Check zero vectors for empty descriptions
        assert torch.all(result["perf:2:1977-05-08"] == 0)
        assert torch.all(result["perf:2:1974-02-24"] == 0)

        # Check non-zero for descriptions that have text
        assert torch.all(result["perf:1:1972-08-27"] == 1.0)

    def test_embeddings_cached_to_disk(self, tmp_path, monkeypatch):
        G = _make_test_graph()
        cache_path = tmp_path / "desc_emb.pt"
        monkeypatch.setattr("deadrecs.features.DESCRIPTION_EMBEDDINGS_PATH", cache_path)

        perf_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "performance"]
        non_empty = [n for n in perf_nodes if G.nodes[n].get("description", "").strip()]

        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((len(non_empty), EMBEDDING_DIM), dtype=np.float32)

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            compute_description_embeddings(G, force=True)

        assert cache_path.exists()

    def test_loads_from_cache(self, tmp_path, monkeypatch):
        G = _make_test_graph()
        cache_path = tmp_path / "desc_emb.pt"
        monkeypatch.setattr("deadrecs.features.DESCRIPTION_EMBEDDINGS_PATH", cache_path)

        # Pre-populate cache
        cached_data = {}
        for n, d in G.nodes(data=True):
            if d["type"] == "performance":
                cached_data[n] = torch.randn(EMBEDDING_DIM)
        torch.save(cached_data, cache_path)

        # Should load from cache without importing SentenceTransformer
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            result = compute_description_embeddings(G)
            mock_st.assert_not_called()

        assert len(result) == len(cached_data)

    def test_embedding_stored_on_graph_nodes(self, tmp_path, monkeypatch):
        G = _make_test_graph()
        monkeypatch.setattr(
            "deadrecs.features.DESCRIPTION_EMBEDDINGS_PATH",
            tmp_path / "desc_emb.pt",
        )

        perf_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "performance"]
        non_empty = [n for n in perf_nodes if G.nodes[n].get("description", "").strip()]

        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((len(non_empty), EMBEDDING_DIM), dtype=np.float32)

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            compute_description_embeddings(G, force=True)

        for n in perf_nodes:
            assert "embedding" in G.nodes[n]
            assert G.nodes[n]["embedding"].shape == (EMBEDDING_DIM,)
