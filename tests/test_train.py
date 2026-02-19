"""Tests for GNN training loop."""

from __future__ import annotations

import torch
import networkx as nx
import pytest

from deadrecs.features import EMBEDDING_DIM, compute_idf
from deadrecs.model import convert_to_hetero_data
from deadrecs.train import (
    _split_edges,
    _compute_loss,
    _compute_val_auc,
    train_model,
    SUPERVISED_EDGE_TYPES,
)


def _make_test_graph() -> nx.DiGraph:
    """Build a small test graph for training tests."""
    G = nx.DiGraph()

    # Songs
    for i in range(1, 4):
        G.add_node(f"song:{i}", type="song", name=f"Song {i}", headyversion_id=i, slug=f"song-{i}")

    # Shows
    dates = ["1972-08-27", "1977-05-08", "1974-02-24"]
    venues = ["Veneta, OR", "Ithaca, NY", "San Francisco, CA"]
    for date, venue in zip(dates, venues):
        G.add_node(f"show:{date}", type="show", date=date, venue=venue)

    # Performances — enough edges for train/val split
    perf_data = [
        ("perf:1:1972-08-27", "show:1972-08-27", "song:1", 135, "Amazing"),
        ("perf:2:1972-08-27", "show:1972-08-27", "song:2", 50, "Classic"),
        ("perf:1:1977-05-08", "show:1977-05-08", "song:1", 98, "Great"),
        ("perf:3:1977-05-08", "show:1977-05-08", "song:3", 80, "Solid"),
        ("perf:1:1974-02-24", "show:1974-02-24", "song:1", 120, "Wall of sound"),
        ("perf:2:1974-02-24", "show:1974-02-24", "song:2", 40, ""),
        ("perf:3:1974-02-24", "show:1974-02-24", "song:3", 60, "Smooth"),
    ]

    for perf_id, show_id, song_id, votes, desc in perf_data:
        emb = torch.randn(EMBEDDING_DIM) if desc else torch.zeros(EMBEDDING_DIM)
        G.add_node(perf_id, type="performance", votes=votes, description=desc, embedding=emb)
        G.add_edge(show_id, perf_id, type="HAS_PERFORMANCE")
        G.add_edge(perf_id, song_id, type="OF_SONG")

    # SETLIST_NEIGHBOR edges
    G.add_edge("show:1972-08-27", "show:1974-02-24", type="SETLIST_NEIGHBOR", weight=0.8)
    G.add_edge("show:1977-05-08", "show:1974-02-24", type="SETLIST_NEIGHBOR", weight=0.6)
    G.add_edge("show:1974-02-24", "show:1972-08-27", type="SETLIST_NEIGHBOR", weight=0.7)

    # TRANSITIONED_TO
    G.add_edge("song:1", "song:2", type="TRANSITIONED_TO")

    G.graph["show_songs"] = {
        "1972-08-27": {1, 2},
        "1977-05-08": {1, 3},
        "1974-02-24": {1, 2, 3},
    }

    compute_idf(G)
    return G


class TestEdgeSplit:
    def test_split_preserves_node_features(self):
        G = _make_test_graph()
        data, _ = convert_to_hetero_data(G)
        train_data, val_pos, val_neg = _split_edges(data)

        # Node features should be unchanged
        assert torch.equal(train_data["show"].x, data["show"].x)
        assert torch.equal(train_data["song"].x, data["song"].x)

    def test_split_produces_train_and_val(self):
        G = _make_test_graph()
        data, _ = convert_to_hetero_data(G)
        train_data, val_pos, val_neg = _split_edges(data)

        # At least one supervised edge type should have validation edges
        has_val = any(etype in val_pos for etype in SUPERVISED_EDGE_TYPES)
        assert has_val

    def test_val_pos_and_neg_same_size(self):
        G = _make_test_graph()
        data, _ = convert_to_hetero_data(G)
        _, val_pos, val_neg = _split_edges(data)

        for etype in val_pos:
            assert val_pos[etype].shape == val_neg[etype].shape


class TestTrainingLoop:
    def test_one_epoch_completes(self, monkeypatch, tmp_path):
        """Training loop completes one epoch on small fixture."""
        monkeypatch.setattr("deadrecs.train.MODEL_PATH", tmp_path / "model.pt")
        monkeypatch.setattr("deadrecs.train.EMBEDDINGS_PATH", tmp_path / "embeddings.pt")

        G = _make_test_graph()
        model, embeddings = train_model(G, epochs=1, hidden_dim=32, out_dim=32)

        assert model is not None
        assert len(embeddings) > 0

    def test_saves_model_and_embeddings(self, monkeypatch, tmp_path):
        model_path = tmp_path / "model.pt"
        emb_path = tmp_path / "embeddings.pt"
        monkeypatch.setattr("deadrecs.train.MODEL_PATH", model_path)
        monkeypatch.setattr("deadrecs.train.EMBEDDINGS_PATH", emb_path)

        G = _make_test_graph()
        train_model(G, epochs=2, hidden_dim=32, out_dim=32)

        assert model_path.exists()
        assert emb_path.exists()

    def test_embeddings_cover_all_node_types(self, monkeypatch, tmp_path):
        monkeypatch.setattr("deadrecs.train.MODEL_PATH", tmp_path / "model.pt")
        monkeypatch.setattr("deadrecs.train.EMBEDDINGS_PATH", tmp_path / "embeddings.pt")

        G = _make_test_graph()
        _, embeddings = train_model(G, epochs=1, hidden_dim=32, out_dim=32)

        # Should have embeddings for shows, songs, and performances
        has_show = any(k.startswith("show:") for k in embeddings)
        has_song = any(k.startswith("song:") for k in embeddings)
        has_perf = any(k.startswith("perf:") for k in embeddings)
        assert has_show
        assert has_song
        assert has_perf

    def test_embedding_dimensions(self, monkeypatch, tmp_path):
        monkeypatch.setattr("deadrecs.train.MODEL_PATH", tmp_path / "model.pt")
        monkeypatch.setattr("deadrecs.train.EMBEDDINGS_PATH", tmp_path / "embeddings.pt")

        G = _make_test_graph()
        _, embeddings = train_model(G, epochs=1, hidden_dim=32, out_dim=32)

        for key, emb in embeddings.items():
            assert emb.shape == (32,), f"Wrong shape for {key}: {emb.shape}"
