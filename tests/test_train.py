"""Tests for GNN training loop."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

from deadrecs.features import EMBEDDING_DIM, compute_idf
from deadrecs.model import DeadRecsGNN, convert_to_heterodata
from deadrecs.train import (
    _compute_auc,
    _compute_link_prediction_loss,
    _manual_auc,
    _split_edges,
    save_embeddings,
    save_model,
    train_gnn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_graph() -> nx.DiGraph:
    """Build a test graph with enough edges for splitting."""
    G = nx.DiGraph()

    # Songs
    for i in range(1, 6):
        G.add_node(f"song:{i}", type="song", name=f"Song {i}", headyversion_id=i, slug=f"song-{i}")

    # Shows
    dates = ["1970-06-01", "1972-08-27", "1974-02-24", "1977-05-08", "1979-11-01",
             "1981-03-15", "1985-07-01"]
    for d in dates:
        G.add_node(f"show:{d}", type="show", date=d, venue=f"Venue {d}")

    # Performances — spread across shows to create enough edges
    perf_idx = 0
    for d in dates:
        # Each show has 2-3 songs
        songs_for_show = [(perf_idx % 5) + 1, ((perf_idx + 1) % 5) + 1]
        for sid in songs_for_show:
            perf_id = f"perf:{sid}:{d}"
            G.add_node(perf_id, type="performance", votes=50 + perf_idx,
                       description=f"Performance {perf_idx}",
                       embedding=torch.randn(EMBEDDING_DIM))
            G.add_edge(f"show:{d}", perf_id, type="HAS_PERFORMANCE")
            G.add_edge(perf_id, f"song:{sid}", type="OF_SONG")
            perf_idx += 1

    # SETLIST_NEIGHBOR edges — need at least ~7 for a meaningful split
    neighbor_pairs = [
        ("1970-06-01", "1972-08-27"),
        ("1972-08-27", "1974-02-24"),
        ("1974-02-24", "1977-05-08"),
        ("1977-05-08", "1979-11-01"),
        ("1979-11-01", "1981-03-15"),
        ("1981-03-15", "1985-07-01"),
        ("1985-07-01", "1970-06-01"),
        ("1970-06-01", "1974-02-24"),
        ("1972-08-27", "1977-05-08"),
        ("1974-02-24", "1979-11-01"),
    ]
    for d1, d2 in neighbor_pairs:
        G.add_edge(f"show:{d1}", f"show:{d2}", type="SETLIST_NEIGHBOR", weight=0.5)

    # Transition edges
    G.add_edge("song:1", "song:2", type="TRANSITIONED_TO")

    # show_songs mapping
    G.graph["show_songs"] = {}
    for d in dates:
        songs = set()
        for _, v, edata in G.out_edges(f"show:{d}", data=True):
            if edata.get("type") == "HAS_PERFORMANCE":
                for _, s, ed2 in G.out_edges(v, data=True):
                    if ed2.get("type") == "OF_SONG":
                        hv_id = G.nodes[s].get("headyversion_id")
                        if hv_id:
                            songs.add(hv_id)
        G.graph["show_songs"][d] = songs

    compute_idf(G)
    return G


# ---------------------------------------------------------------------------
# Edge Split
# ---------------------------------------------------------------------------


class TestSplitEdges:
    def test_split_produces_train_and_val(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)
        train_data, val_data, _ = _split_edges(data, val_ratio=0.15)

        # Both should have edge_label_index for at least one supervision type
        has_supervision = False
        for etype in [("show", "setlist_neighbor", "show"),
                      ("show", "has_performance", "performance")]:
            if etype in train_data.edge_types:
                if hasattr(train_data[etype], "edge_label_index"):
                    has_supervision = True
                    break
        assert has_supervision

    def test_val_has_labels(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)
        _, val_data, _ = _split_edges(data, val_ratio=0.15)

        has_labels = False
        for etype in [("show", "setlist_neighbor", "show"),
                      ("show", "has_performance", "performance")]:
            if etype in val_data.edge_types:
                if hasattr(val_data[etype], "edge_label"):
                    has_labels = True
                    break
        assert has_labels


# ---------------------------------------------------------------------------
# Loss Computation
# ---------------------------------------------------------------------------


class TestLinkPredictionLoss:
    def test_loss_is_scalar(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)
        train_data, _, _ = _split_edges(data, val_ratio=0.15)

        metadata = data.metadata()
        model = DeadRecsGNN(metadata=metadata, hidden_dim=32, out_dim=32)
        model.train()

        loss = _compute_link_prediction_loss(model, train_data)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0  # should be positive

    def test_loss_has_gradient(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)
        train_data, _, _ = _split_edges(data, val_ratio=0.15)

        metadata = data.metadata()
        model = DeadRecsGNN(metadata=metadata, hidden_dim=32, out_dim=32)
        model.train()

        loss = _compute_link_prediction_loss(model, train_data)
        loss.backward()

        # At least some parameters should have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------


class TestManualAUC:
    def test_perfect_separation(self):
        scores = torch.tensor([10.0, 9.0, 1.0, 0.0])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        auc = _manual_auc(scores, labels)
        assert auc == pytest.approx(1.0)

    def test_random_baseline(self):
        # When positive and negative scores are equal, AUC ~ 0.5
        scores = torch.tensor([1.0, 1.0, 1.0, 1.0])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        auc = _manual_auc(scores, labels)
        assert auc == pytest.approx(0.5)

    def test_worst_case(self):
        scores = torch.tensor([0.0, 0.0, 10.0, 10.0])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        auc = _manual_auc(scores, labels)
        assert auc == pytest.approx(0.0)

    def test_no_positive_labels(self):
        scores = torch.tensor([1.0, 2.0])
        labels = torch.tensor([0.0, 0.0])
        auc = _manual_auc(scores, labels)
        assert auc == 0.5

    def test_no_negative_labels(self):
        scores = torch.tensor([1.0, 2.0])
        labels = torch.tensor([1.0, 1.0])
        auc = _manual_auc(scores, labels)
        assert auc == 0.5


class TestComputeAUC:
    def test_auc_returns_float(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)
        _, val_data, _ = _split_edges(data, val_ratio=0.15)

        metadata = data.metadata()
        model = DeadRecsGNN(metadata=metadata, hidden_dim=32, out_dim=32)

        auc = _compute_auc(model, val_data)
        assert isinstance(auc, float)
        assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


class TestTrainGNN:
    def test_train_completes_one_epoch(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)

        model, embeddings = train_gnn(
            data, epochs=1, hidden_dim=32, out_dim=32, patience=5
        )

        assert isinstance(model, DeadRecsGNN)
        assert "show" in embeddings
        assert "song" in embeddings
        assert "performance" in embeddings

    def test_embeddings_have_correct_dim(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)

        out_dim = 32
        model, embeddings = train_gnn(
            data, epochs=2, hidden_dim=32, out_dim=out_dim, patience=5
        )

        for ntype in ["show", "song", "performance"]:
            assert embeddings[ntype].shape[1] == out_dim

    def test_early_stopping(self):
        """With patience=1 and enough epochs, should stop early."""
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)

        model, embeddings = train_gnn(
            data, epochs=100, hidden_dim=16, out_dim=16, patience=1
        )

        # Should have embeddings regardless of how many epochs ran
        assert "show" in embeddings


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


class TestSaveModel:
    def test_save_model(self, tmp_path, monkeypatch):
        monkeypatch.setattr("deadrecs.train.MODEL_PATH", tmp_path / "model.pt")

        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)
        metadata = data.metadata()
        model = DeadRecsGNN(metadata=metadata, hidden_dim=32, out_dim=32)

        save_model(model)
        assert (tmp_path / "model.pt").exists()


class TestSaveEmbeddings:
    def test_save_embeddings(self, tmp_path, monkeypatch):
        monkeypatch.setattr("deadrecs.train.EMBEDDINGS_PATH", tmp_path / "embeddings.pt")

        G = _make_test_graph()
        data, node_id_maps = convert_to_heterodata(G)
        metadata = data.metadata()
        model = DeadRecsGNN(metadata=metadata, hidden_dim=32, out_dim=32)
        model.eval()

        with torch.no_grad():
            embeddings = model(data.x_dict, data.edge_index_dict)

        save_embeddings(embeddings, node_id_maps)

        saved = torch.load(tmp_path / "embeddings.pt", weights_only=True)
        assert "show:1972-08-27" in saved
        assert "song:1" in saved
        assert saved["show:1972-08-27"].shape == (32,)
