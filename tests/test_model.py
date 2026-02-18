"""Tests for PyG data conversion and GNN model."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

from deadrecs.features import EMBEDDING_DIM, compute_idf
from deadrecs.model import (
    DeadRecsGNN,
    _era_onehot,
    convert_to_heterodata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_graph() -> nx.DiGraph:
    """Build a small test graph with IDF and embeddings set."""
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
    G.add_node("perf:1:1972-08-27", type="performance", votes=135, description="Amazing",
               embedding=torch.randn(EMBEDDING_DIM))
    G.add_edge("show:1972-08-27", "perf:1:1972-08-27", type="HAS_PERFORMANCE")
    G.add_edge("perf:1:1972-08-27", "song:1", type="OF_SONG")

    G.add_node("perf:2:1972-08-27", type="performance", votes=50, description="Classic",
               embedding=torch.randn(EMBEDDING_DIM))
    G.add_edge("show:1972-08-27", "perf:2:1972-08-27", type="HAS_PERFORMANCE")
    G.add_edge("perf:2:1972-08-27", "song:2", type="OF_SONG")

    # Performances for show B (1977-05-08): songs 2, 3
    G.add_node("perf:2:1977-05-08", type="performance", votes=30, description="",
               embedding=torch.zeros(EMBEDDING_DIM))
    G.add_edge("show:1977-05-08", "perf:2:1977-05-08", type="HAS_PERFORMANCE")
    G.add_edge("perf:2:1977-05-08", "song:2", type="OF_SONG")

    G.add_node("perf:3:1977-05-08", type="performance", votes=80, description="Great Eyes",
               embedding=torch.randn(EMBEDDING_DIM))
    G.add_edge("show:1977-05-08", "perf:3:1977-05-08", type="HAS_PERFORMANCE")
    G.add_edge("perf:3:1977-05-08", "song:3", type="OF_SONG")

    # Performances for show C (1974-02-24): songs 1, 2, 3
    G.add_node("perf:1:1974-02-24", type="performance", votes=98, description="Wall of sound",
               embedding=torch.randn(EMBEDDING_DIM))
    G.add_edge("show:1974-02-24", "perf:1:1974-02-24", type="HAS_PERFORMANCE")
    G.add_edge("perf:1:1974-02-24", "song:1", type="OF_SONG")

    G.add_node("perf:2:1974-02-24", type="performance", votes=40, description="",
               embedding=torch.zeros(EMBEDDING_DIM))
    G.add_edge("show:1974-02-24", "perf:2:1974-02-24", type="HAS_PERFORMANCE")
    G.add_edge("perf:2:1974-02-24", "song:2", type="OF_SONG")

    G.add_node("perf:3:1974-02-24", type="performance", votes=60, description="Smooth",
               embedding=torch.randn(EMBEDDING_DIM))
    G.add_edge("show:1974-02-24", "perf:3:1974-02-24", type="HAS_PERFORMANCE")
    G.add_edge("perf:3:1974-02-24", "song:3", type="OF_SONG")

    # SETLIST_NEIGHBOR edges
    G.add_edge("show:1972-08-27", "show:1974-02-24", type="SETLIST_NEIGHBOR", weight=0.6)
    G.add_edge("show:1977-05-08", "show:1974-02-24", type="SETLIST_NEIGHBOR", weight=0.5)
    G.add_edge("show:1974-02-24", "show:1972-08-27", type="SETLIST_NEIGHBOR", weight=0.6)
    G.add_edge("show:1974-02-24", "show:1977-05-08", type="SETLIST_NEIGHBOR", weight=0.5)

    # TRANSITIONED_TO edge
    G.add_edge("song:1", "song:2", type="TRANSITIONED_TO")

    G.graph["show_songs"] = {
        "1972-08-27": {1, 2},
        "1977-05-08": {2, 3},
        "1974-02-24": {1, 2, 3},
    }

    # Compute IDF
    compute_idf(G)

    return G


# ---------------------------------------------------------------------------
# Era One-Hot Encoding
# ---------------------------------------------------------------------------


class TestEraOnehot:
    def test_pre_1970(self):
        assert _era_onehot("1967-01-14") == [1, 0, 0, 0, 0, 0]

    def test_early_70s(self):
        assert _era_onehot("1972-08-27") == [0, 1, 0, 0, 0, 0]

    def test_mid_70s(self):
        assert _era_onehot("1977-05-08") == [0, 0, 1, 0, 0, 0]

    def test_late_70s(self):
        assert _era_onehot("1979-06-15") == [0, 0, 0, 1, 0, 0]

    def test_80s(self):
        assert _era_onehot("1985-07-01") == [0, 0, 0, 0, 1, 0]

    def test_90s(self):
        assert _era_onehot("1993-03-20") == [0, 0, 0, 0, 0, 1]


# ---------------------------------------------------------------------------
# PyG Conversion
# ---------------------------------------------------------------------------


class TestConvertToHeterodata:
    def test_produces_valid_heterodata(self):
        G = _make_test_graph()
        data, node_id_maps = convert_to_heterodata(G)

        assert "show" in data.node_types
        assert "song" in data.node_types
        assert "performance" in data.node_types

    def test_node_counts(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)

        assert data["show"].x.shape[0] == 3
        assert data["song"].x.shape[0] == 3
        assert data["performance"].x.shape[0] == 7

    def test_show_feature_dimensions(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)

        # Show features: [num_perfs, mean_vote, era_onehot(6)] = 8
        assert data["show"].x.shape[1] == 8

    def test_song_feature_dimensions(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)

        # Song features: [idf, total_votes, num_performances] = 3
        assert data["song"].x.shape[1] == 3

    def test_performance_feature_dimensions(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)

        # Perf features: [norm_votes, rank, embedding(384)] = 386
        assert data["performance"].x.shape[1] == 2 + EMBEDDING_DIM

    def test_edge_types_present(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)

        edge_types = data.edge_types
        assert ("show", "has_performance", "performance") in edge_types
        assert ("performance", "of_song", "song") in edge_types
        assert ("show", "setlist_neighbor", "show") in edge_types
        assert ("song", "transitioned_to", "song") in edge_types

    def test_has_performance_edge_count(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)

        ei = data[("show", "has_performance", "performance")].edge_index
        assert ei.shape[1] == 7  # 2 + 2 + 3

    def test_of_song_edge_count(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)

        ei = data[("performance", "of_song", "song")].edge_index
        assert ei.shape[1] == 7

    def test_setlist_neighbor_edge_count(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)

        ei = data[("show", "setlist_neighbor", "show")].edge_index
        assert ei.shape[1] == 4

    def test_transition_edge_count(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)

        ei = data[("song", "transitioned_to", "song")].edge_index
        assert ei.shape[1] == 1

    def test_node_id_maps_correct(self):
        G = _make_test_graph()
        _, node_id_maps = convert_to_heterodata(G)

        assert len(node_id_maps["show"]) == 3
        assert len(node_id_maps["song"]) == 3
        assert len(node_id_maps["performance"]) == 7

    def test_show_features_values(self):
        G = _make_test_graph()
        data, node_id_maps = convert_to_heterodata(G)

        # Show 1972-08-27 has 2 performances with votes 135, 50
        idx = node_id_maps["show"]["show:1972-08-27"]
        features = data["show"].x[idx]
        assert features[0].item() == 2.0  # num_perfs
        assert features[1].item() == pytest.approx(92.5)  # mean vote
        # Era: early_70s -> index 1
        assert features[3].item() == 1.0

    def test_song_features_have_idf(self):
        G = _make_test_graph()
        data, node_id_maps = convert_to_heterodata(G)

        # Song 1 should have non-zero IDF
        idx = node_id_maps["song"]["song:1"]
        idf_val = data["song"].x[idx, 0].item()
        assert idf_val > 0

    def test_empty_graph(self):
        G = nx.DiGraph()
        G.graph["show_songs"] = {}
        data, node_id_maps = convert_to_heterodata(G)

        # Should produce empty tensors
        assert data["show"].x.shape[0] == 0
        assert data["song"].x.shape[0] == 0
        assert data["performance"].x.shape[0] == 0


# ---------------------------------------------------------------------------
# GNN Model
# ---------------------------------------------------------------------------


class TestDeadRecsGNN:
    def test_forward_pass(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)
        metadata = data.metadata()

        model = DeadRecsGNN(metadata=metadata, hidden_dim=64, out_dim=64)
        model.eval()

        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)

        assert "show" in out
        assert "song" in out
        assert "performance" in out

    def test_output_dimensions(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)
        metadata = data.metadata()

        out_dim = 64
        model = DeadRecsGNN(metadata=metadata, hidden_dim=32, out_dim=out_dim)
        model.eval()

        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)

        assert out["show"].shape == (3, out_dim)
        assert out["song"].shape == (3, out_dim)
        assert out["performance"].shape == (7, out_dim)

    def test_training_mode_has_dropout(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)
        metadata = data.metadata()

        model = DeadRecsGNN(metadata=metadata, hidden_dim=64, out_dim=64, dropout=0.5)
        model.train()

        # Run twice in training mode — outputs should differ due to dropout
        out1 = model(data.x_dict, data.edge_index_dict)
        out2 = model(data.x_dict, data.edge_index_dict)

        # At least one node type should have different outputs
        any_different = False
        for ntype in out1:
            if not torch.allclose(out1[ntype], out2[ntype]):
                any_different = True
                break
        assert any_different

    def test_default_dimensions(self):
        G = _make_test_graph()
        data, _ = convert_to_heterodata(G)
        metadata = data.metadata()

        model = DeadRecsGNN(metadata=metadata)
        model.eval()

        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)

        # Default out_dim = 128
        assert out["show"].shape[1] == 128
