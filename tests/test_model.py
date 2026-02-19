"""Tests for PyG data conversion and GNN model."""

from __future__ import annotations

import torch
import networkx as nx
import pytest

from deadrecs.features import EMBEDDING_DIM, compute_idf
from deadrecs.model import (
    PERF_FEAT_DIM,
    SHOW_FEAT_DIM,
    SONG_FEAT_DIM,
    DeadRecsGNN,
    convert_to_hetero_data,
    _date_to_era_onehot,
)


def _make_test_graph() -> nx.DiGraph:
    """Build a small test graph with embeddings and IDF computed."""
    G = nx.DiGraph()

    # Songs
    G.add_node("song:1", type="song", name="Dark Star", headyversion_id=1, slug="dark-star")
    G.add_node("song:2", type="song", name="Truckin'", headyversion_id=2, slug="truckin")

    # Shows
    G.add_node("show:1972-08-27", type="show", date="1972-08-27", venue="Veneta, OR")
    G.add_node("show:1977-05-08", type="show", date="1977-05-08", venue="Ithaca, NY")

    # Performances
    G.add_node("perf:1:1972-08-27", type="performance", votes=135, description="Amazing",
               embedding=torch.randn(EMBEDDING_DIM))
    G.add_edge("show:1972-08-27", "perf:1:1972-08-27", type="HAS_PERFORMANCE")
    G.add_edge("perf:1:1972-08-27", "song:1", type="OF_SONG")

    G.add_node("perf:2:1972-08-27", type="performance", votes=50, description="Classic",
               embedding=torch.randn(EMBEDDING_DIM))
    G.add_edge("show:1972-08-27", "perf:2:1972-08-27", type="HAS_PERFORMANCE")
    G.add_edge("perf:2:1972-08-27", "song:2", type="OF_SONG")

    G.add_node("perf:1:1977-05-08", type="performance", votes=98, description="",
               embedding=torch.zeros(EMBEDDING_DIM))
    G.add_edge("show:1977-05-08", "perf:1:1977-05-08", type="HAS_PERFORMANCE")
    G.add_edge("perf:1:1977-05-08", "song:1", type="OF_SONG")

    # SETLIST_NEIGHBOR edge
    G.add_edge("show:1972-08-27", "show:1977-05-08", type="SETLIST_NEIGHBOR", weight=0.8)

    # TRANSITIONED_TO edge
    G.add_edge("song:1", "song:2", type="TRANSITIONED_TO")

    G.graph["show_songs"] = {
        "1972-08-27": {1, 2},
        "1977-05-08": {1},
    }

    compute_idf(G)
    return G


class TestDateToEra:
    def test_classic_era(self):
        vec = _date_to_era_onehot("1972-08-27")
        assert vec[1] == 1.0  # Classic era (1966-1974)
        assert sum(vec) == 1.0

    def test_comeback_era(self):
        vec = _date_to_era_onehot("1977-05-08")
        assert vec[3] == 1.0  # Comeback (1977-1979)

    def test_primal_era(self):
        vec = _date_to_era_onehot("1965-06-01")
        assert vec[0] == 1.0

    def test_final_era(self):
        vec = _date_to_era_onehot("1993-03-15")
        assert vec[5] == 1.0


class TestConvertToHeteroData:
    def test_produces_valid_heterodata(self):
        G = _make_test_graph()
        data, node_id_maps = convert_to_hetero_data(G)

        assert "show" in data.node_types
        assert "song" in data.node_types
        assert "performance" in data.node_types

    def test_node_counts(self):
        G = _make_test_graph()
        data, node_id_maps = convert_to_hetero_data(G)

        assert data["show"].x.shape[0] == 2
        assert data["song"].x.shape[0] == 2
        assert data["performance"].x.shape[0] == 3

    def test_feature_dimensions(self):
        G = _make_test_graph()
        data, _ = convert_to_hetero_data(G)

        assert data["show"].x.shape[1] == SHOW_FEAT_DIM
        assert data["song"].x.shape[1] == SONG_FEAT_DIM
        assert data["performance"].x.shape[1] == PERF_FEAT_DIM

    def test_edge_types_present(self):
        G = _make_test_graph()
        data, _ = convert_to_hetero_data(G)

        # Forward edges
        assert ("show", "has_performance", "performance") in data.edge_types
        assert ("performance", "of_song", "song") in data.edge_types
        assert ("show", "setlist_neighbor", "show") in data.edge_types
        assert ("song", "transitioned_to", "song") in data.edge_types
        # Reverse edges
        assert ("performance", "rev_has_performance", "show") in data.edge_types
        assert ("song", "rev_of_song", "performance") in data.edge_types
        assert ("song", "rev_transitioned_to", "song") in data.edge_types

    def test_edge_counts(self):
        G = _make_test_graph()
        data, _ = convert_to_hetero_data(G)

        # 3 HAS_PERFORMANCE edges + 3 reverse
        assert data["show", "has_performance", "performance"].edge_index.shape[1] == 3
        assert data["performance", "rev_has_performance", "show"].edge_index.shape[1] == 3
        # 3 OF_SONG edges + 3 reverse
        assert data["performance", "of_song", "song"].edge_index.shape[1] == 3
        assert data["song", "rev_of_song", "performance"].edge_index.shape[1] == 3
        # 1 SETLIST_NEIGHBOR edge (no separate reverse)
        assert data["show", "setlist_neighbor", "show"].edge_index.shape[1] == 1
        # 1 TRANSITIONED_TO edge + 1 reverse
        assert data["song", "transitioned_to", "song"].edge_index.shape[1] == 1
        assert data["song", "rev_transitioned_to", "song"].edge_index.shape[1] == 1

    def test_node_id_maps(self):
        G = _make_test_graph()
        _, node_id_maps = convert_to_hetero_data(G)

        assert len(node_id_maps["show"]) == 2
        assert len(node_id_maps["song"]) == 2
        assert len(node_id_maps["performance"]) == 3
        assert "show:1972-08-27" in node_id_maps["show"]

    def test_performance_features_normalized(self):
        """LayerNorm should be applied to performance features."""
        G = _make_test_graph()
        data, _ = convert_to_hetero_data(G)

        # After LayerNorm, features should have roughly zero mean per row
        perf_x = data["performance"].x
        row_means = perf_x.mean(dim=1)
        # LayerNorm centers each row — means should be near zero
        assert torch.allclose(row_means, torch.zeros_like(row_means), atol=1e-5)


class TestDeadRecsGNN:
    def test_forward_pass(self):
        """Model forward pass runs without errors on small graph."""
        G = _make_test_graph()
        data, _ = convert_to_hetero_data(G)

        model = DeadRecsGNN(hidden_dim=32, out_dim=32)
        model.eval()

        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)

        assert "show" in out
        assert "song" in out
        assert "performance" in out
        assert out["show"].shape == (2, 32)
        assert out["song"].shape == (2, 32)
        assert out["performance"].shape == (3, 32)

    def test_output_is_differentiable(self):
        G = _make_test_graph()
        data, _ = convert_to_hetero_data(G)

        model = DeadRecsGNN(hidden_dim=32, out_dim=32)
        out = model(data.x_dict, data.edge_index_dict)

        # Sum all outputs and backprop
        loss = sum(v.sum() for v in out.values())
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
