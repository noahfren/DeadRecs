"""Tests for the recommendation engine."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

from deadrecs.recommend import (
    _is_date,
    _resolve_performance,
    _resolve_show,
    _resolve_song,
    parse_inputs,
    recommend,
)


def _make_test_graph() -> nx.DiGraph:
    """Build a small test graph for recommendation tests."""
    G = nx.DiGraph()

    # Songs
    G.add_node("song:1", type="song", name="Dark Star", headyversion_id=1, slug="dark-star")
    G.add_node("song:2", type="song", name="Playing in the Band", headyversion_id=2, slug="playing-in-the-band")
    G.add_node("song:3", type="song", name="Scarlet Begonias", headyversion_id=3, slug="scarlet-begonias")

    # Shows
    G.add_node("show:1972-08-27", type="show", date="1972-08-27", venue="Veneta, OR")
    G.add_node("show:1977-05-08", type="show", date="1977-05-08", venue="Ithaca, NY")
    G.add_node("show:1977-05-09", type="show", date="1977-05-09", venue="Buffalo, NY")

    # Performances
    G.add_node("perf:1:1972-08-27", type="performance", votes=135, description="Amazing", archive_url="https://archive.org/details/gd1972-08-27")
    G.add_edge("show:1972-08-27", "perf:1:1972-08-27", type="HAS_PERFORMANCE")
    G.add_edge("perf:1:1972-08-27", "song:1", type="OF_SONG")

    G.add_node("perf:2:1977-05-08", type="performance", votes=98, description="Great", archive_url="https://archive.org/details/gd1977-05-08")
    G.add_edge("show:1977-05-08", "perf:2:1977-05-08", type="HAS_PERFORMANCE")
    G.add_edge("perf:2:1977-05-08", "song:2", type="OF_SONG")

    G.add_node("perf:3:1977-05-09", type="performance", votes=80, description="Solid", archive_url="")
    G.add_edge("show:1977-05-09", "perf:3:1977-05-09", type="HAS_PERFORMANCE")
    G.add_edge("perf:3:1977-05-09", "song:3", type="OF_SONG")

    return G


def _make_mock_embeddings(dim: int = 32) -> dict[str, torch.Tensor]:
    """Create mock embeddings for all nodes in the test graph."""
    torch.manual_seed(42)
    ids = [
        "show:1972-08-27", "show:1977-05-08", "show:1977-05-09",
        "song:1", "song:2", "song:3",
        "perf:1:1972-08-27", "perf:2:1977-05-08", "perf:3:1977-05-09",
    ]
    return {nid: torch.randn(dim) for nid in ids}


class TestIsDate:
    def test_full_date(self):
        assert _is_date("1977-05-08")

    def test_year_month(self):
        assert _is_date("1977-05")

    def test_year_only(self):
        assert _is_date("1977")

    def test_song_name(self):
        assert not _is_date("Dark Star")

    def test_partial_string(self):
        assert not _is_date("may 1977")


class TestResolveShow:
    def test_exact_match(self):
        G = _make_test_graph()
        result = _resolve_show(G, "1977-05-08")
        assert result == ["show:1977-05-08"]

    def test_partial_year_month(self):
        G = _make_test_graph()
        result = _resolve_show(G, "1977-05")
        assert set(result) == {"show:1977-05-08", "show:1977-05-09"}

    def test_partial_year(self):
        G = _make_test_graph()
        result = _resolve_show(G, "1977")
        assert set(result) == {"show:1977-05-08", "show:1977-05-09"}

    def test_no_match(self):
        G = _make_test_graph()
        result = _resolve_show(G, "1999-01-01")
        assert result == []


class TestResolveSong:
    def test_exact_match(self):
        G = _make_test_graph()
        result = _resolve_song(G, "Dark Star")
        assert result == ["song:1"]

    def test_case_insensitive(self):
        G = _make_test_graph()
        result = _resolve_song(G, "dark star")
        assert result == ["song:1"]

    def test_substring_match(self):
        G = _make_test_graph()
        result = _resolve_song(G, "Scarlet")
        assert result == ["song:3"]

    def test_no_match(self):
        G = _make_test_graph()
        result = _resolve_song(G, "Truckin")
        assert result == []


class TestResolvePerformance:
    def test_exact_match(self):
        G = _make_test_graph()
        result = _resolve_performance(G, "Dark Star", "1972-08-27")
        assert result == "perf:1:1972-08-27"

    def test_substring_song_match(self):
        G = _make_test_graph()
        result = _resolve_performance(G, "Scarlet", "1977-05-09")
        assert result == "perf:3:1977-05-09"

    def test_wrong_date(self):
        G = _make_test_graph()
        result = _resolve_performance(G, "Dark Star", "1999-01-01")
        assert result is None

    def test_wrong_song(self):
        G = _make_test_graph()
        result = _resolve_performance(G, "Truckin", "1972-08-27")
        assert result is None


class TestParseInputs:
    def test_date_input(self):
        G = _make_test_graph()
        result = parse_inputs(G, ("1977-05-08",))
        assert result == ["show:1977-05-08"]

    def test_song_input(self):
        G = _make_test_graph()
        result = parse_inputs(G, ("Dark Star",))
        assert result == ["song:1"]

    def test_mixed_inputs(self):
        G = _make_test_graph()
        result = parse_inputs(G, ("1977-05-08", "Dark Star"))
        assert result == ["show:1977-05-08", "song:1"]

    def test_unrecognized_date_raises(self):
        G = _make_test_graph()
        with pytest.raises(Exception, match="No show found"):
            parse_inputs(G, ("1999-01-01",))

    def test_unrecognized_song_raises(self):
        G = _make_test_graph()
        with pytest.raises(Exception, match="No song found"):
            parse_inputs(G, ("Nonexistent Song",))

    def test_performance_input(self):
        G = _make_test_graph()
        result = parse_inputs(G, ("Dark Star @ 1972-08-27",))
        assert result == ["perf:1:1972-08-27"]

    def test_performance_not_found_raises(self):
        G = _make_test_graph()
        with pytest.raises(Exception, match="No performance found"):
            parse_inputs(G, ("Dark Star @ 1999-01-01",))

    def test_performance_bad_date_raises(self):
        G = _make_test_graph()
        with pytest.raises(Exception, match="Invalid date"):
            parse_inputs(G, ("Dark Star @ not-a-date",))

    def test_mixed_with_performance(self):
        G = _make_test_graph()
        result = parse_inputs(G, ("1977-05-08", "Dark Star @ 1972-08-27"))
        assert result == ["show:1977-05-08", "perf:1:1972-08-27"]


class TestRecommend:
    def _setup_test_data(self, monkeypatch, tmp_path):
        """Save test graph and embeddings, patch paths."""
        import pickle

        G = _make_test_graph()
        embeddings = _make_mock_embeddings()

        graph_path = tmp_path / "graph.pickle"
        emb_path = tmp_path / "embeddings.pt"
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        torch.save(embeddings, emb_path)

        monkeypatch.setattr("deadrecs.recommend.GRAPH_PATH", graph_path)
        monkeypatch.setattr("deadrecs.recommend.EMBEDDINGS_PATH", emb_path)

    def test_recommend_by_show(self, monkeypatch, tmp_path):
        self._setup_test_data(monkeypatch, tmp_path)
        recommend(like_args=("1972-08-27",), num_results=2)

    def test_recommend_by_song(self, monkeypatch, tmp_path):
        self._setup_test_data(monkeypatch, tmp_path)
        recommend(like_args=("Dark Star",), num_results=2)

    def test_recommend_by_performance(self, monkeypatch, tmp_path):
        self._setup_test_data(monkeypatch, tmp_path)
        recommend(like_args=("Dark Star @ 1972-08-27",), num_results=2)

    def test_recommend_mixed_inputs(self, monkeypatch, tmp_path):
        self._setup_test_data(monkeypatch, tmp_path)
        recommend(like_args=("1977-05-08", "Dark Star"), num_results=2)

    def test_missing_graph_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("deadrecs.recommend.GRAPH_PATH", tmp_path / "missing.pickle")
        monkeypatch.setattr("deadrecs.recommend.EMBEDDINGS_PATH", tmp_path / "missing.pt")

        with pytest.raises(Exception, match="No graph found"):
            recommend(like_args=("1977-05-08",))

    def test_missing_embeddings_raises(self, monkeypatch, tmp_path):
        import pickle
        G = _make_test_graph()
        graph_path = tmp_path / "graph.pickle"
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)

        monkeypatch.setattr("deadrecs.recommend.GRAPH_PATH", graph_path)
        monkeypatch.setattr("deadrecs.recommend.EMBEDDINGS_PATH", tmp_path / "missing.pt")

        with pytest.raises(Exception, match="No embeddings found"):
            recommend(like_args=("1977-05-08",))
