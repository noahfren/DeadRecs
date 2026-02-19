"""Tests for the info command."""

from __future__ import annotations

import pickle

import networkx as nx
import pytest

from deadrecs.info import run_info


def _make_test_graph() -> nx.DiGraph:
    """Build a small test graph for info tests."""
    G = nx.DiGraph()

    # Songs
    G.add_node("song:1", type="song", name="Dark Star", headyversion_id=1, slug="dark-star")
    G.add_node("song:2", type="song", name="Playing in the Band", headyversion_id=2, slug="playing-in-the-band")

    # Shows
    G.add_node("show:1972-08-27", type="show", date="1972-08-27", venue="Veneta, OR")
    G.add_node("show:1977-05-08", type="show", date="1977-05-08", venue="Ithaca, NY")

    # Performances
    G.add_node(
        "perf:1:1972-08-27", type="performance", votes=135,
        description="One of the greatest Dark Stars ever played.",
        archive_url="https://archive.org/details/gd1972-08-27",
    )
    G.add_edge("show:1972-08-27", "perf:1:1972-08-27", type="HAS_PERFORMANCE")
    G.add_edge("perf:1:1972-08-27", "song:1", type="OF_SONG")

    G.add_node(
        "perf:2:1977-05-08", type="performance", votes=98,
        description="Great show.",
        archive_url="https://archive.org/details/gd1977-05-08",
    )
    G.add_edge("show:1977-05-08", "perf:2:1977-05-08", type="HAS_PERFORMANCE")
    G.add_edge("perf:2:1977-05-08", "song:2", type="OF_SONG")

    # Dark Star also played on 1977-05-08
    G.add_node(
        "perf:1:1977-05-08", type="performance", votes=45,
        description="", archive_url="",
    )
    G.add_edge("show:1977-05-08", "perf:1:1977-05-08", type="HAS_PERFORMANCE")
    G.add_edge("perf:1:1977-05-08", "song:1", type="OF_SONG")

    return G


def _setup_graph(monkeypatch, tmp_path):
    """Save test graph and patch GRAPH_PATH."""
    G = _make_test_graph()
    graph_path = tmp_path / "graph.pickle"
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)
    monkeypatch.setattr("deadrecs.info.GRAPH_PATH", graph_path)
    return G


class TestShowInfo:
    def test_show_info_displays_date_and_venue(self, monkeypatch, tmp_path, capsys):
        _setup_graph(monkeypatch, tmp_path)
        run_info("1972-08-27")
        output = capsys.readouterr().out
        assert "1972-08-27" in output
        assert "Veneta, OR" in output

    def test_show_info_displays_performances(self, monkeypatch, tmp_path, capsys):
        _setup_graph(monkeypatch, tmp_path)
        run_info("1972-08-27")
        output = capsys.readouterr().out
        assert "Dark Star" in output
        assert "135 votes" in output

    def test_show_info_displays_description(self, monkeypatch, tmp_path, capsys):
        _setup_graph(monkeypatch, tmp_path)
        run_info("1972-08-27")
        output = capsys.readouterr().out
        assert "greatest Dark Stars" in output

    def test_show_info_displays_archive_url(self, monkeypatch, tmp_path, capsys):
        _setup_graph(monkeypatch, tmp_path)
        run_info("1972-08-27")
        output = capsys.readouterr().out
        assert "archive.org" in output

    def test_show_info_multiple_performances(self, monkeypatch, tmp_path, capsys):
        _setup_graph(monkeypatch, tmp_path)
        run_info("1977-05-08")
        output = capsys.readouterr().out
        assert "Playing in the Band" in output
        assert "Dark Star" in output
        assert "2 rated performance" in output

    def test_show_info_partial_date(self, monkeypatch, tmp_path, capsys):
        _setup_graph(monkeypatch, tmp_path)
        run_info("1977")
        output = capsys.readouterr().out
        assert "1977-05-08" in output

    def test_show_not_found(self, monkeypatch, tmp_path):
        _setup_graph(monkeypatch, tmp_path)
        with pytest.raises(Exception, match="No show found"):
            run_info("1999-01-01")


class TestSongInfo:
    def test_song_info_displays_name(self, monkeypatch, tmp_path, capsys):
        _setup_graph(monkeypatch, tmp_path)
        run_info("Dark Star")
        output = capsys.readouterr().out
        assert "Dark Star" in output

    def test_song_info_displays_performances_ranked(self, monkeypatch, tmp_path, capsys):
        _setup_graph(monkeypatch, tmp_path)
        run_info("Dark Star")
        output = capsys.readouterr().out
        assert "135 votes" in output
        assert "45 votes" in output
        # 135 votes should appear before 45 votes (sorted descending)
        pos_135 = output.index("135 votes")
        pos_45 = output.index("45 votes")
        assert pos_135 < pos_45

    def test_song_info_displays_show_dates(self, monkeypatch, tmp_path, capsys):
        _setup_graph(monkeypatch, tmp_path)
        run_info("Dark Star")
        output = capsys.readouterr().out
        assert "1972-08-27" in output
        assert "1977-05-08" in output

    def test_song_info_displays_venues(self, monkeypatch, tmp_path, capsys):
        _setup_graph(monkeypatch, tmp_path)
        run_info("Dark Star")
        output = capsys.readouterr().out
        assert "Veneta, OR" in output

    def test_song_not_found(self, monkeypatch, tmp_path):
        _setup_graph(monkeypatch, tmp_path)
        with pytest.raises(Exception, match="No song found"):
            run_info("Truckin")

    def test_song_case_insensitive(self, monkeypatch, tmp_path, capsys):
        _setup_graph(monkeypatch, tmp_path)
        run_info("dark star")
        output = capsys.readouterr().out
        assert "Dark Star" in output

    def test_song_substring_match(self, monkeypatch, tmp_path, capsys):
        _setup_graph(monkeypatch, tmp_path)
        run_info("Playing")
        output = capsys.readouterr().out
        assert "Playing in the Band" in output


class TestInfoErrorHandling:
    def test_missing_graph(self, monkeypatch, tmp_path):
        monkeypatch.setattr("deadrecs.info.GRAPH_PATH", tmp_path / "missing.pickle")
        with pytest.raises(Exception, match="No graph found"):
            run_info("1977-05-08")
