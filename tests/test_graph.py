"""Tests for graph construction."""

from __future__ import annotations

import json
import pickle

import networkx as nx
import pytest

from deadrecs.graph import build_graph, load_graph, save_graph, _add_transition_edges


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SONGS_INDEX = [
    {"id": 1, "name": "Dark Star", "slug": "dark-star", "url": "/song/1/grateful-dead/dark-star/"},
    {"id": 2, "name": "Truckin'", "slug": "truckin", "url": "/song/2/grateful-dead/truckin/"},
    {
        "id": 3,
        "name": "China Cat Sunflower -> I Know You Rider",
        "slug": "china-cat-sunflower-i-know-you-rider",
        "url": "/song/3/grateful-dead/china-cat-sunflower-i-know-you-rider/",
        "transitions": ["China Cat Sunflower", "I Know You Rider"],
    },
    {"id": 4, "name": "China Cat Sunflower", "slug": "china-cat-sunflower", "url": "/song/4/grateful-dead/china-cat-sunflower/"},
    {"id": 5, "name": "I Know You Rider", "slug": "i-know-you-rider", "url": "/song/5/grateful-dead/i-know-you-rider/"},
]

SONG_1_DATA = {
    "song_id": 1,
    "name": "Dark Star",
    "performances": [
        {
            "votes": 135,
            "date": "1972-08-27",
            "show_id": 100,
            "venue": "Old Renaissance Faire Grounds Veneta, OR",
            "description": "The greatest Dark Star ever played",
            "archive_url": "https://headyversion.com/show/100/archive/",
        },
        {
            "votes": 98,
            "date": "1974-02-24",
            "show_id": 101,
            "venue": "Winterland Arena San Francisco, CA",
            "description": "Incredible wall of sound Dark Star",
            "archive_url": "https://headyversion.com/show/101/archive/",
        },
    ],
}

SONG_2_DATA = {
    "song_id": 2,
    "name": "Truckin'",
    "performances": [
        {
            "votes": 50,
            "date": "1972-08-27",
            "show_id": 100,
            "venue": "Old Renaissance Faire Grounds Veneta, OR",
            "description": "Classic Truckin'",
            "archive_url": "",
        },
        {
            "votes": 30,
            "date": "1977-05-08",
            "show_id": 200,
            "venue": "Barton Hall Ithaca, NY",
            "description": "",
            "archive_url": "",
        },
    ],
}

SONG_3_DATA = {
    "song_id": 3,
    "name": "China Cat Sunflower -> I Know You Rider",
    "performances": [
        {
            "votes": 200,
            "date": "1977-05-08",
            "show_id": 200,
            "venue": "Barton Hall Ithaca, NY",
            "description": "Definitive China>Rider",
            "archive_url": "",
        },
    ],
}

SONG_4_DATA = {
    "song_id": 4,
    "name": "China Cat Sunflower",
    "performances": [],
}

SONG_5_DATA = {
    "song_id": 5,
    "name": "I Know You Rider",
    "performances": [],
}


@pytest.fixture
def data_dir(tmp_path):
    """Create a temporary data directory with fixture song files."""
    songs_dir = tmp_path / "data" / "raw" / "songs"
    songs_dir.mkdir(parents=True)

    # Write songs index
    index_path = tmp_path / "data" / "raw" / "songs_index.json"
    with open(index_path, "w") as f:
        json.dump(SONGS_INDEX, f)

    # Write song data files
    for song_data in [SONG_1_DATA, SONG_2_DATA, SONG_3_DATA, SONG_4_DATA, SONG_5_DATA]:
        slug = next(s["slug"] for s in SONGS_INDEX if s["id"] == song_data["song_id"])
        path = songs_dir / f"{song_data['song_id']}_{slug}.json"
        with open(path, "w") as f:
            json.dump(song_data, f)

    return tmp_path


@pytest.fixture
def graph(data_dir, monkeypatch):
    """Build a test graph from fixture data."""
    monkeypatch.setattr("deadrecs.graph.SONGS_INDEX_PATH", data_dir / "data" / "raw" / "songs_index.json")
    monkeypatch.setattr("deadrecs.graph.SONGS_DIR", data_dir / "data" / "raw" / "songs")
    return build_graph()


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_creates_song_nodes(self, graph):
        song_nodes = [n for n, d in graph.nodes(data=True) if d["type"] == "song"]
        assert len(song_nodes) == 5

    def test_creates_show_nodes(self, graph):
        show_nodes = [n for n, d in graph.nodes(data=True) if d["type"] == "show"]
        # 3 unique dates: 1972-08-27, 1974-02-24, 1977-05-08
        assert len(show_nodes) == 3

    def test_creates_performance_nodes(self, graph):
        perf_nodes = [n for n, d in graph.nodes(data=True) if d["type"] == "performance"]
        # 2 (dark star) + 2 (truckin) + 1 (china>rider) = 5
        assert len(perf_nodes) == 5

    def test_show_date_deduplication(self, graph):
        """1972-08-27 appears in both Dark Star and Truckin — should be one node."""
        show_1972 = graph.nodes["show:1972-08-27"]
        assert show_1972["date"] == "1972-08-27"
        assert show_1972["venue"] == "Old Renaissance Faire Grounds Veneta, OR"

    def test_has_performance_edges(self, graph):
        hp_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["type"] == "HAS_PERFORMANCE"]
        assert len(hp_edges) == 5

    def test_of_song_edges(self, graph):
        os_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["type"] == "OF_SONG"]
        assert len(os_edges) == 5

    def test_transition_edges(self, graph):
        trans_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["type"] == "TRANSITIONED_TO"]
        # China Cat Sunflower -> I Know You Rider
        assert len(trans_edges) == 1
        u, v = trans_edges[0]
        assert u == "song:4"  # China Cat Sunflower
        assert v == "song:5"  # I Know You Rider

    def test_performance_node_attributes(self, graph):
        perf = graph.nodes["perf:1:1972-08-27"]
        assert perf["votes"] == 135
        assert perf["description"] == "The greatest Dark Star ever played"

    def test_song_node_attributes(self, graph):
        song = graph.nodes["song:1"]
        assert song["name"] == "Dark Star"
        assert song["headyversion_id"] == 1

    def test_show_songs_stored(self, graph):
        show_songs = graph.graph["show_songs"]
        assert 1 in show_songs["1972-08-27"]  # Dark Star
        assert 2 in show_songs["1972-08-27"]  # Truckin'
        assert 2 in show_songs["1977-05-08"]  # Truckin'
        assert 3 in show_songs["1977-05-08"]  # China>Rider


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_save_and_load(self, graph, tmp_path):
        path = tmp_path / "test_graph.pickle"
        save_graph(graph, path)
        loaded = load_graph(path)

        assert loaded.number_of_nodes() == graph.number_of_nodes()
        assert loaded.number_of_edges() == graph.number_of_edges()
        assert loaded.nodes["song:1"]["name"] == "Dark Star"


# ---------------------------------------------------------------------------
# Transition Edges
# ---------------------------------------------------------------------------


class TestTransitionEdges:
    def test_no_transition_without_combo(self):
        G = nx.DiGraph()
        songs = [
            {"id": 1, "name": "Dark Star", "slug": "dark-star", "url": "/song/1/grateful-dead/dark-star/"},
        ]
        G.add_node("song:1", type="song", name="Dark Star", headyversion_id=1, slug="dark-star")
        _add_transition_edges(G, songs)
        trans = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "TRANSITIONED_TO"]
        assert len(trans) == 0

    def test_unresolvable_transition_skipped(self):
        G = nx.DiGraph()
        songs = [
            {
                "id": 10,
                "name": "A -> B",
                "slug": "a-b",
                "url": "/song/10/grateful-dead/a-b/",
                "transitions": ["A", "B"],
            },
        ]
        G.add_node("song:10", type="song", name="A -> B", headyversion_id=10, slug="a-b")
        _add_transition_edges(G, songs)
        trans = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "TRANSITIONED_TO"]
        # "A" and "B" are not standalone songs → no edges
        assert len(trans) == 0
