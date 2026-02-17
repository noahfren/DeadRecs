"""Tests for utility functions."""

from pathlib import Path

from deadrecs.utils import (
    DATA_DIR,
    EMBEDDINGS_PATH,
    GRAPH_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
    RAW_DIR,
    SONGS_DIR,
    SONGS_INDEX_PATH,
    ensure_data_dirs,
    song_path,
)


def test_project_root_exists():
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_data_paths_are_under_project_root():
    assert str(DATA_DIR).startswith(str(PROJECT_ROOT))
    assert str(RAW_DIR).startswith(str(DATA_DIR))
    assert str(SONGS_DIR).startswith(str(RAW_DIR))


def test_artifact_paths():
    assert SONGS_INDEX_PATH == RAW_DIR / "songs_index.json"
    assert GRAPH_PATH == DATA_DIR / "graph.pickle"
    assert MODEL_PATH == DATA_DIR / "model.pt"
    assert EMBEDDINGS_PATH == DATA_DIR / "embeddings.pt"


def test_song_path():
    p = song_path(58, "dark-star")
    assert p == SONGS_DIR / "58_dark-star.json"


def test_ensure_data_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr("deadrecs.utils.SONGS_DIR", tmp_path / "data" / "raw" / "songs")
    ensure_data_dirs()
    assert (tmp_path / "data" / "raw" / "songs").exists()
