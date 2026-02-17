"""Utility functions for data path resolution and directory management."""

from pathlib import Path

# Project root is the parent of the deadrecs package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SONGS_DIR = RAW_DIR / "songs"

SONGS_INDEX_PATH = RAW_DIR / "songs_index.json"
GRAPH_PATH = DATA_DIR / "graph.pickle"
DESCRIPTION_EMBEDDINGS_PATH = DATA_DIR / "description_embeddings.pt"
MODEL_PATH = DATA_DIR / "model.pt"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.pt"


def ensure_data_dirs() -> None:
    """Create the data directory structure if it doesn't exist."""
    SONGS_DIR.mkdir(parents=True, exist_ok=True)


def song_path(song_id: int, slug: str) -> Path:
    """Return the file path for a song's scraped data."""
    return SONGS_DIR / f"{song_id}_{slug}.json"
