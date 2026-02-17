"""Graph construction from scraped Headyversion data.

Builds a NetworkX graph with typed nodes (Show, Song, SongPerformance)
and typed edges (HAS_PERFORMANCE, OF_SONG, TRANSITIONED_TO, SETLIST_NEIGHBOR).
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import click
import networkx as nx

from deadrecs.utils import GRAPH_PATH, RAW_DIR, SONGS_DIR, SONGS_INDEX_PATH

logger = logging.getLogger(__name__)


def load_songs_index() -> list[dict]:
    """Load the song index from disk."""
    with open(SONGS_INDEX_PATH) as f:
        return json.load(f)


def load_song_performances(song_id: int, slug: str) -> dict:
    """Load a single song's performance data from disk."""
    path = SONGS_DIR / f"{song_id}_{slug}.json"
    with open(path) as f:
        return json.load(f)


def build_graph(songs_index: list[dict] | None = None) -> nx.DiGraph:
    """Build the graph from scraped data on disk.

    Creates three node types (Show, Song, SongPerformance) and three edge
    types (HAS_PERFORMANCE, OF_SONG, TRANSITIONED_TO).

    SETLIST_NEIGHBOR edges are added later by features.add_setlist_neighbor_edges().
    """
    G = nx.DiGraph()

    if songs_index is None:
        songs_index = load_songs_index()

    # Track shows by date for deduplication — pick the first venue seen.
    show_venues: dict[str, str] = {}
    # Track which songs appear at which shows (for IDF later).
    show_songs: dict[str, set[int]] = defaultdict(set)

    click.echo(f"Building graph from {len(songs_index)} songs...")

    for song_entry in songs_index:
        song_id = song_entry["id"]
        slug = song_entry["slug"]
        name = song_entry["name"]

        # Add Song node
        song_node_id = f"song:{song_id}"
        G.add_node(
            song_node_id,
            type="song",
            name=name,
            headyversion_id=song_id,
            slug=slug,
        )

        # Add TRANSITIONED_TO edges for combo songs
        if "transitions" in song_entry:
            transitions = song_entry["transitions"]
            for i in range(len(transitions) - 1):
                # These are name-based references; we resolve them after all
                # songs are added.
                pass

        # Load performances
        perf_path = SONGS_DIR / f"{song_id}_{slug}.json"
        if not perf_path.exists():
            logger.warning("No performance data for %s (%s)", name, perf_path)
            continue

        song_data = load_song_performances(song_id, slug)

        for perf in song_data.get("performances", []):
            date = perf.get("date")
            if not date:
                continue

            venue = perf.get("venue", "")
            votes = perf.get("votes", 0)
            description = perf.get("description", "")
            archive_url = perf.get("archive_url", "")

            # Add Show node (deduplicated by date)
            show_node_id = f"show:{date}"
            if show_node_id not in G:
                G.add_node(
                    show_node_id,
                    type="show",
                    date=date,
                    venue=venue,
                )
                show_venues[date] = venue

            # Track song-show association
            show_songs[date].add(song_id)

            # Add SongPerformance node
            perf_node_id = f"perf:{song_id}:{date}"
            G.add_node(
                perf_node_id,
                type="performance",
                votes=votes,
                description=description,
                archive_url=archive_url,
            )

            # HAS_PERFORMANCE: Show -> SongPerformance
            G.add_edge(show_node_id, perf_node_id, type="HAS_PERFORMANCE")

            # OF_SONG: SongPerformance -> Song
            G.add_edge(perf_node_id, song_node_id, type="OF_SONG")

    # Add TRANSITIONED_TO edges (Song -> Song)
    _add_transition_edges(G, songs_index)

    # Store show_songs mapping as graph attribute for later use
    G.graph["show_songs"] = dict(show_songs)

    _print_graph_stats(G)
    return G


def _add_transition_edges(G: nx.DiGraph, songs_index: list[dict]) -> None:
    """Add TRANSITIONED_TO edges between songs that form combos.

    For combo songs like "China Cat Sunflower -> I Know You Rider", we
    create directed edges between the individual song nodes, matching
    by name to find the correct song nodes.
    """
    # Build name-to-node lookup
    name_to_node: dict[str, str] = {}
    for song in songs_index:
        name_to_node[song["name"].lower()] = f"song:{song['id']}"

    for song in songs_index:
        if "transitions" not in song:
            continue
        parts = song["transitions"]
        for i in range(len(parts) - 1):
            src_name = parts[i].lower()
            dst_name = parts[i + 1].lower()
            src_node = name_to_node.get(src_name)
            dst_node = name_to_node.get(dst_name)
            if src_node and dst_node:
                G.add_edge(src_node, dst_node, type="TRANSITIONED_TO")
                logger.debug("Transition: %s -> %s", parts[i], parts[i + 1])
            else:
                logger.debug(
                    "Could not resolve transition: %s -> %s (src=%s, dst=%s)",
                    parts[i], parts[i + 1], src_node, dst_node,
                )


def save_graph(G: nx.DiGraph, path: Path | None = None) -> Path:
    """Serialize the graph to disk as a pickle file."""
    path = path or GRAPH_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    click.echo(f"Graph saved to {path}")
    return path


def load_graph(path: Path | None = None) -> nx.DiGraph:
    """Load a previously saved graph from disk."""
    path = path or GRAPH_PATH
    with open(path, "rb") as f:
        return pickle.load(f)


def _print_graph_stats(G: nx.DiGraph) -> None:
    """Print graph statistics to the console."""
    node_types: dict[str, int] = defaultdict(int)
    for _, data in G.nodes(data=True):
        node_types[data.get("type", "unknown")] += 1

    edge_types: dict[str, int] = defaultdict(int)
    for _, _, data in G.edges(data=True):
        edge_types[data.get("type", "unknown")] += 1

    click.echo("\nGraph statistics:")
    click.echo(f"  Total nodes: {G.number_of_nodes()}")
    for ntype, count in sorted(node_types.items()):
        click.echo(f"    {ntype}: {count}")
    click.echo(f"  Total edges: {G.number_of_edges()}")
    for etype, count in sorted(edge_types.items()):
        click.echo(f"    {etype}: {count}")
