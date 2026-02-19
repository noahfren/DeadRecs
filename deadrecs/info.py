"""Info command for displaying show and song details from the graph."""

from __future__ import annotations

import click
import networkx as nx

from deadrecs.graph import load_graph
from deadrecs.recommend import _is_date, _resolve_show, _resolve_song
from deadrecs.utils import GRAPH_PATH


def _get_song_name_for_perf(G: nx.DiGraph, perf_id: str) -> str:
    """Get the song name for a performance node."""
    for _, target, edata in G.out_edges(perf_id, data=True):
        if edata.get("type") == "OF_SONG":
            return G.nodes[target].get("name", "")
    return ""


def _display_show_info(G: nx.DiGraph, show_id: str) -> None:
    """Display detailed info about a show."""
    attrs = G.nodes[show_id]
    date = attrs.get("date", "")
    venue = attrs.get("venue", "")

    click.echo(f"\n  {date} — {venue}")
    click.echo(f"  {'=' * 50}")

    # Collect performances
    performances = []
    for _, target, edata in G.out_edges(show_id, data=True):
        if edata.get("type") != "HAS_PERFORMANCE":
            continue
        perf_attrs = G.nodes[target]
        votes = perf_attrs.get("votes", 0)
        description = perf_attrs.get("description", "")
        archive_url = perf_attrs.get("archive_url", "")
        song_name = _get_song_name_for_perf(G, target)

        performances.append({
            "song": song_name,
            "votes": votes,
            "description": description,
            "archive_url": archive_url,
        })

    performances.sort(key=lambda p: p["votes"], reverse=True)

    if not performances:
        click.echo("  No rated performances found.")
        return

    click.echo(f"  {len(performances)} rated performance(s):\n")
    for p in performances:
        vote_str = f"({p['votes']} votes)" if p["votes"] > 0 else "(no votes)"
        click.echo(f"    {p['song']}  {vote_str}")
        if p["description"]:
            # Truncate long descriptions
            desc = p["description"]
            if len(desc) > 120:
                desc = desc[:117] + "..."
            click.echo(f"      {desc}")
        if p["archive_url"]:
            click.echo(f"      {p['archive_url']}")

    click.echo()


def _display_song_info(G: nx.DiGraph, song_id: str) -> None:
    """Display detailed info about a song and its top performances."""
    attrs = G.nodes[song_id]
    name = attrs.get("name", "")

    click.echo(f"\n  {name}")
    click.echo(f"  {'=' * 50}")

    # Find all performances of this song
    performances = []
    for source, _, edata in G.in_edges(song_id, data=True):
        if edata.get("type") != "OF_SONG":
            continue
        perf_attrs = G.nodes[source]
        votes = perf_attrs.get("votes", 0)
        description = perf_attrs.get("description", "")
        archive_url = perf_attrs.get("archive_url", "")

        # Find the show date/venue for this performance
        show_date = ""
        show_venue = ""
        for show_source, _, sedata in G.in_edges(source, data=True):
            if sedata.get("type") == "HAS_PERFORMANCE":
                show_attrs = G.nodes[show_source]
                show_date = show_attrs.get("date", "")
                show_venue = show_attrs.get("venue", "")
                break

        performances.append({
            "date": show_date,
            "venue": show_venue,
            "votes": votes,
            "description": description,
            "archive_url": archive_url,
        })

    performances.sort(key=lambda p: p["votes"], reverse=True)

    if not performances:
        click.echo("  No rated performances found.")
        return

    click.echo(f"  {len(performances)} rated performance(s):\n")
    for p in performances:
        vote_str = f"({p['votes']} votes)" if p["votes"] > 0 else "(no votes)"
        click.echo(f"    {p['date']} — {p['venue']}  {vote_str}")
        if p["description"]:
            desc = p["description"]
            if len(desc) > 120:
                desc = desc[:117] + "..."
            click.echo(f"      {desc}")
        if p["archive_url"]:
            click.echo(f"      {p['archive_url']}")

    click.echo()


def run_info(query: str) -> None:
    """Look up and display info about a show or song.

    Args:
        query: A show date (YYYY-MM-DD) or song name.
    """
    if not GRAPH_PATH.exists():
        raise click.ClickException(
            "No graph found. Run `deadrecs train` first."
        )

    G = load_graph(GRAPH_PATH)

    if _is_date(query):
        matches = _resolve_show(G, query)
        if not matches:
            raise click.ClickException(f"No show found for date '{query}'.")
        for show_id in matches:
            _display_show_info(G, show_id)
    else:
        matches = _resolve_song(G, query)
        if not matches:
            raise click.ClickException(f"No song found matching '{query}'.")
        if len(matches) > 1:
            best = G.nodes[matches[0]].get("name", matches[0])
            others = [G.nodes[m].get("name", m) for m in matches[1:3]]
            click.echo(
                f"  '{query}' matched '{best}' "
                f"(also: {', '.join(others)}"
                f"{'...' if len(matches) > 3 else ''})."
            )
        _display_song_info(G, matches[0])
