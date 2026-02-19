"""Recommendation engine for DeadRecs.

Parses user inputs (show dates or song names), loads pre-computed GNN
embeddings, and returns similar shows via cosine similarity.
"""

from __future__ import annotations

import logging
import re

import click
import networkx as nx
import torch
import torch.nn.functional as F

from deadrecs.graph import load_graph
from deadrecs.utils import EMBEDDINGS_PATH, GRAPH_PATH

logger = logging.getLogger(__name__)

# Regex for full or partial date patterns.
DATE_PATTERN = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


def _is_date(value: str) -> bool:
    """Check if a string looks like a date (YYYY, YYYY-MM, or YYYY-MM-DD)."""
    return DATE_PATTERN.match(value) is not None


def _resolve_show(G: nx.DiGraph, date_str: str) -> list[str]:
    """Find show node(s) matching a date string.

    Supports exact match (YYYY-MM-DD) or prefix match (YYYY or YYYY-MM).
    Returns list of matching show node IDs.
    """
    exact = f"show:{date_str}"
    if exact in G and G.nodes[exact].get("type") == "show":
        return [exact]

    # Prefix match for partial dates
    matches = [
        nid for nid, attrs in G.nodes(data=True)
        if attrs.get("type") == "show" and attrs.get("date", "").startswith(date_str)
    ]
    return sorted(matches)


def _resolve_song(G: nx.DiGraph, name: str) -> list[str]:
    """Find song node(s) matching a name via case-insensitive substring match.

    Returns list of matching song node IDs, sorted by name length (prefer
    exact or shorter matches).
    """
    query = name.lower()
    matches = []
    for nid, attrs in G.nodes(data=True):
        if attrs.get("type") != "song":
            continue
        song_name = attrs.get("name", "").lower()
        if query == song_name:
            return [nid]
        if query in song_name:
            matches.append((nid, len(song_name)))

    # Sort by name length (prefer shorter/more precise matches)
    matches.sort(key=lambda x: x[1])
    return [nid for nid, _ in matches]


def _resolve_performance(G: nx.DiGraph, song_name: str, date_str: str) -> str | None:
    """Find a performance node matching a song name and date.

    Returns the performance node ID, or None if not found.
    """
    # Resolve the song first
    song_matches = _resolve_song(G, song_name)
    if not song_matches:
        return None

    # Extract the numeric song ID from the song node ID (e.g. "song:42" -> "42")
    song_node = song_matches[0]
    song_id = song_node.split(":", 1)[1]

    # Try exact performance node
    perf_id = f"perf:{song_id}:{date_str}"
    if perf_id in G and G.nodes[perf_id].get("type") == "performance":
        return perf_id

    return None


def parse_inputs(
    G: nx.DiGraph, like_args: tuple[str, ...]
) -> list[str]:
    """Parse --like arguments into graph node IDs.

    Supported formats:
      - "YYYY-MM-DD" (or partial) → Show node(s)
      - "Song Name" → Song node
      - "Song Name @ YYYY-MM-DD" → Specific performance node

    Returns a list of resolved node IDs. Raises click.ClickException
    for unresolvable inputs.
    """
    resolved: list[str] = []

    for arg in like_args:
        # Check for performance format: "Song Name @ YYYY-MM-DD"
        if " @ " in arg:
            parts = arg.rsplit(" @ ", 1)
            song_part = parts[0].strip()
            date_part = parts[1].strip()

            if not _is_date(date_part):
                raise click.ClickException(
                    f"Invalid date in '{arg}'. Expected format: Song Name @ YYYY-MM-DD"
                )

            perf_id = _resolve_performance(G, song_part, date_part)
            if not perf_id:
                raise click.ClickException(
                    f"No performance found for '{song_part}' on {date_part}."
                )
            resolved.append(perf_id)

        elif _is_date(arg):
            matches = _resolve_show(G, arg)
            if not matches:
                raise click.ClickException(
                    f"No show found for date '{arg}'."
                )
            if len(matches) > 5:
                click.echo(
                    f"  Found {len(matches)} shows matching '{arg}', "
                    f"using all of them."
                )
            resolved.extend(matches)
        else:
            matches = _resolve_song(G, arg)
            if not matches:
                raise click.ClickException(
                    f"No song found matching '{arg}'."
                )
            if len(matches) > 1:
                # Warn about ambiguity, use the best match
                best = G.nodes[matches[0]].get("name", matches[0])
                others = [G.nodes[m].get("name", m) for m in matches[1:3]]
                click.echo(
                    f"  '{arg}' matched '{best}' "
                    f"(also: {', '.join(others)}"
                    f"{'...' if len(matches) > 3 else ''})."
                )
            resolved.append(matches[0])

    return resolved


def _get_show_details(G: nx.DiGraph, show_id: str) -> dict:
    """Get show details including top-rated performances."""
    attrs = G.nodes[show_id]
    date = attrs.get("date", "")
    venue = attrs.get("venue", "")

    # Find performances at this show
    performances = []
    for _, target, edata in G.out_edges(show_id, data=True):
        if edata.get("type") != "HAS_PERFORMANCE":
            continue
        perf_attrs = G.nodes[target]
        votes = perf_attrs.get("votes", 0)
        archive_url = perf_attrs.get("archive_url", "")

        # Find the song name
        song_name = ""
        for _, song_target, sedata in G.out_edges(target, data=True):
            if sedata.get("type") == "OF_SONG":
                song_name = G.nodes[song_target].get("name", "")
                break

        performances.append({
            "song": song_name,
            "votes": votes,
            "archive_url": archive_url,
        })

    # Sort by votes descending
    performances.sort(key=lambda p: p["votes"], reverse=True)

    # Pick the best archive URL from top performance
    archive_url = ""
    for p in performances:
        if p["archive_url"]:
            archive_url = p["archive_url"]
            break

    return {
        "date": date,
        "venue": venue,
        "performances": performances,
        "archive_url": archive_url,
    }


def _format_show_result(rank: int, show_id: str, similarity: float, G: nx.DiGraph) -> str:
    """Format a single show recommendation for display."""
    details = _get_show_details(G, show_id)
    lines = []

    header = f"#{rank:<3d} {details['date']} — {details['venue']} (similarity: {similarity:.2f})"
    lines.append(header)

    # Show top 3 notable performances
    top_perfs = [p for p in details["performances"] if p["votes"] > 0][:3]
    if top_perfs:
        notable = ", ".join(
            f"{p['song']} ({p['votes']} votes)" for p in top_perfs
        )
        lines.append(f"    Notable: {notable}")

    if details["archive_url"]:
        lines.append(f"    Listen: {details['archive_url']}")

    return "\n".join(lines)


def recommend(
    like_args: tuple[str, ...],
    num_results: int = 10,
) -> None:
    """Run the recommendation engine and print show results.

    Args:
        like_args: User-provided --like values (dates or song names).
        num_results: Number of recommendations to return.
    """
    # Load graph and embeddings
    if not GRAPH_PATH.exists():
        raise click.ClickException(
            "No graph found. Run `deadrecs train` first."
        )
    if not EMBEDDINGS_PATH.exists():
        raise click.ClickException(
            "No embeddings found. Run `deadrecs train` first."
        )

    G = load_graph(GRAPH_PATH)
    embeddings: dict[str, torch.Tensor] = torch.load(
        EMBEDDINGS_PATH, weights_only=False
    )

    # Parse inputs
    input_ids = parse_inputs(G, like_args)
    click.echo(f"Using {len(input_ids)} input(s) for recommendations.\n")

    # Collect input embeddings
    input_embs = []
    for nid in input_ids:
        if nid not in embeddings:
            click.echo(f"  Warning: no embedding for '{nid}', skipping.")
            continue
        input_embs.append(embeddings[nid])

    if not input_embs:
        raise click.ClickException(
            "None of the inputs have embeddings. Try different inputs or retrain."
        )

    # Average into a single query vector
    query = torch.stack(input_embs).mean(dim=0)
    query = F.normalize(query.unsqueeze(0), dim=1)

    # Collect candidate show embeddings
    exclude_set = set(input_ids)
    candidates: list[tuple[str, torch.Tensor]] = []
    for nid, emb in embeddings.items():
        if nid.startswith("show:") and nid not in exclude_set:
            candidates.append((nid, emb))

    if not candidates:
        click.echo("No show candidates found.\n")
        return

    # Stack and compute cosine similarity
    cand_ids = [c[0] for c in candidates]
    cand_embs = torch.stack([c[1] for c in candidates])
    cand_embs = F.normalize(cand_embs, dim=1)

    similarities = (cand_embs @ query.T).squeeze(1)

    # Top-n
    k = min(num_results, len(cand_ids))
    top_scores, top_indices = torch.topk(similarities, k)

    # Print results
    click.echo(f"{'=' * 60}")
    click.echo(f"  Top {k} Show Recommendations")
    click.echo(f"{'=' * 60}\n")

    for i in range(k):
        idx = top_indices[i].item()
        score = top_scores[i].item()
        nid = cand_ids[idx]
        click.echo(_format_show_result(i + 1, nid, score, G))
        click.echo()
