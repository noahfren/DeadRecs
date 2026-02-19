"""DeadRecs CLI — command-line interface for the recommendation engine."""

import logging

import click

from deadrecs import __version__


@click.group()
@click.version_option(version=__version__, prog_name="deadrecs")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def main(verbose):
    """DeadRecs — A Grateful Dead show recommendation engine."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@main.command()
@click.option("--delay", default=1.0, type=float, help="Seconds between requests.")
def scrape(delay):
    """Scrape performance data from Headyversion.com."""
    from deadrecs.scraper import run_scraper

    run_scraper(delay=delay)


@main.command()
@click.option("--epochs", default=150, type=int, help="Number of training epochs.")
@click.option("--k-neighbors", default=10, type=int, help="Top-k setlist neighbors per show.")
def train(epochs, k_neighbors):
    """Build the graph and train the GNN model."""
    from deadrecs.graph import build_graph, save_graph
    from deadrecs.features import (
        add_setlist_neighbor_edges,
        compute_description_embeddings,
        compute_idf,
    )
    from deadrecs.utils import SONGS_INDEX_PATH

    if not SONGS_INDEX_PATH.exists():
        raise click.ClickException(
            "No scraped data found. Run `deadrecs scrape` first."
        )

    # Phase 3: Graph construction
    G = build_graph()
    compute_idf(G)
    compute_description_embeddings(G)
    add_setlist_neighbor_edges(G, k=k_neighbors)
    save_graph(G)

    # Phase 4: GNN training
    from deadrecs.train import train_model

    train_model(G, epochs=epochs)


@main.command()
@click.option("--like", multiple=True, required=True, help="A show date or song name you enjoy (repeatable).")
@click.option("--n", "num_results", default=10, type=int, help="Number of recommendations.")
@click.option(
    "--type",
    "rec_type",
    type=click.Choice(["shows", "songs", "both"], case_sensitive=False),
    default="both",
    help="Type of recommendations to return.",
)
def recommend(like, num_results, rec_type):
    """Get show and song recommendations."""
    click.echo("Recommendations not yet implemented.")


@main.command()
@click.argument("query")
def info(query):
    """Display details about a show (by date) or song (by name)."""
    click.echo("Info not yet implemented.")
