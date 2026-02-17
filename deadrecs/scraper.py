"""Scraper for Headyversion.com Grateful Dead performance data."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import click
import requests
from bs4 import BeautifulSoup

from deadrecs.utils import SONGS_INDEX_PATH, ensure_data_dirs, song_path

logger = logging.getLogger(__name__)

BASE_URL = "https://headyversion.com"
SONG_INDEX_URL = f"{BASE_URL}/search/all/?order=count"

# Patterns that indicate a transition/combo song name.
_ARROW_PATTERN = re.compile(r"\s*(?:->|→|>)\s*")


def parse_song_index(html: str) -> list[dict[str, Any]]:
    """Parse the song index page HTML and return a list of song dicts.

    Each dict contains: id, name, slug, url, versions, transitions.
    """
    soup = BeautifulSoup(html, "html.parser")
    songs: list[dict[str, Any]] = []

    for div in soup.find_all("div", class_="big_link"):
        anchor = div.find("a")
        if anchor is None:
            continue
        href = anchor["href"]
        name = anchor.get_text(strip=True)

        # Parse id and slug from href like /song/207/grateful-dead/playin-in-the-band/
        match = re.match(r"/song/(\d+)/grateful-dead/([^/]+)/", href)
        if match is None:
            logger.warning("Could not parse song URL: %s", href)
            continue

        song_id = int(match.group(1))
        slug = match.group(2)

        song: dict[str, Any] = {
            "id": song_id,
            "name": name,
            "slug": slug,
            "url": href,
        }

        # Parse transitions for combo songs.
        transitions = parse_transitions(name)
        if transitions:
            song["transitions"] = transitions

        songs.append(song)

    return songs


def parse_transitions(name: str) -> list[str] | None:
    """Parse a combo song name into individual song names.

    Detects arrows (->  →  >) separating songs and returns the list of
    individual song names.  Returns None if the name is not a combo.

    Examples:
        "China Cat Sunflower -> I Know You Rider" → ["China Cat Sunflower", "I Know You Rider"]
        "Help On The Way > Slipknot > Franklin's Tower" → ["Help On The Way", "Slipknot", "Franklin's Tower"]
        "Dark Star" → None
    """
    parts = _ARROW_PATTERN.split(name)
    if len(parts) <= 1:
        return None
    return [p.strip() for p in parts if p.strip()]


def _parse_date_from_show_url(href: str) -> str | None:
    """Extract a YYYY-MM-DD date from a show URL like /show/25/grateful-dead/1977-05-09/."""
    match = re.search(r"/(\d{4}-\d{2}-\d{2})/", href)
    return match.group(1) if match else None


def _parse_venue_from_text(text: str) -> str:
    """Extract venue from show date text like 'May 9, 1977 - War Memorial Buffalo, NY'.

    Splits on the first ' - ' and returns everything after it.
    """
    parts = text.split(" - ", 1)
    return parts[1].strip() if len(parts) > 1 else ""


def parse_performances(html: str) -> tuple[list[dict[str, Any]], str | None]:
    """Parse a song performance page and return (performances, next_page_url).

    Each performance dict contains: votes, date, venue, description, archive_url, show_id.
    next_page_url is the relative URL for the next page, or None if no more pages.
    """
    soup = BeautifulSoup(html, "html.parser")
    performances: list[dict[str, Any]] = []

    for row in soup.find_all("div", class_="s2s_submission"):
        perf: dict[str, Any] = {}

        # Vote count
        score_div = row.find("div", class_="score")
        if score_div:
            perf["votes"] = int(score_div.get_text(strip=True))
        else:
            perf["votes"] = 0

        # Date and venue from the show_date div
        date_div = row.find("div", class_="show_date")
        if date_div:
            anchor = date_div.find("a")
            if anchor:
                href = anchor["href"]
                date = _parse_date_from_show_url(href)
                perf["date"] = date

                # Extract show_id from href like /show/25/grateful-dead/1977-05-09/
                show_match = re.match(r"/show/(\d+)/", href)
                if show_match:
                    perf["show_id"] = int(show_match.group(1))

                perf["venue"] = _parse_venue_from_text(anchor.get_text(strip=True))

        # Description
        desc_div = row.find("div", class_="show_description")
        if desc_div:
            perf["description"] = desc_div.get_text(strip=True)
        else:
            perf["description"] = ""

        # Archive URL — stored as a relative path on headyversion
        links_div = row.find("div", class_="show_links")
        if links_div:
            archive_link = links_div.find("a", string=re.compile(r"Listen on archive"))
            if archive_link:
                perf["archive_url"] = BASE_URL + archive_link["href"]

        performances.append(perf)

    # Check for next page
    next_page_url = None
    endless_link = soup.find("a", class_="endless_more")
    if endless_link:
        next_page_url = endless_link["href"]

    return performances, next_page_url


def scrape_song_index(session: requests.Session | None = None) -> list[dict[str, Any]]:
    """Fetch and parse the Headyversion song index.

    Returns the list of song dicts and saves them to disk.
    """
    sess = session or requests.Session()
    logger.info("Fetching song index from %s", SONG_INDEX_URL)
    resp = sess.get(SONG_INDEX_URL, timeout=30)
    resp.raise_for_status()

    songs = parse_song_index(resp.text)
    logger.info("Found %d songs", len(songs))

    ensure_data_dirs()
    with open(SONGS_INDEX_PATH, "w") as f:
        json.dump(songs, f, indent=2)

    return songs


def scrape_song_performances(
    song: dict[str, Any],
    session: requests.Session | None = None,
    delay: float = 1.0,
) -> list[dict[str, Any]]:
    """Fetch all performances for a single song, handling pagination.

    Returns the list of performance dicts and saves them to disk.
    """
    sess = session or requests.Session()
    song_id = song["id"]
    slug = song["slug"]
    url = f"{BASE_URL}{song['url']}"

    all_performances: list[dict[str, Any]] = []
    page = 1

    while url:
        logger.debug("Fetching %s (page %d)", song["name"], page)
        resp = sess.get(url, timeout=30)
        resp.raise_for_status()

        performances, next_page = parse_performances(resp.text)
        all_performances.extend(performances)

        if next_page:
            url = f"{BASE_URL}{next_page}"
            page += 1
            time.sleep(delay)
        else:
            url = None  # type: ignore[assignment]

    # Save to disk
    out_path = song_path(song_id, slug)
    ensure_data_dirs()
    with open(out_path, "w") as f:
        json.dump(
            {"song_id": song_id, "name": song["name"], "performances": all_performances},
            f,
            indent=2,
        )

    return all_performances


def run_scraper(delay: float = 1.0) -> None:
    """Run the full scraping pipeline: song index + all performances.

    Supports resuming — skips songs whose data files already exist.
    """
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "DeadRecs/0.1 (Grateful Dead recommendation engine)"}
    )

    # Step 1: Song index
    if SONGS_INDEX_PATH.exists():
        click.echo(f"Song index already exists at {SONGS_INDEX_PATH}, loading...")
        with open(SONGS_INDEX_PATH) as f:
            songs = json.load(f)
        click.echo(f"Loaded {len(songs)} songs from existing index.")
    else:
        click.echo("Scraping song index...")
        songs = scrape_song_index(session)
        click.echo(f"Found {len(songs)} songs.")

    # Step 2: Song performances
    skipped = 0
    scraped = 0
    errors = 0

    with click.progressbar(songs, label="Scraping performances", show_pos=True) as bar:
        for song in bar:
            out = song_path(song["id"], song["slug"])
            if out.exists():
                skipped += 1
                continue

            try:
                perfs = scrape_song_performances(song, session=session, delay=delay)
                scraped += 1
                logger.info("Scraped %s: %d performances", song["name"], len(perfs))
            except requests.RequestException as exc:
                errors += 1
                logger.error("Error scraping %s: %s", song["name"], exc)

            time.sleep(delay)

    click.echo(f"\nDone! Scraped: {scraped}, Skipped (already cached): {skipped}, Errors: {errors}")
