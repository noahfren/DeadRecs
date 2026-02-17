"""Tests for the Headyversion scraper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deadrecs.scraper import (
    parse_performances,
    parse_song_index,
    parse_transitions,
    scrape_song_performances,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


# ---------------------------------------------------------------------------
# parse_song_index
# ---------------------------------------------------------------------------

class TestParseSongIndex:
    def test_parses_all_songs(self):
        html = _read_fixture("song_index.html")
        songs = parse_song_index(html)
        assert len(songs) == 4

    def test_song_fields(self):
        html = _read_fixture("song_index.html")
        songs = parse_song_index(html)
        dark_star = songs[0]
        assert dark_star["id"] == 58
        assert dark_star["name"] == "Dark Star"
        assert dark_star["slug"] == "dark-star"
        assert dark_star["url"] == "/song/58/grateful-dead/dark-star/"

    def test_non_combo_song_has_no_transitions(self):
        html = _read_fixture("song_index.html")
        songs = parse_song_index(html)
        # Dark Star and Eyes Of The World are not combo songs
        dark_star = songs[0]
        eyes = songs[3]
        assert "transitions" not in dark_star
        assert "transitions" not in eyes

    def test_combo_song_has_transitions(self):
        html = _read_fixture("song_index.html")
        songs = parse_song_index(html)
        # China Cat Sunflower -> I Know You Rider
        china_cat = songs[1]
        assert china_cat["transitions"] == ["China Cat Sunflower", "I Know You Rider"]

    def test_multi_arrow_combo(self):
        html = _read_fixture("song_index.html")
        songs = parse_song_index(html)
        # Help On The Way > Slipknot > Franklin's Tower
        hotw = songs[2]
        assert hotw["transitions"] == ["Help On The Way", "Slipknot", "Franklin's Tower"]


# ---------------------------------------------------------------------------
# parse_transitions
# ---------------------------------------------------------------------------

class TestParseTransitions:
    def test_single_song_returns_none(self):
        assert parse_transitions("Dark Star") is None

    def test_arrow_dash_gt(self):
        result = parse_transitions("China Cat Sunflower -> I Know You Rider")
        assert result == ["China Cat Sunflower", "I Know You Rider"]

    def test_arrow_unicode(self):
        result = parse_transitions("Scarlet Begonias → Fire On The Mountain")
        assert result == ["Scarlet Begonias", "Fire On The Mountain"]

    def test_arrow_gt_only(self):
        result = parse_transitions("Help On The Way > Slipknot > Franklin's Tower")
        assert result == ["Help On The Way", "Slipknot", "Franklin's Tower"]

    def test_drums_space(self):
        result = parse_transitions("Drums -> Space")
        assert result == ["Drums", "Space"]

    def test_no_arrow(self):
        assert parse_transitions("Not Fade Away") is None

    def test_song_with_gt_in_name_but_spaced(self):
        # The regex requires whitespace around the arrow
        # A bare > with spaces on both sides is treated as an arrow
        result = parse_transitions("A > B")
        assert result == ["A", "B"]


# ---------------------------------------------------------------------------
# parse_performances
# ---------------------------------------------------------------------------

class TestParsePerformances:
    def test_parses_performances_from_page(self):
        html = _read_fixture("song_performances.html")
        perfs, next_page = parse_performances(html)
        assert len(perfs) == 2

    def test_performance_fields(self):
        html = _read_fixture("song_performances.html")
        perfs, _ = parse_performances(html)
        first = perfs[0]
        assert first["votes"] == 135
        assert first["date"] == "1972-08-27"
        assert first["venue"] == "Old Renaissance Faire Grounds Veneta, OR"
        assert first["description"] == "The greatest Dark Star ever played"
        assert first["archive_url"] == "https://headyversion.com/show/100/archive/"
        assert first["show_id"] == 100

    def test_second_performance(self):
        html = _read_fixture("song_performances.html")
        perfs, _ = parse_performances(html)
        second = perfs[1]
        assert second["votes"] == 98
        assert second["date"] == "1974-02-24"
        assert second["venue"] == "Winterland Arena San Francisco, CA"
        assert second["show_id"] == 101

    def test_next_page_link(self):
        html = _read_fixture("song_performances.html")
        _, next_page = parse_performances(html)
        assert next_page == "/song/58/grateful-dead/dark-star/?page=2"

    def test_last_page_no_next(self):
        html = _read_fixture("song_performances_page2.html")
        perfs, next_page = parse_performances(html)
        assert len(perfs) == 1
        assert next_page is None

    def test_page2_performance(self):
        html = _read_fixture("song_performances_page2.html")
        perfs, _ = parse_performances(html)
        assert perfs[0]["votes"] == 45
        assert perfs[0]["date"] == "1968-02-14"


# ---------------------------------------------------------------------------
# scrape_song_performances (with mocked HTTP)
# ---------------------------------------------------------------------------

class TestScrapeSongPerformances:
    def test_handles_pagination(self, tmp_path, monkeypatch):
        """Verify that scrape_song_performances follows pagination links."""
        page1_html = _read_fixture("song_performances.html")
        page2_html = _read_fixture("song_performances_page2.html")

        # Mock requests.Session
        mock_session = MagicMock()
        responses = [
            MagicMock(text=page1_html, status_code=200),
            MagicMock(text=page2_html, status_code=200),
        ]
        mock_session.get = MagicMock(side_effect=responses)

        # Point song_path to tmp_path
        monkeypatch.setattr(
            "deadrecs.scraper.song_path",
            lambda sid, slug: tmp_path / f"{sid}_{slug}.json",
        )
        monkeypatch.setattr("deadrecs.scraper.ensure_data_dirs", lambda: None)

        song = {"id": 58, "name": "Dark Star", "slug": "dark-star", "url": "/song/58/grateful-dead/dark-star/"}
        perfs = scrape_song_performances(song, session=mock_session, delay=0)

        assert len(perfs) == 3  # 2 from page 1 + 1 from page 2
        assert mock_session.get.call_count == 2

        # Verify file was written
        out_file = tmp_path / "58_dark-star.json"
        assert out_file.exists()


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------

class TestScrapeCLI:
    def test_scrape_command_runs(self):
        """Verify the scrape command is wired up (no longer a placeholder)."""
        from click.testing import CliRunner

        from deadrecs.cli import main

        runner = CliRunner()
        with patch("deadrecs.scraper.run_scraper") as mock_run:
            result = runner.invoke(main, ["scrape", "--delay", "0.5"])
            assert result.exit_code == 0
            mock_run.assert_called_once_with(delay=0.5)
