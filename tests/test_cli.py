"""Tests for the CLI skeleton."""

from unittest.mock import patch

from click.testing import CliRunner

from deadrecs.cli import main


def test_main_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "DeadRecs" in result.output


def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_scrape_invokes_run_scraper():
    runner = CliRunner()
    with patch("deadrecs.scraper.run_scraper") as mock_run:
        result = runner.invoke(main, ["scrape"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(delay=1.0)


def test_train_requires_scraped_data():
    runner = CliRunner()
    with patch("deadrecs.utils.SONGS_INDEX_PATH") as mock_path:
        mock_path.exists.return_value = False
        result = runner.invoke(main, ["train"])
        assert result.exit_code != 0
        assert "scrape" in result.output.lower()


def test_recommend_placeholder():
    runner = CliRunner()
    result = runner.invoke(main, ["recommend", "--like", "1977-05-08"])
    assert result.exit_code == 0
    assert "Recommendations not yet implemented." in result.output


def test_recommend_requires_like():
    runner = CliRunner()
    result = runner.invoke(main, ["recommend"])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()


def test_info_placeholder():
    runner = CliRunner()
    result = runner.invoke(main, ["info", "1977-05-08"])
    assert result.exit_code == 0
    assert "Info not yet implemented." in result.output


def test_info_requires_query():
    runner = CliRunner()
    result = runner.invoke(main, ["info"])
    assert result.exit_code != 0
