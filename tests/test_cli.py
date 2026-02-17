"""Tests for the CLI skeleton."""

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


def test_scrape_placeholder():
    runner = CliRunner()
    result = runner.invoke(main, ["scrape"])
    assert result.exit_code == 0
    assert "Scraper not yet implemented." in result.output


def test_train_placeholder():
    runner = CliRunner()
    result = runner.invoke(main, ["train"])
    assert result.exit_code == 0
    assert "Training not yet implemented." in result.output


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
