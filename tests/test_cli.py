from cate_benchmark import cli


def test_list_shows_every_experiment(capsys):
    cli.main(["list"])
    out = capsys.readouterr().out
    for name in ("full-pool", "sample-size", "sample-size-large", "contamination"):
        assert name in out


def test_run_rejects_unknown_method():
    import pytest

    with pytest.raises(SystemExit):
        cli.main(["run", "full-pool", "--method", "nope"])
