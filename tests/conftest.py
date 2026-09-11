import pytest

from cate_benchmark import config


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    """Point the results directory at a temporary folder."""
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "DATASET_STATS", tmp_path / "results" / "datasets.json")
    return tmp_path / "results"
