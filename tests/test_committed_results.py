"""Integrity checks on the results committed to the repository."""
import pytest

from cate_benchmark import config, tables
from cate_benchmark.config import EXPERIMENTS
from cate_benchmark.results import load_dataset_stats, load_rows
from cate_benchmark.runner import planned_cells

# Table 3 of the poster (computed there with the oracle rounded to 3 decimals).
POSTER_TABLE_3 = {
    "tabpfn_t": {"ihdp": 0.417, "acic2016": 0.102, "lalonde_cps": 0.837, "lalonde_psid": 0.770},
    "causalpfn": {"ihdp": 0.197, "acic2016": 0.176, "lalonde_cps": 0.827, "lalonde_psid": 0.784},
    "dopfn": {"ihdp": 2.125, "acic2016": 1.040, "lalonde_cps": 1.134, "lalonde_psid": 1.225},
}


@pytest.mark.parametrize("name", list(EXPERIMENTS))
@pytest.mark.parametrize("method", list(config.METHODS))
def test_every_planned_cell_has_exactly_one_result(name, method):
    experiment = EXPERIMENTS[name]
    planned = set(planned_cells(experiment, experiment.datasets, range(experiment.reps)))
    assert set(load_rows(experiment, method)) == planned


def test_record_counts_match_the_readme():
    def count(*names):
        return sum(len(load_rows(EXPERIMENTS[n], m)) for n in names for m in config.METHODS)

    assert count("sample-size", "sample-size-large") == 7_500
    assert count("contamination") == 4_800


def test_full_pool_table_matches_the_poster():
    values = tables.full_pool_normalized()
    for method, row in POSTER_TABLE_3.items():
        for dataset, expected in row.items():
            assert values[method][dataset] == pytest.approx(expected, abs=0.0011)


def test_dataset_statistics_match_the_poster():
    stats = load_dataset_stats()
    expected = {  # Table 2 of the poster: pool, test, covariates, treated %
        "ihdp": (672, 75, 25, 18.30), "acic2016": (4321, 481, 58, 17.80),
        "lalonde_cps": (14559, 1618, 8, 1.06), "lalonde_psid": (2407, 268, 8, 5.86),
    }
    for ds, (pool, test, d, treated) in expected.items():
        s = stats[ds]
        assert (s["n_pool"], s["n_test"], s["n_covariates"]) == (pool, test, d)
        assert 100 * s["treated_fraction"] == pytest.approx(treated, abs=0.05)  # poster rounds to 0.1%


def test_committed_tables_are_up_to_date(tmp_path):
    for path in tables.write_all(tmp_path):
        committed = tables.TABLES_DIR / path.name
        assert committed.read_text(encoding="utf-8") == path.read_text(encoding="utf-8"), path.name
