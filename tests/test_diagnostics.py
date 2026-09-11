import json

from cate_benchmark import config
from cate_benchmark.diagnostics import effect_scale


def test_committed_effect_scale_table_is_up_to_date():
    committed = config.RESULTS_DIR / "diagnostics" / "effect_scale.md"
    assert committed.read_text(encoding="utf-8") == effect_scale.table()


def test_dopfn_shrinks_effects_where_they_are_large():
    rows = {(r["method"], r["dataset"]): r for r in effect_scale.compute()}
    assert rows[("dopfn", "ihdp")]["true_scale"] < 0.5
    assert 0.9 < rows[("dopfn", "ihdp")]["shrinkage"] < 1.1
    for ds in ("acic2016", "lalonde_cps", "lalonde_psid"):
        assert rows[("dopfn", ds)]["true_scale"] > 0.8
        assert rows[("dopfn", ds)]["shrinkage"] < 0.25


def test_dopfn_width_results_cover_every_width():
    rows = json.loads((config.RESULTS_DIR / "diagnostics" / "dopfn_width.json").read_text())
    ihdp = {r["d"]: r["pehe_mean"] for r in rows if r["dataset"] == "ihdp"}
    assert set(ihdp) == {2, 4, 6, 8, 12, 16, 20, 25}
    assert min(ihdp, key=ihdp.get) in (4, 6)  # narrowing to ~6 covariates helps Do-PFN
    assert ihdp[4] < ihdp[25]
