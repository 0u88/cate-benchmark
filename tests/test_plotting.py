import pytest

pytest.importorskip("matplotlib")


def test_both_figures_render(tmp_path):
    from cate_benchmark.plotting import contamination, sample_size

    for module, name in ((sample_size, "fig1.png"), (contamination, "fig2.png")):
        out = module.plot(tmp_path / name)
        assert out.stat().st_size > 50_000
