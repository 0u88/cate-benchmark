"""Command-line entry point: `cate-bench <command>` (or `python -m cate_benchmark`).

    cate-bench list                                  show the experiments
    cate-bench run EXPERIMENT --method METHOD        run (or resume) an experiment
    cate-bench plot                                  rebuild the two figures
    cate-bench tables                                rebuild results/tables/*.md
    cate-bench diagnose effect-scale | dopfn-width   Do-PFN failure analysis
    cate-bench datasets                              rebuild results/datasets.json

`plot`, `tables` and `diagnose effect-scale` only read committed results; they
need neither the models nor the datasets.
"""
from __future__ import annotations

import argparse

from . import config


def _list(_args) -> None:
    for e in config.EXPERIMENTS.values():
        cells = sum(len(v) for v in e.train_sizes.values()) * max(len(e.sigmas), 1) * e.reps
        print(f"{e.name:18s} {cells:5d} cells per method   {e.title}")


def _run(args) -> None:
    from .runner import run

    run(
        config.EXPERIMENTS[args.experiment],
        args.method,
        reps=args.reps,
        rep_start=args.rep_start,
        datasets=args.datasets,
        device=args.device,
        force=args.force,
    )


def _plot(_args) -> None:
    from .plotting import contamination, sample_size

    for module in (sample_size, contamination):
        print(f"wrote {module.plot()}")


def _tables(_args) -> None:
    from .tables import write_all

    for path in write_all():
        print(f"wrote {path}")


def _diagnose(args) -> None:
    if args.name == "effect-scale":
        from .diagnostics import effect_scale

        effect_scale.main()
    else:
        from .diagnostics import dopfn_width

        dopfn_width.main(args.rest)


def _datasets(_args) -> None:
    from .data import dataset_stats, load_pool
    from .results import save_dataset_stats

    save_dataset_stats({ds: dataset_stats(load_pool(ds)) for ds in config.DATASETS})
    print(f"wrote {config.DATASET_STATS}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cate-bench", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show the experiments").set_defaults(func=_list)

    p = sub.add_parser("run", help="run or resume an experiment for one method")
    p.add_argument("experiment", choices=list(config.EXPERIMENTS))
    p.add_argument("--method", required=True, choices=list(config.METHODS))
    p.add_argument("--reps", type=int, default=None, help="run reps [rep-start, reps)")
    p.add_argument("--rep-start", type=int, default=0)
    p.add_argument("--datasets", nargs="*", choices=list(config.DATASETS), default=None)
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--force", action="store_true", help="recompute cells already on disk")
    p.set_defaults(func=_run)

    sub.add_parser("plot", help="rebuild the figures").set_defaults(func=_plot)
    sub.add_parser("tables", help="rebuild the result tables").set_defaults(func=_tables)

    p = sub.add_parser("diagnose", help="Do-PFN failure analysis")
    p.add_argument("name", choices=["effect-scale", "dopfn-width"])
    p.add_argument("rest", nargs=argparse.REMAINDER, help="options for dopfn-width")
    p.set_defaults(func=_diagnose)

    sub.add_parser("datasets", help="rebuild results/datasets.json").set_defaults(func=_datasets)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
