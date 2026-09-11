"""Why Do-PFN fails, part 2: it shrinks large effects.

Reads only committed results (Experiment 1, full training pool), so it runs
without models or data.

For each dataset it compares the true effect scale with the predicted one:

  true scale        sd(tau) / sd(y)       how large the heterogeneous effect is
                                          relative to the outcome's own spread
  shrinkage         sd(tau_hat) / sd(tau) 1 = predicted effects vary as much as
                                          the true ones; < 1 = compressed
  vs oracle         PEHE / sd(tau)        < 1 = beats a constant predictor that
                                          knows the true average effect
  residual share    residual / sd(tau)    the error left after removing the
                                          average-effect bias (PEHE^2 = bias^2 +
                                          residual^2); > 1 means the per-unit
                                          predictions add error, not information

Do-PFN's prior is a structural causal model in which the treatment is one node
among many, so large effects relative to the outcome's spread are rare in its
pre-training data. The hypothesis is that it compresses effects more, the
larger the true scale is.
"""
from __future__ import annotations

from .. import config
from ..config import EXPERIMENTS
from ..metrics import decompose
from ..results import load_dataset_stats, load_rows


def compute() -> list[dict]:
    stats = load_dataset_stats()
    experiment = EXPERIMENTS["full-pool"]
    rows = []
    for method in config.METHODS:
        results = load_rows(experiment, method)
        for ds in experiment.datasets:
            s = stats[ds]
            r = results[(ds, experiment.train_sizes[ds][0], None, 0)]
            _bias, residual = decompose(r["pehe"], r["cate_pred_mean"], s["tau_mean"])
            rows.append({
                "method": method,
                "dataset": ds,
                "n_covariates": s["n_covariates"],
                "true_scale": s["tau_sd"] / s["y_sd"],
                "shrinkage": r["cate_pred_sd"] / s["tau_sd"],
                "vs_oracle": r["pehe"] / s["tau_sd"],
                "residual_share": residual / s["tau_sd"],
            })
    return rows


def table() -> str:
    rows = compute()
    lines = [
        "# Effect-scale diagnostic (Experiment 1, full training pool)",
        "",
        "| Dataset | Covariates | True scale sd(τ)/sd(y) | Method | Shrinkage sd(τ̂)/sd(τ) "
        "| PEHE / sd(τ) | Residual / sd(τ) |",
        "|---|---|---|---|---|---|---|",
    ]
    for ds in config.DATASETS:
        for i, r in enumerate(x for x in rows if x["dataset"] == ds):
            head = (
                f"| {config.DATASETS[ds]} | {r['n_covariates']} | {r['true_scale']:.2f} |"
                if i == 0 else "| | | |"
            )
            lines.append(
                f"{head} {config.METHODS[r['method']]} | {r['shrinkage']:.2f} | "
                f"{r['vs_oracle']:.2f} | {r['residual_share']:.2f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    out = config.RESULTS_DIR / "diagnostics" / "effect_scale.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table(), encoding="utf-8")
    print(table())
    print(f"wrote {out}")
