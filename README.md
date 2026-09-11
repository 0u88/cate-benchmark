# Benchmarking Tabular Foundation Models for CATE Estimation

How well do **tabular foundation models** estimate heterogeneous treatment effects when data is
scarce or noisy? This project benchmarks three prior-data fitted networks (PFNs) on
**conditional average treatment effect (CATE)** estimation:

- **TabPFN v3**: a general-purpose tabular foundation model, used as a T-learner
- **CausalPFN**: pretrained specifically for causal effect estimation (Balazadeh et al., 2025)
- **Do-PFN**: pretrained on structural causal models (Robertson et al., 2025)

The benchmark covers four standard causal-inference datasets: IHDP, ACIC 2016, Lalonde CPS, and
Lalonde PSID.

**Poster:** [`poster.pdf`](poster.pdf) (A0) summarizes the project.

## Research questions

1. How does a **general-purpose** tabular foundation model compare with **causally-pretrained**
   ones for CATE estimation?
2. How does CATE accuracy **scale with training-set size** in the small-sample regime?
3. How **robust** are these models when the observed covariates are **contaminated** with
   measurement noise?

## Results

All errors are root-PEHE (lower is better). Two model-free reference lines appear in every figure:

- **Predict-0** (dashed): the error of always predicting τ̂ = 0.
- **Oracle** (dotted): the error of a constant predictor that already knows the true average
  treatment effect. A method below this line has captured genuine effect heterogeneity.

### Experiment 1: Baseline CATE estimation

This experiment uses each dataset's full training pool and fixed test set. Each value is root-PEHE
divided by the oracle error, so values below 1 beat the oracle constant predictor.

| Model | IHDP | ACIC 2016 | Lalonde CPS | Lalonde PSID |
|---|---|---|---|---|
| TabPFN v3 (T-learner) | 0.417 | **0.102** | 0.837 | **0.770** |
| CausalPFN | **0.197** | 0.176 | **0.827** | 0.784 |
| Do-PFN | 2.125 | 1.040 | 1.134 | 1.225 |

TabPFN v3 and CausalPFN perform similarly. Do-PFN does not beat the oracle on any dataset, meaning
it doesn't recover the heterogeneity of the treatment effect.

### Experiment 2: CATE vs. training-set size

Training sets range from n = 100 to 500. Lalonde CPS uses n = 2,000 to 10,000 instead, because
only 1% of its units are treated. Each cell is repeated over 100 independent training draws.

![PEHE vs. training-set size, 100 draws per cell](figures/boxplot_all_datasets.png)

For all three models, median error drops sharply at first and then levels off at a
dataset-specific floor. Beyond that point, more data still narrows the spread across draws and
reduces worst-case error. On IHDP, TabPFN is better below n ≈ 300 and CausalPFN is better above it.

### Experiment 3: CATE vs. covariate contamination

Training size is fixed (n = 2,000 for Lalonde CPS, n = 500 otherwise). Noise is added to the
covariates only, from σ = 0.01 (essentially clean) to σ = 10 (about 10% of the original signal
left).

![PEHE vs. covariate contamination, 100 draws per cell](figures/contamination_all_datasets.png)

Causal pretraining gives no robustness advantage. TabPFN and CausalPFN degrade at similar rates on
three of the four datasets; on IHDP, CausalPFN degrades faster. Under extreme noise, their
estimates approach the oracle constant predictor. On ACIC 2016, severe contamination pushes both
models above the predict-0 line, so they do worse than assuming no treatment effect at all.

Per-dataset numbers (median and interquartile range over 100 draws) are in
[`results/summary_boxplot.md`](results/summary_boxplot.md),
[`results/summary_boxplot_large_n.md`](results/summary_boxplot_large_n.md), and
[`results/summary_contamination.md`](results/summary_contamination.md).

## Experimental design

| | |
|---|---|
| **Datasets** | IHDP (25 covariates), ACIC 2016 (58), Lalonde CPS (8), Lalonde PSID (8). Splits and ground-truth CATE follow the CausalPFN benchmark loaders, realization 0. |
| **Training sizes** | n ∈ {100, 200, 300, 400, 500}. Lalonde CPS also uses n ∈ {2,000, …, 10,000}, because only ~1% of units are treated. |
| **Repetitions** | 100 independent uniform training draws per cell, nested in n. The test set is identical across draws, so the spread reflects training-sample variability only. |
| **Metric** | root-PEHE = √mean((τ − τ̂)²), using CausalPFN's definition so numbers are comparable to the published tables. |
| **Reference lines** | predict-0 = √mean(τ²) and oracle = sd(τ). Both depend only on the test set. |
| **Contamination** | Numeric covariates: additive Gaussian noise `x + σ·sd(x)·ε`. Binary/categorical covariates: resampled from the empirical marginal with probability `p = 1 − 1/√(1+σ²)`, which matches the same correlation with the clean value. |

All three methods read the exact same pre-computed training rows. Because the random draw
(`np.random.default_rng(r).permutation`) was verified to give bit-identical results under both
numpy versions used, every method sees the same data even though they run in different Python
environments.

## Repository layout

```
cate-benchmark/
├── scripts/
│   ├── common.py                 shared constants, data loading, PEHE
│   ├── prepare_data.py           build the train/test splits (.npz)
│   ├── run_reps.py               main runner: one method × all (dataset, n, draw) cells
│   ├── run_contamination.py      runner for the contamination sweep
│   ├── contaminate.py            covariate-noise model
│   ├── run_{tabpfn,causalpfn,dopfn}.py   single-draw runners (Experiment 1)
│   ├── plot_boxplot.py           Figure 1
│   ├── plot_contamination.py     Figure 2
│   └── plot_style.py             shared figure style and reference lines
├── results/
│   ├── reps/                     per-draw results (7,500 records)
│   ├── contamination/            per-draw results (4,800 records)
│   ├── *.json                    single-draw runs at every training size (Experiment 1 table)
│   └── summary_*.md              median [Q1, Q3] tables for Experiments 2 and 3
├── figures/                      the two poster figures shown above
├── poster.pdf                    A0 research poster
├── requirements/                 frozen environments
└── third_party/                  Do-PFN compatibility patch; upstream repos are cloned here
```

Every runner is **resumable**: results are merged by `(dataset, n_train, rep)`, so re-running a
command skips cells that are already on disk.

## Reproducing

All per-draw results are committed in `results/`. To only redraw the figures from them, do steps
1–3 and 5; this doesn't need a GPU. Step 4 re-runs the experiments themselves.

**1. Clone the two upstream model repositories** at the commits used in this study:

```bash
git clone https://github.com/vdblm/CausalPFN third_party/CausalPFN
git -C third_party/CausalPFN checkout 7da4afa

git clone https://github.com/jr2021/Do-PFN third_party/Do-PFN
git -C third_party/Do-PFN checkout 90d6743
git -C third_party/Do-PFN apply ../dopfn.patch
```

`third_party/dopfn.patch` makes three small compatibility fixes to Do-PFN: it replaces hard-coded
cluster data paths, adds a missing `typing.Optional` import, and updates for the scikit-learn
`force_all_finite` → `ensure_all_finite` rename. To use clones stored somewhere else, set the
`CAUSALPFN_REPO` and `DOPFN_REPO` environment variables.

**2. Create two Python environments.** CausalPFN's dependencies need Python 3.10:

```bash
# TabPFN v3 + Do-PFN
python3.12 -m venv .venv-tabpfn && .venv-tabpfn/bin/pip install -r requirements/tabpfn-dopfn.txt
# CausalPFN
python3.10 -m venv .venv-causalpfn && .venv-causalpfn/bin/pip install -r requirements/causalpfn.txt

export PY_DOPFN=$PWD/.venv-tabpfn/bin/python
export PY_CAUSALPFN=$PWD/.venv-causalpfn/bin/python
```

TabPFN v3 weights require an access token from [Prior Labs](https://priorlabs.ai).
CausalPFN weights are downloaded automatically on first use. Do-PFN weights are included in its
repository.

**3. Prepare the data splits.** This reads the four datasets through the CausalPFN benchmark
loaders and writes the train/test splits to `data/`. ACIC 2016 is downloaded on first use.

```bash
cd scripts
$PY_DOPFN prepare_data.py
CATE_TRAIN_SIZES=2000,4000,6000,8000,10000 $PY_DOPFN prepare_data.py --datasets lalonde_cps
```

**4. Run the experiments.** Experiment 1 uses the full training pool of each dataset:

```bash
for pair in ihdp:672 acic2016:4321 lalonde_cps:14559 lalonde_psid:2407; do
  ds=${pair%%:*}; n=${pair##*:}
  CATE_TRAIN_SIZES=$n $PY_DOPFN     prepare_data.py --datasets $ds
  CATE_TRAIN_SIZES=$n $PY_DOPFN     run_tabpfn.py --learner t --datasets $ds
  CATE_TRAIN_SIZES=$n $PY_DOPFN     run_dopfn.py --datasets $ds
  CATE_TRAIN_SIZES=$n $PY_CAUSALPFN run_causalpfn.py --datasets $ds
done
```

The table divides each root-PEHE by the dataset's oracle error, sd(τ) on the test set.

Experiment 2, the sample-size study:

```bash
$PY_DOPFN     run_reps.py --method tabpfn_t  --reps 100
$PY_DOPFN     run_reps.py --method dopfn     --reps 100
$PY_CAUSALPFN run_reps.py --method causalpfn --reps 100

# Lalonde CPS at large n
export CATE_TRAIN_SIZES=2000,4000,6000,8000,10000 CATE_FIG_SUFFIX=_large_n
$PY_DOPFN     run_reps.py --method tabpfn_t  --reps 100 --datasets lalonde_cps
$PY_DOPFN     run_reps.py --method dopfn     --reps 100 --datasets lalonde_cps
$PY_CAUSALPFN run_reps.py --method causalpfn --reps 100 --datasets lalonde_cps
```

Experiment 3, the contamination study:

```bash
unset CATE_TRAIN_SIZES CATE_FIG_SUFFIX
for m in tabpfn_t dopfn; do
  $PY_DOPFN run_contamination.py --method $m --reps 100 --datasets lalonde_cps lalonde_psid ihdp acic2016
done
$PY_CAUSALPFN run_contamination.py --method causalpfn --reps 100 --datasets lalonde_cps lalonde_psid ihdp acic2016
```

Hardware used: one NVIDIA RTX 5090 Laptop GPU (24 GB). Do-PFN runs on CPU because of a dtype bug
in its CUDA forward pass.

> **Note:** Don't pipe runner output (e.g. `| tail`) when running in the background. Once the parent
> shell exits, the process blocks forever writing to stdout.

**5. Plot.** This writes the figures to `figures/`:

```bash
unset CATE_TRAIN_SIZES CATE_FIG_SUFFIX
$PY_DOPFN plot_boxplot.py                                    # Figure 1 + per-dataset panels
$PY_DOPFN plot_contamination.py --datasets ihdp acic2016 lalonde_cps lalonde_psid   # Figure 2
```

## Limitations

- A single realization of each dataset is used. Box spreads reflect training-sample variability,
  not variability of the data-generating process.
- Only the public Do-PFN **v1** checkpoint could be evaluated. The v1.1 model reported in the
  paper has not been released.
- Experiment 1 is a single training run per dataset, so its table has no error bars.

## References

- Hollmann et al. (2025). *Accurate predictions on small data with a tabular foundation model.* Nature.
- Balazadeh et al. (2025). *CausalPFN: Amortized Causal Effect Estimation via In-Context Learning.* arXiv:2506.07918.
- Robertson et al. (2025). *Do-PFN: In-Context Learning for Causal Effect Estimation.* NeurIPS 2025.
- Hill (2011). *Bayesian Nonparametric Modeling for Causal Inference.* (IHDP)
- Dorie et al. (2019). *Automated versus Do-It-Yourself Methods for Causal Inference.* (ACIC 2016)
- LaLonde (1986); Dehejia & Wahba (1999). (Lalonde CPS / PSID)

The datasets are not redistributed here. `prepare_data.py` loads them from the CausalPFN
repository and the public ACIC 2016 source, under their original terms.

## Author

**Chia-Yu Ou**, Department of Information Management and Finance, National Yang Ming Chiao Tung
University (NYCU).
Summer research project, 2026, supervised by Dr. Tso-Jung Yen.

## License

Code is released under the [MIT License](LICENSE).
