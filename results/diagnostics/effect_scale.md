# Effect-scale diagnostic (Experiment 1, full training pool)

| Dataset | Covariates | True scale sd(τ)/sd(y) | Method | Shrinkage sd(τ̂)/sd(τ) | PEHE / sd(τ) | Residual / sd(τ) |
|---|---|---|---|---|---|---|
| IHDP | 25 | 0.36 | TabPFN v3 (T-learner) | 1.13 | 0.42 | 0.41 |
| | | | CausalPFN | 0.99 | 0.20 | 0.20 |
| | | | Do-PFN | 1.01 | 2.13 | 1.40 |
| ACIC 2016 | 58 | 0.81 | TabPFN v3 (T-learner) | 0.97 | 0.10 | 0.10 |
| | | | CausalPFN | 0.97 | 0.18 | 0.17 |
| | | | Do-PFN | 0.07 | 1.04 | 1.01 |
| Lalonde CPS | 8 | 1.10 | TabPFN v3 (T-learner) | 0.66 | 0.84 | 0.82 |
| | | | CausalPFN | 0.54 | 0.83 | 0.83 |
| | | | Do-PFN | 0.13 | 1.13 | 0.96 |
| Lalonde PSID | 8 | 1.11 | TabPFN v3 (T-learner) | 0.68 | 0.77 | 0.77 |
| | | | CausalPFN | 0.59 | 0.78 | 0.77 |
| | | | Do-PFN | 0.22 | 1.23 | 0.91 |
