# Experiment 1: oracle-normalized root-PEHE on the full training pool

Each value is root-PEHE divided by sd(tau), the error of a constant predictor that
knows the true average effect. Values below 1 beat that predictor. Best per dataset in bold.

| Model | IHDP | ACIC 2016 | Lalonde CPS | Lalonde PSID |
|---|---|---|---|---|
| TabPFN v3 (T-learner) | 0.417 | **0.102** | 0.837 | **0.770** |
| CausalPFN | **0.198** | 0.176 | **0.827** | 0.784 |
| Do-PFN | 2.126 | 1.040 | 1.134 | 1.225 |
