# Day 11 — Joint OOS Validation Summary

## 1. Research objective

Day 11 evaluates whether the nine frozen network characteristics provide incremental strict out-of-sample information for future idiosyncratic volatility beyond the frozen traditional-characteristic baseline.

No model is refitted in Step 6. All statistics below are assembled from the frozen outputs of Steps 2, 4, and 5.

## 2. Frozen factor redundancy

| Window | Native effective rank | Common-date effective rank | Native PC1 share | Native condition number |
|---:|---:|---:|---:|---:|
| 60 | 3.751 | 3.781 | 43.7% | 178.7 |
| 120 | 4.030 | 4.027 | 40.9% | 127.3 |
| 252 | 4.429 | 4.429 | 36.4% | 91.8 |

## 3. Strict OOS incremental performance

| Scope | W | Baseline IC | Network IC | Delta IC | Positive months | Positive years | Relative OOS R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRIMARY | 60 | 0.4902 | 0.5039 | 0.0137 | 84.4% | 90.0% | 2.7% |
| PRIMARY | 120 | 0.4916 | 0.5036 | 0.0119 | 77.8% | 100.0% | 2.3% |
| PRIMARY | 252 | 0.4923 | 0.5054 | 0.0130 | 82.4% | 100.0% | 2.6% |
| FULL_RANK | 60 | 0.4949 | 0.5049 | 0.0100 | 83.5% | 85.7% | 2.0% |
| FULL_RANK | 120 | 0.4981 | 0.5060 | 0.0078 | 74.7% | 100.0% | 1.6% |
| FULL_RANK | 252 | 0.5005 | 0.5109 | 0.0104 | 79.7% | 100.0% | 2.1% |

## 4. Temporal stability

| Scope | W | Years | Positive-year share | Worst year | Worst-year Delta IC | Best year | Best-year Delta IC |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRIMARY | 60 | 10 | 90.0% | 2021 | -0.0003 | 2018 | 0.0293 |
| PRIMARY | 120 | 10 | 100.0% | 2021 | 0.0009 | 2018 | 0.0273 |
| PRIMARY | 252 | 9 | 100.0% | 2021 | 0.0022 | 2018 | 0.0226 |
| FULL_RANK | 60 | 7 | 85.7% | 2021 | -0.0003 | 2024 | 0.0230 |
| FULL_RANK | 120 | 7 | 100.0% | 2021 | 0.0009 | 2024 | 0.0206 |
| FULL_RANK | 252 | 7 | 100.0% | 2021 | 0.0022 | 2024 | 0.0225 |

## 5. Conditional leave-one-block-out contributions

Positive marginal IC means that removing the block lowers OOS IC conditional on the remaining network blocks. Contributions are not additive.

| Scope | W | Block | Marginal IC | Positive months | Positive years | MSE gain |
|---|---:|---|---:|---:|---:|---:|
| PRIMARY | 60 | LEVEL_POSITION | 0.0057 | 75.2% | 90.0% | 0.0114 |
| PRIMARY | 60 | CROSS_INDUSTRY_COMPOSITION | 0.0000 | 54.1% | 70.0% | 0.0000 |
| PRIMARY | 60 | RECONFIGURATION | 0.0050 | 66.1% | 70.0% | 0.0101 |
| PRIMARY | 60 | NEIGHBOR_STABILITY | 0.0029 | 78.9% | 100.0% | 0.0058 |
| PRIMARY | 120 | LEVEL_POSITION | 0.0057 | 75.0% | 80.0% | 0.0114 |
| PRIMARY | 120 | CROSS_INDUSTRY_COMPOSITION | 0.0000 | 50.9% | 60.0% | 0.0001 |
| PRIMARY | 120 | RECONFIGURATION | 0.0072 | 69.4% | 80.0% | 0.0144 |
| PRIMARY | 120 | NEIGHBOR_STABILITY | 0.0011 | 69.4% | 100.0% | 0.0022 |
| PRIMARY | 252 | LEVEL_POSITION | 0.0048 | 75.5% | 66.7% | 0.0097 |
| PRIMARY | 252 | CROSS_INDUSTRY_COMPOSITION | 0.0000 | 51.0% | 66.7% | 0.0001 |
| PRIMARY | 252 | RECONFIGURATION | 0.0109 | 81.4% | 100.0% | 0.0219 |
| PRIMARY | 252 | NEIGHBOR_STABILITY | 0.0000 | 52.9% | 44.4% | 0.0000 |
| FULL_RANK | 60 | LEVEL_POSITION | 0.0036 | 70.9% | 85.7% | 0.0071 |
| FULL_RANK | 60 | CROSS_INDUSTRY_COMPOSITION | 0.0000 | 55.7% | 85.7% | 0.0001 |
| FULL_RANK | 60 | RECONFIGURATION | 0.0027 | 64.6% | 57.1% | 0.0054 |
| FULL_RANK | 60 | NEIGHBOR_STABILITY | 0.0028 | 79.7% | 100.0% | 0.0056 |
| FULL_RANK | 120 | LEVEL_POSITION | 0.0028 | 68.4% | 71.4% | 0.0055 |
| FULL_RANK | 120 | CROSS_INDUSTRY_COMPOSITION | 0.0000 | 54.4% | 71.4% | 0.0001 |
| FULL_RANK | 120 | RECONFIGURATION | 0.0040 | 64.6% | 71.4% | 0.0079 |
| FULL_RANK | 120 | NEIGHBOR_STABILITY | 0.0012 | 69.6% | 100.0% | 0.0025 |
| FULL_RANK | 252 | LEVEL_POSITION | 0.0032 | 70.9% | 57.1% | 0.0065 |
| FULL_RANK | 252 | CROSS_INDUSTRY_COMPOSITION | 0.0000 | 49.4% | 57.1% | 0.0000 |
| FULL_RANK | 252 | RECONFIGURATION | 0.0087 | 78.5% | 100.0% | 0.0175 |
| FULL_RANK | 252 | NEIGHBOR_STABILITY | -0.0001 | 51.9% | 42.9% | -0.0001 |

## 6. Integrated interpretation

- The frozen all-network model has positive mean incremental OOS rank IC at all three network windows in the PRIMARY evaluation.
- The positive incremental OOS rank IC also holds at all three windows on the FULL_RANK numerical-robustness sample.
- The incremental OOS improvement is positive in a majority of calendar years at all three windows.
- Block-level statistics are conditional LOBO diagnostics and must not be interpreted as an additive decomposition or as a new feature-selection rule.

## 7. Formal QA

- all_formal_qa_pass: `True`
- joint_summary_row_count: `6`
- block_summary_row_count: `24`
- No new model fitting, factor selection, window selection, or outcome-driven specification change is performed in Step 6.