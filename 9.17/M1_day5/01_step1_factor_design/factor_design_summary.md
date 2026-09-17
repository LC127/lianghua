# Day 5 Step 1 — Frozen Network Factor Design

## Research objective

Convert the validated full-market stock network into a reproducible stock-level factor panel for subsequent Alpha and Risk research.

## Primary network

- Primary network: `M1_RESIDUAL_POSITIVE`
- Benchmark network: `B0_RAW_POSITIVE`
- Primary density: `1.0%`
- Robustness density grid: `[0.2, 0.5, 1.0, 2.0]%`
- Windows: `[60, 120, 252]`
- Main industry definition: `industry_id1`

## Timing rule

All factor values at analysis date `t` may use information available through `t` only.

Future outcomes start strictly after `t` and end at the next analysis date.

Therefore the empirical relation will follow:

\[
X_{i,t} \rightarrow Y_{i,t+1}
\]

rather than contemporaneous feature-outcome matching.

## Factor families

1. Centrality
2. Raw–Residual differential
3. Industry backbone / cross-industry decomposition
4. Dynamic network features
5. Community / bridge features

Total predefined candidate factors: **20**

Core factors: **18**

## Important restrictions

- No factor sign is selected using future returns.
- No best window is selected in Step 1.
- No best density is selected using future outcomes.
- Missing lagged features are kept missing, not filled with zero.
- Current stock universe is never defined using future return availability.
- Industry and size neutralization are deferred to formal factor diagnostics.
- The Residual network remains an association network, not a conditional-independence network.

## Input coverage

|   window |   analysis_date_count |   all_sources_complete_dates |   all_sources_complete_share |   has_step2_node_universe_count |   has_step2_raw_edge_master_count |   has_step2_residual_edge_master_count |   has_step3_raw_node_metrics_count |   has_step3_residual_node_metrics_count |   has_step4_raw_community_count |   has_step4_residual_community_count |
|---------:|----------------------:|-----------------------------:|-----------------------------:|--------------------------------:|----------------------------------:|---------------------------------------:|-----------------------------------:|----------------------------------------:|--------------------------------:|-------------------------------------:|
|       60 |                   138 |                          138 |                            1 |                             138 |                               138 |                                    138 |                                138 |                                     138 |                             138 |                                  138 |
|      120 |                   135 |                          135 |                            1 |                             135 |                               135 |                                    135 |                                135 |                                     135 |                             135 |                                  135 |
|      252 |                   129 |                          129 |                            1 |                             129 |                               129 |                                    129 |                                129 |                                     129 |                             129 |                                  129 |

## Design hash

`b2f66e9ac2c999e7ceeef0ed453b3bf500fa727eb9913cccfac0b77b8a9caa8e`

This hash should be stored with later factor-panel outputs to confirm that all subsequent analyses use the same frozen design.
