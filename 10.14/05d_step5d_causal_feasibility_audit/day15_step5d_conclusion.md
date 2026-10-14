# Day15 Step5-D — Causal Identification Feasibility Audit

**Verdict: CAUSAL_IDENTIFICATION_NOT_ESTABLISHED.** This is a robustness audit, not a causal-effect estimate.

- Residual W120 POST_SECTOR association: 0.005709 (0.571 percentage points), BH q=0.00177268.
- Residual W120 PRE_SECTOR placebo: 0.001271, BH q=0.550569; exploratory +/-0.5 pp equivalence: WITHIN EXPLORATORY TOLERANCE (TOST p=0.04001).
- Residual–Raw POST_SECTOR contrast BH q=0.398684.
- First 51 dates POST_SECTOR mean: 0.007837; last 51: 0.003580; this retrospective split is NOT independent out-of-sample confirmation.
- Step5-C: 0/16 walk-forward forecast tests pass BH 5%; fresh untouched holdout unavailable.
- Source shocks are endogenous, market/news common shocks and interference remain, and post/pre have different risk sets. Do not subtract them as a causal DiD estimator.

**Next research requirement:** externally justified stock-specific shocks, suitable untreated controls, credible exclusion/exchangeability arguments, and genuinely unseen validation dates.
