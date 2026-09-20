# M1 A-Share Stock Association Network
## Integrated Research Summary: Day 1–Day 7

**Generated:** 2026-09-20 14:58:31

**Step-6 design hash:** `fc19604666c835006f862148573c6df3d1186dbd553ab6a404f5259c1771a22d`

---

# 1. Research Objective

The M1 project studies whether the cross-sectional position of A-share
stocks in rolling association networks contains economically and
statistically meaningful information about subsequent stock returns
and risks.

The research pipeline deliberately separates:

1. network construction;
2. network-feature engineering;
3. future-outcome construction;
4. return validation;
5. risk validation;
6. robustness and incremental-information tests;
7. turnover and transaction-cost diagnostics.

The central empirical question is therefore not merely whether a
network variable is correlated with a future outcome, but whether the
relation survives alternative market states, traditional stock
characteristics, conditional cross-sectional regressions, and economic
implementation constraints.

---

# 2. Core Research Discipline

The empirical pipeline follows four fixed restrictions:

- No future outcome is used to construct the current network or current
  network feature.
- The nine-factor set is frozen before the main future return and risk
  validation.
- Factor signs are not reversed after observing outcomes.
- W60, W120 and W252 are all retained; no optimal window is selected
  ex post.

Accordingly, the final robustness matrix is an **evidence map rather
than a factor-ranking exercise**.

---

# 3. Return Conventions

Two return definitions are intentionally kept separate.

For network construction, the project uses the validated network
return available only when adjacent market dates are both valid
trading observations.

For future holding-period labels, the stock holding return is

\[
r^{hold}_{i,s}
=
\begin{cases}
0, & \text{stock suspended on day }s,\\
r^{last-trade}_{i,s}, & \text{stock trades with a valid last-trade return},\\
NA, & \text{otherwise}.
\end{cases}
\]

This allows suspended positions to remain marked at stale value while
the first reopening observation captures the full return accumulated
since the last actual trade.

---

# 4. Day 1–Day 7 Research Pipeline

| Day | Stage | Research question | Guardrail |
|---|---|---|---|
| 1 | Full-market data preparation and QA | Can a clean point-in-time A-share panel and consistent return system be constructed? | Network construction return and holding-period return are kept as two distinct conventions. |
| 2 | Initial market association-network construction | What structure is visible in broad stock-return association networks? | Only contemporaneously available historical returns enter network construction. |
| 3 | Residual-network construction | How does network structure change after removing common market-related dependence? | Residual correlation is interpreted as a conditional-association proxy, not as conditional independence. |
| 4 | Full-market network structural validation | Are observed network structures stable across construction choices? | Network-construction robustness is separated from subsequent return/risk validation. |
| 5 | Network feature engineering | Which interpretable node-level characteristics summarize network position and dynamics? | The nine-factor set is frozen before future-return and future-risk labels are used. |
| 6 | Initial forward return and risk validation | Do frozen network variables contain forward return or risk information? | No outcome-based sign flip, factor deletion or network-window selection. |
| 7 | Robustness and incremental validation | Do Day-6 relations survive regimes, traditional characteristics, conditional regressions and transaction costs? | Robustness diagnostics do not re-freeze factors or select an optimal window. |

---

# 5. Day-6 Forward Validation Framework

The primary Alpha target is future excess return over the interval

\[
(analysis\_date, next\_analysis\_date].
\]

The initial cross-sectional Rank IC is

\[
IC_{t,W,f}
=
Corr_S
\left(
F_{i,t,W,f},
Y_{i,t,W}
\right).
\]

Portfolio validation uses five equal-weight portfolios, where Q1
contains the lowest factor values and Q5 the highest. The canonical
spread is always

\[
R_{Q5}-R_{Q1},
\]

and negative factors are not flipped after observing outcomes.

Risk validation considers:

- future realized volatility;
- future downside volatility;
- future maximum drawdown;
- future idiosyncratic volatility.

Industry and size neutralization is implemented cross-sectionally
using contemporaneous information.

---

# 6. Day-7 Robustness Chain

Day 7 extends the initial validation through four increasingly strict
tests.

## 6.1 Regime robustness

Relations are evaluated across:

- 2015–2018;
- 2019–2022;
- 2023–2026;
- Bull / Bear states;
- High / Low volatility states;
- their joint regimes.

Regimes are defined only using information available by the current
analysis date.

## 6.2 Traditional stock characteristics

The full specification controls for:

\[
Industry
+
Size
+
Momentum
+
Reversal
+
PastVolatility
+
Turnover.
\]

The main comparison is

\[
Industry+Size
\rightarrow
FullControls.
\]

A common stock sample is used so that changes in the estimated signal
cannot be attributed merely to changes in cross-sectional coverage.

## 6.3 Fama–MacBeth regression

For every factor separately,

\[
R_{i,t+1}
=
\alpha_{Industry(i,t)}
+
\beta_t F_{i,t}
+
\gamma_t^\top X_{i,t}
+
\varepsilon_{i,t+1}.
\]

The final network coefficient is

\[
\bar\beta
=
\frac1T
\sum_{t=1}^T
\hat\beta_t,
\]

with HAC inference applied to the monthly coefficient sequence.

## 6.4 Turnover and transaction costs

For target portfolio weights \(w_{i,t}\), the pre-trade weight is

\[
w^-_{i,t}
=
\frac{
w_{i,t-1}
(1+R_{i,t-1\rightarrow t})
}{
\sum_j
w_{j,t-1}
(1+R_{j,t-1\rightarrow t})
}.
\]

The drift-adjusted half-\(L_1\) turnover is

\[
TO_t
=
\frac12
\sum_i
|w_{i,t}-w^-_{i,t}|.
\]

For one-way transaction cost \(c\), the cost drag of the two-leg
Q5−Q1 portfolio is

\[
Cost_t
=
2c
\left(
TO_{Q5,t}
+
TO_{Q1,t}
\right).
\]

The project reports 5, 10, 20 and 30 bp scenarios.

---

# 7. Final Factor Robustness Matrix

| Factor | Family | Evidence role | Full-control mean IC | M3 FMB beta | Mean spread | Mean IVOL IC | Stable risks | Cost |
|---|---|---|---|---|---|---|---|---|
| residual_degree_percentile | LEVEL_STRUCTURAL | RETURN_AND_RISK_EVIDENCE | 0.0087 | 8.61 bp/SD | 26.02 bp | -0.0583 | 4/4 | MODERATE_COST_SENSITIVITY |
| delta_degree_percentile | LEVEL_STRUCTURAL | RETURN_AND_RISK_EVIDENCE | -0.0063 | -9.77 bp/SD | -24.41 bp | 0.0363 | 1/4 | MODERATE_COST_SENSITIVITY |
| cross_industry_degree_percentile | LEVEL_STRUCTURAL | RETURN_AND_RISK_EVIDENCE | 0.0087 | 5.96 bp/SD | 20.62 bp | -0.0586 | 4/4 | MODERATE_COST_SENSITIVITY |
| cross_industry_degree_ratio | LEVEL_STRUCTURAL | WEAK_OR_MIXED_EVIDENCE | -0.0006 | -6.81 bp/SD | -15.64 bp | 0.0056 | 3/4 | MODERATE_COST_SENSITIVITY |
| delta_residual_degree_percentile_1m | DYNAMIC_1M | WEAK_OR_MIXED_EVIDENCE | 0.0018 | 2.65 bp/SD | 7.23 bp | -0.0020 | 0/4 | HIGH_COST_SENSITIVITY |
| delta_cross_industry_degree_percentile_1m | DYNAMIC_1M | WEAK_OR_MIXED_EVIDENCE | 0.0024 | 1.94 bp/SD | 6.44 bp | -0.0034 | 0/4 | HIGH_COST_SENSITIVITY |
| neighbor_jaccard_1m | DYNAMIC_1M | RETURN_AND_RISK_EVIDENCE | 0.0059 | 3.23 bp/SD | 9.10 bp | -0.0418 | 1/4 | HIGH_COST_SENSITIVITY |
| neighbor_retention_1m | DYNAMIC_1M | WEAK_OR_MIXED_EVIDENCE | 0.0050 | 3.15 bp/SD | 9.58 bp | -0.0320 | 0/4 | HIGH_COST_SENSITIVITY |
| outside_community_degree_percentile | LEVEL_STRUCTURAL | RISK_DOMINANT_EVIDENCE | 0.0069 | 1.05 bp/SD | 5.06 bp | -0.0727 | 4/4 | HIGH_COST_SENSITIVITY |

The table summarizes return information, conditional Fama–MacBeth
coefficients, future-IVOL information, cross-risk stability and
transaction-cost sensitivity. These dimensions should be interpreted
jointly rather than reduced to a single score.

---

# 8. Cross-Factor Findings

| Topic | Finding | Interpretation |
|---|---|---|
| Return consistency | 9 of 9 factors have the same sign for full-control mean Rank IC and mean M3 Fama-MacBeth beta. | Rank-based and conditional-linear return evidence can therefore be compared directly for most factors, while disagreements should be treated as mixed evidence. |
| Risk robustness | 3 factors preserve the same direction across W60/W120/W252 for all four future-risk targets. | Risk evidence is assessed separately from return evidence and is not reduced to a single Alpha criterion. |
| Transaction-cost sensitivity | 5 of 9 factors are classified as HIGH_COST_SENSITIVITY under the fixed Step-5 descriptive rule. | Statistical predictability and economic implementability are treated as separate dimensions. |
| Structural vs dynamic features | Mean round-trip traded-notional ratio is 1.551 for level/structural variables and 2.958 for one-month dynamic variables. | Short-horizon network dynamics require more portfolio rebalancing on average. |
| Evidence-role distribution | RETURN_AND_RISK_EVIDENCE=4; WEAK_OR_MIXED_EVIDENCE=4; RISK_DOMINANT_EVIDENCE=1 | The final research output is a role-based evidence map rather than a best-to-worst factor ranking. |

---

# 9. Factor-Level Evidence Profiles

### `residual_degree_percentile`
- Structural family: `LEVEL_STRUCTURAL`.
- Step-5 evidence role: `RETURN_AND_RISK_EVIDENCE`; cost profile: `MODERATE_COST_SENSITIVITY`.
- Full-control mean Alpha Rank IC = 0.0087; M3 Fama–MacBeth mean coefficient = 8.61 bp/SD; mean full-control Q5−Q1 spread = 26.02 bp.
- Mean future-IVOL Rank IC = -0.0583; 4/4 risk targets preserve the same sign across W60/W120/W252.
- Strict regime consistency: Alpha=False, IVOL=True.
- Mean round-trip traded-notional ratio = 1.355; mean descriptive one-way break-even cost = 25.06 bp.

### `delta_degree_percentile`
- Structural family: `LEVEL_STRUCTURAL`.
- Step-5 evidence role: `RETURN_AND_RISK_EVIDENCE`; cost profile: `MODERATE_COST_SENSITIVITY`.
- Full-control mean Alpha Rank IC = -0.0063; M3 Fama–MacBeth mean coefficient = -9.77 bp/SD; mean full-control Q5−Q1 spread = -24.41 bp.
- Mean future-IVOL Rank IC = 0.0363; 1/4 risk targets preserve the same sign across W60/W120/W252.
- Strict regime consistency: Alpha=True, IVOL=True.
- Mean round-trip traded-notional ratio = 1.564; mean descriptive one-way break-even cost = 15.13 bp.

### `cross_industry_degree_percentile`
- Structural family: `LEVEL_STRUCTURAL`.
- Step-5 evidence role: `RETURN_AND_RISK_EVIDENCE`; cost profile: `MODERATE_COST_SENSITIVITY`.
- Full-control mean Alpha Rank IC = 0.0087; M3 Fama–MacBeth mean coefficient = 5.96 bp/SD; mean full-control Q5−Q1 spread = 20.62 bp.
- Mean future-IVOL Rank IC = -0.0586; 4/4 risk targets preserve the same sign across W60/W120/W252.
- Strict regime consistency: Alpha=False, IVOL=True.
- Mean round-trip traded-notional ratio = 1.459; mean descriptive one-way break-even cost = 17.67 bp.

### `cross_industry_degree_ratio`
- Structural family: `LEVEL_STRUCTURAL`.
- Step-5 evidence role: `WEAK_OR_MIXED_EVIDENCE`; cost profile: `MODERATE_COST_SENSITIVITY`.
- Full-control mean Alpha Rank IC = -0.0006; M3 Fama–MacBeth mean coefficient = -6.81 bp/SD; mean full-control Q5−Q1 spread = -15.64 bp.
- Mean future-IVOL Rank IC = 0.0056; 3/4 risk targets preserve the same sign across W60/W120/W252.
- Strict regime consistency: Alpha=False, IVOL=False.
- Mean round-trip traded-notional ratio = 1.479; mean descriptive one-way break-even cost = 11.07 bp.

### `delta_residual_degree_percentile_1m`
- Structural family: `DYNAMIC_1M`.
- Step-5 evidence role: `WEAK_OR_MIXED_EVIDENCE`; cost profile: `HIGH_COST_SENSITIVITY`.
- Full-control mean Alpha Rank IC = 0.0018; M3 Fama–MacBeth mean coefficient = 2.65 bp/SD; mean full-control Q5−Q1 spread = 7.23 bp.
- Mean future-IVOL Rank IC = -0.0020; 0/4 risk targets preserve the same sign across W60/W120/W252.
- Strict regime consistency: Alpha=False, IVOL=True.
- Mean round-trip traded-notional ratio = 3.206; mean descriptive one-way break-even cost = 2.84 bp.

### `delta_cross_industry_degree_percentile_1m`
- Structural family: `DYNAMIC_1M`.
- Step-5 evidence role: `WEAK_OR_MIXED_EVIDENCE`; cost profile: `HIGH_COST_SENSITIVITY`.
- Full-control mean Alpha Rank IC = 0.0024; M3 Fama–MacBeth mean coefficient = 1.94 bp/SD; mean full-control Q5−Q1 spread = 6.44 bp.
- Mean future-IVOL Rank IC = -0.0034; 0/4 risk targets preserve the same sign across W60/W120/W252.
- Strict regime consistency: Alpha=False, IVOL=True.
- Mean round-trip traded-notional ratio = 3.270; mean descriptive one-way break-even cost = 2.26 bp.

### `neighbor_jaccard_1m`
- Structural family: `DYNAMIC_1M`.
- Step-5 evidence role: `RETURN_AND_RISK_EVIDENCE`; cost profile: `HIGH_COST_SENSITIVITY`.
- Full-control mean Alpha Rank IC = 0.0059; M3 Fama–MacBeth mean coefficient = 3.23 bp/SD; mean full-control Q5−Q1 spread = 9.10 bp.
- Mean future-IVOL Rank IC = -0.0418; 1/4 risk targets preserve the same sign across W60/W120/W252.
- Strict regime consistency: Alpha=True, IVOL=True.
- Mean round-trip traded-notional ratio = 2.539; mean descriptive one-way break-even cost = 3.96 bp.

### `neighbor_retention_1m`
- Structural family: `DYNAMIC_1M`.
- Step-5 evidence role: `WEAK_OR_MIXED_EVIDENCE`; cost profile: `HIGH_COST_SENSITIVITY`.
- Full-control mean Alpha Rank IC = 0.0050; M3 Fama–MacBeth mean coefficient = 3.15 bp/SD; mean full-control Q5−Q1 spread = 9.58 bp.
- Mean future-IVOL Rank IC = -0.0320; 0/4 risk targets preserve the same sign across W60/W120/W252.
- Strict regime consistency: Alpha=True, IVOL=True.
- Mean round-trip traded-notional ratio = 2.816; mean descriptive one-way break-even cost = 4.24 bp.

### `outside_community_degree_percentile`
- Structural family: `LEVEL_STRUCTURAL`.
- Step-5 evidence role: `RISK_DOMINANT_EVIDENCE`; cost profile: `HIGH_COST_SENSITIVITY`.
- Full-control mean Alpha Rank IC = 0.0069; M3 Fama–MacBeth mean coefficient = 1.05 bp/SD; mean full-control Q5−Q1 spread = 5.06 bp.
- Mean future-IVOL Rank IC = -0.0727; 4/4 risk targets preserve the same sign across W60/W120/W252.
- Strict regime consistency: Alpha=True, IVOL=True.
- Mean round-trip traded-notional ratio = 1.900; mean descriptive one-way break-even cost = 8.19 bp.



---

# 10. Main Integrated Interpretation

The complete Day 1–7 evidence shows that stock-network characteristics
contain heterogeneous information.

A recurring pattern is that several level or structural network
variables preserve forward-risk information after controlling for
industry, size, momentum, short-term reversal, historical volatility
and turnover.

By contrast, a substantial portion of the initial return predictability
of several network variables is attenuated once traditional stock
characteristics are introduced. This means the empirical results do
not support treating every network characteristic as an independent
Alpha factor.

The distinction between return information and risk information is
therefore central to the final interpretation.

In particular, the project should distinguish:

\[
\text{predictive return information}
\neq
\text{predictive risk information}
\neq
\text{tradable strategy performance}.
\]

The transaction-cost exercise additionally shows that short-horizon
dynamic network variables can require substantially greater
rebalancing than slower-moving structural characteristics.

---

# 11. Interpretation of Statistical Evidence

HAC \(t\)-statistics are reported as exploratory nominal diagnostics.

The project has evaluated multiple:

- factors;
- network windows;
- targets;
- portfolio specifications;
- regimes.

Therefore a nominal threshold such as

\[
|t|\ge1.96
\]

must not be interpreted as a multiple-testing-adjusted discovery rule.

Similarly, a QA PASS means that the computational design and internal
consistency checks succeeded. It does **not** imply that an economic
or statistical hypothesis has been established.

---

# 12. Important Limitations

Several limitations should remain explicit.

First, all current results are descriptive predictive associations.
They do not establish causality between network position and future
stock outcomes.

Second, the association network is based on rolling historical return
dependence. Residual correlation should not be interpreted as
conditional independence.

Third, longer rolling windows mechanically overlap more strongly,
which can contribute to persistence in network features.

Fourth, transaction-cost calculations are stylized equal-weight
diagnostics. They do not yet include all real trading frictions such as
market impact, price limits, liquidity constraints, short-sale
availability or implementation delay.

Fifth, the current traditional controls are intentionally limited to
variables whose point-in-time construction is well defined in the
current database. Additional accounting characteristics require a
separate PIT-data validation stage.

---

# 13. Current Research Conclusion

The Day 1–7 pipeline supports a more nuanced conclusion than a simple
"network Alpha" interpretation.

The empirical evidence suggests that network position and network
reorganization can contain information about subsequent stock
outcomes, but the strongest and most persistent component is often
associated with future risk rather than independent return Alpha.

After controlling for traditional stock characteristics, some return
relations remain modest and directionally stable, while several
future-risk relations remain considerably more persistent.

Accordingly, a natural continuation of M1 is to investigate whether

\[
\boxed{
\text{stock-network structure contains incremental forward-looking
idiosyncratic-risk information}
}
\]

and to study the economic mechanism behind that relation.

---

# 14. Suggested Next Research Directions

The next phase should not simply add more network variables. More
informative extensions include:

1. genuine out-of-sample or expanding-window validation;
2. deeper investigation of idiosyncratic-risk mechanisms;
3. interaction between network characteristics and liquidity,
   size or market stress;
4. portfolio applications designed ex ante rather than obtained by
   reversing an observed in-sample factor sign;
5. alternative network estimators as robustness checks;
6. formal multiple-testing or false-discovery control if a larger
   factor search is resumed.

---

# 15. Reproducibility and Governance

The final Day-7 summary inherits only outputs whose formal upstream QA
passed.

No factor was removed in Step 6.

No factor orientation was changed.

No network window was selected.

No additional statistical test was introduced during report
generation.

The report is therefore a deterministic summary of the completed
Day 1–7 research pipeline.
