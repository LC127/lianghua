# Day 3 Step 1 — Formal Network Construction Design

## 1. Objective

Freeze the formal construction rules before producing full-market
Raw and Market-Residual networks.

The central comparison is:

\[
G^{Raw}_{t,W,q}
\quad	ext{vs}\quad
G^{Residual}_{t,W,q}.
\]

The comparison must use the same date, rolling window, node universe,
candidate-pair universe, and edge budget.

## 2. Primary network

Primary candidate:

**PIT leave-one-out market-residual Pearson positive network**

Baseline:

**Raw Pearson positive network**

Primary sparsification:

\[
q=1\%.
\]

Density robustness:

\[
q\in\{0.2\%,0.5\%,1\%,2\%\}.
\]

## 3. Comparable-pair principle

At each \((t,W)\), first construct one common valid pair universe.
Only pairs for which both Raw and Residual correlations are available
are eligible for the matched comparison.

The exact edge count is then

\[
K_{t,W,q}
=
\left\lfloor
q M_{t,W}^{common}
ightfloor .
\]

Both Raw and Residual networks retain exactly \(K_{t,W,q}\) edges.

## 4. Positive network versus signed network

The primary network ranks pairs by positive correlation.

Signed robustness ranks pairs by absolute correlation and retains the
original sign as the edge weight.

These are treated as distinct economic objects rather than mixed in a
single main network.

## 5. Local-network robustness

Per-node Top-k robustness uses:

\[
k\in\{10,25,50\}
\]

with OR symmetrization.

## 6. Primary q=1% approximate network scale

- W=60: mean stocks ≈ 3904, mean edges ≈ 81,948, mean degree ≈ 39.0
- W=120: mean stocks ≈ 3875, mean edges ≈ 81,108, mean degree ≈ 38.7
- W=252: mean stocks ≈ 3780, mean edges ≈ 77,693, mean degree ≈ 37.8

The exact edge budget will be recomputed in Step 2 from the common
valid-pair universe; the numbers above are only scale diagnostics based
on \(inom{p}{2}\).

## 7. Industry information

Industry labels are not used to construct the graph.

They are retained only for post-construction validation such as:

- same-industry edge share;
- industry enrichment;
- industry assortativity;
- community–industry alignment.

This avoids circular validation.

## 8. Design QA

Passed checks:

**6/6**

All checks should pass before Step 2 is executed.

## 9. Next step

Day 3 Step 2 will construct matched-density Raw and Market-Residual
networks at each month-end and save complete edge tables for subsequent
structural, industry, and dynamic-stability validation.
