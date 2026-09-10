# Day 2 Step 7 — Network Candidate Evidence Matrix

## 1. Objective

Step 7 integrates the empirical evidence from the raw-correlation,
common-factor, industry, and market-residual analyses to determine the
appropriate role of competing network definitions.

No arbitrary weighted total score is used. The selection is based on
economically interpretable evidence gates.

## 2. Main empirical evidence

Across the 60-, 120-, and 252-day windows:

- Corrected raw mean correlation ranges from **0.318–0.323**.
- Market-residual mean correlation falls to **0.0026–0.0035**.
- The fraction of the positive average-correlation baseline removed by
  the market factor ranges from **0.987–0.991**.
- The residual Same-minus-Cross industry correlation gap remains
  **0.085–0.089**.
- Same-industry enrichment among the residual Top-1% strongest
  associations ranges from **4.881–7.083**.
- Raw and market-residual Top-1% edge overlap is only
  **0.332–0.336**.

These results indicate that broad market co-movement materially affects
the raw network, while economically meaningful industry structure
persists after market removal.

## 3. Evidence gates

**8/8** diagnostic evidence gates pass.

## 4. Candidate roles

### C1 — Raw Pearson Network

Retain as an **unconditional baseline**. It is simple and transparent,
but its edge magnitudes and strongest-edge rankings are substantially
affected by the market-wide common mode.

### C2 — Market-Residual Pearson Network

Selection status: **PRIMARY CANDIDATE**.

This is the preferred interpretable association-network candidate for
the next stage because it removes the broad market component while
preserving strong industry-level economic structure.

### C3 — Market + Industry Residual Network

Retain as a future robustness extension if the objective becomes
stock-specific association beyond both market and industry structure.

### C4 — Sparse Conditional Association

Retain as the main **formal challenger**. Because the full A-share
problem satisfies p >> W, conditional-network estimation should use
high-dimensional sparse methods rather than an unrestricted covariance
inverse.

## 5. Sparsification

A fixed absolute correlation threshold should not be the primary edge
rule because correlation distributions vary substantially across
market regimes and rolling-window lengths.

The preferred first comparison is therefore:

**Market-Residual Pearson + fixed-density Top-q% sparsification**

with:

**Per-node Top-k**

as a local-network robustness check.

## 6. Day 2 conclusion

The evidence supports the following working hierarchy:

Raw Pearson → baseline unconditional co-movement

Market-Residual Pearson → primary interpretable association network

Sparse conditional association on market residuals → next-stage
high-dimensional challenger

The final network should still be selected based on downstream network
stability, turnover, Alpha-factor performance, risk linkage, and
computational scalability rather than Step 7 diagnostics alone.
