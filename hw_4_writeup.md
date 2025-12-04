# FINMATH 36701 — Homework 4 Writeup

## Q1: GAN training and basic evaluation

### Experimental setup (brief)

- **Tickers:** 0050, 0056, 2330
- **Training months:** 2023-10, 2023-11, 2023-12
- **Hyperparameters (per ticker):**
  - 0050: lr_G = 0.00375, lr_D = 0.00100
  - 0056: lr_G = 0.00375, lr_D = 0.00100
  - 2330: lr_G = 0.00300, lr_D = 0.00100
  - Batch size: 50; epochs: 200; seed: 307

The learning rates were tuned to get stable, non-divergent adversarial training; 2330 uses a slightly lower generator rate because higher rates produced more unstable losses.

### Training behavior (loss curves)

Using the CSVs in `q1_notebook_outputs/<ticker>/*_train_g_d.csv` and `*_eval_g_d.csv`, the loss trajectories look numerically as follows:

- **0050:**
  - `train_g` decreases from about **1.29** at the start to **≈1.08** by the final column, while `eval_g` decreases from **≈1.12** down to **≈0.98**.
  - `train_d` and `eval_d` start very close to **0.25** and drift down to the **0.19–0.23** range.
- **0056:**
  - `train_g` starts near **1.11**, dips toward **≈0.79–0.81** late in training, and `eval_g` moves from **≈0.98** down to **≈0.66–0.72**.
  - `train_d` begins around **0.25** and climbs into the **0.22–0.30** range, with some variability; `eval_d` also drifts upward into the **0.25–0.29** band.
- **2330:**
  - `train_g` moves from **≈0.48** down to the **≈0.18–0.20** region late in training; `eval_g` decreases from **≈0.49** early on to **≈0.22–0.26** in the later columns.
  - `train_d` and `eval_d` start near **0.25** and gradually increase toward **≈0.29–0.30** in the later part of training.

Across all three tickers, losses evolve smoothly (no spikes to huge values or collapses to 0), and generator losses move from higher initial levels into lower, more stable ranges. Discriminator losses shift from their initial 0.25 baseline into new but still moderate bands, indicating that the adversarial game has settled into a reasonably stable regime rather than diverging.

### Distributional behavior of returns

From the return-comparison figures (real vs synthetic) in `q1_notebook_outputs/<ticker>/<ticker>_return.png`:

- For 0050 and 0056, synthetic intraday returns cluster in roughly the same **±0.5%–1%** region as the real returns, with only slightly lighter tails.
- For 2330, the synthetic distribution again matches the bulk (central mass) of the empirical distribution but appears to under-represent the most extreme positive and negative returns.

Interpretation: the GAN captures the first-order behavior of returns (location and scale) reasonably well for all three tickers, but it is less accurate in reproducing rare, large moves. This is consistent with many GAN applications, where tail events are difficult to learn from finite samples.

---

## Q2: Discriminator-based anomaly detection and microstructure comparison

### Conceptual setup

The trained discriminator assigns a scalar score to each trading day’s 265×20 LOB tensor. Real days with **low scores** are interpreted as **abnormal**, in the sense that their dynamics look less like the bulk of the training data. Using a threshold of 0.5:

- Days with scores ≤ 0.5 are labeled **abnormal**.
- Days with scores > 0.5 are labeled **normal**.

On top of this labeling, we look at minute-level microstructure variables and compare their empirical distributions across abnormal vs normal days. The per-ticker and combined summaries are written to
`q2_notebook_outputs/q2_all_tickers_summary.csv`.

### Microstructure variables

For each minutely snapshot, the following variables are examined:

- **Trade price return:** within-day percent change of the last trade price.
- **Midquote return:** within-day percent change of the midquote.
- **Trade size.**
- **Bid-ask spread:** SP1 − BP1.
- **First difference of spread:** one-step change of the spread within a day.
- **Order-flow pressure at level 1:** (BV1 − SV1) / (BV1 + SV1).
- **Order-flow pressure at levels 1–5:** 
  (∑ BVi − ∑ SVi) / (∑ BVi + ∑ SVi) over levels i = 1,…,5.

Moments and KS statistics comparing abnormal vs normal days for each of these appear as rows in
`q2_all_tickers_summary.csv`.

### Interpretation of abnormal vs normal days

Key patterns from `q2_all_tickers_summary.csv` can be summarized as follows.

**Returns (variance and KS tests)**

| Ticker | Variable              | Abn. var          | Norm. var         | KS stat | KS p‑value      |
|--------|-----------------------|-------------------|-------------------|---------|-----------------|
| 50     | trade_price_returns   | 2.14×10⁻⁷         | 1.87×10⁻⁷         | 0.165   | 1.2×10⁻⁶        |
| 2330   | trade_price_returns   | 9.88×10⁻⁷         | 1.09×10⁻⁶         | 0.103   | 3.8×10⁻⁵        |

These rows show that the return distributions on abnormal days differ measurably from those on normal days, mainly via changes in variance and tails, with KS tests strongly rejecting equality.

**Spreads**

| Ticker | Variable        | Abn. mean | Norm. mean | KS stat | KS p‑value |
|--------|-----------------|-----------|------------|---------|-----------:|
| 50     | bid_ask_spread  | 0.0696    | 0.0675     | 0.042   | 0.74       |
| 56     | bid_ask_spread  | 0.0108    | 0.0125     | 0.107   | 0.005      |

For 56, abnormal days exhibit a clearly different spread distribution (despite a slightly smaller mean), while for 50 the mean difference is smaller and the KS test is not significant.

**Trade size (means and KS tests)**

| Ticker | Abn. mean size | Norm. mean size | KS stat | KS p‑value |
|--------|----------------|-----------------|---------|-----------:|
| 50     | 238            | 722             | 0.019   | ≈1.0       |
| 56     | 8,713          | 2,279           | 0.033   | 0.93       |

Trade size is extremely heavy‑tailed in both groups; even with large mean differences, the KS tests indicate that size alone does not sharply separate abnormal from normal days.

**Order‑flow pressure (level 1 and 5)**

| Ticker | Variable          | Abn. mean | Norm. mean | KS stat | KS p‑value    |
|--------|-------------------|-----------|------------|---------|--------------:|
| 50     | pressure_level_5  | 0.107     | 0.060      | 0.167   | 8.2×10⁻⁷      |
| 56     | pressure_level_1  | 0.125     | −0.143     | 0.195   | 4.0×10⁻⁹      |
| 2330   | pressure_level_5  | 0.046     | −0.138     | 0.376   | 2.1×10⁻⁶⁵     |

Order‑flow pressure metrics show some of the strongest differences between abnormal and normal days: means often change sign or magnitude substantially, and KS p‑values are extremely small, especially for 2330.

Overall, abnormal days, as identified by the discriminator, tend to be those with:

- Measurably different return dynamics (variance and tails),
- Distinct spread distributions for some tickers, and
- Very pronounced changes in order-flow pressure, particularly at the 5-level aggregate.

These properties are economically plausible characteristics of “unusual” trading days, reinforcing the idea that the discriminator is capturing meaningful deviations from typical market conditions rather than pure noise.

---

## Q3: Quality of synthetic order books

### High-level evaluation approach

The generator is evaluated by comparing **real vs synthetic order books** at both the day and snapshot levels, for the test months 2024-01 to 2024-03 and all three tickers.

The empirical basis for the discussion here is:

- Time-series plots in `q3_notebook_outputs/<ticker>/*_<date>_timeseries.png`.
- Depth-curve snapshots in `q3_notebook_outputs/<ticker>/*_<date>_snapshot_*.png`.
- Structural diagnostics summarized in `q3_notebook_outputs/q3_synthetic_quality_diagnostics.csv`.

### Visual diagnostics: spreads, pressures, and depth

From the representative days and snapshots examined in `q3_notebook_outputs/`:

- **Spreads:**
  - For 0050, on days like **2024-01-02** and **2024-03-29**, synthetic spreads are of similar magnitude to real spreads and broadly follow intraday patterns such as higher spreads near the open and close and narrower spreads mid-session.
  - For 0056 and 2330, the same qualitative pattern holds, but the synthetic series are visibly smoother, with fewer abrupt jumps.

- **Order-flow pressures:**
  - Real and synthetic pressure_1 and pressure_5 series occupy similar numeric ranges and show comparable volatility across the 265 minutes.
  - The generator reproduces broad swings between buy- and sell-dominant order flow, though the timing and exact amplitudes of swings do not match one-for-one.

- **Depth curves:**
  - For most snapshots (e.g., minute 0, 132, and 264 on 0050-2024-01-02), synthetic bid and ask depth curves have realistic shapes: cumulative volume increases smoothly with distance from the midprice, and price levels are ordered as expected on each side.

### Structural diagnostics

The file `q3_synthetic_quality_diagnostics.csv` quantifies key structural properties for three representative days per ticker:

| Ticker | Date       | frac_crossed_book | frac_neg_vol | frac_ask_mono | frac_bid_mono |
|--------|------------|-------------------|--------------|---------------|---------------|
| 0050   | 2024-01-02 | 0.00              | 0.00         | 0.0075        | 0.00          |
| 0050   | 2024-02-20 | 0.00              | 0.00         | 0.00          | 0.00          |
| 0050   | 2024-03-29 | 0.00              | 0.00         | 0.00          | 0.00          |
| 0056   | 2024-01-02 | 0.68              | 0.00         | 0.00          | 0.00          |
| 0056   | 2024-02-20 | 0.64              | 0.00         | 0.00          | 0.00          |
| 0056   | 2024-03-29 | 0.73              | 0.00         | 0.00          | 0.00          |
| 2330   | 2024-01-02 | 0.26              | 0.00         | 0.0415        | 0.75          |
| 2330   | 2024-02-20 | 0.25              | 0.00         | 0.0113        | 0.00          |
| 2330   | 2024-03-29 | 0.21              | 0.00         | 0.0151        | 0.0113        |

Interpretation:

- For 0050, the generator respects basic structural constraints almost perfectly: no crossed books, no negative volumes, and very rare monotonicity violations.
- For 0056, the generator produces a **large fraction of crossed books** (more than 60% of minutes), which is a serious structural deficiency, even though volumes remain non‑negative.
- For 2330, crossed-book frequencies are lower than for 0056 but still substantial (about 20–26% of minutes), and ask monotonicity is only occasionally satisfied, although bid monotonicity is good on one of the three days.

### Overall assessment

Putting the visual and structural evidence together:

- The GAN produces **structurally coherent** synthetic books for 0050 (and partly for 2330) in terms of non-negative volumes and mostly non-crossed quotes, and it captures **coarse intraday patterns** in spreads and order-flow pressures for all three tickers.
- However, the very high crossed-book fractions for 0056 and the non-trivial crossed-book rates for 2330 highlight that the generator does not uniformly respect price ordering constraints across all instruments.
- It also tends to **smooth out extreme or very short-lived events**, and tail behavior in prices and liquidity remains somewhat under-represented.

From an assignment perspective, these diagnostics show both the **strengths** (0050 in particular) and **limitations** (notably 0056) of the trained GAN as a tool for generating realistic intraday LOB scenarios and suggest that adding explicit structural penalties or constraints could meaningfully improve realism.
