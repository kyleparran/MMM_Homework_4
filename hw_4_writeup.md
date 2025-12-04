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

Across all three tickers, the generator and discriminator loss curves behave as follows:

- Both losses fluctuate but remain within a relatively stable band after an initial transient.
- Neither loss explodes nor collapses to zero; instead, they oscillate around moderate values, consistent with a rough adversarial equilibrium.
- Smoothed generator loss curves (e.g., rolling mean with window 25) do not show a sustained upward or downward drift, suggesting that training does not systematically deteriorate.

Per-ticker behavior:

- **0050:** Losses are comparatively smooth. The discriminator does not dominate for long stretches, and the generator retains enough capacity to keep learning. This indicates a well-balanced game.
- **0056:** Similar to 0050 but with slightly more volatility in the discriminator loss. The system remains stable, with no evidence of collapse.
- **2330:** With the lower generator learning rate, the generator loss is smoother than in higher-rate experiments, and the discriminator still provides a clear learning signal.

Overall, the loss dynamics suggest that the GANs learn reasonable approximations to the LOB dynamics for all three stocks, without obvious training pathologies such as divergence, vanishing gradients, or hard mode collapse.

### Distributional behavior of returns

From the return-comparison figures (real vs synthetic):

- Synthetic return histograms broadly match the shape and scale of the empirical return distributions.
- In several cases, synthetic tails are somewhat lighter, indicating that the generator under-represents extreme intraday returns.
- There is no clear sign of mode collapse: the synthetic return distributions are spread out rather than concentrated in a few spikes.

Interpretation: the GAN captures the first-order behavior of returns (location and scale) reasonably well, but it is less accurate in reproducing rare, large moves. This is consistent with many GAN applications, where tail events are harder to learn from finite data.

---

## Q2: Discriminator-based anomaly detection and microstructure comparison

### Conceptual setup

The trained discriminator assigns a scalar score to each trading day’s 265×20 LOB tensor. Real days with **low scores** are interpreted as **abnormal**, in the sense that their dynamics look less like the bulk of the training data. Using a threshold of 0.5:

- Days with scores ≤ 0.5 are labeled **abnormal**.
- Days with scores > 0.5 are labeled **normal**.

On top of this labeling, we look at minute-level microstructure variables and compare their empirical distributions across abnormal vs normal days.

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

We then compute moments (mean, variance, skewness, kurtosis) and two-sample KS statistics for abnormal vs normal days, aggregated by ticker.

### Interpretation of abnormal vs normal days

Key patterns visible in the summary statistics and KS results:

1. **Return-related variables**
   - Abnormal days typically exhibit **higher return variance** and more pronounced tails (larger absolute skewness and kurtosis) than normal days.
   - KS tests for trade and midquote returns often show sizable statistics and small p-values, indicating a genuine distributional shift.
   - Interpretation: the discriminator tends to flag days with more volatile and more extreme intraday price paths as abnormal.

2. **Bid-ask spread and spread changes**
   - On abnormal days, average spreads are often **wider**, and spread variance is higher.
   - First differences of the spread are more volatile, suggesting unstable liquidity conditions.
   - Interpretation: abnormal days correspond to periods when liquidity is more expensive and less stable, in line with the economic intuition that unusual conditions often coincide with higher transaction costs.

3. **Trade size**
   - Differences in trade size distributions between abnormal and normal days are less systematic across tickers than for returns or spreads.
   - For some stocks, abnormal days show heavier right tails in trade size, consistent with bursts of large orders or block trades.

4. **Order-flow pressure (level 1 and 5)**
   - Pressure metrics on abnormal days show more dispersion and sometimes more skewness, indicating that one-sided order-flow episodes (strong buying or selling dominance) are more common.
   - KS tests frequently detect differences in the pressure distributions between abnormal and normal days.

Overall, abnormal days, as identified by the discriminator, tend to be those with:

- More volatile intraday prices,
- Wider and more volatile spreads, and
- Stronger and more erratic order-flow imbalances.

These properties are economically plausible characteristics of “unusual” trading days, reinforcing the idea that the discriminator is capturing meaningful deviations from typical market conditions rather than pure noise.

---

## Q3: Quality of synthetic order books

### High-level evaluation approach

The generator is evaluated by comparing **real vs synthetic order books** at both the day and snapshot levels, for the test months 2024-01 to 2024-03 and all three tickers.

The comparisons include:

- Intraday time series of spreads and order-flow pressures (1-level and 5-level).
- Depth curves (cumulative volume vs price) constructed from 5-level quote and volume data for selected minutes (e.g., open, midday, close).
- Simple structural diagnostics quantifying how often key consistency constraints are violated (crossed quotes, negative volumes, non-monotone price ladders).

### Visual diagnostics: spreads, pressures, and depth

From the representative days and snapshots examined:

- **Spreads:**
  - Synthetic spreads are of similar magnitude to real spreads and broadly follow intraday patterns such as wider spreads near the open and close, with narrower spreads mid-session.
  - Fine-grained microstructure (short-lived spikes or micro-crashes in liquidity) tends to be smoother in the synthetic series, indicating that very transient events are not perfectly captured.

- **Order-flow pressures:**
  - Real and synthetic pressure_1 and pressure_5 series occupy similar ranges and show comparable levels of volatility.
  - The generator reproduces broad swings between buy- and sell-dominant order flow, even if the exact timing and amplitude of swings differ from the real series (as expected for a generative model).

- **Depth curves:**
  - For most snapshots, synthetic bid and ask depth curves have realistic shapes: cumulative volume increases with distance from the midprice, and price levels form reasonable ladders on each side.
  - Differences tend to be in the fine details of volume steps and exact level spacing, rather than in gross structural features.

### Structural diagnostics

Simple structural metrics computed on the synthetic books show:

- **Crossed books (best bid ≥ best ask):** fractions are essentially zero for the sampled days, meaning the generator almost never produces obviously invalid quotes.
- **Negative volumes:** the share of negative volume entries is also near zero, indicating that the de-normalization and generation pipeline preserve non-negativity.
- **Monotonicity across levels:**
  - A large majority of minutes satisfy monotone asks (SP1 ≤ SP2 ≤ … ≤ SP5) and monotone bids (BP1 ≥ BP2 ≥ … ≥ BP5).
  - Occasional violations do occur but at low frequencies, comparable to (or only slightly above) what might appear in noisy real data.

These diagnostics suggest that the generator has internalized key invariants of a valid limit order book, not just marginal distributions of individual features.

### Overall assessment

Putting the visual and structural evidence together:

- The GAN produces **structurally coherent** synthetic books that respect basic market microstructure constraints.
- It captures **coarse intraday patterns** in spreads and order-flow pressures and yields reasonable depth profiles at the snapshot level.
- It tends to **smooth out extreme or very short-lived events**, and tail behavior in prices and liquidity remains somewhat under-represented.

From an assignment perspective, this level of performance supports the conclusion that the trained GAN is a useful tool for generating realistic intraday LOB scenarios, while also highlighting where more advanced architectures or loss functions (e.g., explicit tail penalties or additional microstructure constraints) might further improve realism.

---

## How the notebook addresses the assignment questions

- **Q1:**
  - Provides trained GAN models for the three tickers and uses loss and return diagnostics to argue that training is stable and that basic return distributions are matched reasonably well.

- **Q2:**
  - Uses the trained discriminator as an anomaly detector for daily LOB sequences, then compares a set of economically meaningful microstructure variables between abnormal and normal days.
  - The discussion focuses on how volatility, spreads, and order-flow pressures differ between the two groups and why those differences are plausible.

- **Q3:**
  - Evaluates synthetic order books via time-series plots, depth-curve comparisons, and structural consistency checks.
  - The interpretation emphasizes both strengths (structural validity, realistic coarse patterns) and weaknesses (smoothing of extremes, limited tail realism).

This writeup is intentionally focused on interpretation and discussion; detailed implementation steps and file paths are documented in the code itself rather than repeated here.
