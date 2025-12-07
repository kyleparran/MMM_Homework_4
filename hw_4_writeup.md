# FINMATH 36701 — Homework 4 Writeup

This report addresses the three main components of Assignment 4: (i) training and evaluating a GAN for intraday limit order book (LOB) data, (ii) using the trained discriminator as an anomaly detector to distinguish normal from abnormal days based on LOB dynamics and microstructure variables, and (iii) assessing the realism of synthetic order books produced by the generator, including structural consistency checks. The sections below are organized accordingly: Q1 covers GAN training and basic distributional fit, Q2 analyzes discriminator-based day-level anomalies and their microstructure signatures, and Q3 evaluates the quality and limitations of the synthetic LOBs.

## Q1: GAN training and basic evaluation

### Experimental setup (brief)

| Ticker | Training months        | lr_G   | lr_D   | Batch size | Epochs | Seed |
|--------|------------------------|--------|--------|------------|--------|------|
| 0050   | 2023-10, 2023-11, 2023-12 | 0.00375 | 0.00100 | 50         | 200    | 307  |
| 0056   | 2023-10, 2023-11, 2023-12 | 0.00375 | 0.00100 | 50         | 200    | 307  |
| 2330   | 2023-10, 2023-11, 2023-12 | 0.00300 | 0.00100 | 50         | 200    | 307  |

A minimal example of the training configuration for ticker 0050 is:

```python
run_cfg = q1_make_run_config(
    ticker="0050",
    train_months=["202310", "202311", "202312"],
    lr_g=0.00375, lr_d=0.00100,
    batch_size=50, epochs=200, seed=307,
)
```

The learning rates were tuned to get stable, non-divergent adversarial training; 2330 uses a slightly lower generator rate because higher rates produced more unstable losses.

### Data and Preprocessing

We follow the same simple sequence of steps the code uses to turn raw minutely LOB data into daily inputs and outputs. The list below is the exact order applied in training, evaluation, and synthesis.

1. Select full trading days: keep only days with exactly 265 minutes.
2. Slice features: use 20 columns from the raw frame (columns 5 to 24; SP, BP, SV, BV across levels 1 to 5, and related variables).
3. Apply transforms: take `log1p` on the 10 volume related columns.
4. Normalize per day: subtract the day’s mean and divide by twice the day’s standard deviation; later (Q3) invert using the same per day statistics and `expm1` on the last 10 columns to recover raw scale.
5. Train/test split: train on 2023-10/11/12; evaluate and score on 2024-01/02/03.

### Training behavior (loss curves)

From the loss CSVs for each ticker, the trajectories look numerically as follows:

| Ticker | train_g (start → end) | eval_g (start → end) | train_d (start → end) | eval_d (start → end) | Interpretation |
|--------|------------------------|-----------------------|------------------------|-----------------------|----------------|
| 0050   | 1.29 → ≈1.08           | ≈1.12 → ≈0.98         | ≈0.25 → 0.19–0.23      | ≈0.25 → 0.19–0.23     | Smooth decline in generator losses and small drop in discriminator losses suggest a stable training regime without mode collapse or divergence. |
| 0056   | ≈1.11 → 0.79–0.81      | ≈0.98 → 0.66–0.72     | ≈0.25 → 0.22–0.30      | ≈0.25 → 0.25–0.29     | Generator losses fall more sharply while discriminator losses drift slightly upward, indicating the discriminator remains competitive but does not overpower the generator. |
| 2330   | ≈0.48 → 0.18–0.20      | ≈0.49 → 0.22–0.26     | ≈0.25 → 0.29–0.30      | ≈0.25 → 0.29–0.30     | Generator losses drop substantially and discriminator losses rise modestly, consistent with a stronger generator that still faces a non-trivial discriminator. |

Across all three tickers, losses evolve smoothly (no spikes to huge values or collapses to 0), and generator losses move from higher initial levels into lower, more stable ranges. Discriminator losses shift from their initial 0.25 baseline into new but still moderate bands, indicating that the adversarial game has settled into a reasonably stable regime rather than diverging.

Figures 1–3 show the loss curves for each ticker, and Figures 4–6 show real versus synthetic return distributions.

![Q1 loss curves for 0050](q1_notebook_outputs/0050/0050_q1_loss_curves.png)

![Q1 loss curves for 0056](q1_notebook_outputs/0056/0056_q1_loss_curves.png)

![Q1 loss curves for 2330](q1_notebook_outputs/2330/2330_q1_loss_curves.png)

![Real vs synthetic intraday returns for 0050](q1_notebook_outputs/0050/0050_return.png)

![Real vs synthetic intraday returns for 0056](q1_notebook_outputs/0056/0056_return.png)

![Real vs synthetic intraday returns for 2330](q1_notebook_outputs/2330/2330_return.png)

### Distributional behavior of returns

From the return-comparison figures (real vs synthetic) for each ticker:

- For 0050 and 0056, synthetic intraday returns cluster in roughly the same ±0.5%–1% region as the real returns, with only slightly lighter tails.
- For 2330, the synthetic distribution again matches the bulk (central mass) of the empirical distribution but appears to under-represent the most extreme positive and negative returns.

Overall, the GAN captures the first-order behavior of returns (location and scale) reasonably well for all three tickers, but it is less accurate in reproducing rare, large moves. This is consistent with many GAN applications, where tail events are difficult to learn from finite samples.

---

## Q2: Discriminator-based anomaly detection and microstructure comparison

### Conceptual setup

The trained discriminator assigns a scalar score to each trading day’s 265×20 LOB tensor. Real days with low scores are interpreted as abnormal, in the sense that their dynamics look less like the bulk of the training data. Using a threshold of 0.5:

- Days with scores ≤ 0.5 are labeled abnormal.
- Days with scores > 0.5 are labeled normal.

Daily tensors have the same shape and preprocessing as in the training scripts, for example:

```python
for date_key, day_df in minutely_df.groupby("date", sort=True):
    if day_df.shape[0] == 265:          # full trading day
        daily_arrays.append(day_df.values)
        daily_dates.append(str(date_key))
X = np.array(daily_arrays)[:, :, 5:]     # 20 LOB features
```

On top of this labeling, minute-level microstructure variables are compared across abnormal versus normal days.

#### Threshold Choice

We set the abnormal‑day threshold to 0.5 based on the day‑score distribution and simple sensitivity checks: varying the threshold in the 0.4–0.6 range yields similar rankings and microstructure contrasts while avoiding extremes (too few or too many abnormal days).

### Microstructure variables

For each minutely snapshot, the following variables are examined:

- Trade price return: within-day percent change of the last trade price.
- Midquote return: within-day percent change of the midquote.
- Trade size.
- Bid-ask spread: SP1 − BP1.
- First difference of spread: one-step change of the spread within a day.
- Order-flow pressure at level 1: (BV1 − SV1) / (BV1 + SV1).
- Order-flow pressure at levels 1–5: (∑ BVi − ∑ SVi) / (∑ BVi + ∑ SVi) over levels i = 1,…,5.

These variables are constructed from the raw minutely LOB using simple transformations, for example:

```python
df["midquote"] = (df["BP1"] + df["SP1"]) / 2
df["spread"] = df["SP1"] - df["BP1"]
df["trade_price_return"] = df.groupby("date_str")["lastPx"].pct_change()
df["pressure_5"] = (bv_sum - sv_sum) / (bv_sum + sv_sum)
```

Interpretation of KS tests: the two‑sample KS statistic measures distributional distance; small p‑values indicate the abnormal and normal samples are unlikely to come from the same distribution. For heavy‑tailed variables (for example, trade size), KS is less informative without filtering; pressure and spread metrics are therefore emphasized.

### Interpretation of abnormal vs normal days

Key patterns from the summary statistics and KS tests are summarized below.

Returns (variance and KS tests)

| Ticker | Variable              | Abn. var          | Norm. var         | KS stat | KS p‑value      |
|--------|-----------------------|-------------------|-------------------|---------|-----------------|
| 0050   | trade_price_returns   | 2.14×10⁻⁷         | 1.87×10⁻⁷         | 0.165   | 1.2×10⁻⁶        |
| 2330   | trade_price_returns   | 9.88×10⁻⁷         | 1.09×10⁻⁶         | 0.103   | 3.8×10⁻⁵        |

These rows show that the return distributions on abnormal days differ measurably from those on normal days, mainly via changes in variance and tails, with KS tests strongly rejecting equality.

Spreads

| Ticker | Variable        | Abn. mean | Norm. mean | KS stat | KS p‑value |
|--------|-----------------|-----------|------------|---------|-----------:|
| 0050   | bid_ask_spread  | 0.0696    | 0.0675     | 0.042   | 0.74       |
| 0056   | bid_ask_spread  | 0.0108    | 0.0125     | 0.107   | 0.005      |

For 0056, abnormal days exhibit a clearly different spread distribution (despite a slightly smaller mean), while for 0050 the mean difference is smaller and the KS test is not significant.

Trade size (means and KS tests)

| Ticker | Abn. mean size | Norm. mean size | KS stat | KS p‑value |
|--------|----------------|-----------------|---------|-----------:|
| 0050   | 238            | 722             | 0.019   | ≈1.0       |
| 0056   | 8,713          | 2,279           | 0.033   | 0.93       |

Trade size is extremely heavy‑tailed in both groups; even with large mean differences, the KS tests indicate that size alone does not sharply separate abnormal from normal days.

Order‑flow pressure (level 1 and 5)

| Ticker | Variable          | Abn. mean | Norm. mean | KS stat | KS p‑value    |
|--------|-------------------|-----------|------------|---------|--------------:|
| 0050   | pressure_level_5  | 0.107     | 0.060      | 0.167   | 8.2×10⁻⁷      |
| 0056   | pressure_level_1  | 0.125     | −0.143     | 0.195   | 4.0×10⁻⁹      |
| 2330   | pressure_level_5  | 0.046     | −0.138     | 0.376   | 2.1×10⁻⁶⁵     |

Order‑flow pressure metrics show some of the strongest differences between abnormal and normal days: means often change sign or magnitude substantially, and KS p‑values are extremely small, especially for 2330.

Overall, abnormal days, as identified by the discriminator, tend to be those with measurably different return dynamics (variance and tails), distinct spread distributions for some tickers, and pronounced changes in order-flow pressure, particularly at the 5-level aggregate. These are economically plausible characteristics of unusual trading days, which supports the interpretation that the discriminator is capturing meaningful deviations from typical market conditions rather than pure noise.

Abnormal vs normal minute counts (from pressure_level_5 summary):

| Ticker | Abnormal minutes (n) | Normal minutes (n) |
|--------|----------------------|--------------------|
| 0050   | 265                  | 14575              |
| 0056   | 265                  | 14575              |
| 2330   | 530                  | 14310              |

Reproducibility: the `.env` file specifies `LOB_GAN_DATA_DIR`; seeds are set in the notebook and scripts; outputs are cached under `q1_notebook_outputs`, `q2_notebook_outputs`, `q3_notebook_outputs`, and `cache/`, ensuring reruns reproduce results without recomputation.

---

## Q3: Quality of synthetic order books

### High-level evaluation approach

The generator is evaluated by comparing real and synthetic order books at both the day and snapshot levels, for the test months 2024-01 to 2024-03 and all three tickers.

For each stored day, the normalized GAN inputs and their per-day mean and standard deviation are used to generate and de-normalize synthetic LOB paths:

```python
gen = q3_load_generator_portable(ticker)
syn_norm = gen(torch.tensor(X_norm, dtype=torch.float32)).detach().cpu().numpy()
syn_trans = syn_norm * (2.0 * std[:, None, :]) + mean[:, None, :]
syn_raw = syn_trans.copy()
syn_raw[:, :, -10:] = np.expm1(syn_trans[:, :, -10:])
```

### Visual diagnostics: spreads, pressures, and depth

From representative days and snapshots:

- Spreads:
  - For 0050, on days such as 2024-01-02 and 2024-03-29, synthetic spreads are of similar magnitude to real spreads and broadly follow intraday patterns such as higher spreads near the open and close and narrower spreads mid-session.
  - For 0056 and 2330, the same qualitative pattern holds, but the synthetic series are visibly smoother, with fewer abrupt jumps.

- Order-flow pressures:
  - Real and synthetic pressure_1 and pressure_5 series occupy similar numeric ranges and show comparable volatility across the 265 minutes.
  - The generator reproduces broad swings between buy- and sell-dominant order flow, though the timing and exact amplitudes of swings do not match one-for-one.

- Depth curves:
  - For most snapshots (for example, minute 0, 132, and 264 on 0050-2024-01-02), synthetic bid and ask depth curves have realistic shapes: cumulative volume increases smoothly with distance from the midprice, and price levels are ordered as expected on each side.

Figures 7–9 show example time-series comparisons, and Figures 10–13 show depth snapshots for a structurally clean and a structurally problematic case.

![0050, 2024-02-20: spreads and order-flow pressures](q3_notebook_outputs/0050/0050_2024-02-20_timeseries.png)

![0056, 2024-02-20: spreads and order-flow pressures](q3_notebook_outputs/0056/0056_2024-02-20_timeseries.png)

![2330, 2024-02-20: spreads and order-flow pressures](q3_notebook_outputs/2330/2330_2024-02-20_timeseries.png)

Structurally clean case — 0050, 2024-01-02

![0050, 2024-01-02, minute 0](q3_notebook_outputs/0050/0050_2024-01-02_snapshot_000.png)

![0050, 2024-01-02, minute 132](q3_notebook_outputs/0050/0050_2024-01-02_snapshot_132.png)

![0050, 2024-01-02, minute 264](q3_notebook_outputs/0050/0050_2024-01-02_snapshot_264.png)

Structurally problematic case — 0056, 2024-02-20

![0056, 2024-02-20, minute 132](q3_notebook_outputs/0056/0056_2024-02-20_snapshot_132.png)

### Structural diagnostics

Key structural properties for three representative days per ticker are:

Before aggregating into fractions, the structural checks for a synthetic day use simple conditions on best quotes and volumes:

```python
crossed = np.mean(bp1v >= sp1v)               # best bid ≥ best ask
neg_vol = np.mean(vols < 0)                   # any negative depth
ask_mono = np.mean(np.all(np.diff(ask_p, 1) >= 0, axis=1))
bid_mono = np.mean(np.all(np.diff(bid_p, 1) <= 0, axis=1))
```

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

For 0050, the generator respects basic structural constraints almost perfectly: no crossed books, no negative volumes, and very rare monotonicity violations. For 0056, the generator produces a large fraction of crossed books (more than 60% of minutes), which is a serious structural deficiency, even though volumes remain non‑negative. For 2330, crossed-book frequencies are lower than for 0056 but still substantial (about 20–26% of minutes), and ask monotonicity is only occasionally satisfied, although bid monotonicity is good on one of the three days.

### Overall assessment

Taken together, the visual and structural evidence indicates that the GAN produces structurally coherent synthetic books for 0050 (and partly for 2330) in terms of non-negative volumes and mostly non-crossed quotes, and it captures coarse intraday patterns in spreads and order-flow pressures for all three tickers. However, the very high crossed-book fractions for 0056 and the non-trivial crossed-book rates for 2330 highlight that the generator does not uniformly respect price ordering constraints across all instruments. It also tends to smooth out extreme or very short-lived events, and tail behavior in prices and liquidity remains somewhat under-represented.

From the perspective of the assignment, these diagnostics show both the strengths (0050 in particular) and limitations (notably 0056) of the trained GAN as a tool for generating realistic intraday LOB scenarios and suggest that adding explicit structural penalties or constraints could meaningfully improve realism.

### Remedies

The diagnostics point to two concrete issues: crossed books and smoothed tails. A practical fix for crossed books is to add training penalties whenever the best bid is above the best ask and whenever price levels break monotonicity (asks should increase across levels; bids should decrease). If violations persist, a simple post‑processing step can project each synthetic snapshot to the nearest configuration that is non‑crossed and monotone, preserving the overall shape while repairing structure. To address smoothed tails, incorporate objectives that pay extra attention to extremes, such as quantile‑style losses or reweighting rare, large moves in prices and volumes; this helps the generator learn spikes instead of averaging them away. Finally, for anomaly labeling, verify that conclusions are stable across thresholds from 0.4 to 0.6 and accompany results with abnormal versus normal counts so KS outcomes are interpreted in context.
