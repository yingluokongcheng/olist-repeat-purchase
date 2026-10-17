# Olist Repeat Purchase: Fulfillment and Regional Purchase Mix

[中文说明](README.zh-CN.md) · [Methods](docs/methodology.md) · [Experiment history](docs/experiment_history.md) · [Reproduction](docs/reproduction.md)

An independent portfolio study of whether fulfillment experience and regional purchase mix help predict a customer's next purchase in the Olist public dataset.

**Main finding:** removing poorly supported month-cycle features reduced probability underestimation. Intercept calibration did not improve the later evaluation. Regional features showed no stable incremental benefit, and the final LightGBM comparison did not establish a clear advantage over a simpler logistic regression.

This is an observational analysis, not a claim that improving delivery causes repeat purchases. The full experiment sequence—including unsuccessful changes—is retained. Analysis is closed; all recorded stages and unsuccessful changes are preserved in this repository.

## Research question

Among customers whose first observed order has been delivered and who have not ordered again by that point:

> Do fulfillment experience and the historical product-category mix in the customer's state improve prediction of an approved new order within 90 days?

The business motivation is whether the platform serves customers in different regions well. The measurable variables are narrower: delivery time, lateness, freight burden and observed transaction mix. Completed purchases cannot separately identify consumer preferences, available supply, product exposure or local culture.

## Data and cohort

Source: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). The downloaded snapshot contains 99,441 orders and 96,096 unique customers. Use `customer_unique_id`, not the per-order `customer_id`, to follow a customer.

| Audit step | Customers remaining |
|---|---:|
| Unique customers | 96,096 |
| First order delivered; valid chronology; no other order before delivery | 92,081 |
| Complete 90-day follow-up under the chosen cutoff | 65,493 |
| At least 180 days of calendar history | 62,733 |
| History window entirely within continuously populated months from January 2017 | **51,981** |

The conservative cohort contains **567 repeat purchasers (1.091%)**. Follow-up ends before **2018-08-01**: late-August volume tapers and September–October contain only sparse orders. This is an analytical boundary, not an official guarantee of complete data capture. Alternative cutoff and horizon counts are retained in the [audit results](reports/audit/audit_results.json) and [sensitivity table](reports/audit/sensitivity.csv).

The prediction landmark is first-order delivery. A positive label requires another order placed after that landmark and approved within 90 days. A later cancellation does not erase the approval event. Customers with tied first-order timestamps or another order placed before/at delivery are excluded. “First” and “repeat” refer only to what this dataset observes.

## Analysis design

1. **Audit before modeling:** verify join cardinality, chronology, observation windows, geographic support and repeat-event counts.
2. **Background model:** first-order category, amount, item count, product dimensions/weight, approximate seller–customer distance, state and selected time variables.
3. **Fulfillment increment:** delivery duration, late indicator, late days and freight-to-item-value ratio.
4. **Regional increment:** first-purchase category share within the state and its share relative to the national mix, using only the preceding 180 days with approval already known. Multi-category orders share one unit of weight. Sparse state estimates shrink toward the national mix.
5. **Preserved revisions:** diagnose month extrapolation, test intercept-only calibration, then run one fixed LightGBM comparison. No additional experiments followed.

There are **two separate training regimes**. Their scores must not be combined into a best-model leaderboard.

| Stage | Training customers / events | Development or calibration | Evaluation |
|---|---:|---|---|
| V1 final logistic comparison | 25,780 / 303, after the initial validation stage | December 2017; initial fit used 6,459 / 71 | 7,893 / 87 |
| V2 fixed A/B/C revision | 6,459 / 71 | December: 6,862 / 60 | Same 7,893 / 87 |
| V3 fixed LightGBM comparison | 6,459 / 71 | Same December cohort, early stopping only | Same 7,893 / 87 |

Labels mature before the next stage starts. **V2 and V3 reuse an already-viewed evaluation set and are exploratory revisions, not new independent tests.** Their plans were frozen before each run, not before the first evaluation results were seen.

## Results, including unsuccessful changes

All evaluation tables below use the same 7,893 customers, with an observed repeat rate of **1.102%**. AP is noninterpolated Average Precision, not accuracy or trapezoidal PR-AUC. “Top 10%” means exactly 790 customers.

### V1: fulfillment and regional information

| Logistic model | AP ↑ | Mean predicted probability | Top-790 repeat purchasers |
|---|---:|---:|---:|
| M1: background | 0.018493 | 0.544% | 17 |
| M2: + fulfillment | 0.019266 | 0.540% | 19 |
| M3: + regional mix | 0.019296 | 0.540% | 19 |

Fulfillment added a small ranking gain; its paired AP interval included zero. Regional features added almost nothing in this specification. All three underestimated probabilities, and the constant training-rate baseline had better Brier and log loss. Diagnostic decomposition identified month-cycle terms as the main model-internal contributor to the downward shift when extrapolating from second-half-year training into spring.

### V2: a fixed month-removal and calibration diagnostic

| Version | Mean predicted probability | Log loss ↓ | AP ↑ | Top-790 hits |
|---|---:|---:|---:|---:|
| A: retain month terms | 0.139% | 0.074706 | 0.015916 | 12 |
| B: remove month terms | 0.942% | 0.061213 | 0.015668 | 11 |
| C: B + intercept calibration | 0.819% | 0.061531 | 0.015668 | 11 |

Removing month terms improved probability losses in this diagnostic, without improving ranking. In December, B overpredicted the observed rate, so the fitted calibration intercept shifted probabilities downward. That shift made subsequent underprediction worse. **B and C had identical complete rankings and top-790 lists.** Calibration aligned the calibration-period mean by construction; this did not establish future calibration.

### V3: final fixed nonlinear comparison

| Model | AP ↑ | Log loss ↓ | Mean predicted probability | Top-790 hits |
|---|---:|---:|---:|---:|
| LR: frozen V2 B | **0.015668** | 0.061213 | 0.942% | 11 |
| G1: LightGBM, same information | 0.012260 | 0.060634 | 1.099% | 13* |
| G2: G1 + regional mix | 0.015311 | **0.060511** | 1.127% | **18** |

G1 selected round 1; G2 selected round 146 using the same December log-loss early-stopping rule. G2 improved over G1 in point estimates, but both customer- and state-bootstrap AP intervals included zero. G2's AP remained slightly below LR despite more hits at the fixed selection size and lower probability-loss point estimates. These are tradeoffs, not clear overall superiority.

\* G1 assigns the highest score to 1,548 customers. The fixed customer-ID tie-break selected 790 of them and found 13 repeat purchasers. Uniform selection within that tied group would yield **11.23 expected hits**. The model cannot rank customers within that group.

![Final exploratory comparison of ranking and calibration](reports/v3/model_comparison.png)

Full [V1 metrics](reports/v1/model_metrics.csv), [V2 metrics](reports/v2/metrics.csv), [V3 metrics](reports/v3/metrics.csv), learning curves and uncertainty intervals remain in the stage folders. HTML reports are readable after download; GitHub displays HTML source rather than hosting the rendered report.

## Two descriptive category cases

Bed/bath/table and sports/leisure were the two largest single-category groups in the pre-evaluation training pool. Neither met the original training-period repeat-event threshold for a standalone model, so they remain descriptive cases.

| Category | On-time repeats / customers | Late repeats / customers |
|---|---:|---:|
| Bed, bath and table | 74 / 4,480 (1.65%) | 5 / 478 (1.05%) |
| Sports and leisure | 52 / 4,002 (1.30%) | 3 / 341 (0.88%) |

These unadjusted, full-cohort comparisons have wide intervals and very few late-group events. They do not establish a delivery intervention effect. Regional-match quartiles showed no consistent monotonic pattern in these cases. See [descriptive counts and intervals](reports/v1/descriptive_cases.csv).

## Limits that matter

- **Rare outcomes:** only 87 evaluation events; detailed regional/category interactions are poorly supported.
- **Selection:** results apply to delivered first orders with no intervening order, not canceled or undelivered customers.
- **Incomplete market visibility:** Olist transactions do not capture customers' purchases elsewhere, unmet demand, exposure, inventory or full supply.
- **Evaluation reuse:** later revisions are exploratory. Bootstrap intervals do not remove adaptive-analysis bias or include model-fitting/early-stopping uncertainty.
- **Probability versus ranking:** a good average rate, AP improvement or top-list hit count alone is insufficient to declare a model operationally useful.
- **No business-effect claim:** no randomized retention intervention, verified profit calculation, causal lift estimate or production deployment is included.

The evidence did not establish a stable regional increment or a clear reason to replace the simpler model with LightGBM. This is not proof that region or nonlinear models never matter. No further experiments are planned on this evaluation set.

## Repository and reproducibility

```text
analysis/       Preserved historical Python scripts, organized by stage
configs/        Frozen V1/V2/V3 experimental protocols
reports/        Recorded metrics, figures, HTML reports and audit evidence
docs/           Methods, experiment chronology, provenance and data attribution
scripts/        Packaging verification and isolated numerical replay entrypoint
data/raw/       User-supplied Olist ZIP only; ignored by Git
runs/           Isolated replay outputs, caches and predictions; ignored by Git
```

**Review without data or training:**

```bash
python scripts/verify_package.py
```

This checks recorded evidence, hashes, links and source syntax; it does not run experiments. Recorded experiments ran with Python 3.12 and pinned dependencies in [requirements.txt](requirements.txt). The new isolated replay wrapper has been statically checked but **not executed end to end**, honoring the decision to stop experiments. Exact numerical replay across environments is not guaranteed. See [reproduction instructions](docs/reproduction.md) before explicitly requesting a replay.

## Attribution and scope

Olist's publisher page labels the dataset **CC BY-NC-SA 4.0**. Raw data and customer-level predictions are not distributed here. Source attribution, snapshot SHA-256 and the separation of code/data licensing are documented in [data and licensing](docs/data_and_license.md).

This is an independent public-data portfolio study. Research design was developed through iterative discussion; implementation, checks and documentation were assisted by Codex. It does not claim proprietary business access, new algorithms or production experience.
