# Methods and definitions

## Unit, selection and outcome

One row represents a `customer_unique_id`. The first observed order is selected by purchase timestamp. Tied earliest timestamps are excluded rather than broken arbitrarily. The first order must have delivered status, an actual delivery time, approval at/after purchase and delivery at/after approval. Any other order placed before or at first delivery excludes that customer, including orders not yet approved; this is intentionally stricter than checking approved purchases only.

The prediction landmark `t0` is first delivery. The outcome is an additional order placed strictly after `t0`, with approval at/after its purchase timestamp and at/before `t0 + 90 days`. Final cancellation status does not change this approval-based label. The entire 90-day interval must end before 2018-08-01. The conservative main cohort also requires first purchase at/after 2017-06-30, allowing a 180-day window entirely within the continuously populated months beginning January 2017.

This selection answers a question about customers who completed a first observed transaction and had not already reordered. It does not estimate churn among all platform visitors or all first-time buyers.

## Input definitions

| Group | Variables and treatment |
|---|---|
| Order context | Item value, item count, category, multiple-seller indicator |
| Shipping difficulty | Total weight/volume, mean item-level seller-to-customer straight-line distance |
| Geography | First-order customer state; low-frequency levels pooled by training-only encoder |
| Time, V1 and V2 A only | Sine/cosine of purchase month |
| Fulfillment | Purchase-to-delivery days, delivery-date lateness indicator, nonnegative late days, freight/item value |
| Regional structure | Smoothed local category share and log local/national relative share |

Long-tailed numeric measures use `log1p` before scaling. The item amount excludes freight. Lateness compares calendar dates so arrival on the promised date is not counted as late. Coordinate medians by ZIP prefix avoid multiplicative joins; distances are approximate straight-line distances, not routes. If any item has missing dimensions or seller geography, corresponding order-level measurements are flagged missing instead of treating incomplete sums as complete.

Multiple-category first orders use a distinct category code in the models. Their regional features average the values for each distinct category. Descriptive cases use single, known-category orders only. Product attributes and ZIP coordinates come from the released snapshot; historical changes to those static tables cannot be verified.

## Point-in-time regional features

For a first purchase at time `t`, history contains purchases in `[t-180 days, t)` whose approval is already known strictly before `t`. Each order contributes total category weight 1, split equally across its distinct categories. Since the focal purchase is the customer's first observed order, the preceding history cannot contain that customer's earlier observed purchase; this is checked on sampled direct calculations.

For category `c` and state `s`:

```text
p_national(c) = (national_category_weight + 0.5)
                / (national_order_weight + 0.5 * 74)

p_state(c) = (state_category_weight + 100 * p_national(c))
             / (state_order_weight + 100)

relative_share = log(p_state(c) / p_national(c))
```

The 74-category vocabulary is the released taxonomy plus unknown/untranslated labels. It is a static catalog, not a target-derived encoding. Smoothing constants were fixed before model evaluation, not optimized. The audit additionally counts unweighted historical order support for state/category combinations; these support counts are diagnostics, not the weighted model feature itself.

## Chronology and model stages

Initial training landmarks fall in July–August 2017; their labels are mature before December. December labels mature before April evaluation. V1's final refit uses landmarks before 2018-01-01 (25,780 customers) and evaluates April 1–May 2. V2/V3 keep the smaller initial training pool (6,459 customers), allowing December to be used for calibration or early stopping without post-development refitting.

V1 uses L2 logistic regression with `C=0.1`, `lbfgs`, `max_iter=3000`, `tol=1e-8`, no class weights. V2 uses the same settings, removing only month features for B; C adds an intercept-only correction with slope fixed at 1, fitted on December predictions from the frozen B. V3 uses one conservative LightGBM configuration and log-loss-only early stopping. See immutable [V1](../configs/v1/protocol.json), [V2](../configs/v2/protocol.json), [V3](../configs/v3/protocol.json) settings.

## Metrics and uncertainty

Average Precision uses noninterpolated precision/recall increments. Brier and log loss evaluate probabilities. ROC-AUC is supplementary. The top-10% list contains `ceil(0.1*n)` customers; equal scores are ordered by customer identifier, and boundary ties are disclosed.

Customer-level and state-cluster paired bootstrap intervals use fixed predictions. V2 separately bootstraps calibration observations to characterize intercept uncertainty. These do not refit bottom models and are not joint intervals covering all uncertainty. With only 27 states, unequal cluster sizes and few events, intervals are diagnostics. None removes repeated-evaluation/adaptive-analysis bias.

Calibration plots use five equal-count groups. Equal scores can be split across groups, particularly for G1; differing observed rates within such groups are not distinctions learned by the model. Descriptive-case intervals are customer-level Wilson intervals and do not adjust for state clustering or confounding.
