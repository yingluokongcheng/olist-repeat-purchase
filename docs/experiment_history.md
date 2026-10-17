# Preserved experiment history

This record is descriptive, not a claim of preregistration before seeing the original evaluation set. The project stopped after V3. Packaging did not run new experiments.

| Stage | Motivation and scope | Retained outcome |
|---|---|---|
| Audit | Test feasibility before model fitting | 51,981 customers / 567 repeats; standalone category models unsupported |
| V1 | Compare background, fulfillment and regional inputs in logistic regression | Small fulfillment gain; negligible regional increment; probability underestimation |
| V2 | Diagnose month extrapolation and intercept calibration | Month removal reduced losses, not AP; calibration worsened later-period point estimates |
| V3 | One final nonlinear and regional comparison | G1 stopped at one tree; G2 gained over G1 in point estimates without stable increment; no clear overall win over LR |

## Decisions that constrain interpretation

1. Olist was selected for named geography and fulfillment fields, not because it promised a large repeat-purchase signal.
2. Culture was not treated as directly measured. The hypothesis was narrowed to observed regional purchase mix and fulfillment experience.
3. Sample audit reduced scope: state-level pooled analysis, two descriptive category cases, no standalone city/category models.
4. V1 evaluation was viewed and diagnosed. Subsequent V2/V3 comparisons used the same evaluation cohort and were explicitly labeled exploratory.
5. Each revision used a saved protocol before its own run. This limits further tuning but cannot undo earlier information exposure.
6. No change was made after viewing a revision's evaluation to improve that revision's score. V3 was explicitly the last algorithm attempt.

## Negative and mixed findings are part of the result

- V1 regional features barely changed metrics.
- V2 calibration aligned the development-period mean but worsened the later evaluation's mean error and loss point estimates.
- V2 removing month features did not improve ranking.
- V3 G1's near-correct mean was accompanied by weak discrimination and a large tied top-score group.
- V3 G2's 18 top-list hits did not correspond to a higher overall AP than LR.
- Neither region nor model complexity was established as a stable improvement under the fixed criteria.

## Recorded evidence

Each `reports/` stage includes recorded numeric outputs, figures and the original stage narrative. Stage reports are historical snapshots: a V1 report saying LightGBM had not been run was true at that stage; it is not a statement about the completed V3 project.

Original numerical tables, JSON records, figures, analysis scripts and selected LightGBM text models are copied byte-for-byte. Training logs retain their warnings and content, with personal absolute environment paths redacted. HTML adds only an archive banner and a repository reproduction link. [Provenance](provenance.json) records original and packaged hashes and identifies these presentation changes. Older preservation records use the original local relative paths as historical evidence; they are not links to files expected in this repository.

## What this portfolio does and does not demonstrate

It demonstrates cohort design, time-aware features, rare-event evaluation, constrained model comparison, calibration diagnosis and honest interpretation. The implementation is Python-based. It does not claim an implemented SQL warehouse, deployed retention system, randomized intervention, business revenue lift or a novel algorithm.
