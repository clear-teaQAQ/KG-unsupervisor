# V15 SWDF CPU Audit

This is a read-only audit. It does not change GED cost, column-3 labels,
preference supervision, model code, checkpoints, or the primary MAE/ACC results.

## V15 paired result

| Metric | Raw | Projected input | Change |
|---|---:|---:|---:|
| MAE | 0.263 | 0.242 | -0.021 |
| ACC | 0.757 | 0.773 | +0.016 |
| MSE | 0.306 | 0.273 | -0.033 |
| FEA | 0.925 | 0.921 | -0.004 |
| RHO | 0.927 | 0.935 | +0.008 |
| TAU | 0.888 | 0.898 | +0.010 |

## Test-pair structure

- Test pairs: 10000
- Unique test graphs involved: 200
- Pairs exposed to at least one collapsed graph: 9231 (92.31%)
- Pair-weighted raw edges removed by last-write projection: 18.26%
- Full-multigraph lower-bound certificates: 293
- Simple-graph count lower-bound violations: 0
- Official GED mean/range: 9.7258 / [0, 18]

## Aggregate error constraints

The old result files did not save per-pair mappings. The following counts
are therefore inferred from the rounded aggregate metrics, not reconstructed
per-pair predictions.

| Quantity | Raw | Projected input |
|---|---:|---:|
| Exact pairs | 7570 | 7730 |
| Non-exact pairs | 2430 | 2270 |
| Below-GT reports | 750 | 790 |
| Above-GT reports | 1680 | 1480 |
| Absolute-error mass beyond all-wrong-by-one minimum | 200 | 150 |

## Interpretation

The projection addresses a real input/target mismatch, but nearly every test
pair is exposed to collapsed edges. Exposure is not the same as an error, so
17.5% fewer unique input edges cannot be translated directly into 17.5 ACC
points. The projected run adds about 160 exact pairs and removes about 50 units
of aggregate absolute-error mass. Most remaining errors are tightly concentrated
near one edit, while the below-GT reports show that a frozen-mapping executable-
path audit is still needed before attributing every miss to the model.
