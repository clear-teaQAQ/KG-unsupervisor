# V14 Results

Status: implementation validation and paired LUBM smoke complete; formal runs
pending.

## Validation

| Check | Result |
| --- | --- |
| Python compilation | passed |
| Shell syntax | passed |
| Exact/wrong/broken event tests | passed |
| Hard unit edge-cost reconstruction | passed |
| Soft matching gradient | passed |
| Zero-gate V11 discriminator parity | passed exactly |
| LUBM baseline/full paired smoke | passed |

With identical smoke inputs, baseline and matched-edge begin with the same
`D loss=1729.174`, mapping loss, candidate GED, and best GED. The full gate
moves from `0.0` to approximately `-0.01` after one epoch, confirming that the
new evidence participates in training only after the initially disabled
residual is learned.

## Hard references

| Dataset | V11 MAE | V11 ACC |
| --- | ---: | ---: |
| LUBM | 0.092 | 0.916 |
| SWDF | 0.263 | 0.756 |

No improvement claim is allowed from smoke runs. A V14 claim requires paired
baseline/full runs, independent fixed-seed evaluation, and MAE/ACC comparison.
