# V18 Results

Status: implementation validation complete; formal training pending. Smoke
metrics are plumbing checks only and must not be interpreted as effectiveness
results.

Validation:

| Check | Result |
| --- | --- |
| Python compilation and shell syntax | passed |
| Exact/wrong/broken official-edge events | passed |
| Batched reasoner equals pairwise reasoner | passed |
| Soft-mapping gradient | passed |
| Zero-gate V11 discriminator parity | passed exactly |
| LUBM matched-edge smoke | passed |
| SWDF matched-edge smoke | passed |
| LUBM same-path baseline smoke | passed |
| Alpha-zero fast path | passed |
| Runtime cache fallbacks / shape mismatches | 0 / 0 |

On a 16-pair CPU plumbing run, the first training phase processed about 112
pairs/s and the alpha-zero fast path processed about 365 pairs/s. This is not a
GPU benchmark or an effectiveness result; it only verifies that the redundant
second-half discriminator computation is removed.

Hard references:

| Dataset | Required baseline | MAE | ACC |
| --- | --- | ---: | ---: |
| LUBM | V11 independent seed 0 | 0.092 | 0.916 |
| SWDF | V16 official graph seed 0 | 0.070 | 0.934 |
