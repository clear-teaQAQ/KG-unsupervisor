# V15 Results

Status: implementation and paired smoke validation complete; formal SWDF runs
pending.

## Validation

| Check | Result |
| --- | --- |
| Python compilation | passed |
| Shell syntax | passed |
| Projection and preserved-cost tests | 4/4 passed |
| SWDF paired 16-pair smoke | passed |
| LUBM no-drop parity smoke | passed exactly |
| Independent checkpoint discovery/evaluation | passed |

SWDF projection diagnostics:

```text
graphs=1000
affected_graphs=678
raw_edges=8124
projected_edges=6701
dropped_edges=1423 (17.516%)
parallel_endpoint_pairs=1217
max_multiplicity=4
```

In the paired SWDF smoke, raw and projected modes used the same candidate-cost
view and produced identical aggregate `pred GED=275`, `current best GED=265`,
and `gt GED=188`. Their generator/discriminator losses differed, confirming
that projection affected representation learning without changing candidate
unit-GED values.

LUBM dropped zero edges. Raw and projected modes matched exactly on generator,
discriminator, mapping, and GED losses, all candidate-GED aggregates, and test
MAE/ACC. This establishes the required no-drop parity control.

## Hard references

| Dataset | V11 MAE | V11 ACC |
| --- | ---: | ---: |
| LUBM | 0.092 | 0.916 |
| SWDF | 0.263 | 0.756 |

Smoke metrics are plumbing checks only. No effectiveness result has been
recorded yet.
