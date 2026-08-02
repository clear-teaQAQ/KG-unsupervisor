# V6 Results

Status: four-dataset functional smoke passed; unrestricted cosine is not
retained after the V7 evidence audit.

No training or GED cost change is part of this version.

## Affected-dataset smoke

This is a functional smoke test, not a generalization estimate. The loader
takes the first 100 entries of `test_GEDINFO.json` without random or stratified
sampling. Those entries cover only four YAGO source graphs and two WIKIDATA
source graphs. Accuracy values in this section must not be used as formal test
results.

Command:

```bash
SMOKE=1 DATASETS="YAGO WIKIDATA" \
  bash experiments/seabed_versions/v6_semantic_tie_break/run.sh
```

Both datasets used 100 test pairs, `test_k=5`, and 20 maximum semantic local
search iterations. YAGO loaded the V4 checkpoint ending in
`20260801_193451.pt`; WIKIDATA loaded the V4 checkpoint ending in
`20260801_231901.pt`.

| Dataset | Alignment before | Alignment after | Changed pairs | Improved pairs | Harmed pairs | Semantic seconds/pair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YAGO | 1,976 / 2,050 (96.39%) | 2,050 / 2,050 (100%) | 41 | 41 | 0 | 0.091388 |
| WIKIDATA | 2,680 / 2,700 (99.26%) | 2,700 / 2,700 (100%) | 17 | 17 | 0 | 0.142844 |

Path and cost invariants:

| Dataset | Simple validity / consistency / replay | Multi validity / consistency / replay | Simple cost unchanged | Multi cost unchanged |
| --- | --- | --- | ---: | ---: |
| YAGO | 1 / 1 / 1 | 1 / 1 / 1 | 1 | 1 |
| WIKIDATA | 1 / 1 / 1 | 1 / 1 / 1 | 1 | 1 |

All 200 saved path records were also checked directly. Their operation-cost
breakdowns sum to the recorded total costs, and their final shared-entity
alignment agrees with the aggregate results. The runtime dual-cost assertion
passed on every semantic move.

Smoke path quality was exact on this non-representative prefix sample:

| Dataset | Simple MAE / ACC / FEA | Multi MAE / ACC / FEA |
| --- | --- | --- |
| YAGO | 0 / 1 / 1 | 0 / 1 / 1 |
| WIKIDATA | 0 / 1 / 1 | 0 / 1 / 1 |

Result files:

```text
results/result_SEABED_v6_semantic_tie_break_YAGO_test_k5_two_swap_20260802_124924.json
results/result_SEABED_v6_semantic_tie_break_WIKIDATA_test_k5_two_swap_20260802_125059.json
```

Decision: the affected-dataset smoke is accepted.

## Control-dataset smoke

Command:

```bash
SMOKE=1 DATASETS="LUBM SWDF" \
  bash experiments/seabed_versions/v6_semantic_tie_break/run.sh
```

| Dataset | Alignment before | Alignment after | Changed pairs | Improved pairs | Harmed pairs | Semantic seconds/pair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LUBM | 28 / 43 (65.12%) | 30 / 43 (69.77%) | 83 | 2 | 0 | 0.008423 |
| SWDF | 0 / 1 (0%) | 0 / 1 (0%) | 78 | 0 | 0 | 0.007156 |

Both simple and multirelation paths retained validity, cost consistency, and
replay rates of 1.0. Both protected costs remained unchanged on every pair.
All 200 saved path cost breakdowns also matched their recorded total costs.

The control result is a safety pass, not sufficient semantic evidence. LUBM
has only 43 shared entities across the 100 pairs, and SWDF has only one. Most
cosine-selected mapping changes therefore cannot be judged by the available
entity-ID proxy: LUBM changes 83 pairs but verifies improvements in only two;
SWDF changes 78 pairs with no measurable improvement or harm. A full V6 run
must wait for an embedding-identity and nearest-neighbor audit.

Control result files:

```text
results/result_SEABED_v6_semantic_tie_break_LUBM_test_k5_two_swap_20260802_125848.json
results/result_SEABED_v6_semantic_tie_break_SWDF_test_k5_two_swap_20260802_125903.json
```

## Interpretation boundary

V6 cannot improve GED or executable-path accuracy over its own pre-semantic
mapping: every accepted mapping must have exactly the same simple and
multirelation costs. Its measured effect is only correspondence selection
inside the equal-cost solution set. The before/after GED values are therefore
identical by construction.

The SEABED derived pairs also retain the same node embedding for a copied
entity. In the V5 diagnostic, embedding cosine reached the exact entity-ID
local oracle on all observed improvements. This makes cosine a useful and safe
identity-preserving tie-break for explanations, but it is not evidence of new
semantic reasoning. Formal reporting must keep structural GED quality and
semantic correspondence quality as separate metrics.
