# V13: V11-Preserving Gated Anchor Residual

## Goal

Preserve the original V11 unit-cost GED task and improve only within that task.
This version does **not** redefine the GED cost, does **not** introduce a
semantic-cost target, and treats V11 MAE/ACC as the hard baseline.

The first requirement is parity:

- `V13_MODE=baseline` should behave as the V11 relation-aware model, with only
  output paths and metadata changed.
- Any new module must be gated or disabled by default so it cannot silently
  invalidate the V11 baseline.

## Modes

- `baseline`: exact V11 relation-aware model/trainer path.
- `gated_anchor`: V11 plus a scalar residual bias on exact-anchor candidate
  mapping logits:

  ```text
  mapping_logit = v11_mapping_logit + anchor_gate * exact_anchor_mask
  ```

  `anchor_gate` is initialized from `V13_ANCHOR_GATE_INIT`, default `0.0`.
  With the default initialization, the first forward pass is identical to V11.

This residual is intentionally unit-cost aligned: exact label/feature matches
are the part of the available semantic signal that the original unit-cost GED
objective actually recognizes.

## Reproducible execution

Smoke baseline parity:

```bash
SMOKE=1 DATASETS=LUBM V13_MODE=baseline \
  bash experiments/seabed_versions/v13_v11_preserving_gated_anchor_residual/train.sh
```

Smoke gated-anchor run:

```bash
SMOKE=1 DATASETS=LUBM V13_MODE=gated_anchor \
  bash experiments/seabed_versions/v13_v11_preserving_gated_anchor_residual/train.sh
```

Formal runs should only be started after the baseline mode is checked against
V11 on the same MAE/ACC reporting path.

