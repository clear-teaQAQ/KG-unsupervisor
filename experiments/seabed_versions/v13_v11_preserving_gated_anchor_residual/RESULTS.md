# V13 Results

Status: scaffolded.

Primary metrics are MAE and ACC under the original unit-cost GED setting
(`ged_column=3`, `cost_mode=unit`). V11 remains the baseline to beat:

- LUBM V11 training: MAE 0.094, ACC 0.913
- SWDF V11 training: MAE 0.266, ACC 0.754

Do not claim improvement until `V13_MODE=baseline` has reproduced V11 closely
and `V13_MODE=gated_anchor` improves or preserves MAE/ACC.

