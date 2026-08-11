# V12 Results

Status: bootstrap implementation complete; LUBM one-epoch smoke and the four
control smoke runs (full/raw, no_critic/raw, full/constant, full/shuffled) all
completed successfully on 2026-08-10. These small runs validate execution only
and are not reported as method results. Formal attribution still requires the
same four configurations at 200 epochs on LUBM and SWDF.

Smoke output is written to `checkpoints/` and `training_results/`. The shuffled
control can emit a `ConstantInputWarning` for rank correlation when a tiny
sample produces constant predictions; this does not stop evaluation.
