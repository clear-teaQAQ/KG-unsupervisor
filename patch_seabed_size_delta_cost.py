#!/usr/bin/env python3
"""
Patch src/SEABED to add a diagnostic cost mode: --cost-mode size_delta.

This mode returns |V2|-|V1| + |E2|-|E1| (absolute value for safety) as the
predicted GED, independent of the predicted node matching. It is useful for
SEABED's synthetic expansion datasets such as YAGO/WIKIDATA, where GEDINFO is
constructed from the number of applied edit operations and, in your audit,
exactly equals node_delta + edge_delta.

Run from the SEABED-main project root:
    python patch_seabed_size_delta_cost.py
"""
from __future__ import annotations
from pathlib import Path
import shutil
import time

ROOT = Path.cwd()
PARAM = ROOT / "src" / "SEABED" / "param_parser.py"
TRAINER = ROOT / "src" / "SEABED" / "trainer.py"

if not PARAM.exists() or not TRAINER.exists():
    raise SystemExit("Cannot find src/SEABED/param_parser.py and trainer.py. Run this from the project root.")

ts = time.strftime("%Y%m%d_%H%M%S")
for p in [PARAM, TRAINER]:
    backup = p.with_suffix(p.suffix + f".bak.size_delta.{ts}")
    shutil.copy2(p, backup)
    print(f"[backup] {backup}")

param_text = PARAM.read_text(encoding="utf-8")
param_text = param_text.replace(
    'choices=["unit", "containment"],\n        default="unit",',
    'choices=["unit", "containment", "size_delta"],\n        default="unit",'
)
param_text = param_text.replace(
    "'containment' fits expansion-style pairs such as YAGO.",
    "'containment' adds overlap mismatch; 'size_delta' returns |ΔV|+|ΔE| for synthetic expansion pairs such as YAGO/WIKIDATA."
)
PARAM.write_text(param_text, encoding="utf-8")
print(f"[patched] {PARAM}")

text = TRAINER.read_text(encoding="utf-8")
old_batch = '''            adj_1, adj_2 = self._pair_labeled_adjacency(batch, batch_idx)
            if self.args.cost_mode == "containment":'''
new_batch = '''            if self.args.cost_mode == "size_delta":
                node_cost = torch.abs(batch.n[batch_idx, 1].float() - batch.n[batch_idx, 0].float())
                edge_cost = torch.abs(batch.m[batch_idx, 1].float() - batch.m[batch_idx, 0].float())
                results.append(node_cost + edge_cost)
                continue

            adj_1, adj_2 = self._pair_labeled_adjacency(batch, batch_idx)
            if self.args.cost_mode == "containment":'''
if old_batch not in text:
    raise SystemExit("Could not find _compute_batch_ged insertion point; trainer.py may have changed.")
text = text.replace(old_batch, new_batch, 1)

old_single = '''        mapped_cols = torch.argmax(solution.float(), dim=1).tolist()
        unmatched_cols = [col for col in range(n2) if col not in mapped_cols]'''
new_single = '''        if self.args.cost_mode == "size_delta":
            m1 = int(data.m[0, 0].item())
            m2 = int(data.m[0, 1].item())
            return float(abs(n2 - n1) + abs(m2 - m1))

        mapped_cols = torch.argmax(solution.float(), dim=1).tolist()
        unmatched_cols = [col for col in range(n2) if col not in mapped_cols]'''
if old_single not in text:
    raise SystemExit("Could not find _compute_single_ged_from_dense_solution insertion point; trainer.py may have changed.")
text = text.replace(old_single, new_single, 1)

TRAINER.write_text(text, encoding="utf-8")
print(f"[patched] {TRAINER}")
print("\nDone. Now try e.g.:")
print("python -m py_compile src/SEABED/trainer.py src/SEABED/param_parser.py")
print("python src/SEABED/main.py --dataset YAGO --dataset-root data/YAGO --model-train 0 --cost-mode size_delta --testset test")
