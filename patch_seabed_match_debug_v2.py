#!/usr/bin/env python3
from __future__ import annotations
import re
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path.cwd()
TRAINER = ROOT / "src" / "SEABED" / "trainer.py"
PARAM = ROOT / "src" / "SEABED" / "param_parser.py"

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for p in [TRAINER, PARAM]:
    if not p.exists():
        raise SystemExit(f"Cannot find {p}. Run this from GEDRanker-main project root.")
    bak = p.with_suffix(p.suffix + f".bak.match_debug_v2.{stamp}")
    shutil.copy2(p, bak)
    print(f"[backup] {bak}")

# param_parser: add flag if needed
text = PARAM.read_text(encoding="utf-8")
if "--debug-match-metrics" not in text:
    arg = '''    parser.add_argument(
        "--debug-match-metrics",
        action="store_true",
        help="Diagnostic: compare model-predicted matching against entity-oracle matching during evaluation.",
    )
'''
    # Prefer after --eval-mapping, then --init-mapping, then before topk
    m = re.search(r"parser\.add_argument\(\s*\"--eval-mapping\"[\s\S]*?\n\s*\)\n", text)
    if not m:
        m = re.search(r"parser\.add_argument\(\s*\"--init-mapping\"[\s\S]*?\n\s*\)\n", text)
    if m:
        text = text[:m.end()] + arg + text[m.end():]
    else:
        marker = '    parser.add_argument("--topk-approach"'
        idx = text.find(marker)
        if idx < 0:
            raise SystemExit("Cannot find insertion point in param_parser.py")
        text = text[:idx] + arg + text[idx:]
    PARAM.write_text(text, encoding="utf-8")
    print(f"[patched] {PARAM}")
else:
    print(f"[skip] --debug-match-metrics already present in {PARAM}")

text = TRAINER.read_text(encoding="utf-8")
if "def _init_mapping_label" not in text:
    raise SystemExit("trainer.py does not contain _init_mapping_label. Run patch_seabed_entity_init.py first, then rerun this patch.")

# Insert helper before diffusion_ged_parallel
if "def _debug_matching_metrics" not in text:
    marker = "    def diffusion_ged_parallel(self, batch, test_k=100):\n"
    if marker not in text:
        raise SystemExit("Cannot find insertion point before diffusion_ged_parallel.")
    helper = r'''    def _entity_oracle_solution_for_data(self, data):
        """Construct n1 x n2 entity-oracle matching for one graph pair Data object."""
        n1 = int(data.n[0, 0].item())
        n2 = int(data.n[0, 1].item())
        id_1 = int(data.i_j[0, 0].item())
        id_2 = int(data.i_j[0, 1].item())
        old_mode = getattr(self.args, "init_mapping", "random")
        self.args.init_mapping = "entity"
        try:
            label = self._init_mapping_label(id_1, id_2, n1, n2)
        finally:
            self.args.init_mapping = old_mode
        if label is None:
            return None
        return label.view(n1, n2).to(self.device)

    def _debug_matching_metrics(self, pred_solution, batch, gt):
        """Compare model-predicted n1 x n2 solution with entity-oracle matching."""
        data = batch[0]
        if pred_solution is None or (hasattr(pred_solution, "numel") and pred_solution.numel() == 0):
            return None
        oracle = self._entity_oracle_solution_for_data(data)
        if oracle is None:
            return None
        pred = pred_solution.to(self.device).float()
        if pred.dim() != 2:
            pred = pred.view_as(oracle)
        if pred.shape != oracle.shape:
            return None
        pred_cols = torch.argmax(pred, dim=1)
        oracle_cols = torch.argmax(oracle.float(), dim=1)
        row_hits = (pred_cols == oracle_cols)
        oracle_ged = self._compute_single_ged_from_dense_solution(oracle, data)
        pred_ged = self._compute_single_ged_from_dense_solution(pred, data)
        return {
            "row_acc": float(row_hits.float().mean().item()),
            "exact_pair": float(row_hits.all().item()),
            "oracle_ged": float(oracle_ged),
            "pred_ged_from_solution": float(pred_ged),
            "oracle_ged_abs_err": float(abs(oracle_ged - gt)),
        }

'''
    text = text.replace(marker, helper + marker)
    print("[patched] inserted _debug_matching_metrics helpers")
else:
    print("[skip] debug helpers already present")

# Ensure score() initializes debug arrays
if "match_row_acc = []" not in text:
    text = text.replace(
        "        pres = {}\n        gts = {}\n",
        "        pres = {}\n        gts = {}\n        match_row_acc = []\n        match_exact_pair = []\n        match_oracle_ged_mae = []\n        match_pred_ged = []\n"
    )
    print("[patched] inserted debug metric accumulators")

# Replace inference return usage in score if still using underscore.
text2 = text.replace(
'''            if top_k_approach == "parallel":
                pre_ged, _, running_time = self.diffusion_ged_parallel(batch, test_k)
            else:
                pre_ged, _, running_time = self.diffusion_ged_sequential(batch, test_k)
''',
'''            if top_k_approach == "parallel":
                pre_ged, pred_solution, running_time = self.diffusion_ged_parallel(batch, test_k)
            else:
                pre_ged, pred_solution, running_time = self.diffusion_ged_sequential(batch, test_k)
''')
if text2 != text:
    text = text2
    print("[patched] score() now keeps pred_solution")

# Insert debug collection after inference block, before num += 1
if "dbg = self._debug_matching_metrics(pred_solution, batch, gt)" not in text:
    anchor = "\n            num += 1\n"
    block = '''
            if getattr(self.args, "debug_match_metrics", False):
                dbg = self._debug_matching_metrics(pred_solution, batch, gt)
                if dbg is not None:
                    match_row_acc.append(dbg["row_acc"])
                    match_exact_pair.append(dbg["exact_pair"])
                    match_oracle_ged_mae.append(dbg["oracle_ged_abs_err"])
                    match_pred_ged.append(dbg["pred_ged_from_solution"])
'''
    if anchor not in text:
        raise SystemExit("Cannot find 'num += 1' anchor in score().")
    text = text.replace(anchor, block + anchor, 1)
    print("[patched] inserted debug metric collection")

# Insert debug print after normal result print.
if "MATCH_DEBUG" not in text:
    anchor = '''        print(*self.results[-2], sep="\t")
        print(*self.results[-1], sep="\t")
'''
    block = '''        print(*self.results[-2], sep="\t")
        print(*self.results[-1], sep="\t")
        if getattr(self.args, "debug_match_metrics", False) and match_row_acc:
            debug_summary = {
                "model_vs_entity_oracle_row_acc": round(float(np.mean(match_row_acc)), 4),
                "model_vs_entity_oracle_exact_pair_rate": round(float(np.mean(match_exact_pair)), 4),
                "entity_oracle_ged_mae": round(float(np.mean(match_oracle_ged_mae)), 4),
                "pred_solution_ged_mean": round(float(np.mean(match_pred_ged)), 4),
            }
            print("MATCH_DEBUG", json.dumps(debug_summary, ensure_ascii=False))
'''
    if anchor not in text:
        raise SystemExit("Cannot find result-print anchor in score().")
    text = text.replace(anchor, block, 1)
    print("[patched] inserted MATCH_DEBUG print")

TRAINER.write_text(text, encoding="utf-8")
print(f"[patched] {TRAINER}")
print("\nRun checks:")
print("grep -n 'MATCH_DEBUG\\|debug_match_metrics\\|_debug_matching_metrics' src/SEABED/trainer.py src/SEABED/param_parser.py")
print("python -m py_compile src/SEABED/trainer.py src/SEABED/param_parser.py")
