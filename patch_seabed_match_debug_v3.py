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
    bak = p.with_suffix(p.suffix + f".bak.match_debug_v3.{stamp}")
    shutil.copy2(p, bak)
    print(f"[backup] {bak}")

# 1) Make sure CLI flag exists.
param = PARAM.read_text(encoding="utf-8")
if "--debug-match-metrics" not in param:
    arg = '''    parser.add_argument(
        "--debug-match-metrics",
        action="store_true",
        help="Diagnostic: compare model-predicted matching against entity-oracle matching during evaluation.",
    )
'''
    # Put after --eval-mapping or --init-mapping if possible.
    m = re.search(r"parser\.add_argument\(\s*['\"]--eval-mapping['\"][\s\S]*?\n\s*\)\n", param)
    if not m:
        m = re.search(r"parser\.add_argument\(\s*['\"]--init-mapping['\"][\s\S]*?\n\s*\)\n", param)
    if m:
        param = param[:m.end()] + arg + param[m.end():]
    else:
        # Fallback: insert before return args / args = parser.parse_args
        idx = param.find("args = parser.parse_args")
        if idx < 0:
            idx = param.find("return parser.parse_args")
        if idx < 0:
            raise SystemExit("Cannot find insertion point in param_parser.py")
        param = param[:idx] + arg + param[idx:]
    PARAM.write_text(param, encoding="utf-8")
    print("[patched] added --debug-match-metrics to param_parser.py")
else:
    print("[skip] --debug-match-metrics already present")

text = TRAINER.read_text(encoding="utf-8")
if "def _init_mapping_label" not in text:
    raise SystemExit("trainer.py does not contain _init_mapping_label. Run patch_seabed_entity_init.py first.")

# 2) Insert helper methods before diffusion_ged_parallel.
if "def _debug_matching_metrics" not in text:
    marker = "    def diffusion_ged_parallel(self, batch, test_k=100):\n"
    if marker not in text:
        raise SystemExit("Cannot find insertion point before diffusion_ged_parallel in trainer.py")
    helper = r'''    def _entity_oracle_solution_for_data(self, data):
        """Construct n1 x n2 entity-oracle matching for one graph-pair Data object."""
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
        """Compare model-predicted solution with entity-oracle matching."""
        data = batch[0]
        if pred_solution is None:
            return None
        if isinstance(pred_solution, list) and len(pred_solution) == 0:
            return None
        if hasattr(pred_solution, "numel") and pred_solution.numel() == 0:
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
        row_hits = pred_cols == oracle_cols
        try:
            oracle_ged = self._compute_single_ged_from_dense_solution(oracle, data)
            pred_ged = self._compute_single_ged_from_dense_solution(pred, data)
        except Exception:
            oracle_ged = float("nan")
            pred_ged = float("nan")
        return {
            "row_acc": float(row_hits.float().mean().item()),
            "exact_pair": float(row_hits.all().item()),
            "oracle_ged_abs_err": float(abs(oracle_ged - gt)) if oracle_ged == oracle_ged else float("nan"),
            "pred_ged_from_solution": float(pred_ged) if pred_ged == pred_ged else float("nan"),
        }

'''
    text = text.replace(marker, helper + marker)
    print("[patched] inserted _debug_matching_metrics helper methods")
else:
    print("[skip] debug helper methods already present")

# 3) Patch only the body of score().
score_pos = text.find("    def score(self,")
if score_pos < 0:
    raise SystemExit("Cannot find score() in trainer.py")
next_def = text.find("\n    def ", score_pos + 1)
if next_def < 0:
    next_def = len(text)
pre = text[:score_pos]
score = text[score_pos:next_def]
post = text[next_def:]

# accumulators after pres/gts
if "match_row_acc = []" not in score:
    if "        pres = {}\n        gts = {}\n" in score:
        score = score.replace(
            "        pres = {}\n        gts = {}\n",
            "        pres = {}\n        gts = {}\n"
            "        match_row_acc = []\n"
            "        match_exact_pair = []\n"
            "        match_oracle_ged_mae = []\n"
            "        match_pred_ged = []\n",
            1,
        )
    else:
        raise SystemExit("Cannot find pres/gts initialization inside score().")
    print("[patched] inserted debug accumulators inside score()")

# Keep pred_solution returned by inference.
score_new = re.sub(
    r"pre_ged,\s*_,\s*running_time\s*=\s*self\.diffusion_ged_parallel\(batch,\s*test_k\)",
    "pre_ged, pred_solution, running_time = self.diffusion_ged_parallel(batch, test_k)",
    score,
)
score_new = re.sub(
    r"pre_ged,\s*_,\s*running_time\s*=\s*self\.diffusion_ged_sequential\(batch,\s*test_k\)",
    "pre_ged, pred_solution, running_time = self.diffusion_ged_sequential(batch, test_k)",
    score_new,
)
if score_new != score:
    score = score_new
    print("[patched] score() now keeps pred_solution")
else:
    print("[skip/warn] did not find inference lines with '_' to replace; maybe already patched")

# Add collection before num += 1
if "dbg = self._debug_matching_metrics(pred_solution, batch, gt)" not in score:
    anchor = "\n            num += 1\n"
    block = '''
            if getattr(self.args, "debug_match_metrics", False):
                dbg = self._debug_matching_metrics(pred_solution, batch, gt)
                if dbg is not None:
                    match_row_acc.append(dbg["row_acc"])
                    match_exact_pair.append(dbg["exact_pair"])
                    if dbg["oracle_ged_abs_err"] == dbg["oracle_ged_abs_err"]:
                        match_oracle_ged_mae.append(dbg["oracle_ged_abs_err"])
                    if dbg["pred_ged_from_solution"] == dbg["pred_ged_from_solution"]:
                        match_pred_ged.append(dbg["pred_ged_from_solution"])
'''
    if anchor not in score:
        raise SystemExit("Cannot find 'num += 1' anchor inside score().")
    score = score.replace(anchor, block + anchor, 1)
    print("[patched] inserted debug metric collection inside score()")

# Add print after normal result line, robust to variations.
if "MATCH_DEBUG" not in score:
    line = '        print(*self.results[-1], sep="\\t")\n'
    if line not in score:
        # tolerate single quotes
        line = "        print(*self.results[-1], sep='\\t')\n"
    if line not in score:
        raise SystemExit("Cannot find final result print line inside score().")
    block = line + '''        if getattr(self.args, "debug_match_metrics", False) and match_row_acc:
            debug_summary = {
                "model_vs_entity_oracle_row_acc": round(float(np.mean(match_row_acc)), 4),
                "model_vs_entity_oracle_exact_pair_rate": round(float(np.mean(match_exact_pair)), 4),
                "entity_oracle_ged_mae": round(float(np.mean(match_oracle_ged_mae)), 4) if match_oracle_ged_mae else None,
                "pred_solution_ged_mean": round(float(np.mean(match_pred_ged)), 4) if match_pred_ged else None,
            }
            print("MATCH_DEBUG", json.dumps(debug_summary, ensure_ascii=False))
'''
    score = score.replace(line, block, 1)
    print("[patched] inserted MATCH_DEBUG print inside score()")

text = pre + score + post
TRAINER.write_text(text, encoding="utf-8")
print(f"[patched] {TRAINER}")
print("\nRecommended checks:")
print("grep -n 'MATCH_DEBUG\\|debug_match_metrics\\|_debug_matching_metrics' src/SEABED/trainer.py src/SEABED/param_parser.py")
print("python -m py_compile src/SEABED/trainer.py src/SEABED/param_parser.py")
