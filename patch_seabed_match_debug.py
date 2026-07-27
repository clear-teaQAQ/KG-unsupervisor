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
    bak = p.with_suffix(p.suffix + f".bak.match_debug.{stamp}")
    shutil.copy2(p, bak)
    print(f"[backup] {bak}")

# ---- param_parser.py: add debug flag ----
text = PARAM.read_text(encoding="utf-8")
if "--debug-match-metrics" not in text:
    arg = '''    parser.add_argument(
        "--debug-match-metrics",
        action="store_true",
        help="During evaluation, compare the model-predicted matching with the entity-oracle matching and print row-level matching accuracy. Diagnostic only.",
    )
'''
    m = re.search(r"parser\.add_argument\(\s*\"--eval-mapping\"[\s\S]*?\n\s*\)\n", text)
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

# ---- trainer.py: add helpers and collect metrics ----
text = TRAINER.read_text(encoding="utf-8")
if "def _debug_matching_metrics" not in text:
    if "def _init_mapping_label" not in text:
        raise SystemExit("trainer.py does not contain _init_mapping_label. Run patch_seabed_entity_init.py first.")
    marker = "    def diffusion_ged_parallel(self, batch, test_k=100):\n"
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

    def _debug_matching_metrics(self, pred_solution, batch):
        """Return row accuracy and exact-pair flag vs entity-oracle matching.

        pred_solution is the n1 x n2 matrix returned by model inference.
        Row accuracy answers: among G1 nodes, how many are matched to the
        same G2 local node as the raw-entity oracle matching?
        """
        data = batch[0]
        if pred_solution is None or len(pred_solution) == 0:
            return None
        oracle = self._entity_oracle_solution_for_data(data)
        if oracle is None:
            return None
        pred = pred_solution.to(self.device).float()
        if pred.dim() != 2:
            pred = pred.view_as(oracle)
        pred_cols = torch.argmax(pred, dim=1)
        oracle_cols = torch.argmax(oracle.float(), dim=1)
        row_hits = (pred_cols == oracle_cols)
        row_acc = row_hits.float().mean().item()
        exact_pair = float(row_hits.all().item())
        oracle_ged = self._compute_single_ged_from_dense_solution(oracle, data)
        pred_ged = self._compute_single_ged_from_dense_solution(pred, data)
        return {
            "row_acc": row_acc,
            "exact_pair": exact_pair,
            "oracle_ged": oracle_ged,
            "pred_ged_from_solution": pred_ged,
        }

'''
    if marker not in text:
        raise SystemExit("Cannot find insertion point before diffusion_ged_parallel.")
    text = text.replace(marker, helper + marker)

# add collections in score
old = '''        pres = {}
        gts = {}

        for batch in tqdm(loader, file=sys.stdout):
'''
new = '''        pres = {}
        gts = {}
        match_row_acc = []
        match_exact_pair = []
        match_oracle_ged_mae = []

        for batch in tqdm(loader, file=sys.stdout):
'''
if old in text:
    text = text.replace(old, new)
elif "match_row_acc = []" in text:
    pass
else:
    raise SystemExit("Cannot patch score metric initialization; inspect trainer.py score().")

old = '''            if top_k_approach == "parallel":
                pre_ged, _, running_time = self.diffusion_ged_parallel(batch, test_k)
            else:
                pre_ged, _, running_time = self.diffusion_ged_sequential(batch, test_k)

            num += 1
'''
new = '''            if top_k_approach == "parallel":
                pre_ged, pred_solution, running_time = self.diffusion_ged_parallel(batch, test_k)
            else:
                pre_ged, pred_solution, running_time = self.diffusion_ged_sequential(batch, test_k)

            if getattr(self.args, "debug_match_metrics", False):
                dbg = self._debug_matching_metrics(pred_solution, batch)
                if dbg is not None:
                    match_row_acc.append(dbg["row_acc"])
                    match_exact_pair.append(dbg["exact_pair"])
                    match_oracle_ged_mae.append(abs(dbg["oracle_ged"] - gt))

            num += 1
'''
if old in text:
    text = text.replace(old, new)
elif "pred_solution, running_time" in text and "debug_match_metrics" in text:
    pass
else:
    raise SystemExit("Cannot patch inference return usage in score(); inspect trainer.py score().")

# print debug metrics after existing table print
old = '''        print(*self.results[-2], sep="\t")
        print(*self.results[-1], sep="\t")

        with open(self.result_dir / f"result_SEABED_{self.args.dataset}_{testing_graph_set}_{self.args.unsupervised_approach}.json", "w", encoding="utf-8") as handle:
'''
new = '''        print(*self.results[-2], sep="\t")
        print(*self.results[-1], sep="\t")
        if getattr(self.args, "debug_match_metrics", False) and match_row_acc:
            debug_summary = {
                "model_vs_entity_oracle_row_acc": round(float(np.mean(match_row_acc)), 4),
                "model_vs_entity_oracle_exact_pair_rate": round(float(np.mean(match_exact_pair)), 4),
                "entity_oracle_ged_mae": round(float(np.mean(match_oracle_ged_mae)), 4),
            }
            print("MATCH_DEBUG", json.dumps(debug_summary, ensure_ascii=False))

        with open(self.result_dir / f"result_SEABED_{self.args.dataset}_{testing_graph_set}_{self.args.unsupervised_approach}.json", "w", encoding="utf-8") as handle:
'''
if old in text:
    text = text.replace(old, new)
elif "MATCH_DEBUG" in text:
    pass
else:
    raise SystemExit("Cannot patch debug print; inspect score() ending.")

TRAINER.write_text(text, encoding="utf-8")
print(f"[patched] {TRAINER}")
print("\nDone. Run:")
print("python -m py_compile src/SEABED/trainer.py src/SEABED/param_parser.py")
print("python main.py --dataset YAGO --dataset-root /root/autodl-tmp/SEABED-main/data/YAGO/ --cost-mode unit --init-mapping entity --eval-mapping model --debug-match-metrics --model-train 0 --model-epoch-start <EPOCH> --model-epoch-end <EPOCH>")
