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
    bak = p.with_suffix(p.suffix + f".bak.eval_mapping.{stamp}")
    shutil.copy2(p, bak)
    print(f"[backup] {bak}")

# Add --eval-mapping
text = PARAM.read_text(encoding="utf-8")
if "--eval-mapping" not in text:
    arg = '''    parser.add_argument(
        "--eval-mapping",
        choices=["model", "entity", "feature"],
        default="model",
        help="Diagnostic evaluation mode. model uses diffusion inference; entity/feature bypass the model and evaluate a fixed matching constructed like --init-mapping.",
    )
'''
    # Prefer placing after --init-mapping if present.
    m = re.search(r"parser\.add_argument\(\s*\"--init-mapping\"[\s\S]*?\n\s*\)\n", text)
    if m:
        text = text[:m.end()] + arg + text[m.end():]
    else:
        marker = "    parser.add_argument(\"--topk-approach\""
        idx = text.find(marker)
        if idx < 0:
            raise SystemExit("Cannot find insertion point in param_parser.py")
        text = text[:idx] + arg + text[idx:]
    PARAM.write_text(text, encoding="utf-8")
    print(f"[patched] {PARAM}")
else:
    print(f"[skip] --eval-mapping already present in {PARAM}")

text = TRAINER.read_text(encoding="utf-8")
if "def _eval_fixed_mapping" not in text:
    if "def _init_mapping_label" not in text:
        raise SystemExit("trainer.py does not contain _init_mapping_label. Run patch_seabed_entity_init.py first.")
    marker = "    def diffusion_ged_parallel(self, batch, test_k=100):\n"
    helper = r'''    def _eval_fixed_mapping(self, batch):
        """Diagnostic evaluation: bypass diffusion and score a fixed entity/feature matching.

        This is not a formal model result. It checks whether the data representation,
        fixed matching construction, and unit-cost evaluator agree with the GED labels.
        """
        start_time = time.time()
        data = batch[0]
        n1 = int(data.n[0, 0].item())
        n2 = int(data.n[0, 1].item())
        id_1 = int(data.i_j[0, 0].item())
        id_2 = int(data.i_j[0, 1].item())
        mode = getattr(self.args, "eval_mapping", "model")
        if mode == "model":
            raise ValueError("_eval_fixed_mapping called with eval_mapping=model")

        old_mode = getattr(self.args, "init_mapping", "random")
        self.args.init_mapping = mode
        try:
            label = self._init_mapping_label(id_1, id_2, n1, n2)
        finally:
            self.args.init_mapping = old_mode
        if label is None:
            raise RuntimeError(f"Could not construct fixed eval mapping for mode={mode}")
        solution = label.view(n1, n2).to(self.device)
        pre_ged = self._compute_single_ged_from_dense_solution(solution, data)
        return pre_ged, solution, time.time() - start_time

'''
    if marker not in text:
        raise SystemExit("Cannot find diffusion_ged_parallel insertion point in trainer.py")
    text = text.replace(marker, helper + marker)

# Add early bypass to parallel and sequential inference.
old = '''    def diffusion_ged_parallel(self, batch, test_k=100):
        start_time = time.time()
        num_parallel_sampling = test_k
        data = batch[0]
'''
new = '''    def diffusion_ged_parallel(self, batch, test_k=100):
        if getattr(self.args, "eval_mapping", "model") != "model":
            return self._eval_fixed_mapping(batch)
        start_time = time.time()
        num_parallel_sampling = test_k
        data = batch[0]
'''
if old in text:
    text = text.replace(old, new)
elif 'def diffusion_ged_parallel(self, batch, test_k=100):\n        if getattr(self.args, "eval_mapping", "model") != "model":' in text:
    pass
else:
    raise SystemExit("Cannot patch diffusion_ged_parallel; inspect manually.")

old = '''    def diffusion_ged_sequential(self, batch, test_k=100):
        start_time = time.time()
        mapping_t = torch.randn_like(batch.edge_attr_mapping, device=self.device)
'''
new = '''    def diffusion_ged_sequential(self, batch, test_k=100):
        if getattr(self.args, "eval_mapping", "model") != "model":
            return self._eval_fixed_mapping(batch)
        start_time = time.time()
        mapping_t = torch.randn_like(batch.edge_attr_mapping, device=self.device)
'''
if old in text:
    text = text.replace(old, new)
elif 'def diffusion_ged_sequential(self, batch, test_k=100):\n        if getattr(self.args, "eval_mapping", "model") != "model":' in text:
    pass
else:
    print("[warn] Could not patch diffusion_ged_sequential; parallel mode is enough for diagnostics.")

TRAINER.write_text(text, encoding="utf-8")
print(f"[patched] {TRAINER}")
print("\nDone. Recommended checks:")
print("python -m py_compile src/SEABED/trainer.py src/SEABED/param_parser.py")
print("python main.py --dataset YAGO --dataset-root /root/autodl-tmp/SEABED-main/data/YAGO/ --cost-mode unit --init-mapping entity --eval-mapping entity --model-train 1 --model-epoch-start 0 --model-epoch-end 1")
