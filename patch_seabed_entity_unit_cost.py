#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path.cwd()
UTILS = ROOT / "src" / "SEABED" / "utils.py"
TRAINER = ROOT / "src" / "SEABED" / "trainer.py"
PARAM = ROOT / "src" / "SEABED" / "param_parser.py"

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for p in [UTILS, TRAINER, PARAM]:
    if not p.exists():
        raise SystemExit(f"Cannot find {p}. Please run this from GEDRanker-main root.")
    bak = p.with_suffix(p.suffix + f".bak.entity_unit.{stamp}")
    shutil.copy2(p, bak)
    print(f"[backup] {bak}")

# ---------------- utils.py: make sure local_id -> entity URI is loaded ----------------
text = UTILS.read_text(encoding="utf-8")

if "def _infer_node_ids_from_kg" not in text:
    marker = "def _extract_graph(graph_payload, gid, use_raw_features):\n"
    helper = r'''
def _infer_node_ids_from_kg(graph_payload):
    """Infer local node index -> raw entity URI from aligned KG triples and edge_indices."""
    edge_pairs = graph_payload.get("edge_indices", [])
    kg = graph_payload.get("KG", [])
    node_features = graph_payload.get("node_features", [])
    node_ids = [None for _ in range(len(node_features))]
    conflicts = []

    for triple, uv in zip(kg, edge_pairs):
        if len(triple) < 3 or len(uv) < 2:
            continue
        h, _, t = map(str, triple[:3])
        u, v = int(uv[0]), int(uv[1])
        for idx, ent in ((u, h), (v, t)):
            if 0 <= idx < len(node_ids):
                if node_ids[idx] is None:
                    node_ids[idx] = ent
                elif node_ids[idx] != ent:
                    conflicts.append((idx, node_ids[idx], ent))
    return node_ids, conflicts

'''
    if marker not in text:
        raise SystemExit("Cannot find _extract_graph in utils.py")
    text = text.replace(marker, helper + marker)

if '"node_ids"' not in text:
    text = text.replace(
        '    node_features = graph_payload.get("node_features", [])\n',
        '    node_features = graph_payload.get("node_features", [])\n'
        '    node_ids, node_id_conflicts = _infer_node_ids_from_kg(graph_payload)\n'
        '    if node_id_conflicts:\n'
        '        raise ValueError(f"Conflicting local node to raw entity mapping in graph gid={gid}: {node_id_conflicts[:3]}")\n',
        1,
    )

    text = text.replace(
        '        "features": features,\n',
        '        "features": features,\n'
        '        "node_ids": node_ids,\n',
        1,
    )

UTILS.write_text(text, encoding="utf-8")
print(f"[patched] {UTILS}")

# ---------------- param_parser.py: add entity_unit cost mode ----------------
text = PARAM.read_text(encoding="utf-8")

if "entity_unit" not in text:
    m = re.search(r'(parser\.add_argument\(\s*"--cost-mode"[\s\S]*?choices=\[)([^\]]+)(\])', text)
    if not m:
        raise SystemExit("Cannot find --cost-mode choices in param_parser.py")
    choices_text = m.group(2)
    new_choices_text = choices_text.rstrip() + ', "entity_unit"'
    text = text[:m.start(2)] + new_choices_text + text[m.end(2):]

    text = text.replace(
        "'unit' keeps the current edge-mismatch style cost; 'containment' fits expansion-style pairs such as YAGO.",
        "'unit' keeps the current indexed dense edge-mismatch cost; 'entity_unit' scores the model-predicted matching using KG entity/predicate triples; 'containment' fits expansion-style pairs such as YAGO.",
    )

PARAM.write_text(text, encoding="utf-8")
print(f"[patched] {PARAM}")

# ---------------- trainer.py: add entity_unit evaluator ----------------
text = TRAINER.read_text(encoding="utf-8")

if "from collections import Counter" not in text:
    text = text.replace("import json\n", "import json\nfrom collections import Counter\n", 1)

# Ensure self.node_ids is available.
if "self.node_ids =" not in text:
    text = text.replace(
        '        self.gm = [graph["m"] for graph in self.graphs]\n',
        '        self.gm = [graph["m"] for graph in self.graphs]\n'
        '        self.node_ids = [graph.get("node_ids", []) for graph in self.graphs]\n',
        1,
    )

if "def _compute_entity_unit_ged" not in text:
    marker = "    def _pair_labeled_adjacency(self, batch, batch_idx):\n"
    helper = r'''    def _node_token(self, graph_id, local_idx):
        """Return raw entity URI for a local node index; fallback keeps it graph-local."""
        local_idx = int(local_idx)
        ids = self.node_ids[graph_id] if hasattr(self, "node_ids") else []
        if 0 <= local_idx < len(ids) and ids[local_idx] is not None:
            return str(ids[local_idx])
        return f"__graph_{graph_id}_node_{local_idx}"

    def _edge_predicate(self, graph_id, edge_pos):
        graph = self.graphs[graph_id]
        edge_ids = graph.get("edge_ids", [])
        if edge_pos < len(edge_ids):
            return str(edge_ids[edge_pos])
        return f"__edge_label_{edge_pos}"

    def _target_edge_counter(self, graph_id):
        """KG directed labeled edges of the target graph, represented with raw entity URIs."""
        graph = self.graphs[graph_id]
        counter = Counter()
        for epos, uv in enumerate(graph.get("graph", [])):
            if len(uv) < 2:
                continue
            u, v = int(uv[0]), int(uv[1])
            p = self._edge_predicate(graph_id, epos)
            counter[(self._node_token(graph_id, u), p, self._node_token(graph_id, v))] += 1
        return counter

    def _mapped_source_edge_counter(self, source_id, target_id, mapped_cols):
        """Map source graph edges into target entity space using the MODEL-predicted matching."""
        graph = self.graphs[source_id]
        counter = Counter()
        bad_edges = 0
        for epos, uv in enumerate(graph.get("graph", [])):
            if len(uv) < 2:
                continue
            u, v = int(uv[0]), int(uv[1])
            if u >= len(mapped_cols) or v >= len(mapped_cols):
                bad_edges += 1
                continue
            mu = int(mapped_cols[u])
            mv = int(mapped_cols[v])
            p = self._edge_predicate(source_id, epos)
            counter[(self._node_token(target_id, mu), p, self._node_token(target_id, mv))] += 1
        return counter, bad_edges

    def _compute_entity_unit_ged(self, id_1, id_2, mapped_cols):
        """Unit KG-GED proxy under the model-predicted node mapping.

        This does NOT use oracle matching. It only changes how the already predicted
        matching is scored: directed KG triples are compared in raw entity/predicate space.
        """
        n1 = int(self.gn[id_1])
        n2 = int(self.gn[id_2])

        mapped_source_edges, bad_edges = self._mapped_source_edge_counter(id_1, id_2, mapped_cols)
        target_edges = self._target_edge_counter(id_2)

        common = sum((mapped_source_edges & target_edges).values())
        edge_cost = max(sum(mapped_source_edges.values()), sum(target_edges.values())) - common
        edge_cost += bad_edges

        node_cost = n2 - n1
        return float(node_cost + edge_cost)

'''
    if marker not in text:
        raise SystemExit("Cannot find _pair_labeled_adjacency insertion point in trainer.py")
    text = text.replace(marker, helper + marker)

# Patch batched training GED computation.
batch_marker = '''            mapped_cols = torch.argmax(pair_solution.float(), dim=1).tolist()
            unmatched_cols = [col for col in range(n2) if col not in mapped_cols]
'''
batch_insert = '''            mapped_cols = torch.argmax(pair_solution.float(), dim=1).tolist()
            if self.args.cost_mode == "entity_unit":
                id_1 = int(batch.i_j[batch_idx, 0].item())
                id_2 = int(batch.i_j[batch_idx, 1].item())
                val = self._compute_entity_unit_ged(id_1, id_2, mapped_cols)
                results.append(torch.tensor(val, dtype=torch.float, device=batch.edge_index.device))
                continue
            unmatched_cols = [col for col in range(n2) if col not in mapped_cols]
'''
if 'if self.args.cost_mode == "entity_unit":\n                id_1 = int(batch.i_j[batch_idx, 0].item())' not in text:
    if batch_marker not in text:
        raise SystemExit("Cannot patch _compute_batch_ged mapped_cols block.")
    text = text.replace(batch_marker, batch_insert, 1)

# Patch single evaluation GED computation.
single_marker = '''        mapped_cols = torch.argmax(solution.float(), dim=1).tolist()
        unmatched_cols = [col for col in range(n2) if col not in mapped_cols]
'''
single_insert = '''        mapped_cols = torch.argmax(solution.float(), dim=1).tolist()
        if self.args.cost_mode == "entity_unit":
            id_1 = int(data.i_j[0, 0].item())
            id_2 = int(data.i_j[0, 1].item())
            return float(self._compute_entity_unit_ged(id_1, id_2, mapped_cols))
        unmatched_cols = [col for col in range(n2) if col not in mapped_cols]
'''
if 'if self.args.cost_mode == "entity_unit":\n            id_1 = int(data.i_j[0, 0].item())' not in text:
    if single_marker not in text:
        raise SystemExit("Cannot patch _compute_single_ged_from_dense_solution mapped_cols block.")
    text = text.replace(single_marker, single_insert, 1)

TRAINER.write_text(text, encoding="utf-8")
print(f"[patched] {TRAINER}")

print("\nDone. Recommended checks:")
print("python -m py_compile src/SEABED/utils.py src/SEABED/trainer.py src/SEABED/param_parser.py")
print("python src/SEABED/main.py --dataset YAGO --dataset-root /root/autodl-tmp/SEABED-main/data/YAGO/ --cost-mode entity_unit --init-mapping random --eval-mapping model --debug-match-metrics --model-train 1 --model-epoch-start 0 --model-epoch-end 1")
