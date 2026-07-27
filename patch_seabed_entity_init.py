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
        raise SystemExit(f"Cannot find {p}. Run this from GEDRanker-main project root.")
    bak = p.with_suffix(p.suffix + f".bak.entity_init.{stamp}")
    shutil.copy2(p, bak)
    print(f"[backup] {bak}")

# ---------------- utils.py: infer and store local index -> raw entity URI ----------------
text = UTILS.read_text(encoding="utf-8")
if "def _infer_node_ids_from_kg" not in text:
    marker = "def _extract_graph(graph_payload, gid, use_raw_features):\n"
    helper = r'''
def _infer_node_ids_from_kg(graph_payload):
    """Infer local node index -> raw entity URI from aligned KG triples and edge_indices.

    SEABED JSON stores both raw triples (KG) and local indexed edges (edge_indices).
    For each aligned row i: KG[i] = (h, p, t), edge_indices[i] = (u, v),
    therefore local node u corresponds to raw entity h and v corresponds to raw entity t.
    """
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
    text = text.replace(marker, helper + marker)

if '"node_ids": node_ids,' not in text:
    old = '''def _extract_graph(graph_payload, gid, use_raw_features):
    edge_pairs = graph_payload.get("edge_indices", [])
    edge_features = graph_payload.get("edge_features", [])
    node_features = graph_payload.get("node_features", [])
    if use_raw_features and node_features:
        features = [np.asarray(node["embedding"], dtype=float).reshape(-1).tolist() for node in node_features]
    else:
        features = [[2.0] for _ in range(len(node_features))]

    return {
        "gid": gid,
        "n": len(node_features),
        "m": len(edge_pairs),
        "graph": edge_pairs,
        "edge_ids": [edge["id"] for edge in edge_features],
        "features": features,
    }
'''
    new = '''def _extract_graph(graph_payload, gid, use_raw_features):
    edge_pairs = graph_payload.get("edge_indices", [])
    edge_features = graph_payload.get("edge_features", [])
    node_features = graph_payload.get("node_features", [])
    node_ids, node_id_conflicts = _infer_node_ids_from_kg(graph_payload)
    if node_id_conflicts:
        raise ValueError(f"Conflicting local node to raw entity mapping in graph gid={gid}: {node_id_conflicts[:3]}")
    if use_raw_features and node_features:
        features = [np.asarray(node["embedding"], dtype=float).reshape(-1).tolist() for node in node_features]
    else:
        features = [[2.0] for _ in range(len(node_features))]

    return {
        "gid": gid,
        "n": len(node_features),
        "m": len(edge_pairs),
        "graph": edge_pairs,
        "edge_ids": [edge["id"] for edge in edge_features],
        "features": features,
        "node_ids": node_ids,
    }
'''
    if old not in text:
        raise SystemExit("utils.py pattern not found; inspect _extract_graph manually.")
    text = text.replace(old, new)
UTILS.write_text(text, encoding="utf-8")
print(f"[patched] {UTILS}")

# ---------------- param_parser.py: add --init-mapping ----------------
text = PARAM.read_text(encoding="utf-8")
if "--init-mapping" not in text:
    insert_after = re.search(r"parser\.add_argument\(\s*\"--cost-mode\"[\s\S]*?\n\s*\)\n", text)
    arg = '''    parser.add_argument(
        "--init-mapping",
        choices=["random", "entity", "feature"],
        default="random",
        help="Initial best mapping used by GEDRanker roll-out. random keeps the original unsupervised setting; entity uses raw URI identity for debugging/warm start; feature greedily matches node embeddings without raw URI labels.",
    )
'''
    if insert_after:
        pos = insert_after.end()
        text = text[:pos] + arg + text[pos:]
    else:
        marker = "    parser.add_argument(\"--topk-approach\""
        idx = text.find(marker)
        if idx < 0:
            raise SystemExit("Cannot find insertion point in param_parser.py")
        text = text[:idx] + arg + text[idx:]
PARAM.write_text(text, encoding="utf-8")
print(f"[patched] {PARAM}")

# ---------------- trainer.py: store node_ids and initialize best_mapping_label ----------------
text = TRAINER.read_text(encoding="utf-8")
if "self.node_ids =" not in text:
    text = text.replace(
        '        self.gm = [graph["m"] for graph in self.graphs]\n',
        '        self.gm = [graph["m"] for graph in self.graphs]\n        self.node_ids = [graph.get("node_ids", []) for graph in self.graphs]\n'
    )

if "def _init_mapping_label" not in text:
    marker = "    def _pair_labeled_adjacency(self, batch, batch_idx):\n"
    method = r'''    def _init_mapping_label(self, id_1, id_2, n1, n2):
        """Return an n1*n2 binary mapping label for initializing GEDRanker's best mapping.

        random: keep original random initialization.
        entity: match local nodes with the same raw entity URI, inferred from KG + edge_indices.
        feature: greedy cosine matching using node features only; no raw URI labels.
        """
        mode = getattr(self.args, "init_mapping", "random")
        if mode == "random":
            return None

        label = torch.zeros((n1 * n2, 1), dtype=torch.float)
        used_cols = set()

        if mode == "entity":
            ids1 = self.node_ids[id_1]
            ids2 = self.node_ids[id_2]
            ent_to_col = {}
            duplicate = set()
            for j, ent in enumerate(ids2):
                if ent is None:
                    continue
                if ent in ent_to_col:
                    duplicate.add(ent)
                else:
                    ent_to_col[ent] = j
            for i, ent in enumerate(ids1):
                j = ent_to_col.get(ent)
                if ent is not None and ent not in duplicate and j is not None and j not in used_cols:
                    label[i * n2 + j, 0] = 1.0
                    used_cols.add(j)

        elif mode == "feature":
            x1 = self.features[id_1].float()
            x2 = self.features[id_2].float()
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
            sim = x1 @ x2.t()
            # greedy maximum-weight one-to-one matching, enough for initialization/debug
            for _ in range(min(n1, n2)):
                flat_idx = torch.argmax(sim).item()
                i = flat_idx // n2
                j = flat_idx % n2
                if sim[i, j].item() == float("-inf"):
                    break
                label[i * n2 + j, 0] = 1.0
                used_cols.add(j)
                sim[i, :] = float("-inf")
                sim[:, j] = float("-inf")
        else:
            raise ValueError(f"Unknown init_mapping mode: {mode}")

        # Make it a full row-wise assignment so roll_out/mapping_loss always have one target per row.
        for i in range(n1):
            start = i * n2
            end = start + n2
            if label[start:end].sum().item() == 0:
                for j in range(n2):
                    if j not in used_cols:
                        label[start + j, 0] = 1.0
                        used_cols.add(j)
                        break
        return label

'''
    text = text.replace(marker, method + marker)

old = '        new_data.best_mapping_label = torch.rand_like(new_data.edge_attr_mapping)\n        return new_data\n'
new = '''        init_mapping_label = self._init_mapping_label(id_1, id_2, n1, n2)
        if init_mapping_label is None:
            new_data.best_mapping_label = torch.rand_like(new_data.edge_attr_mapping)
        else:
            new_data.best_mapping_label = init_mapping_label
        return new_data
'''
if old in text and "init_mapping_label = self._init_mapping_label" not in text:
    text = text.replace(old, new)
elif "init_mapping_label = self._init_mapping_label" not in text:
    raise SystemExit("trainer.py best_mapping_label pattern not found; inspect pack_graph_pair manually.")

TRAINER.write_text(text, encoding="utf-8")
print(f"[patched] {TRAINER}")

print("\nDone. Recommended checks:")
print("python -m py_compile src/SEABED/utils.py src/SEABED/trainer.py src/SEABED/param_parser.py")
print("python main.py --dataset YAGO --dataset-root /root/autodl-tmp/SEABED-main/data/YAGO/ --cost-mode unit --init-mapping entity --model-train 1 --model-epoch-start 0 --model-epoch-end 1")
print("python main.py --dataset YAGO --dataset-root /root/autodl-tmp/SEABED-main/data/YAGO/ --cost-mode unit --init-mapping feature --model-train 1 --model-epoch-start 0 --model-epoch-end 1")
