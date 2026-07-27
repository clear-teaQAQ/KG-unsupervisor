#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter, defaultdict


def load_payload(path: Path):
    obj = json.loads(path.read_text(encoding='utf-8'))
    return obj.get('0', obj)


def get_node_count(g):
    return len(g.get('node_features', []))


def infer_local_entity_map(g):
    """Infer local node index -> raw entity URI from aligned KG triples and edge_indices.
    Assumption: KG[i] corresponds to edge_indices[i] and edge_features[i].
    """
    kg = g.get('KG', [])
    edge_indices = g.get('edge_indices', [])
    edge_features = g.get('edge_features', [])
    idx2ent = {}
    conflicts = []
    pred_mismatch = 0
    bad_edge_idx = 0
    for i, (tri, uv) in enumerate(zip(kg, edge_indices)):
        if len(tri) < 3 or len(uv) < 2:
            continue
        h, p, t = map(str, tri[:3])
        u, v = int(uv[0]), int(uv[1])
        for idx, ent in [(u, h), (v, t)]:
            old = idx2ent.get(idx)
            if old is None:
                idx2ent[idx] = ent
            elif old != ent:
                conflicts.append((idx, old, ent, i))
        if i < len(edge_features):
            pf = str(edge_features[i].get('id')) if isinstance(edge_features[i], dict) else str(edge_features[i])
            if pf != p:
                pred_mismatch += 1
        else:
            bad_edge_idx += 1
    ent2idx = defaultdict(list)
    for idx, ent in idx2ent.items():
        ent2idx[ent].append(idx)
    dup_ents = {e: xs for e, xs in ent2idx.items() if len(xs) > 1}
    return idx2ent, dict(ent2idx), conflicts, dup_ents, pred_mismatch, bad_edge_idx


def triples_from_indices_with_inferred_map(g, idx2ent):
    out = Counter()
    missing_node = 0
    edge_indices = g.get('edge_indices', [])
    edge_features = g.get('edge_features', [])
    for i, uv in enumerate(edge_indices):
        if len(uv) < 2:
            continue
        u, v = int(uv[0]), int(uv[1])
        if u not in idx2ent or v not in idx2ent:
            missing_node += 1
            continue
        if i < len(edge_features) and isinstance(edge_features[i], dict):
            p = str(edge_features[i].get('id'))
        elif i < len(g.get('KG', [])):
            p = str(g['KG'][i][1])
        else:
            p = ''
        out[(idx2ent[u], p, idx2ent[v])] += 1
    return out, missing_node


def kg_triples(g):
    return Counter(tuple(map(str, tri[:3])) for tri in g.get('KG', []))


def subset(a, b):
    return sum((a & b).values()) == sum(a.values())


def build_oracle_mapping(idx2ent1, ent2idx2):
    mapping = {}
    missing = []
    ambiguous = []
    for i, ent in idx2ent1.items():
        js = ent2idx2.get(ent, [])
        if len(js) == 1:
            mapping[i] = js[0]
        elif len(js) == 0:
            missing.append((i, ent))
        else:
            mapping[i] = js[0]
            ambiguous.append((i, ent, js))
    return mapping, missing, ambiguous


def mapped_edge_cost(g1, g2, mapping):
    # Only edge insertion/deletion/relabel as exact directed labeled edges after mapping G1 local nodes into G2 local nodes.
    def p_at(g, i):
        ef = g.get('edge_features', [])
        if i < len(ef) and isinstance(ef[i], dict):
            return str(ef[i].get('id'))
        return str(g.get('KG', [])[i][1]) if i < len(g.get('KG', [])) else ''

    mapped1 = Counter()
    bad_map_edges = 0
    for i, uv in enumerate(g1.get('edge_indices', [])):
        u, v = int(uv[0]), int(uv[1])
        if u not in mapping or v not in mapping:
            bad_map_edges += 1
            continue
        mapped1[(mapping[u], p_at(g1, i), mapping[v])] += 1
    e2 = Counter()
    for i, uv in enumerate(g2.get('edge_indices', [])):
        u, v = int(uv[0]), int(uv[1])
        e2[(u, p_at(g2, i), v)] += 1
    common = sum((mapped1 & e2).values())
    # unit insert/delete/relabel proxy; for subset case this should be len(e2)-len(mapped1)
    edge_cost = max(sum(mapped1.values()), sum(e2.values())) - common
    return edge_cost, bad_map_edges, mapped1, e2


def eval_pair(root, split, f1, f2, gt_raw):
    g1 = load_payload(root / split / f1)
    g2 = load_payload(root / split / f2)
    if get_node_count(g1) > get_node_count(g2):
        g1, g2 = g2, g1
        f1, f2 = f2, f1
    gt = int(float(gt_raw))
    idx2ent1, ent2idx1, conf1, dup1, pm1, bad1 = infer_local_entity_map(g1)
    idx2ent2, ent2idx2, conf2, dup2, pm2, bad2 = infer_local_entity_map(g2)
    recon1, missn1 = triples_from_indices_with_inferred_map(g1, idx2ent1)
    recon2, missn2 = triples_from_indices_with_inferred_map(g2, idx2ent2)
    kg1, kg2 = kg_triples(g1), kg_triples(g2)
    mapping, missing, ambiguous = build_oracle_mapping(idx2ent1, ent2idx2)
    edge_cost, bad_map_edges, mapped1, e2 = mapped_edge_cost(g1, g2, mapping)
    node_cost = get_node_count(g2) - get_node_count(g1)
    total_cost = node_cost + edge_cost
    return {
        'pair': (f1, f2), 'gt': gt,
        'n': (get_node_count(g1), get_node_count(g2)),
        'm': (len(g1.get('edge_indices', [])), len(g2.get('edge_indices', []))),
        'infer_conflicts': (len(conf1), len(conf2)),
        'predicate_mismatch': (pm1, pm2),
        'reconstructed_eq_KG': (recon1 == kg1, recon2 == kg2),
        'reconstructed_subset': subset(recon1, recon2),
        'oracle_mapping_coverage': (len(mapping), len(idx2ent1), len(missing), len(ambiguous)),
        'mapped_edge_subset': subset(mapped1, e2),
        'inferred_oracle_cost': total_cost,
        'inferred_oracle_cost_eq_gt': total_cost == gt,
        'node_cost': node_cost,
        'edge_cost': edge_cost,
        'examples_mapping': list(mapping.items())[:10],
        'missing_entities': missing[:5],
        'missing_mapped_edges': list((mapped1 - e2).elements())[:5],
        'extra_edges': list((e2 - mapped1).elements())[:5],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', type=Path, required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--max-pairs', type=int, default=6000)
    ap.add_argument('--show-examples', type=int, default=3)
    args = ap.parse_args()
    pairs = json.loads((args.dataset_root / f'{args.split}_GEDINFO.json').read_text(encoding='utf-8'))['pairs_info']
    total = 0
    stats = Counter()
    examples = []
    for entry in pairs[:args.max_pairs]:
        r = eval_pair(args.dataset_root, args.split, entry[0], entry[1], entry[2])
        total += 1
        for k in ['reconstructed_eq_KG','reconstructed_subset','mapped_edge_subset','inferred_oracle_cost_eq_gt']:
            val = r[k]
            if isinstance(val, tuple):
                stats[k] += int(all(val))
            else:
                stats[k] += int(bool(val))
        stats['has_no_infer_conflicts'] += int(r['infer_conflicts'] == (0,0))
        stats['has_no_predicate_mismatch'] += int(r['predicate_mismatch'] == (0,0))
        stats['full_mapping_coverage'] += int(r['oracle_mapping_coverage'][0] == r['oracle_mapping_coverage'][1] and r['oracle_mapping_coverage'][2] == 0)
        if len(examples) < args.show_examples and not r['inferred_oracle_cost_eq_gt']:
            examples.append(r)
    print(f'split={args.split}, checked_pairs={total}')
    for k in ['has_no_infer_conflicts','has_no_predicate_mismatch','reconstructed_eq_KG','reconstructed_subset','full_mapping_coverage','mapped_edge_subset','inferred_oracle_cost_eq_gt']:
        print(f'{k}: {stats[k]}/{total} = {stats[k]/total if total else 0:.3f}')
    print('\nExamples where inferred oracle cost != gt:')
    for r in examples:
        print(json.dumps(r, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
