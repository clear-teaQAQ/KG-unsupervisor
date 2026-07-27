#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter, defaultdict


def load_payload(path: Path):
    obj = json.loads(path.read_text(encoding='utf-8'))
    return obj.get('0', obj)


def node_ids(g):
    return [str(x.get('id')) for x in g.get('node_features', [])]


def edges(g):
    return [(int(uv[0]), int(uv[1]), str(feat.get('id'))) for uv, feat in zip(g.get('edge_indices', []), g.get('edge_features', []))]


def kg_triples(g):
    if 'KG' in g:
        return [tuple(map(str, t)) for t in g['KG']]
    ids = node_ids(g)
    out = []
    for u, v, p in edges(g):
        if 0 <= u < len(ids) and 0 <= v < len(ids):
            out.append((ids[u], p, ids[v]))
    return out


def bucket(g, mapping=None, undirected=True):
    b = defaultdict(Counter)
    for u, v, p in edges(g):
        if mapping is not None:
            if u >= len(mapping) or v >= len(mapping):
                # invalid endpoint, count under impossible key
                key = (10**9 + u, 10**9 + v)
                b[key][p] += 1
                continue
            u, v = mapping[u], mapping[v]
        key = tuple(sorted((u, v))) if undirected else (u, v)
        b[key][p] += 1
    return b


def edge_cost(b1, b2):
    cost = 0
    for k in set(b1) | set(b2):
        c1, c2 = b1.get(k, Counter()), b2.get(k, Counter())
        common = sum((c1 & c2).values())
        cost += max(sum(c1.values()), sum(c2.values())) - common
    return cost


def ged_like(g1, g2, mapping, undirected=True, containment=False):
    n1, n2 = len(node_ids(g1)), len(node_ids(g2))
    m1, m2 = len(edges(g1)), len(edges(g2))
    if containment:
        # matches trainer.py containment mode: node additions + edge-count diff + overlap mismatches
        return (n2 - n1) + abs(m2 - m1) + edge_cost(bucket(g1, mapping, undirected), bucket(g2, None, undirected)) - abs(m2 - m1)
    return (n2 - n1) + edge_cost(bucket(g1, mapping, undirected), bucket(g2, None, undirected))


def id_mapping(g1, g2):
    ids2 = {x: i for i, x in enumerate(node_ids(g2))}
    out, used = [], set()
    for x in node_ids(g1):
        if x not in ids2 or ids2[x] in used:
            return None
        out.append(ids2[x]); used.add(ids2[x])
    return out


def prefix_mapping(g1, g2):
    n1, n2 = len(node_ids(g1)), len(node_ids(g2))
    if n1 > n2:
        return None
    return list(range(n1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', required=True, type=Path)
    ap.add_argument('--split', default='test', choices=['train','val','test'])
    ap.add_argument('--directed', action='store_true')
    ap.add_argument('--max-examples', type=int, default=10)
    args = ap.parse_args()
    root = args.dataset_root
    graphs = {p.name: load_payload(p) for p in sorted((root / args.split).glob('*.json'))}
    pairs = json.loads((root / f'{args.split}_GEDINFO.json').read_text(encoding='utf-8'))['pairs_info']
    undirected = not args.directed

    total = 0
    size_ok = 0
    prefix_ok = 0
    id_ok = 0
    kg_subset_ok = 0
    kg_size_delta_ok = 0
    examples = []

    for entry in pairs:
        f1, f2, gt_raw = entry[:3]
        if f1 not in graphs or f2 not in graphs:
            continue
        g1, g2 = graphs[f1], graphs[f2]
        if len(node_ids(g1)) > len(node_ids(g2)):
            g1, g2 = g2, g1
        gt = int(float(gt_raw))
        n1, n2 = len(node_ids(g1)), len(node_ids(g2))
        m1, m2 = len(edges(g1)), len(edges(g2))
        size_delta = (n2 - n1) + (m2 - m1)
        pm = prefix_mapping(g1, g2)
        im = id_mapping(g1, g2)
        prefix_cost = None if pm is None else ged_like(g1, g2, pm, undirected=undirected)
        id_cost = None if im is None else ged_like(g1, g2, im, undirected=undirected)
        t1, t2 = Counter(kg_triples(g1)), Counter(kg_triples(g2))
        common_triples = sum((t1 & t2).values())
        subset = common_triples == sum(t1.values())
        kg_delta = sum(t2.values()) - sum(t1.values())

        total += 1
        size_ok += int(size_delta == gt)
        prefix_ok += int(prefix_cost == gt)
        id_ok += int(id_cost == gt)
        kg_subset_ok += int(subset)
        kg_size_delta_ok += int((n2 - n1) + kg_delta == gt)
        if len(examples) < args.max_examples and not (size_delta == gt and prefix_cost == gt):
            examples.append({
                'pair': (f1, f2), 'gt': gt, 'n/m': (n1,n2,m1,m2), 'size_delta': size_delta,
                'prefix_cost': prefix_cost, 'id_cost': id_cost,
                'kg_common/base': f'{common_triples}/{sum(t1.values())}',
                'kg_subset': subset, 'kg_delta_nodes_edges': (n2-n1, kg_delta)
            })

    def show(name, val):
        print(f'{name}: {val}/{total} = {val/total if total else 0:.3f}')
    print(f'split={args.split}, total_pairs={total}, directed={args.directed}')
    show('gt == (n2-n1)+(m2-m1)', size_ok)
    show('prefix_mapping_cost == gt', prefix_ok)
    show('id_mapping_cost == gt', id_ok)
    show('KG triples of G1 subset of G2', kg_subset_ok)
    show('gt == node_delta + KG_triple_delta', kg_size_delta_ok)
    print('examples where size_delta/prefix not both ok:')
    for ex in examples:
        print(ex)

if __name__ == '__main__':
    main()
