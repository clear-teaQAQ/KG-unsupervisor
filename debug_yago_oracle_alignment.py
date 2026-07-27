#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter


def load_payload(path: Path):
    obj = json.loads(path.read_text(encoding='utf-8'))
    return obj.get('0', obj)


def node_ids(g):
    return [str(x.get('id')) for x in g.get('node_features', [])]


def kg_field_triples(g):
    return Counter(tuple(map(str, t)) for t in g.get('KG', []))


def indexed_triples(g):
    ids = node_ids(g)
    edge_indices = g.get('edge_indices', [])
    edge_features = g.get('edge_features', [])
    out = []
    bad = 0
    for uv, feat in zip(edge_indices, edge_features):
        u, v = int(uv[0]), int(uv[1])
        p = str(feat.get('id'))
        if 0 <= u < len(ids) and 0 <= v < len(ids):
            out.append((ids[u], p, ids[v]))
        else:
            bad += 1
    return Counter(out), bad


def multiset_subset(a: Counter, b: Counter) -> bool:
    return sum((a & b).values()) == sum(a.values())


def edge_delta_cost(a: Counter, b: Counter) -> int:
    # If using triples as atomic labeled directed edges, minimum insert/delete/relabel on exact triples
    # for subset-expansion pairs should equal |b|-|a| when a subset b.
    common = sum((a & b).values())
    return max(sum(a.values()), sum(b.values())) - common


def one_pair(root: Path, split: str, f1: str, f2: str, gt_raw):
    g1 = load_payload(root / split / f1)
    g2 = load_payload(root / split / f2)
    if len(node_ids(g1)) > len(node_ids(g2)):
        g1, g2 = g2, g1
        f1, f2 = f2, f1
    gt = int(float(gt_raw))
    n1, n2 = len(node_ids(g1)), len(node_ids(g2))
    kg1, kg2 = kg_field_triples(g1), kg_field_triples(g2)
    ix1, bad1 = indexed_triples(g1)
    ix2, bad2 = indexed_triples(g2)
    size_delta_kg = (n2 - n1) + (sum(kg2.values()) - sum(kg1.values()))
    size_delta_ix = (n2 - n1) + (sum(ix2.values()) - sum(ix1.values()))
    cost_kg = (n2 - n1) + edge_delta_cost(kg1, kg2)
    cost_ix = (n2 - n1) + edge_delta_cost(ix1, ix2)
    return {
        'pair': (f1, f2), 'gt': gt, 'n': (n1, n2),
        'kg_edges': (sum(kg1.values()), sum(kg2.values())),
        'indexed_edges': (sum(ix1.values()), sum(ix2.values())),
        'bad_indexed_edges': (bad1, bad2),
        'kg_subset': multiset_subset(kg1, kg2),
        'indexed_subset': multiset_subset(ix1, ix2),
        'gt_eq_kg_size_delta': gt == size_delta_kg,
        'gt_eq_indexed_size_delta': gt == size_delta_ix,
        'kg_oracle_cost_eq_gt': cost_kg == gt,
        'indexed_oracle_cost_eq_gt': cost_ix == gt,
        'kg_cost': cost_kg,
        'indexed_cost': cost_ix,
        'missing_indexed_examples': list((ix1 - ix2).elements())[:10],
        'extra_indexed_examples': list((ix2 - ix1).elements())[:10],
        'missing_kg_examples': list((kg1 - kg2).elements())[:10],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', required=True, type=Path)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--max-pairs', type=int, default=6000)
    ap.add_argument('--show-examples', type=int, default=3)
    args = ap.parse_args()
    root = args.dataset_root
    pairs = json.loads((root / f'{args.split}_GEDINFO.json').read_text(encoding='utf-8'))['pairs_info']
    stats = Counter()
    examples = []
    total = 0
    for entry in pairs[:args.max_pairs]:
        r = one_pair(root, args.split, entry[0], entry[1], entry[2])
        total += 1
        for k in ['kg_subset','indexed_subset','gt_eq_kg_size_delta','gt_eq_indexed_size_delta','kg_oracle_cost_eq_gt','indexed_oracle_cost_eq_gt']:
            stats[k] += int(bool(r[k]))
        if len(examples) < args.show_examples and not r['indexed_oracle_cost_eq_gt']:
            examples.append(r)
    print(f'split={args.split}, checked_pairs={total}')
    for k in ['kg_subset','indexed_subset','gt_eq_kg_size_delta','gt_eq_indexed_size_delta','kg_oracle_cost_eq_gt','indexed_oracle_cost_eq_gt']:
        print(f'{k}: {stats[k]}/{total} = {stats[k]/total if total else 0:.3f}')
    print('\nExamples where indexed oracle cost != gt:')
    for r in examples:
        keep = {k:v for k,v in r.items() if k not in ['missing_indexed_examples','extra_indexed_examples','missing_kg_examples']}
        print(json.dumps(keep, ensure_ascii=False, indent=2))
        print('missing_indexed_examples:', r['missing_indexed_examples'])
        print('extra_indexed_examples:', r['extra_indexed_examples'])
        print('missing_kg_examples:', r['missing_kg_examples'])

if __name__ == '__main__':
    main()
