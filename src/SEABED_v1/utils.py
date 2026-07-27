"""Utilities for loading SEABED-format graph pairs."""

from glob import glob
from os.path import basename
import json
from pathlib import Path
import numpy as np

from texttable import Texttable


def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    table = Texttable()
    rows = [["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys]
    table.add_rows(rows)
    print(table.draw())


def sorted_nicely(items):
    def try_int(text):
        try:
            return int(text)
        except ValueError:
            return text

    import re

    def alphanum_key(text):
        return [try_int(chunk) for chunk in re.split(r"([0-9]+)", text)]

    return sorted(items, key=alphanum_key)


def get_file_paths(directory, file_format="json"):
    directory = directory.rstrip("/\\")
    return sorted_nicely(glob(directory + "/*." + file_format))


def _load_graph_payload(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "0" in payload:
        return payload["0"]
    return payload


def _extract_graph(graph_payload, gid, use_raw_features):
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


def load_split_graphs(split_dir, gid_start, use_raw_features):
    graphs = []
    name_to_gid = {}
    current_gid = gid_start
    for file_path in get_file_paths(str(split_dir), "json"):
        payload = _load_graph_payload(file_path)
        graph = _extract_graph(payload, current_gid, use_raw_features)
        graphs.append(graph)
        name_to_gid[basename(file_path)] = current_gid
        current_gid += 1
    return graphs, name_to_gid, current_gid


def load_pair_file(file_path, name_to_gid):
    with open(file_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    pairs = []
    for entry in payload["pairs_info"]:
        file_1, file_2 = entry[:2]
        if len(entry) >= 4:
            ged = entry[3]
        else:
            ged = entry[2]
        gid_1 = name_to_gid[file_1]
        gid_2 = name_to_gid[file_2]
        pairs.append((gid_1, gid_2, float(ged)))
    return pairs


def load_dataset(dataset_root, use_raw_features=True):
    dataset_root = Path(dataset_root)
    all_graphs = []
    split_ranges = {}
    split_name_to_gid = {}
    global_name_to_gid = {}
    current_gid = 0

    for split in ["train", "val", "test"]:
        split_dir = dataset_root / split
        graphs, name_to_gid, current_gid = load_split_graphs(split_dir, current_gid, use_raw_features)
        split_ranges[split] = (len(all_graphs), len(all_graphs) + len(graphs))
        split_name_to_gid[split] = name_to_gid
        for file_name, gid in name_to_gid.items():
            if file_name in global_name_to_gid and global_name_to_gid[file_name] != gid:
                raise ValueError(f"Duplicate graph filename across splits: {file_name}")
            global_name_to_gid[file_name] = gid
        all_graphs.extend(graphs)

    pairs = {}
    for split in ["train", "val", "test"]:
        pair_file = dataset_root / f"{split}_GEDINFO.json"
        pairs[split] = load_pair_file(pair_file, global_name_to_gid)

    return all_graphs, pairs, split_ranges
