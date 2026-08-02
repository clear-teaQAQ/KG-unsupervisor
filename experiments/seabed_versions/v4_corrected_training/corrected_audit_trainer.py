"""Apply the unchanged V3 inference/path audit to a V4-trained checkpoint."""

import json
from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
V3_DIR = CURRENT_DIR.parent / "v3_topology_feature_reindex"
if str(V3_DIR) not in sys.path:
    sys.path.insert(0, str(V3_DIR))

from reindexed_trainer import TopologyFeatureReindexTrainer


class CorrectedTrainingAuditTrainer(TopologyFeatureReindexTrainer):
    version = "v4_corrected_training"

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        result = super().score(testing_graph_set, test_k, top_k_approach)
        result["frozen_inference_source"] = (
            "v4 corrected-feature checkpoint + v1 deterministic_dense_v4/two_swap"
        )
        result["checkpoint_training"] = {
            "version": self.version,
            "feature_revision": self.feature_revision,
            "trained_from_scratch": True,
        }

        result_stem = (
            f"result_SEABED_{self.version}_{self.args.dataset}_{testing_graph_set}"
            f"_k{test_k}_{self.args.repair_mode}"
        )
        result_path = self._result_file_path(result_stem)
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print("Recorded V4 checkpoint provenance:", result_path)
        return result
