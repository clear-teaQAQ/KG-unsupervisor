import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Data

from relation_models import RelationAwareDiffMatch
from relation_trainer import normalize_relation_embedding, relation_mode_vectors


class RelationDataTest(unittest.TestCase):
    def test_nested_swdf_embedding_is_flattened(self):
        vector = normalize_relation_embedding([[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(vector, np.array([1.0, 2.0, 3.0]))

    def test_constant_and_shuffled_controls_preserve_shape(self):
        vectors = np.arange(12, dtype=float).reshape(4, 3)
        constant = relation_mode_vectors(vectors, "constant", graph_index=7)
        shuffled_1 = relation_mode_vectors(vectors, "shuffled", graph_index=7)
        shuffled_2 = relation_mode_vectors(vectors, "shuffled", graph_index=7)
        self.assertEqual(constant.shape, vectors.shape)
        self.assertTrue(np.all(constant == 0))
        self.assertTrue(np.array_equal(shuffled_1, shuffled_2))
        self.assertFalse(np.array_equal(shuffled_1, vectors))


class RelationModelTest(unittest.TestCase):
    def test_generator_consumes_relation_attributes(self):
        torch.manual_seed(0)
        args = SimpleNamespace(hidden_dim=[8, 8])
        model = RelationAwareDiffMatch(
            args,
            number_of_labels=4,
            relation_dim=3,
        ).eval()
        data = Data()
        data.x = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        data.edge_index = torch.tensor(
            [
                [0, 1, 0, 1, 2, 3, 2, 3],
                [1, 0, 0, 1, 3, 2, 2, 3],
            ],
            dtype=torch.long,
        )
        data.edge_index_mapping = torch.tensor(
            [[0, 0, 1, 1], [2, 3, 2, 3]],
            dtype=torch.long,
        )
        data.x_indicator = torch.tensor([[0], [0], [1], [1]])
        data.batch = torch.zeros(4, dtype=torch.long)
        data.edge_attr = torch.zeros((8, 3))
        noise = torch.zeros((4, 1))
        timestep = torch.tensor([1.0])

        with torch.no_grad():
            no_relation = model(data, noise, timestep)
            data.edge_attr[:2, 0] = 3.0
            data.edge_attr[4:6, 1] = -2.0
            with_relation = model(data, noise, timestep)

        self.assertEqual(no_relation.shape, (4, 1))
        self.assertFalse(torch.allclose(no_relation, with_relation))


if __name__ == "__main__":
    unittest.main()
