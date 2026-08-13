import sys
from pathlib import Path

import torch
from types import SimpleNamespace
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v17_models import CrossGraphSinkhornMatcher


def test_output_shape_and_gradient():
    args = SimpleNamespace(hidden_dim=[16, 8])
    model = CrossGraphSinkhornMatcher(args, 4, 3)
    data = Data(
        x=torch.randn(5, 4),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 1], [1, 0, 3, 2, 4, 4]]),
        edge_attr=torch.randn(6, 3),
        batch=torch.zeros(5, dtype=torch.long),
        x_indicator=torch.tensor([[0.], [0.], [1.], [1.], [1.]]),
        n=torch.tensor([[2, 3]]),
    )
    output = model(data)
    assert output.shape == (6, 1)
    output.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


if __name__ == "__main__":
    test_output_shape_and_gradient()
    print("V17 tests passed")
