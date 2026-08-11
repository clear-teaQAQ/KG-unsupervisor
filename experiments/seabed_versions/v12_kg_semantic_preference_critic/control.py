"""Control helpers for the isolated V12 experiment line."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class V12Control:
    control_mode: str = "full"
    relation_mode: str = "raw"
    enable_semantic_critic: bool = True
    use_teacher_consistency: bool = True
    adaptive_explore: bool = True
    sparse_matching: bool = True
    semantic_weight: float = 0.15
    keep_loss_weight: float = 0.10
    explore_weight: float = 0.05
    sparse_top_r: int = 32
    sparse_random_epsilon: float = 0.05
    semantic_hidden_dim: tuple[int, ...] = (64, 32)
    feature_dim: int = 5


def resolve_control(args) -> V12Control:
    mode = getattr(args, "control_mode", "full")
    relation_mode = getattr(args, "relation_mode", "raw")
    semantic_hidden_dim = tuple(getattr(args, "semantic_hidden_dim", (64, 32)))
    keep_loss_weight = float(getattr(args, "keep_loss_weight", 0.10))
    explore_weight = float(getattr(args, "explore_weight", 0.05))
    semantic_weight = float(getattr(args, "semantic_weight", 0.15))
    sparse_matching = bool(int(getattr(args, "sparse_matching", 1)))
    adaptive_explore = bool(int(getattr(args, "adaptive_explore", 1)))

    if mode == "no_critic":
        semantic_weight = 0.0
    elif mode == "no_keep":
        keep_loss_weight = 0.0
    elif mode == "no_adapt":
        adaptive_explore = False
    elif mode == "no_sparse":
        sparse_matching = False
    elif mode in {"raw", "constant", "shuffled"}:
        relation_mode = mode
        mode = "full"

    return V12Control(
        control_mode=mode,
        relation_mode=relation_mode,
        enable_semantic_critic=semantic_weight > 0,
        use_teacher_consistency=keep_loss_weight > 0,
        adaptive_explore=adaptive_explore,
        sparse_matching=sparse_matching,
        semantic_weight=semantic_weight,
        keep_loss_weight=keep_loss_weight,
        explore_weight=explore_weight,
        sparse_top_r=int(getattr(args, "sparse_top_r", 32)),
        sparse_random_epsilon=float(getattr(args, "sparse_random_epsilon", 0.05)),
        semantic_hidden_dim=semantic_hidden_dim,
    )


def control_as_dict(control: V12Control) -> dict:
    return asdict(control)
