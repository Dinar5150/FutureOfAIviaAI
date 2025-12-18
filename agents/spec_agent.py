from dataclasses import dataclass
from typing import List

from tools.dataset_inspector import find_datasets


@dataclass
class Spec:
    datasets: List[str]
    max_pairs: int = 200_000
    epochs: int = 20
    hidden_dim: int = 64
    batch_size: int = 2048
    lr: float = 3e-4
    seed: int = 42


class SpecAgent:
    """Plans which datasets to run and with which hyperparameters."""

    def __init__(self, root: str = ".") -> None:
        self.root = root

    def build_spec(self, limit: int | None = 3) -> Spec:
        datasets = find_datasets(self.root)
        if limit is not None:
            datasets = datasets[:limit]
        return Spec(datasets=datasets)
