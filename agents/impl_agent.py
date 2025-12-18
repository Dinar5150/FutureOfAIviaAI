from dataclasses import asdict
from typing import Dict

from agents.spec_agent import Spec


class ImplAgent:
    """Turns a Spec into a concrete config dict for experiment runners."""

    def __init__(self, spec: Spec) -> None:
        self.spec = spec

    def materialize(self) -> Dict:
        cfg = {
            "epochs": self.spec.epochs,
            "batch_size": self.spec.batch_size,
            "hidden_dim": self.spec.hidden_dim,
            "max_pairs": self.spec.max_pairs,
            "lr": self.spec.lr,
            "seed": self.spec.seed,
        }
        cfg["datasets"] = self.spec.datasets
        return cfg
