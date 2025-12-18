import os
from typing import Dict, List

from all_models.M9_PyG_GCN.model import run_link_prediction


def run_m9_experiment(dataset_paths: List[str], config: Dict | None = None) -> List[Dict]:
    results = []
    for path in dataset_paths:
        print(f"[M9] Running dataset: {os.path.basename(path)}")
        res = run_link_prediction(path, config)
        results.append(res)
    return results
