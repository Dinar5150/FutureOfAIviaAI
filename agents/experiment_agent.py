import os
import time
from typing import Dict, List

from tools.experiment_runner import run_m9_experiment
from tools.plotter import plot_auc


class ExperimentAgent:
    """Runs experiments and stores results/plots."""

    def __init__(self, out_dir: str = "results") -> None:
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def run(self, datasets: List[str], config: Dict) -> List[Dict]:
        t0 = time.time()
        results = run_m9_experiment(datasets, config)
        dur = time.time() - t0
        print(f"[M9] Completed {len(results)} runs in {dur:.1f}s")
        self._persist(results)
        return results

    def _persist(self, results: List[Dict]) -> None:
        if not results:
            return
        csv_path = os.path.join(self.out_dir, "auc_summary.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            headers = [
                "data_path",
                "auc_val",
                "auc_full",
                "year_start",
                "years_delta",
                "vertex_degree_cutoff",
                "min_edges",
            ]
            f.write(",".join(headers) + "\n")
            for r in results:
                row = [str(r.get(h, "")) for h in headers]
                f.write(",".join(row) + "\n")
        plot_path = os.path.join(self.out_dir, "auc_bar.png")
        plot_auc(results, plot_path)
