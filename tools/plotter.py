import os
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def plot_auc(results: List[dict], out_path: str) -> None:
    if not results:
        return
    labels = [os.path.basename(r.get("data_path", "")) for r in results]
    aucs = [r.get("auc_full", 0.0) for r in results]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(labels)), aucs)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("AUC (full pairs)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
