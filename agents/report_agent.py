import os
from datetime import datetime
from typing import List


class ReportAgent:
    """Generates a lightweight Markdown report summarizing experiments."""

    def __init__(self, out_dir: str = "report") -> None:
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def write(self, results: List[dict], model_name: str = "M9_GCN") -> str:
        report_path = os.path.join(self.out_dir, "report.md")
        ts = datetime.utcnow().isoformat()
        lines = [f"# Automated Report ({model_name})", f"_Generated: {ts} UTC_", ""]
        if not results:
            lines.append("No results to report.")
        else:
            lines.append("## Results")
            for r in results:
                name = os.path.basename(r.get("data_path", ""))
                auc_val = r.get("auc_val", 0.0)
                auc_full = r.get("auc_full", 0.0)
                lines.append(f"- **{name}**: val AUC={auc_val:.4f}, full AUC={auc_full:.4f}")

            lines.append("")
            lines.append("## Method (summary)")
            lines.append(
                "- 2-layer GCN on degree features; dot-product decoder; BCE loss on provided pairs."
            )
            lines.append(
                "- Train/val split on provided unconnected pairs; edges up to `year_start` build the graph."
            )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return report_path
