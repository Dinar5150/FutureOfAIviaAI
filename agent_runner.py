"""
Single entrypoint to run automated experiments with the M9 GCN model.

Example:
    python agent_runner.py --task full_run --dataset-limit 2
"""

from __future__ import annotations

import argparse
import os
from typing import List

from agents.impl_agent import ImplAgent
from agents.report_agent import ReportAgent
from agents.spec_agent import SpecAgent
from agents.experiment_agent import ExperimentAgent


def run_full(task_args) -> None:
    spec_agent = SpecAgent(root=".")
    spec = spec_agent.build_spec(limit=task_args.dataset_limit)
    impl = ImplAgent(spec)
    cfg = impl.materialize()

    datasets: List[str] = cfg.pop("datasets", [])
    if not datasets:
        print("No datasets found matching SemanticGraph_*.pkl")
        return

    exp_agent = ExperimentAgent(out_dir="results")
    results = exp_agent.run(datasets, cfg)

    report_agent = ReportAgent(out_dir="report")
    report_path = report_agent.write(results, model_name="M9_GCN")
    print(f"Report written to {report_path}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Agent runner for GCN experiments.")
    parser.add_argument(
        "--task",
        choices=["full_run"],
        default="full_run",
        help="Which pipeline to execute.",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=2,
        help="Max number of SemanticGraph_*.pkl files to process (-1 for all).",
    )
    args = parser.parse_args()

    if args.task == "full_run":
        # Translate -1 to None for unlimited processing.
        if args.dataset_limit is not None and args.dataset_limit < 0:
            args.dataset_limit = None
        run_full(args)
    else:
        raise ValueError(f"Unknown task {args.task}")


if __name__ == "__main__":
    cli()
