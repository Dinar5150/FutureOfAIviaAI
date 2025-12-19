# Proposal: Graph Neural Network (GCN) for Science4Cast Link Prediction (Model M9)

## Objective
Extend the `FutureOfAIviaAI` repository by implementing and evaluating a **new** method for link prediction based on **Graph Neural Networks (GNNs)**, and compare it against existing baselines using **ROC-AUC**.

This work targets the Science4Cast-style task: given a temporal semantic network and a set of candidate (currently unconnected) node pairs, predict which pairs will form sufficiently strong connections in the future.

## Proposed Method (Not Present in the Official Repo)
**Model M9: 2-layer Graph Convolutional Network (GCN)** for link prediction, with skip connections.

- **Graph encoder:** a standard 2-layer GCN operating on the graph snapshot up to `year_start`.
- **Node features:** normalized node degree (1D), chosen for simplicity, speed, and as a strong baseline feature under preferential attachment dynamics.
- **Edge scoring / decoder:** dot-product between the learned node embeddings of a candidate pair.
- **Training objective:** binary classification over provided candidate pairs using `BCEWithLogitsLoss`, with `pos_weight = #neg/#pos` to handle extreme class imbalance.
- **Metric:** ROC-AUC.

## Why This Method
- GCNs are a canonical GNN architecture covered in graph ML coursework and are directly applicable to link prediction.
- The method is **more expressive** than purely hand-crafted scores (e.g., PA/CN) because it learns neighborhood aggregation weights.
- The implementation can remain minimal and reproducible (pure PyTorch, no external graph libs required).

## Expected Benefit
- **Improved performance** compared to simple degree-based heuristics on some settings, especially where neighborhood context matters.
- **Learned representations** that capture local graph structure and can generalize across different `delta/cutoff/minedge` configurations.
- A clean, maintainable implementation integrated into the repo's existing dataset and evaluation format.

## Potential Challenges and Mitigations
- **Severe class imbalance** (very low positive rate for some tasks):
  - Use `pos_weight` in BCE loss.
  - Use a capped random subset of pairs for faster iteration (`max_pairs`), while preserving class presence.
- **Runtime on CPU:**
  - Keep the model small (2 layers), reduce epochs (default 10), and cap pairs when needed (default is no cap).
  - Allow optional GPU usage if available.

## Implementation Plan
1. **Add M9 model module** in `all_models/M9/model.py`
   - GCN layers, training loop, and scoring for candidate pairs.
2. **Add evaluation entrypoint** in `all_models/M9/evaluate_model.py`
   - Mirror the style of `all_models/M4/evaluate_model.py` / `all_models/M5/evaluate_model.py`.
   - Loop over the dataset grid and write a summary file with AUC results.
3. **Ensure reproducibility**
   - Fixed seeds, deterministic settings where possible, and clearly documented defaults.

## Evaluation Plan
- Use the provided datasets: `SemanticGraph_delta_{delta}_cutoff_{cutoff}_minedge_{minedge}.pkl`.
- Evaluate across the standard grid:
  - delta in {1,3,5}, cutoff in {0,5,25}, minedge in {1,3}.
- Report ROC-AUC and compare against existing models (at least the baseline scores already present in the repository).

## Deliverables
- **New method implementation:** `all_models/M9/model.py`
- **Evaluation script:** `all_models/M9/evaluate_model.py`
- **Results summary:** `AUC Summary M9.txt`
- **Documentation:** updated `README.md` including M9 results + brief discussion

## How To Run (for grading)
From the repository root (with `SemanticGraph_*.pkl` datasets present):

```bash
python all_models/M9/evaluate_model.py
```

Optional single-dataset run:

```bash
python all_models/M9/model.py --data SemanticGraph_delta_1_cutoff_25_minedge_1.pkl
```
