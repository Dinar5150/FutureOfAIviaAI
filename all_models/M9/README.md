# Model M9: GCN (pure PyTorch)

This folder contains a minimal Graph Convolutional Network (GCN) model for the Science4Cast/FutureOfAIviaAI link prediction task, implemented in pure PyTorch (no `torch_geometric`).

## Files

- `model.py`: GCN implementation + CLI.
- `evaluate_model.py`: runs the M9 model over the standard benchmark grid (delta/cutoff/minedge), computes AUC and writes `AUC Summary M9.txt`.

## Run

Place the dataset files `SemanticGraph_delta_{delta}_cutoff_{cutoff}_minedge_{minedge}.pkl` in the repo root, then run from the repo root:

```bash
python all_models/M9/evaluate_model.py
```

To run a single dataset:

```bash
python all_models/M9/model.py --data SemanticGraph_delta_1_cutoff_25_minedge_1.pkl --epochs 10 --max-pairs 200000
```

