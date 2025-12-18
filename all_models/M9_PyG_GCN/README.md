# M9: PyTorch GCN link predictor

Lightweight 2-layer GCN for the Science4Cast semantic graph, implemented without external graph libraries (pure PyTorch). Uses node-degree features and dot-product decoding for link prediction.

## Usage
```bash
python all_models/M9_PyG_GCN/model.py --data SemanticGraph_delta_1_cutoff_25_minedge_1.pkl \
  --epochs 20 --batch-size 2048 --hidden-dim 64 --max-pairs 200000 --lr 0.0003
```

The script:
- loads the pickled dataset tuple (same format as `evaluate_model.py`),
- builds a normalized adjacency from edges up to `year_start`,
- trains a GCN on a train/val split of `unconnected_vertex_pairs`,
- prints validation and full-pair AUCs.

## Notes
- Designed to run on CPU; will use GPU if available.
- Pair set is capped via `--max-pairs` to keep runs short; increase for better accuracy.
- Seeds are fixed for reproducibility; change with `--seed` if desired.
