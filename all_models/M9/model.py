"""
Minimal GCN link predictor (pure PyTorch, no torch_geometric).
Trains a 2-layer GCN on node-degree features and scores pairs via dot product.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import date
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_adj(edge_array: np.ndarray, cutoff_year: int, num_nodes: int) -> torch.Tensor:
    origin = date(1990, 1, 1)
    cutoff_day = (date(cutoff_year, 12, 31) - origin).days
    edges = edge_array[edge_array[:, 2] < cutoff_day][:, :2].astype(np.int64)
    if edges.size == 0:
        raise ValueError("No edges before cutoff_year")

    edges_rev = edges[:, [1, 0]]
    self_loops = np.arange(num_nodes, dtype=np.int64)
    loop_edges = np.stack([self_loops, self_loops], axis=1)
    edges_all = np.vstack([edges, edges_rev, loop_edges])

    edge_index = torch.from_numpy(edges_all.T)
    row, col = edge_index
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt.isinf()] = 0.0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    adj = torch.sparse_coo_tensor(edge_index, norm, size=(num_nodes, num_nodes))
    return adj.coalesce()


def build_features(adj: torch.Tensor) -> torch.Tensor:
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg = deg / (deg.max() + 1e-8)
    return deg.unsqueeze(-1)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = torch.sparse.mm(adj, x)
        return self.linear(out)


class LinkPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 64) -> None:
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gcn1(x, adj))
        h = self.gcn2(h, adj)
        return h

    def decode(self, h: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        h_src = h[pairs[:, 0]]
        h_dst = h[pairs[:, 1]]
        return (h_src * h_dst).sum(dim=-1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, adj)
        return self.decode(h, pairs)


class PairDataset(Dataset):
    def __init__(self, pairs: np.ndarray, labels: np.ndarray) -> None:
        self.pairs = torch.from_numpy(pairs.astype(np.int64))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.pairs)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return self.pairs[idx], self.labels[idx]


@dataclass
class TrainConfig:
    hidden_dim: int = 64
    epochs: int = 10
    batch_size: int = 2048
    lr: float = 3e-4
    weight_decay: float = 1e-4
    train_ratio: float = 0.8
    max_pairs: int = 200_000
    seed: int = 42


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = fp = 0
    tps = []
    fps = []
    for lab in sorted_labels:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tps.append(tp)
        fps.append(fp)
    if tp == 0 or fp == 0:
        return 0.5
    tpr = np.array(tps, dtype=np.float64) / tp
    fpr = np.array(fps, dtype=np.float64) / fp
    return float(np.trapz(tpr, fpr))


def split_pairs(
    pairs: np.ndarray, labels: np.ndarray, ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if ratio >= 1.0:
        return pairs, labels, pairs[:0], labels[:0]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    split = int(len(idx) * ratio)
    train_idx = idx[:split]
    val_idx = idx[split:]
    return pairs[train_idx], labels[train_idx], pairs[val_idx], labels[val_idx]


def train_one_dataset(
    edges: np.ndarray,
    pairs: np.ndarray,
    labels: np.ndarray,
    year_start: int,
    num_nodes: int,
    config: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    adj = build_adj(edges, year_start, num_nodes).to(device)
    feats = build_features(adj).to(device)

    if config.max_pairs > 0 and len(pairs) > config.max_pairs:
        rng = np.random.default_rng(config.seed)
        subset_idx = rng.choice(len(pairs), size=config.max_pairs, replace=False)
        pairs = pairs[subset_idx]
        labels = labels[subset_idx]

    train_pairs, train_labels, val_pairs, val_labels = split_pairs(pairs, labels, config.train_ratio, config.seed)

    train_ds = PairDataset(train_pairs, train_labels)
    val_pairs_t = torch.from_numpy(val_pairs.astype(np.int64)).to(device)
    val_labels_t = torch.from_numpy(val_labels.astype(np.float32)).to(device)

    model = LinkPredictor(in_dim=feats.size(1), hidden_dim=config.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train_pos = float(np.sum(train_labels))
    train_neg = float(len(train_labels) - np.sum(train_labels))
    if train_pos > 0:
        pos_weight = torch.tensor([train_neg / train_pos], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
        for batch_pairs, batch_labels in loader:
            batch_pairs = batch_pairs.to(device)
            batch_labels = batch_labels.to(device)
            scores = model(feats, adj, batch_pairs)
            loss = loss_fn(scores, batch_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batches += 1

        model.eval()
        if len(val_pairs) > 0:
            with torch.no_grad():
                val_scores = model(feats, adj, val_pairs_t)
                auc = compute_auc(val_scores.cpu().numpy(), val_labels)
                if auc > best_val_auc:
                    best_val_auc = auc
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        avg_loss = total_loss / max(batches, 1)
        print(f"[Epoch {epoch+1}/{config.epochs}] loss={avg_loss:.4f} val_auc={best_val_auc:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    if len(val_pairs) > 0:
        with torch.no_grad():
            best_val_scores = model(feats, adj, val_pairs_t).cpu().numpy()
        val_auc = compute_auc(best_val_scores, val_labels)
    else:
        val_auc = float("nan")

    with torch.no_grad():
        all_pairs_t = torch.from_numpy(pairs.astype(np.int64)).to(device)
        all_scores = model(feats, adj, all_pairs_t).cpu().numpy()
        full_auc = compute_auc(all_scores, labels)

    return {
        "auc_val": val_auc,
        "auc_full": full_auc,
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "train_pos_rate": float(np.mean(train_labels)) if len(train_labels) else 0.0,
        "val_pos_rate": float(np.mean(val_labels)) if len(val_labels) else 0.0,
    }


def load_dataset(data_path: str):
    with open(data_path, "rb") as f:
        (
            full_dynamic_graph_sparse,
            unconnected_vertex_pairs,
            unconnected_vertex_pairs_solution,
            year_start,
            years_delta,
            vertex_degree_cutoff,
            min_edges,
        ) = pickle.load(f)
    return (
        full_dynamic_graph_sparse,
        unconnected_vertex_pairs,
        np.array(unconnected_vertex_pairs_solution, dtype=np.int64),
        year_start,
        years_delta,
        vertex_degree_cutoff,
        min_edges,
    )


def run_link_prediction(data_path: str, config: Dict | None = None) -> Dict[str, float]:
    cfg = TrainConfig(**(config or {}))
    set_seed(cfg.seed)
    edges, pairs, labels, year_start, years_delta, vertex_degree_cutoff, min_edges = load_dataset(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes = int(max(edges[:, :2].max(), pairs.max())) + 1
    result = train_one_dataset(
        edges=edges,
        pairs=pairs,
        labels=labels,
        year_start=year_start,
        num_nodes=num_nodes,
        config=cfg,
        device=device,
    )
    result.update(
        {
            "data_path": data_path,
            "year_start": year_start,
            "years_delta": years_delta,
            "vertex_degree_cutoff": vertex_degree_cutoff,
            "min_edges": min_edges,
            "device": str(device),
        }
    )
    return result


def run_link_prediction_loaded(
    *,
    full_dynamic_graph_sparse: np.ndarray,
    unconnected_vertex_pairs: np.ndarray,
    unconnected_vertex_pairs_solution: np.ndarray,
    year_start: int,
    config: Dict | None = None,
) -> Dict[str, float]:
    cfg = TrainConfig(**(config or {}))
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes = int(max(full_dynamic_graph_sparse[:, :2].max(), unconnected_vertex_pairs.max())) + 1
    result = train_one_dataset(
        edges=full_dynamic_graph_sparse,
        pairs=unconnected_vertex_pairs,
        labels=np.array(unconnected_vertex_pairs_solution, dtype=np.int64),
        year_start=year_start,
        num_nodes=num_nodes,
        config=cfg,
        device=device,
    )
    result.update({"device": str(device)})
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run GCN link prediction (M9).")
    parser.add_argument("--data", required=True, help="Path to SemanticGraph_*.pkl dataset.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        max_pairs=args.max_pairs,
        lr=args.lr,
        seed=args.seed,
    )
    result = run_link_prediction(args.data, asdict(cfg))
    print("=== M9 GCN Results ===")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    cli()
