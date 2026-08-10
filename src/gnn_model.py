from __future__ import annotations

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

def build_dense_adjacency(graph: nx.Graph, n: int) -> torch.Tensor:
    A = np.zeros((n, n), dtype=np.float32)
    for u, v in graph.edges():
        A[u, v] = 1.0
        A[v, u] = 1.0
    return torch.tensor(A)

def build_sparse_adjacency(graph: nx.Graph, n: int) -> torch.Tensor:

    rows: list[int] = []
    cols: list[int] = []
    for u, v in graph.edges():
        rows.extend((u, v))
        cols.extend((v, u))
    idx = torch.tensor([rows, cols], dtype=torch.long)
    vals = torch.ones(len(rows), dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, vals, (n, n)).coalesce()

def _agg(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    if A.is_sparse:

        if X.dim() == 2:
            return torch.sparse.mm(A, X)
        B, n, F = X.shape
        Xf = X.permute(1, 0, 2).reshape(n, B * F)
        return torch.sparse.mm(A, Xf).reshape(n, B, F).permute(1, 0, 2).contiguous()
    if X.dim() == 2:
        return A @ X
    return torch.einsum("ij,bjk->bik", A, X)

class GINLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        agg = _agg(A, X)
        return self.mlp((1.0 + self.eps) * X + agg)

class GIN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.layers = nn.ModuleList([GINLayer(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        H = X
        for layer, norm in zip(self.layers, self.norms):
            H = layer(H, A)
            H = norm(H)
            H = torch.relu(H)
        return H

class GLocalKDScorer(nn.Module):

    def __init__(self, in_dim: int, hidden: int, n_layers: int):
        super().__init__()
        self.teacher = GIN(in_dim, hidden, n_layers)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.student = GIN(in_dim, hidden, n_layers)

    @staticmethod
    def _graph_readout(H: torch.Tensor) -> torch.Tensor:
        return torch.cat([H.mean(dim=-2), H.max(dim=-2).values], dim=-1)

    @torch.no_grad()
    def teacher_embed(self, X: torch.Tensor, A: torch.Tensor, chunk: int = 64) -> torch.Tensor:

        if X.dim() == 2:
            return self.teacher(X, A)
        outs = [self.teacher(X[s:s + chunk], A) for s in range(0, X.shape[0], chunk)]
        return torch.cat(outs, dim=0)

    def node_discrepancy(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            Ht = self.teacher(X, A)
        Hs = self.student(X, A)
        return ((Ht - Hs) ** 2).mean(dim=-1)

    def distill_loss(self, X: torch.Tensor, A: torch.Tensor, Ht: torch.Tensor | None = None) -> torch.Tensor:
        if Ht is None:
            with torch.no_grad():
                Ht = self.teacher(X, A)
        Hs = self.student(X, A)
        node_disc = ((Ht - Hs) ** 2).mean(dim=-1)
        gt, gs = self._graph_readout(Ht), self._graph_readout(Hs)
        graph_disc = ((gt - gs) ** 2).mean(dim=-1)
        return node_disc.mean() + graph_disc.mean()

class MomentCalibrator:

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    @staticmethod
    def _moments(disc_batch: torch.Tensor) -> np.ndarray:
        m = disc_batch.mean(dim=-1)
        s = disc_batch.std(dim=-1, unbiased=False)
        mx = disc_batch.max(dim=-1).values
        return torch.stack([m, s, mx], dim=-1).detach().numpy()

    def fit(self, disc_batch: torch.Tensor) -> "MomentCalibrator":
        moments = self._moments(disc_batch)
        self.mean_ = moments.mean(axis=0)
        self.std_ = moments.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def score(self, disc_batch: torch.Tensor) -> np.ndarray:
        moments = self._moments(disc_batch)
        z = (moments - self.mean_) / self.std_
        return z.mean(axis=1)

def train_scorer(
    model: GLocalKDScorer,
    A: torch.Tensor,
    train_X: torch.Tensor,
    val_X: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int = 12,
    batch_size: int = 32,
    verbose: bool = False,
) -> dict:
    opt = torch.optim.Adam(model.student.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = train_X.shape[0]

    Ht_train = model.teacher_embed(train_X, A)
    Ht_val = model.teacher_embed(val_X, A)
    best_val = float("inf")
    best_state = None
    epochs_since_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            Xb = train_X[idx]
            opt.zero_grad()
            loss = model.distill_loss(Xb, A, Ht=Ht_train[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_loss = model.distill_loss(val_X, A, Ht=Ht_val).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.student.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                break

    if best_state is not None:
        model.student.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    history["stopped_epoch"] = epoch
    return history
