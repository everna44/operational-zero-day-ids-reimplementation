import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


RANDOM_STATE = 42

EPOCHS = 80
HIDDEN_DIM = 128
DROPOUT = 0.2
LEARNING_RATE = 1e-3


def set_seed(
    seed: int = RANDOM_STATE,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_feature_columns(
    nodes: pd.DataFrame,
) -> list[str]:
    return [
        column
        for column in nodes.columns
        if column.startswith("out_")
        or column.startswith("in_")
    ]


class GCNDetector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim,
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim,
        )

        self.classifier = nn.Linear(
            hidden_dim,
            1,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
    self,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.conv1(
            x,
            edge_index,
            edge_weight=edge_weight,
        )

        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index,
            edge_weight=edge_weight,
        )

        x = torch.relu(x)
        x = self.dropout(x)

        logits = self.classifier(
            x
        ).squeeze(-1)

        return logits


def prepare_graph_data(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    device: str = "cuda",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Convert one split-specific graph into local PyTorch tensors.

    Global node_id values are mapped to local tensor indices.

    Labels:
        BENIGN -> 0
        SEEN   -> 1
        OOD    -> 1
    """
    if nodes.empty:
        raise ValueError(
            "nodes must not be empty."
        )

    if edges.empty:
        raise ValueError(
            "edges must not be empty."
        )

    feature_columns = get_feature_columns(
        nodes
    )

    x_array = nodes[
        feature_columns
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if not np.isfinite(x_array).all():
        raise ValueError(
            "Node features contain NaN or infinity."
        )

    y_array = (
        nodes["session_category"]
        .map({
            "BENIGN": 0.0,
            "SEEN": 1.0,
            "OOD": 1.0,
        })
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    if not np.isfinite(y_array).all():
        raise ValueError(
            "Invalid session_category values."
        )

    node_ids = nodes[
        "node_id"
    ].to_numpy(
        dtype=np.int64,
        copy=True,
    )

    node_index = pd.Index(
        node_ids
    )

    src = node_index.get_indexer(
        edges["src_node_id"].to_numpy(
            dtype=np.int64,
        )
    )

    dst = node_index.get_indexer(
        edges["dst_node_id"].to_numpy(
            dtype=np.int64,
        )
    )

    if (
        (src < 0).any()
        or (dst < 0).any()
    ):
        raise ValueError(
            "Edge contains node IDs outside this split."
        )

    # Stored edge list contains each undirected pair once.
    edge_src = np.concatenate([
        src,
        dst,
    ])

    edge_dst = np.concatenate([
        dst,
        src,
    ])

    edge_index = torch.tensor(
        np.vstack([
            edge_src,
            edge_dst,
        ]),
        dtype=torch.long,
        device=device,
    )

    if "similarity" in edges.columns:
        weights = edges[
            "similarity"
        ].to_numpy(
            dtype=np.float32,
            copy=True,
        )

    elif "interaction_count" in edges.columns:
        weights = edges[
            "interaction_count"
        ].to_numpy(
            dtype=np.float32,
            copy=True,
        )

    else:
        weights = np.ones(
            len(edges),
            dtype=np.float32,
        )

    if not np.isfinite(weights).all():
        raise ValueError(
            "Edge weights contain NaN or infinity."
        )

    edge_weight = torch.tensor(
        np.concatenate([
            weights,
            weights,
        ]),
        dtype=torch.float32,
        device=device,
    )

    x = torch.tensor(
        x_array,
        dtype=torch.float32,
        device=device,
    )

    y = torch.tensor(
        y_array,
        dtype=torch.float32,
        device=device,
    )

    return (
        x,
        y,
        edge_index,
        edge_weight,
    )


def train_gcn(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    epochs: int = EPOCHS,
    hidden_dim: int = HIDDEN_DIM,
    dropout: float = DROPOUT,
    learning_rate: float = LEARNING_RATE,
    seed: int = RANDOM_STATE,
    device: str = "cuda",
) -> GCNDetector:
    """
    Train a GCN on one split-specific graph.

    Training labels:
        BENIGN -> 0
        SEEN   -> 1

    OOD must not be present in the training split.
    Class imbalance is handled using BCE pos_weight.
    """
    set_seed(seed)

    if (
        nodes["session_category"]
        .eq("OOD")
        .any()
    ):
        raise ValueError(
            "OOD samples must not be present "
            "in GCN training data."
        )

    x, y, edge_index, edge_weight = (
        prepare_graph_data(
            nodes,
            edges,
            device=device,
        )
    )

    num_positive = int(
        (y == 1).sum().item()
    )

    num_negative = int(
        (y == 0).sum().item()
    )

    if (
        num_positive == 0
        or num_negative == 0
    ):
        raise ValueError(
            "Training data must contain both "
            "BENIGN and SEEN classes."
        )

    model = GCNDetector(
        input_dim=x.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    pos_weight = torch.tensor(
        [num_negative / num_positive],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    print(
        f"Device: {device} | "
        f"Nodes: {len(nodes):,} | "
        f"Edges: {len(edges):,} | "
        f"BENIGN: {num_negative:,} | "
        f"SEEN: {num_positive:,} | "
        f"pos_weight: {pos_weight.item():.3f}"
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        optimizer.zero_grad()

        logits = model(
            x,
            edge_index,
            edge_weight,
        )

        loss = criterion(
            logits,
            y,
        )

        loss.backward()
        optimizer.step()

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):
            print(
                f"Epoch "
                f"{epoch:03d}/{epochs} | "
                f"Loss: {loss.item():.6f}"
            )

    return model


def predict_gcn_score(
    model: GCNDetector,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    device: str = "cuda",
) -> np.ndarray:
    """
    Return P(malicious) for every node in one
    split-specific graph.
    """
    x, _, edge_index, edge_weight = (
        prepare_graph_data(
            nodes,
            edges,
            device=device,
        )
    )

    model.eval()

    with torch.no_grad():
        logits = model(
            x,
            edge_index,
            edge_weight,
        )

        scores = torch.sigmoid(
            logits
        )

    return (
        scores
        .cpu()
        .numpy()
    )