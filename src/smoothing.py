from pathlib import Path

import numpy as np
import pandas as pd
import torch


DEFAULT_GAMMA = 0.3

PROCESSED_DIR = Path("data/processed")

def get_feature_columns(
    nodes: pd.DataFrame,
) -> list[str]:
    """
    Return standardized directional feature columns.
    """
    return [
        column
        for column in nodes.columns
        if column.startswith("out_")
        or column.startswith("in_")
    ]


def apply_pre_smoothing(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    gamma: float = DEFAULT_GAMMA,
    device: str | None = None,
) -> pd.DataFrame:
    """
    Apply weighted graph pre-smoothing:

        X' = (1 - gamma) X + gamma D^{-1} A X

    Global node_id values do not need to be contiguous.
    Edge endpoints are mapped to local tensor positions
    internally.

    Stored KNN edges are undirected pairs represented once,
    so both message-passing directions are added internally.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(
            "gamma must be between 0 and 1."
        )

    if gamma == 0.0:
        return nodes.copy()

    required_node_columns = {
        "node_id",
    }

    required_edge_columns = {
        "src_node_id",
        "dst_node_id",
        "similarity",
    }

    if not required_node_columns.issubset(
        nodes.columns
    ):
        raise ValueError(
            "Node table must contain 'node_id'."
        )

    if not required_edge_columns.issubset(
        edges.columns
    ):
        raise ValueError(
            "Edge table must contain "
            "'src_node_id', 'dst_node_id', "
            "and 'similarity'."
        )

    if not nodes["node_id"].is_unique:
        raise ValueError(
            "node_id values must be unique."
        )

    if nodes.empty:
        raise ValueError(
            "Node table must not be empty."
        )

    if edges.empty:
        raise ValueError(
            "Edge table must not be empty."
        )

    feature_columns = get_feature_columns(
        nodes
    )

    if not feature_columns:
        raise ValueError(
            "No directional feature columns found."
        )

    feature_array = nodes[
        feature_columns
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if not np.isfinite(
        feature_array
    ).all():
        raise ValueError(
            "Feature matrix contains NaN or infinity."
        )

    node_ids = nodes[
        "node_id"
    ].to_numpy(
        dtype=np.int64,
        copy=True,
    )

    src_global = edges[
        "src_node_id"
    ].to_numpy(
        dtype=np.int64,
        copy=True,
    )

    dst_global = edges[
        "dst_node_id"
    ].to_numpy(
        dtype=np.int64,
        copy=True,
    )

    weight = edges[
        "similarity"
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if not np.isfinite(weight).all():
        raise ValueError(
            "Edge weights contain NaN or infinity."
        )

    node_index = pd.Index(
        node_ids
    )

    src = node_index.get_indexer(
        src_global
    )

    dst = node_index.get_indexer(
        dst_global
    )

    if (
        (src < 0).any()
        or (dst < 0).any()
    ):
        raise ValueError(
            "Edge list contains node IDs "
            "that are not present in nodes."
        )

    num_nodes = len(nodes)

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    x = torch.from_numpy(
        feature_array
    ).to(device)

    src_tensor = torch.from_numpy(
        src.astype(
            np.int64,
            copy=False,
        )
    ).to(device)

    dst_tensor = torch.from_numpy(
        dst.astype(
            np.int64,
            copy=False,
        )
    ).to(device)

    weight_tensor = torch.from_numpy(
        weight
    ).to(device)

    # Stored edges contain each undirected pair once.
    message_src = torch.cat([
        src_tensor,
        dst_tensor,
    ])

    message_dst = torch.cat([
        dst_tensor,
        src_tensor,
    ])

    message_weight = torch.cat([
        weight_tensor,
        weight_tensor,
    ])

    neighbor_sum = torch.zeros_like(
        x
    )

    degree = torch.zeros(
        num_nodes,
        dtype=x.dtype,
        device=device,
    )

    neighbor_sum.index_add_(
        0,
        message_dst,
        x[message_src]
        * message_weight.unsqueeze(1),
    )

    degree.index_add_(
        0,
        message_dst,
        message_weight,
    )

    if (degree == 0).any():
        raise RuntimeError(
            "Graph contains zero-degree nodes."
        )

    neighbor_mean = (
        neighbor_sum
        / degree.unsqueeze(1)
    )

    smoothed = (
        (1.0 - gamma) * x
        + gamma * neighbor_mean
    )

    result = nodes.copy()

    result[feature_columns] = (
        smoothed
        .cpu()
        .numpy()
    )

    return result


def smooth_and_save(
    window_name: str,
    gamma: float = DEFAULT_GAMMA,
    device: str | None = None,
) -> Path:
    """
    Apply KNN-based feature pre-smoothing and save
    the resulting node table.

    Example:
        data/processed/nodes_1m_smoothed_g0.3.csv
    """
    if window_name not in {"1m", "5m"}:
        raise ValueError(
            "window_name must be either '1m' or '5m'"
        )

    node_path = (
        PROCESSED_DIR
        / f"nodes_{window_name}_standardized.csv"
    )

    edge_path = (
        PROCESSED_DIR
        / f"knn_edges_{window_name}_k3.csv"
    )

    nodes = pd.read_csv(node_path)
    edges = pd.read_csv(edge_path)

    smoothed = apply_pre_smoothing(
        nodes,
        edges,
        gamma=gamma,
        device=device,
    )

    output_path = (
        PROCESSED_DIR
        / f"nodes_{window_name}_smoothed_g{gamma}.csv"
    )

    smoothed.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved {len(smoothed):,} nodes "
        f"to {output_path}"
    )

    return output_path


def smooth_split_aware_and_save(
    window_name: str,
    gamma: float = DEFAULT_GAMMA,
    device: str | None = None,
) -> Path:
    """
    Apply pre-smoothing independently inside
    Train / Validation / Test.

    Each split uses its independently reconstructed
    KNN graph, preventing cross-split feature mixing.
    """
    if window_name not in {
        "1m",
        "5m",
    }:
        raise ValueError(
            "window_name must be '1m' or '5m'."
        )

    node_path = (
        PROCESSED_DIR
        / f"nodes_{window_name}_standardized.csv"
    )

    nodes = pd.read_csv(
        node_path
    )

    feature_columns = get_feature_columns(
        nodes
    )

    result = nodes.copy()

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        subset = nodes[
            nodes["split"] == split_name
        ].copy()

        edge_path = (
            PROCESSED_DIR
            / (
                f"knn_edges_"
                f"{window_name}_"
                f"{split_name}.csv"
            )
        )

        edges = pd.read_csv(
            edge_path
        )

        smoothed_subset = apply_pre_smoothing(
            subset,
            edges,
            gamma=gamma,
            device=device,
        )

        result.loc[
            subset.index,
            feature_columns,
        ] = smoothed_subset[
            feature_columns
        ].to_numpy()

        print(
            f"{window_name} | "
            f"{split_name} | "
            f"nodes={len(subset):,} | "
            f"edges={len(edges):,}"
        )

    output_path = (
        PROCESSED_DIR
        / (
            f"nodes_{window_name}_"
            f"smoothed_split_g{gamma}.csv"
        )
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved {len(result):,} nodes "
        f"to {output_path}"
    )

    return output_path
