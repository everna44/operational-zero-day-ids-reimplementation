from pathlib import Path

import pandas as pd

from src.knn_graph import build_knn_similarity_edges


PROCESSED_DIR = Path("data/processed")

DEFAULT_K = 3
DEFAULT_BATCH_SIZE = 512


def filter_edges_within_split(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Keep communication edges whose endpoints both belong
    to the requested split.
    """
    if split_name not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            "split_name must be "
            "'train', 'validation', or 'test'."
        )

    split_node_ids = set(
        nodes.loc[
            nodes["split"] == split_name,
            "node_id",
        ].astype(int)
    )

    filtered = edges[
        edges["src_node_id"].isin(split_node_ids)
        & edges["dst_node_id"].isin(split_node_ids)
    ].copy()

    return filtered.reset_index(drop=True)


def build_split_knn_edges(
    nodes: pd.DataFrame,
    split_name: str,
    k: int = DEFAULT_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> pd.DataFrame:
    """
    Rebuild KNN independently inside one split.

    This is different from constructing a global KNN graph
    and deleting cross-split edges afterward.

    The original global node_id values are retained.
    """
    if split_name not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            "split_name must be "
            "'train', 'validation', or 'test'."
        )

    subset = nodes[
        nodes["split"] == split_name
    ].copy()

    if len(subset) <= k:
        raise ValueError(
            f"Split '{split_name}' must contain "
            f"more than k={k} nodes."
        )

    edges = build_knn_similarity_edges(
        subset,
        k=k,
        batch_size=batch_size,
        device=device,
    )

    return edges


def save_split_edges(
    window_name: str,
    graph_name: str,
    k: int = DEFAULT_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> None:
    """
    Save split-specific graph edges.

    communication:
        Filter the observed communication graph so both
        endpoints belong to the same split.

    knn:
        Reconstruct KNN independently inside each split.
    """
    if window_name not in {"1m", "5m"}:
        raise ValueError(
            "window_name must be '1m' or '5m'."
        )

    if graph_name not in {
        "communication",
        "knn",
    }:
        raise ValueError(
            "graph_name must be "
            "'communication' or 'knn'."
        )

    node_path = (
        PROCESSED_DIR
        / f"nodes_{window_name}_standardized.csv"
    )

    nodes = pd.read_csv(node_path)

    if graph_name == "communication":
        edge_path = (
            PROCESSED_DIR
            / f"edges_{window_name}.csv"
        )

        full_edges = pd.read_csv(
            edge_path
        )

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        if graph_name == "communication":
            edges = filter_edges_within_split(
                nodes,
                full_edges,
                split_name,
            )

        else:
            edges = build_split_knn_edges(
                nodes,
                split_name,
                k=k,
                batch_size=batch_size,
                device=device,
            )

        output_path = (
            PROCESSED_DIR
            / (
                f"{graph_name}_edges_"
                f"{window_name}_"
                f"{split_name}.csv"
            )
        )

        edges.to_csv(
            output_path,
            index=False,
        )

        print(
            f"{window_name} | "
            f"{graph_name} | "
            f"{split_name}: "
            f"{len(edges):,} edges"
        )