from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROCESSED_DIR = Path("data/processed")

DEFAULT_K = 3
DEFAULT_BATCH_SIZE = 256


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


def build_knn_similarity_edges(
    nodes: pd.DataFrame,
    k: int = DEFAULT_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> pd.DataFrame:
    """
    Build an exact cosine-similarity KNN graph.

    Policy:
    - Use standardized node features.
    - Compute exact cosine similarity against all nodes.
    - Select k nearest neighbors for each node.
    - Remove self-loops.
    - Convert directed KNN selections into undirected edges.
    - Merge duplicate undirected pairs.
    - Store cosine similarity as edge weight.

    Computation is batched to avoid materializing the full
    N x N similarity matrix at once.
    """
    if "node_id" not in nodes.columns:
        raise ValueError(
            "Input node table must contain 'node_id'."
        )

    if not nodes["node_id"].is_unique:
        raise ValueError(
            "node_id values must be unique."
        )

    feature_columns = get_feature_columns(nodes)

    if not feature_columns:
        raise ValueError(
            "No directional feature columns were found."
        )

    num_nodes = len(nodes)

    if k < 1 or k >= num_nodes:
        raise ValueError(
            "k must satisfy 1 <= k < number of nodes."
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be positive."
        )

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    feature_array = nodes[
        feature_columns
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if not np.isfinite(feature_array).all():
        raise ValueError(
            "Feature matrix contains NaN or infinity."
        )

    node_ids = nodes[
        "node_id"
    ].to_numpy(
        dtype=np.int64,
        copy=True,
    )

    features = torch.from_numpy(
        feature_array
    ).to(device)

    features = F.normalize(
        features,
        p=2,
        dim=1,
        eps=1e-12,
    )

    src_parts = []
    dst_parts = []
    similarity_parts = []

    total_batches = (
        num_nodes + batch_size - 1
    ) // batch_size

    print(
        f"Building exact KNN graph on {device} "
        f"with {num_nodes:,} nodes, "
        f"k={k}, batch_size={batch_size}"
    )

    with torch.inference_mode():
        for batch_index, start in enumerate(
            range(0, num_nodes, batch_size),
            start=1,
        ):
            end = min(
                start + batch_size,
                num_nodes,
            )

            query = features[start:end]

            similarity = (
                query @ features.T
            )

            local_rows = torch.arange(
                end - start,
                device=device,
            )

            self_columns = torch.arange(
                start,
                end,
                device=device,
            )

            similarity[
                local_rows,
                self_columns,
            ] = -torch.inf

            values, indices = torch.topk(
                similarity,
                k=k,
                dim=1,
                largest=True,
                sorted=False,
            )

            neighbor_indices = (
                indices
                .reshape(-1)
                .cpu()
                .numpy()
            )

            neighbor_similarity = (
                values
                .reshape(-1)
                .cpu()
                .numpy()
            )

            query_ids = np.repeat(
                node_ids[start:end],
                k,
            )

            neighbor_ids = node_ids[
                neighbor_indices
            ]

            src_parts.append(query_ids)
            dst_parts.append(neighbor_ids)
            similarity_parts.append(
                neighbor_similarity
            )

            if (
                batch_index % 100 == 0
                or batch_index == total_batches
            ):
                print(
                    f"  batch "
                    f"{batch_index:,}/"
                    f"{total_batches:,}"
                )

            del similarity
            del values
            del indices

    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    similarity = np.concatenate(
        similarity_parts
    )
    similarity = np.clip(
    similarity,
    -1.0,
    1.0,
    )

    edge_src = np.minimum(
        src,
        dst,
    )

    edge_dst = np.maximum(
        src,
        dst,
    )

    edges = pd.DataFrame({
        "src_node_id": edge_src,
        "dst_node_id": edge_dst,
        "similarity": similarity,
    })

    edges = edges[
        edges["src_node_id"]
        != edges["dst_node_id"]
    ]

    edges = (
        edges
        .groupby(
            ["src_node_id", "dst_node_id"],
            as_index=False,
        )["similarity"]
        .max()
    )

    edges = (
        edges
        .sort_values(
            ["src_node_id", "dst_node_id"]
        )
        .reset_index(drop=True)
    )

    return edges


def build_and_save_knn_graph(
    window_name: str,
    k: int = DEFAULT_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> Path:
    """
    Build and save the cosine-similarity KNN graph.

    Example:
        data/processed/knn_edges_1m_k3.csv
    """
    if window_name not in {"1m", "5m"}:
        raise ValueError(
            "window_name must be either '1m' or '5m'"
        )

    input_path = (
        PROCESSED_DIR
        / f"nodes_{window_name}_standardized.csv"
    )

    nodes = pd.read_csv(
        input_path,
    )

    edges = build_knn_similarity_edges(
        nodes,
        k=k,
        batch_size=batch_size,
        device=device,
    )

    output_path = (
        PROCESSED_DIR
        / f"knn_edges_{window_name}_k{k}.csv"
    )

    edges.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved {len(edges):,} edges "
        f"to {output_path}"
    )

    return output_path