from pathlib import Path

import pandas as pd

from src.calibration import (
    calibrate_target_fpr,
    compute_fpr,
)
from src.evaluation import evaluate_attack_subset
from src.gnn import (
    predict_gnn_score,
    train_gnn,
)


PROCESSED_DIR = Path("data/processed")

TARGET_FPRS = [
    0.001,
    0.01,
    0.05,
]


def load_gnn_setting(
    window: str,
    method: str,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    if window not in {
        "1m",
        "5m",
    }:
        raise ValueError(
            "window must be '1m' or '5m'."
        )

    method = method.lower()

    if method == "base":
        node_path = (
            PROCESSED_DIR
            / f"nodes_{window}_standardized.csv"
        )

        edge_prefix = (
            f"communication_edges_{window}"
        )

    elif method == "ours":
        node_path = (
            PROCESSED_DIR
            / f"nodes_{window}_smoothed_split_g0.3.csv"
        )

        edge_prefix = (
            f"knn_edges_{window}"
        )

    else:
        raise ValueError(
            "method must be 'base' or 'ours'."
        )

    nodes = pd.read_csv(
        node_path
    )

    edges = {}

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        edges[split_name] = pd.read_csv(
            PROCESSED_DIR
            / f"{edge_prefix}_{split_name}.csv"
        )

    return nodes, edges


def run_gnn_experiment(
    window: str,
    method: str,
    backbone: str,
    seed: int,
    epochs: int = 80,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Run one GNN setting for one random seed.

    A model is trained once. The same validation
    and test scores are then evaluated at each
    target-FPR operating point.
    """
    nodes, edges = load_gnn_setting(
        window,
        method,
    )

    train = nodes[
        nodes["split"] == "train"
    ].copy()

    validation = nodes[
        nodes["split"] == "validation"
    ].copy()

    test = nodes[
        nodes["split"] == "test"
    ].copy()

    model = train_gnn(
        train,
        edges["train"],
        backbone=backbone,
        epochs=epochs,
        seed=seed,
        device=device,
    )

    validation_scores = (
        predict_gnn_score(
            model,
            validation,
            edges["validation"],
            device=device,
        )
    )

    test_scores = predict_gnn_score(
        model,
        test,
        edges["test"],
        device=device,
    )

    validation_benign_mask = (
        validation["session_category"]
        .eq("BENIGN")
        .to_numpy()
    )

    validation_benign_scores = (
        validation_scores[
            validation_benign_mask
        ]
    )

    rows = []

    for alpha in TARGET_FPRS:
        threshold = calibrate_target_fpr(
            validation_benign_scores,
            alpha,
        )

        validation_fpr = compute_fpr(
            validation_benign_scores,
            threshold,
        )

        seen = evaluate_attack_subset(
            test,
            test_scores,
            threshold,
            "SEEN",
        )

        ood = evaluate_attack_subset(
            test,
            test_scores,
            threshold,
            "OOD",
        )

        violation_ratio = (
            ood["fpr"] / alpha
        )

        rows.append({
            "window": window,
            "method": method,
            "backbone": backbone,
            "seed": seed,
            "alpha": alpha,
            "threshold": threshold,
            "validation_fpr": validation_fpr,
            "test_fpr": ood["fpr"],
            "violation_ratio": violation_ratio,
            "seen_precision": seen["precision"],
            "seen_recall": seen["recall"],
            "seen_f1": seen["f1"],
            "seen_tp": seen["true_positives"],
            "ood_precision": ood["precision"],
            "ood_recall": ood["recall"],
            "ood_f1": ood["f1"],
            "ood_tp": ood["true_positives"],
            "false_positives": ood["false_positives"],
        })

    return pd.DataFrame(
        rows
    )