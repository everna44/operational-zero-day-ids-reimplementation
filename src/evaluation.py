import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def evaluate_attack_subset(
    nodes: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    attack_category: str,
) -> dict:
    """
    Evaluate one attack category against benign test traffic.

    Examples:
        attack_category="SEEN"
        attack_category="OOD"

    Positive class:
        attack = 1
        BENIGN = 0
    """
    if attack_category not in {"SEEN", "OOD"}:
        raise ValueError(
            "attack_category must be 'SEEN' or 'OOD'."
        )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

    if len(nodes) != len(scores):
        raise ValueError(
            "nodes and scores must have the same length."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "scores contain NaN or infinity."
        )

    if "split" not in nodes.columns:
        raise ValueError(
            "Missing required column: split"
        )

    if "session_category" not in nodes.columns:
        raise ValueError(
            "Missing required column: session_category"
        )

    test_mask = (
        nodes["split"].eq("test")
        & nodes["session_category"].isin(
            ["BENIGN", attack_category]
        )
    )

    subset = nodes.loc[test_mask]
    subset_scores = scores[test_mask.to_numpy()]

    if subset.empty:
        raise ValueError(
            "No eligible test samples were found."
        )

    labels = (
        subset["session_category"]
        .eq(attack_category)
        .astype(np.int64)
        .to_numpy()
    )

    predictions = (
        subset_scores >= threshold
    ).astype(np.int64)

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    benign_mask = labels == 0
    attack_mask = labels == 1

    false_positives = int(
        np.sum(
            predictions[benign_mask] == 1
        )
    )

    true_positives = int(
        np.sum(
            predictions[attack_mask] == 1
        )
    )

    num_benign = int(
        benign_mask.sum()
    )

    num_attack = int(
        attack_mask.sum()
    )

    fpr = (
        false_positives / num_benign
        if num_benign > 0
        else 0.0
    )

    return {
        "attack_category": attack_category,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "num_attack": num_attack,
        "num_benign": num_benign,
    }