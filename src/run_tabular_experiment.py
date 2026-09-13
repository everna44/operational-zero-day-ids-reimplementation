from pathlib import Path

import pandas as pd

from src.calibration import (
    calibrate_target_fpr,
    compute_fpr,
)
from src.evaluation import evaluate_attack_subset
from src.mlp import (
    predict_mlp_score,
    train_mlp,
)
from src.tabular import (
    predict_malicious_score,
    train_logistic_regression,
    train_random_forest,
)


PROCESSED_DIR = Path("data/processed")

TARGET_FPRS = [
    0.001,
    0.01,
    0.05,
]


def load_tabular_nodes(
    window: str,
) -> pd.DataFrame:
    if window not in {
        "1m",
        "5m",
    }:
        raise ValueError(
            "window must be '1m' or '5m'."
        )

    path = (
        PROCESSED_DIR
        / f"nodes_{window}_standardized.csv"
    )

    return pd.read_csv(path)


def train_tabular_model(
    model_name: str,
    nodes: pd.DataFrame,
    seed: int,
    epochs: int = 80,
    device: str = "cuda",
):
    model_name = model_name.lower()

    if model_name == "lr":
        return train_logistic_regression(
            nodes,
            random_state=seed,
        )

    if model_name == "rf":
        return train_random_forest(
            nodes,
            random_state=seed,
        )

    if model_name == "mlp":
        return train_mlp(
            nodes,
            epochs=epochs,
            seed=seed,
            device=device,
        )

    raise ValueError(
        "model_name must be one of: "
        "'lr', 'rf', 'mlp'."
    )


def predict_tabular_score(
    model_name: str,
    model,
    nodes: pd.DataFrame,
):
    model_name = model_name.lower()

    if model_name in {
        "lr",
        "rf",
    }:
        return predict_malicious_score(
            model,
            nodes,
        )

    if model_name == "mlp":
        return predict_mlp_score(
            model,
            nodes,
        )

    raise ValueError(
        "model_name must be one of: "
        "'lr', 'rf', 'mlp'."
    )


def run_tabular_experiment(
    window: str,
    model_name: str,
    seed: int,
    epochs: int = 80,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Run one tabular model for one random seed.

    The model is trained once. Validation and
    test scores are then evaluated at all target
    FPR operating points.
    """
    nodes = load_tabular_nodes(
        window
    )

    validation = nodes[
        nodes["split"] == "validation"
    ].copy()

    test = nodes[
        nodes["split"] == "test"
    ].copy()

    print(
        f"Window: {window} | "
        f"Model: {model_name.upper()} | "
        f"Seed: {seed}"
    )

    model = train_tabular_model(
        model_name=model_name,
        nodes=nodes,
        seed=seed,
        epochs=epochs,
        device=device,
    )

    validation_scores = (
        predict_tabular_score(
            model_name,
            model,
            validation,
        )
    )

    test_scores = predict_tabular_score(
        model_name,
        model,
        test,
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
            "model": model_name.lower(),
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