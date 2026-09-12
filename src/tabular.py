import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 42


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


def train_logistic_regression(
    nodes: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """
    Train a binary Logistic Regression baseline.

    Training policy:
    - BENIGN -> 0
    - SEEN -> 1
    - OOD is completely excluded from training.
    - Only nodes assigned to the Train split are used.
    """
    required_columns = {
        "split",
        "session_category",
    }

    missing_columns = (
        required_columns - set(nodes.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    feature_columns = get_feature_columns(
        nodes
    )

    if not feature_columns:
        raise ValueError(
            "No directional feature columns were found."
        )

    train = nodes[
        (nodes["split"] == "train")
        & nodes["session_category"].isin(
            ["BENIGN", "SEEN"]
        )
    ].copy()

    if train.empty:
        raise ValueError(
            "No eligible Train samples were found."
        )

    x_train = train[
        feature_columns
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    y_train = (
        train["session_category"]
        .map({
            "BENIGN": 0,
            "SEEN": 1,
        })
        .to_numpy(
            dtype=np.int64,
            copy=True,
        )
    )

    if not np.isfinite(x_train).all():
        raise ValueError(
            "Training features contain NaN or infinity."
        )

    if len(np.unique(y_train)) != 2:
        raise ValueError(
            "Training data must contain both "
            "BENIGN and SEEN classes."
        )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
    )

    model.fit(
        x_train,
        y_train,
    )

    return model


def predict_malicious_score(
    model: LogisticRegression,
    nodes: pd.DataFrame,
) -> np.ndarray:
    """
    Return P(malicious) for each supplied host-session node.
    """
    feature_columns = get_feature_columns(
        nodes
    )

    x = nodes[
        feature_columns
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    if not np.isfinite(x).all():
        raise ValueError(
            "Prediction features contain NaN or infinity."
        )

    scores = model.predict_proba(x)[:, 1]

    return scores