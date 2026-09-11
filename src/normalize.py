import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path("data/processed")

def get_feature_columns(
    nodes: pd.DataFrame,
) -> list[str]:
    """
    Return directional numeric feature columns.
    """
    return [
        column
        for column in nodes.columns
        if column.startswith("out_")
        or column.startswith("in_")
    ]


def standardize_features(
    nodes: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize node features using statistics fitted
    only on the Train split.

    The fitted scaler is then applied unchanged to
    Train, Validation, and Test nodes.
    """
    if "split" not in nodes.columns:
        raise ValueError(
            "Input node table must contain a 'split' column."
        )

    feature_columns = get_feature_columns(nodes)

    if not feature_columns:
        raise ValueError(
            "No directional feature columns were found."
        )

    train_mask = nodes["split"] == "train"

    if not train_mask.any():
        raise ValueError(
            "No Train nodes were found."
        )

    scaler = StandardScaler()

    scaler.fit(
        nodes.loc[
            train_mask,
            feature_columns,
        ]
    )

    result = nodes.copy()

    result[feature_columns] = scaler.transform(
        result[feature_columns]
    )

    return result, scaler


def standardize_and_save(
    window_name: str,
) -> Path:
    """
    Standardize node features using Train-only statistics
    and save the resulting node table.

    Example outputs:
        data/processed/nodes_1m_standardized.csv
        data/processed/nodes_5m_standardized.csv
    """
    if window_name not in {"1m", "5m"}:
        raise ValueError(
            "window_name must be either '1m' or '5m'"
        )

    input_path = (
        PROCESSED_DIR
        / f"nodes_{window_name}_split.csv"
    )

    nodes = pd.read_csv(
        input_path,
    )

    standardized, _ = standardize_features(
        nodes,
    )

    output_path = (
        PROCESSED_DIR
        / f"nodes_{window_name}_standardized.csv"
    )

    standardized.to_csv(
        output_path,
        index=False,
    )

    return output_path
