from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROCESSED_DIR = Path("data/processed")

RANDOM_STATE = 42


def assign_train_val_test_split(
    nodes: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Assign reproducible Train / Validation / Test splits.

    Reimplementation policy:
    - BENIGN and SEEN nodes are split 60% / 20% / 20%.
    - Stratification preserves the BENIGN/SEEN class ratio.
    - OOD nodes are excluded from Train and Validation.
    - All OOD nodes are assigned to Test.
    """
    required_columns = {
        "node_id",
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

    result = nodes.copy()

    result["split"] = pd.NA

    known_mask = result[
        "session_category"
    ].isin(["BENIGN", "SEEN"])

    known = result.loc[known_mask].copy()

    train_indices, temp_indices = train_test_split(
        known.index,
        test_size=0.4,
        random_state=random_state,
        stratify=known["session_category"],
    )

    temp = known.loc[temp_indices]

    val_indices, test_indices = train_test_split(
        temp.index,
        test_size=0.5,
        random_state=random_state,
        stratify=temp["session_category"],
    )

    result.loc[train_indices, "split"] = "train"
    result.loc[val_indices, "split"] = "validation"
    result.loc[test_indices, "split"] = "test"

    ood_mask = (
        result["session_category"] == "OOD"
    )

    result.loc[ood_mask, "split"] = "test"

    if result["split"].isna().any():
        raise RuntimeError(
            "Some nodes were not assigned to a split."
        )

    return result



def assign_and_save_split(
    window_name: str,
    random_state: int = RANDOM_STATE,
) -> Path:
    """
    Assign Train / Validation / Test splits and save
    a new node table without modifying the original file.

    Example outputs:
        data/processed/nodes_1m_split.csv
        data/processed/nodes_5m_split.csv
    """
    if window_name not in {"1m", "5m"}:
        raise ValueError(
            "window_name must be either '1m' or '5m'"
        )

    input_path = (
        PROCESSED_DIR
        / f"nodes_{window_name}.csv"
    )

    nodes = pd.read_csv(
        input_path,
    )

    nodes = assign_train_val_test_split(
        nodes,
        random_state=random_state,
    )

    output_path = (
        PROCESSED_DIR
        / f"nodes_{window_name}_split.csv"
    )

    nodes.to_csv(
        output_path,
        index=False,
    )

    return output_path