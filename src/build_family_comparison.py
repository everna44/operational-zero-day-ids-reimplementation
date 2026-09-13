from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")


METRICS = [
    "ood_f1_mean",
    "seen_f1_mean",
    "test_fpr_mean",
    "violation_ratio_mean",
    "violation_ratio_p95",
    "strict_ok",
    "b3_ok",
    "b5_ok",
    "b10_ok",
]


def main() -> None:
    tabular = pd.read_csv(
        RESULTS_DIR
        / "tabular_family_summary.csv"
    )

    gnn = pd.read_csv(
        RESULTS_DIR
        / "gnn_family_summary.csv"
    )

    tabular = tabular.copy()
    tabular["family"] = "tabular"

    gnn = gnn.copy()
    gnn["family"] = gnn[
        "method"
    ].map({
        "base": "gnn_base",
        "ours": "gnn_ours",
    })

    columns = [
        "window",
        "alpha",
        "family",
        *METRICS,
    ]

    combined = pd.concat(
        [
            tabular[columns],
            gnn[columns],
        ],
        ignore_index=True,
    )

    family_order = {
        "tabular": 0,
        "gnn_base": 1,
        "gnn_ours": 2,
    }

    combined["_family_order"] = (
        combined["family"]
        .map(family_order)
    )

    combined = (
        combined
        .sort_values(
            [
                "window",
                "alpha",
                "_family_order",
            ]
        )
        .drop(
            columns="_family_order"
        )
        .reset_index(
            drop=True
        )
    )

    long_path = (
        RESULTS_DIR
        / "family_comparison.csv"
    )

    combined.to_csv(
        long_path,
        index=False,
    )

    wide = combined.pivot(
        index=[
            "window",
            "alpha",
        ],
        columns="family",
        values=METRICS,
    )

    wide.columns = [
        f"{family}_{metric}"
        for metric, family
        in wide.columns
    ]

    wide = (
        wide
        .reset_index()
        .sort_values(
            [
                "window",
                "alpha",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    wide_path = (
        RESULTS_DIR
        / "family_comparison_wide.csv"
    )

    wide.to_csv(
        wide_path,
        index=False,
    )

    print(
        "Long comparison:",
        combined.shape,
        "->",
        long_path,
    )

    print(
        "Wide comparison:",
        wide.shape,
        "->",
        wide_path,
    )

    print()

    print(
        combined[
            [
                "window",
                "alpha",
                "family",
                "ood_f1_mean",
                "seen_f1_mean",
                "violation_ratio_p95",
                "strict_ok",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()