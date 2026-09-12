import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "results/gnn_results.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "results"
)

GROUP_BACKBONE = [
    "window",
    "method",
    "backbone",
    "alpha",
]

GROUP_FAMILY = [
    "window",
    "method",
    "alpha",
]

RUN_METRICS = [
    "threshold",
    "validation_fpr",
    "test_fpr",
    "violation_ratio",
    "seen_precision",
    "seen_recall",
    "seen_f1",
    "ood_precision",
    "ood_recall",
    "ood_f1",
]


def add_mean_std(
    row: dict,
    group: pd.DataFrame,
) -> None:
    for metric in RUN_METRICS:
        values = group[
            metric
        ].astype(float)

        row[f"{metric}_mean"] = (
            float(values.mean())
        )

        row[f"{metric}_std"] = (
            float(values.std(ddof=1))
        )


def add_feasibility(
    row: dict,
    ratios: np.ndarray,
) -> None:
    row["strict_ok"] = float(
        np.mean(ratios <= 1.0)
    )

    row["b3_ok"] = float(
        np.mean(ratios <= 3.0)
    )

    row["b5_ok"] = float(
        np.mean(ratios <= 5.0)
    )

    row["b10_ok"] = float(
        np.mean(ratios <= 10.0)
    )


def summarize_backbones(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for key, group in results.groupby(
        GROUP_BACKBONE,
        sort=True,
    ):
        window, method, backbone, alpha = key

        row = {
            "window": window,
            "method": method,
            "backbone": backbone,
            "alpha": float(alpha),
            "num_seeds": int(
                group["seed"].nunique()
            ),
            "num_runs": int(len(group)),
        }

        add_mean_std(
            row,
            group,
        )

        ratios = group[
            "violation_ratio"
        ].to_numpy(
            dtype=float
        )

        row["violation_ratio_p90"] = float(
            np.percentile(
                ratios,
                90,
            )
        )

        row["violation_ratio_p95"] = float(
            np.percentile(
                ratios,
                95,
            )
        )

        add_feasibility(
            row,
            ratios,
        )

        rows.append(
            row
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            GROUP_BACKBONE
        )
        .reset_index(
            drop=True
        )
    )


def summarize_families(
    results: pd.DataFrame,
    backbone_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for key, group in results.groupby(
        GROUP_FAMILY,
        sort=True,
    ):
        window, method, alpha = key

        row = {
            "window": window,
            "method": method,
            "alpha": float(alpha),
            "num_backbones": int(
                group[
                    "backbone"
                ].nunique()
            ),
            "num_runs": int(
                len(group)
            ),
        }

        add_mean_std(
            row,
            group,
        )

        ratios = group[
            "violation_ratio"
        ].to_numpy(
            dtype=float
        )

        add_feasibility(
            row,
            ratios,
        )

        backbone_group = (
            backbone_summary[
                (
                    backbone_summary[
                        "window"
                    ] == window
                )
                & (
                    backbone_summary[
                        "method"
                    ] == method
                )
                & (
                    backbone_summary[
                        "alpha"
                    ] == float(alpha)
                )
            ]
        )

        row[
            "violation_ratio_p90"
        ] = float(
            backbone_group[
                "violation_ratio_p90"
            ].mean()
        )

        row[
            "violation_ratio_p95"
        ] = float(
            backbone_group[
                "violation_ratio_p95"
            ].mean()
        )

        rows.append(
            row
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            GROUP_FAMILY
        )
        .reset_index(
            drop=True
        )
    )


def build_base_vs_ours(
    family_summary: pd.DataFrame,
) -> pd.DataFrame:
    base = family_summary[
        family_summary["method"] == "base"
    ].copy()

    ours = family_summary[
        family_summary["method"] == "ours"
    ].copy()

    columns = [
        "window",
        "alpha",
        "seen_f1_mean",
        "ood_f1_mean",
        "test_fpr_mean",
        "violation_ratio_mean",
        "violation_ratio_p95",
        "strict_ok",
        "b3_ok",
        "b5_ok",
        "b10_ok",
    ]

    base = base[
        columns
    ].rename(
        columns={
            column: f"base_{column}"
            for column in columns
            if column not in {
                "window",
                "alpha",
            }
        }
    )

    ours = ours[
        columns
    ].rename(
        columns={
            column: f"ours_{column}"
            for column in columns
            if column not in {
                "window",
                "alpha",
            }
        }
    )

    comparison = base.merge(
        ours,
        on=[
            "window",
            "alpha",
        ],
        how="inner",
    )

    comparison[
        "delta_ood_f1"
    ] = (
        comparison[
            "ours_ood_f1_mean"
        ]
        - comparison[
            "base_ood_f1_mean"
        ]
    )

    comparison[
        "delta_seen_f1"
    ] = (
        comparison[
            "ours_seen_f1_mean"
        ]
        - comparison[
            "base_seen_f1_mean"
        ]
    )

    comparison[
        "delta_p95"
    ] = (
        comparison[
            "ours_violation_ratio_p95"
        ]
        - comparison[
            "base_violation_ratio_p95"
        ]
    )

    return (
        comparison
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


def main(
    input_path: Path,
    output_dir: Path,
) -> None:
    results = pd.read_csv(
        input_path
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    backbone_summary = (
        summarize_backbones(
            results
        )
    )

    family_summary = (
        summarize_families(
            results,
            backbone_summary,
        )
    )

    comparison = build_base_vs_ours(
        family_summary
    )

    backbone_path = (
        output_dir
        / "gnn_backbone_summary.csv"
    )

    family_path = (
        output_dir
        / "gnn_family_summary.csv"
    )

    comparison_path = (
        output_dir
        / "gnn_base_vs_ours.csv"
    )

    backbone_summary.to_csv(
        backbone_path,
        index=False,
    )

    family_summary.to_csv(
        family_path,
        index=False,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print(
        "Backbone summary:",
        backbone_summary.shape,
        "->",
        backbone_path,
    )

    print(
        "Family summary:",
        family_summary.shape,
        "->",
        family_path,
    )

    print(
        "Base vs Ours:",
        comparison.shape,
        "->",
        comparison_path,
    )

    print()
    print(
        comparison[
            [
                "window",
                "alpha",
                "base_ood_f1_mean",
                "ours_ood_f1_mean",
                "delta_ood_f1",
                "base_violation_ratio_p95",
                "ours_violation_ratio_p95",
                "delta_p95",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    main(
        input_path=args.input,
        output_dir=args.output_dir,
    )