import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "results/tabular_results.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "results"
)

GROUP_MODEL = [
    "window",
    "model",
    "alpha",
]

GROUP_FAMILY = [
    "window",
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

        row[f"{metric}_mean"] = float(
            values.mean()
        )

        row[f"{metric}_std"] = float(
            values.std(ddof=1)
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


def summarize_models(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for key, group in results.groupby(
        GROUP_MODEL,
        sort=True,
    ):
        window, model, alpha = key

        row = {
            "window": window,
            "model": model,
            "alpha": float(alpha),
            "num_seeds": int(
                group["seed"].nunique()
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
            GROUP_MODEL
        )
        .reset_index(
            drop=True
        )
    )


def summarize_family(
    results: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for key, group in results.groupby(
        GROUP_FAMILY,
        sort=True,
    ):
        window, alpha = key

        row = {
            "window": window,
            "family": "tabular",
            "alpha": float(alpha),
            "num_models": int(
                group["model"].nunique()
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

        model_group = model_summary[
            (
                model_summary[
                    "window"
                ] == window
            )
            & (
                model_summary[
                    "alpha"
                ] == float(alpha)
            )
        ]

        row[
            "violation_ratio_p90"
        ] = float(
            model_group[
                "violation_ratio_p90"
            ].mean()
        )

        row[
            "violation_ratio_p95"
        ] = float(
            model_group[
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

    model_summary = summarize_models(
        results
    )

    family_summary = summarize_family(
        results,
        model_summary,
    )

    model_path = (
        output_dir
        / "tabular_model_summary.csv"
    )

    family_path = (
        output_dir
        / "tabular_family_summary.csv"
    )

    model_summary.to_csv(
        model_path,
        index=False,
    )

    family_summary.to_csv(
        family_path,
        index=False,
    )

    print(
        "Model summary:",
        model_summary.shape,
        "->",
        model_path,
    )

    print(
        "Family summary:",
        family_summary.shape,
        "->",
        family_path,
    )

    print()
    print(
        family_summary[
            [
                "window",
                "alpha",
                "ood_f1_mean",
                "seen_f1_mean",
                "violation_ratio_mean",
                "violation_ratio_p95",
                "strict_ok",
                "b3_ok",
                "b5_ok",
                "b10_ok",
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