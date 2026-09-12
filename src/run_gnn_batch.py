import argparse
import gc
from pathlib import Path

import pandas as pd
import torch

from src.run_gnn_experiment import (
    TARGET_FPRS,
    run_gnn_experiment,
)


DEFAULT_WINDOWS = ["1m", "5m"]
DEFAULT_METHODS = ["base", "ours"]
DEFAULT_BACKBONES = [
    "gcn",
    "gin",
    "sage",
    "gat",
]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

DEFAULT_OUTPUT = Path(
    "results/gnn_results.csv"
)


def save_incremental(
    new_result: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        previous = pd.read_csv(
            output_path
        )

        combined = pd.concat(
            [
                previous,
                new_result,
            ],
            ignore_index=True,
        )
    else:
        combined = new_result.copy()

    key_columns = [
        "window",
        "method",
        "backbone",
        "seed",
        "alpha",
    ]

    combined = (
        combined
        .drop_duplicates(
            subset=key_columns,
            keep="last",
        )
        .sort_values(
            key_columns
        )
        .reset_index(
            drop=True
        )
    )

    combined.to_csv(
        output_path,
        index=False,
    )


def load_completed_settings(
    output_path: Path,
) -> set[tuple[str, str, str, int]]:
    if not output_path.exists():
        return set()

    results = pd.read_csv(
        output_path
    )

    required_columns = {
        "window",
        "method",
        "backbone",
        "seed",
        "alpha",
    }

    if not required_columns.issubset(
        results.columns
    ):
        return set()

    completed = set()

    group_columns = [
        "window",
        "method",
        "backbone",
        "seed",
    ]

    expected_alphas = set(
        TARGET_FPRS
    )

    for key, group in results.groupby(
        group_columns
    ):
        observed_alphas = set(
            group["alpha"]
            .astype(float)
            .tolist()
        )

        if expected_alphas.issubset(
            observed_alphas
        ):
            completed.add(
                (
                    str(key[0]),
                    str(key[1]),
                    str(key[2]),
                    int(key[3]),
                )
            )

    return completed


def run_batch(
    windows: list[str],
    methods: list[str],
    backbones: list[str],
    seeds: list[int],
    epochs: int = 80,
    device: str = "cuda",
    output_path: Path = DEFAULT_OUTPUT,
) -> None:
    total = (
        len(windows)
        * len(methods)
        * len(backbones)
        * len(seeds)
    )

    current = 0
    completed_settings = load_completed_settings(
        output_path
    )

    for window in windows:
        for method in methods:
            for backbone in backbones:
                for seed in seeds:
                    current += 1

                    setting_key = (
                        window,
                        method,
                        backbone,
                        seed,
                    )

                    if setting_key in completed_settings:
                        print(
                            f"[{current}/{total}] SKIP | "
                            f"window={window} | "
                            f"method={method} | "
                            f"backbone={backbone} | "
                            f"seed={seed}"
                        )
                        continue

                    print()
                    print("=" * 72)

                    print(
                        f"[{current}/{total}] "
                        f"window={window} | "
                        f"method={method} | "
                        f"backbone={backbone} | "
                        f"seed={seed}"
                    )

                    result = run_gnn_experiment(
                        window=window,
                        method=method,
                        backbone=backbone,
                        seed=seed,
                        epochs=epochs,
                        device=device,
                    )

                    save_incremental(
                        result,
                        output_path,
                    )

                    completed_settings.add(
                        setting_key
                    )

                    print()
                    print(
                        result[
                            [
                                "alpha",
                                "validation_fpr",
                                "test_fpr",
                                "violation_ratio",
                                "seen_f1",
                                "ood_f1",
                            ]
                        ].to_string(
                            index=False
                        )
                    )

                    print(
                        f"Saved -> "
                        f"{output_path}"
                    )

                    del result

                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    print()
    print("=" * 72)
    print(
        f"Batch complete: {output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run split-aware GNN experiments "
            "and save incremental results."
        )
    )

    parser.add_argument(
        "--windows",
        nargs="+",
        default=DEFAULT_WINDOWS,
        choices=["1m", "5m"],
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=["base", "ours"],
    )

    parser.add_argument(
        "--backbones",
        nargs="+",
        default=DEFAULT_BACKBONES,
        choices=[
            "gcn",
            "gin",
            "sage",
            "gat",
        ],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_batch(
        windows=args.windows,
        methods=args.methods,
        backbones=args.backbones,
        seeds=args.seeds,
        epochs=args.epochs,
        device=args.device,
        output_path=args.output,
    )