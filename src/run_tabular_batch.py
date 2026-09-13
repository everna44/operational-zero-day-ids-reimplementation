import argparse
import gc
from pathlib import Path

import pandas as pd
import torch

from src.run_tabular_experiment import (
    TARGET_FPRS,
    run_tabular_experiment,
)


DEFAULT_WINDOWS = [
    "1m",
    "5m",
]

DEFAULT_MODELS = [
    "lr",
    "rf",
    "mlp",
]

DEFAULT_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

DEFAULT_OUTPUT = Path(
    "results/tabular_results.csv"
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
        "model",
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
) -> set[tuple[str, str, int]]:
    if not output_path.exists():
        return set()

    results = pd.read_csv(
        output_path
    )

    required_columns = {
        "window",
        "model",
        "seed",
        "alpha",
    }

    if not required_columns.issubset(
        results.columns
    ):
        return set()

    expected_alphas = set(
        TARGET_FPRS
    )

    completed = set()

    for key, group in results.groupby(
        [
            "window",
            "model",
            "seed",
        ]
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
                    int(key[2]),
                )
            )

    return completed


def run_batch(
    windows: list[str],
    models: list[str],
    seeds: list[int],
    epochs: int = 80,
    device: str = "cuda",
    output_path: Path = DEFAULT_OUTPUT,
) -> None:
    total = (
        len(windows)
        * len(models)
        * len(seeds)
    )

    completed_settings = (
        load_completed_settings(
            output_path
        )
    )

    current = 0

    for window in windows:
        for model_name in models:
            for seed in seeds:
                current += 1

                setting_key = (
                    window,
                    model_name,
                    seed,
                )

                if setting_key in completed_settings:
                    print(
                        f"[{current}/{total}] SKIP | "
                        f"window={window} | "
                        f"model={model_name} | "
                        f"seed={seed}"
                    )
                    continue

                print()
                print("=" * 72)

                print(
                    f"[{current}/{total}] "
                    f"window={window} | "
                    f"model={model_name} | "
                    f"seed={seed}"
                )

                result = run_tabular_experiment(
                    window=window,
                    model_name=model_name,
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
                    f"Saved -> {output_path}"
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
            "Run tabular baseline experiments "
            "and save incremental results."
        )
    )

    parser.add_argument(
        "--windows",
        nargs="+",
        choices=[
            "1m",
            "5m",
        ],
        default=DEFAULT_WINDOWS,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "lr",
            "rf",
            "mlp",
        ],
        default=DEFAULT_MODELS,
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
        models=args.models,
        seeds=args.seeds,
        epochs=args.epochs,
        device=args.device,
        output_path=args.output,
    )