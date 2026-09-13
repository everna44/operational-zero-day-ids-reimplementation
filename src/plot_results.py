from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


RESULTS_PATH = Path("results/family_comparison.csv")
FIGURE_DIR = Path("figures")

FAMILY_ORDER = [
    "tabular",
    "gnn_base",
    "gnn_ours",
]

FAMILY_LABELS = {
    "tabular": "Tabular",
    "gnn_base": "GNN-base",
    "gnn_ours": "GNN-ours",
}

FAMILY_COLORS = {
    "tabular": "#3B82B8",
    "gnn_base": "#E57A32",
    "gnn_ours": "#2E9B61",
}

ALPHA_MARKERS = {
    0.001: "o",
    0.01: "^",
    0.05: "s",
}


def configure_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#222222",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.titlesize": 19,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)

    required = {
        "window",
        "alpha",
        "family",
        "ood_f1_mean",
        "violation_ratio_p95",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    return df


def save_figure(
    fig: plt.Figure,
    filename: str,
) -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = FIGURE_DIR / f"{filename}.png"
    pdf_path = FIGURE_DIR / f"{filename}.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"Saved -> {png_path}")
    print(f"Saved -> {pdf_path}")


def plot_tradeoff(
    df: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.3),
        sharey=True,
    )

    windows = [
        "1m",
        "5m",
    ]

    for ax, window in zip(
        axes,
        windows,
    ):
        subset = df[
            df["window"] == window
        ].copy()

        for family in FAMILY_ORDER:
            family_df = subset[
                subset["family"] == family
            ]

            for _, row in family_df.iterrows():
                alpha = float(
                    row["alpha"]
                )

                x = float(
                    row[
                        "violation_ratio_p95"
                    ]
                )

                y = float(
                    row[
                        "ood_f1_mean"
                    ]
                )

                ax.scatter(
                    x,
                    y,
                    s=105,
                    marker=ALPHA_MARKERS[
                        alpha
                    ],
                    color=FAMILY_COLORS[
                        family
                    ],
                    edgecolor="white",
                    linewidth=1.1,
                    zorder=3,
                )

                label = (
                    f"α={alpha:g}"
                )

                ax.annotate(
                    label,
                    (x, y),
                    xytext=(7, 7),
                    textcoords="offset points",
                    fontsize=8.5,
                    color=FAMILY_COLORS[
                        family
                    ],
                )

        ax.axvline(
            1.0,
            linestyle="--",
            linewidth=1.4,
            color="#5F6368",
            zorder=1,
        )

        ax.text(
            1.01,
            0.985,
            "operational target",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=9,
            color="#5F6368",
        )

        ax.set_title(
            f"{window} window",
            weight="bold",
            pad=12,
        )

        ax.set_xlabel(
            "p95 FPR violation ratio\n(lower is better)"
        )

        ax.grid(
            axis="both",
            linestyle="--",
            linewidth=0.7,
            alpha=0.20,
        )

        ax.set_ylim(
            -0.015,
            0.49,
        )

    axes[0].set_xlim(
        0.86,
        1.39,
    )

    axes[1].set_xlim(
        0.80,
        2.05,
    )

    axes[0].set_ylabel(
        "Mean OOD F1\n(higher is better)"
    )

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=(
                FAMILY_COLORS[family]
            ),
            markeredgecolor="white",
            markersize=9,
            label=FAMILY_LABELS[
                family
            ],
        )
        for family in FAMILY_ORDER
    ]

    alpha_handles = [
        Line2D(
            [0],
            [0],
            marker=ALPHA_MARKERS[
                alpha
            ],
            color="none",
            markerfacecolor="#555555",
            markeredgecolor="#555555",
            markersize=8,
            label=f"α = {alpha:g}",
        )
        for alpha in [
            0.001,
            0.01,
            0.05,
        ]
    ]

    fig.legend(
        handles=(
            family_handles
            + alpha_handles
        ),
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            -0.02,
        ),
    )

    fig.suptitle(
        "Detection–Robustness Trade-off",
        weight="bold",
        y=1.02,
    )

    fig.text(
        0.5,
        -0.065,
        (
            "Upper-left is preferable: "
            "higher OOD F1 with lower "
            "tail-risk amplification."
        ),
        ha="center",
        fontsize=10,
        color="#555555",
    )

    fig.tight_layout(
        rect=[
            0,
            0.08,
            1,
            0.98,
        ]
    )

    save_figure(
        fig,
        "detection_robustness_tradeoff",
    )

    plt.close(fig)


def plot_tail_risk_reduction(
    df: pd.DataFrame,
) -> None:
    settings = [
        ("1m", 0.001),
        ("1m", 0.01),
        ("1m", 0.05),
        ("5m", 0.001),
        ("5m", 0.01),
        ("5m", 0.05),
    ]

    labels = [
        f"{window}   α={alpha:g}"
        for window, alpha
        in settings
    ]

    fig, ax = plt.subplots(
        figsize=(11.5, 6.5)
    )

    y_positions = np.arange(
        len(settings)
    )[::-1]

    for y, (
        window,
        alpha,
    ) in zip(
        y_positions,
        settings,
    ):
        subset = df[
            (df["window"] == window)
            & (
                np.isclose(
                    df["alpha"],
                    alpha,
                )
            )
        ]

        values = {
            row["family"]:
                float(
                    row[
                        "violation_ratio_p95"
                    ]
                )
            for _, row
            in subset.iterrows()
        }

        tabular = values[
            "tabular"
        ]

        base = values[
            "gnn_base"
        ]

        ours = values[
            "gnn_ours"
        ]

        ax.plot(
            [base, ours],
            [y, y],
            color="#A9A9A9",
            linewidth=2.2,
            zorder=1,
        )

        ax.scatter(
            tabular,
            y,
            s=68,
            color=FAMILY_COLORS[
                "tabular"
            ],
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )

        ax.scatter(
            base,
            y,
            s=96,
            color=FAMILY_COLORS[
                "gnn_base"
            ],
            edgecolor="white",
            linewidth=1,
            zorder=4,
        )

        ax.scatter(
            ours,
            y,
            s=96,
            color=FAMILY_COLORS[
                "gnn_ours"
            ],
            edgecolor="white",
            linewidth=1,
            zorder=4,
        )

        ax.annotate(
            f"{tabular:.3f}",
            (tabular, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=FAMILY_COLORS[
                "tabular"
            ],
        )

        ax.annotate(
            f"{base:.3f}",
            (base, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=FAMILY_COLORS[
                "gnn_base"
            ],
        )

        ax.annotate(
            f"{ours:.3f}",
            (ours, y),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=FAMILY_COLORS[
                "gnn_ours"
            ],
        )

        delta = ours - base

        reduction = (
            (base - ours)
            / base
            * 100.0
        )

        ax.text(
            max(
                base,
                ours,
            ) + 0.045,
            y,
            (
                f"Δ {delta:+.3f}  "
                f"({reduction:.1f}% lower)"
            ),
            va="center",
            fontsize=9,
            color="#276749",
            weight="bold",
        )

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1.4,
        color="#5F6368",
    )

    ax.text(
        1.01,
        1.015,
        "operational target",
        transform=ax.get_xaxis_transform(),
        ha="left",
        fontsize=9,
        color="#5F6368",
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlim(
        0.72,
        2.12,
    )

    ax.set_xlabel(
        "p95 FPR violation ratio (lower is better)"
    )

    ax.set_title(
        "Tail-Risk Reduction: GNN-base → GNN-ours",
        weight="bold",
        pad=18,
    )

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.7,
        alpha=0.22,
    )

    ax.grid(
        axis="y",
        linestyle="-",
        linewidth=0.5,
        alpha=0.10,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=(
                FAMILY_COLORS[family]
            ),
            markeredgecolor="white",
            markersize=9,
            label=FAMILY_LABELS[
                family
            ],
        )
        for family in FAMILY_ORDER
    ]

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
    )

    fig.text(
        0.5,
        0.015,
        (
            "All six settings show lower "
            "p95 tail risk with GNN-ours "
            "than with GNN-base."
        ),
        ha="center",
        fontsize=10,
        color="#555555",
    )

    fig.tight_layout(
        rect=[
            0,
            0.05,
            1,
            1,
        ]
    )

    save_figure(
        fig,
        "tail_risk_reduction",
    )

    plt.close(fig)


def main() -> None:
    configure_style()

    df = load_results()

    plot_tradeoff(df)

    plot_tail_risk_reduction(
        df
    )


if __name__ == "__main__":
    main()