import numpy as np


def calibrate_target_fpr(
    benign_scores: np.ndarray,
    alpha: float,
) -> float:
    """
    Calibrate a conservative decision threshold using
    benign validation scores only.

    Prediction rule:
        malicious if score >= threshold

    The returned threshold guarantees that the empirical
    validation FPR does not exceed alpha, including ties.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    scores = np.asarray(
        benign_scores,
        dtype=np.float64,
    ).reshape(-1)

    if scores.size == 0:
        raise ValueError(
            "benign_scores must not be empty."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "benign_scores contains NaN or infinity."
        )

    num_benign = len(scores)

    max_false_positives = int(
        np.floor(alpha * num_benign)
    )

    sorted_scores = np.sort(
        scores
    )[::-1]

    if max_false_positives == 0:
        threshold = np.nextafter(
            sorted_scores[0],
            np.inf,
        )
    else:
        boundary_score = sorted_scores[
            max_false_positives
        ]

        threshold = np.nextafter(
            boundary_score,
            np.inf,
        )

    realized_fpr = float(
        np.mean(scores >= threshold)
    )

    if realized_fpr > alpha:
        raise RuntimeError(
            "Calibrated threshold exceeds target FPR."
        )

    return float(threshold)


def compute_fpr(
    benign_scores: np.ndarray,
    threshold: float,
) -> float:
    """
    Compute empirical false positive rate at a fixed threshold.
    """
    scores = np.asarray(
        benign_scores,
        dtype=np.float64,
    ).reshape(-1)

    if scores.size == 0:
        raise ValueError(
            "benign_scores must not be empty."
        )

    return float(
        np.mean(scores >= threshold)
    )