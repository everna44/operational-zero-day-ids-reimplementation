import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


EPOCHS = 80
HIDDEN_DIM = 128
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024

def set_seed(
    seed: int = RANDOM_STATE,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def get_feature_columns(
    nodes: pd.DataFrame,
) -> list[str]:
    return [
        column
        for column in nodes.columns
        if column.startswith("out_")
        or column.startswith("in_")
    ]


def prepare_training_data(
    nodes: pd.DataFrame,
    device: str = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_columns = get_feature_columns(nodes)

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
        dtype=np.float32,
        copy=True,
    )

    y_train = (
        train["session_category"]
        .map({
            "BENIGN": 0,
            "SEEN": 1,
        })
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    if not np.isfinite(x_train).all():
        raise ValueError(
            "Training features contain NaN or infinity."
        )

    x_tensor = torch.from_numpy(
        x_train
    ).to(device)

    y_tensor = torch.from_numpy(
        y_train
    ).to(device)

    return x_tensor, y_tensor


def train_mlp(
    nodes: pd.DataFrame,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    hidden_dim: int = HIDDEN_DIM,
    dropout: float = DROPOUT,
    seed: int = RANDOM_STATE,
    device: str = DEVICE,
) -> TabularMLP:
    """
    Train the tabular MLP baseline.

    Training policy:
    - BENIGN -> 0
    - SEEN -> 1
    - OOD is excluded from training.
    - Class imbalance is handled with BCE pos_weight.

    Adam, batch size, and pos_weight are clean-reimplementation
    choices because the original detailed MLP training settings
    are no longer recoverable.
    """
    set_seed(seed)

    feature_columns = get_feature_columns(nodes)

    if not feature_columns:
        raise ValueError(
            "No directional feature columns were found."
        )

    x_train, y_train = prepare_training_data(
        nodes,
        device="cpu",
    )

    dataset = TensorDataset(
        x_train,
        y_train,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = TabularMLP(
        input_dim=len(feature_columns),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    num_positive = int(
        (y_train == 1).sum().item()
    )

    num_negative = int(
        (y_train == 0).sum().item()
    )

    if num_positive == 0 or num_negative == 0:
        raise ValueError(
            "Training data must contain both "
            "BENIGN and SEEN classes."
        )

    pos_weight = torch.tensor(
        [num_negative / num_positive],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    print(
        f"Device: {device} | "
        f"Train: {len(dataset):,} | "
        f"BENIGN: {num_negative:,} | "
        f"SEEN: {num_positive:,} | "
        f"pos_weight: {pos_weight.item():.3f}"
    )

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(x_batch)

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()
            optimizer.step()

            batch_size_actual = len(x_batch)

            total_loss += (
                loss.item()
                * batch_size_actual
            )

            total_samples += batch_size_actual

        average_loss = (
            total_loss / total_samples
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):
            print(
                f"Epoch "
                f"{epoch:03d}/{epochs} | "
                f"Loss: {average_loss:.6f}"
            )

    return model


def predict_mlp_score(
    model: TabularMLP,
    nodes: pd.DataFrame,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Return P(malicious) for supplied host-session nodes.
    """
    feature_columns = get_feature_columns(nodes)

    x = nodes[
        feature_columns
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if not np.isfinite(x).all():
        raise ValueError(
            "Prediction features contain NaN or infinity."
        )

    device = next(
        model.parameters()
    ).device

    model.eval()

    scores = []

    with torch.no_grad():
        for start in range(
            0,
            len(x),
            batch_size,
        ):
            end = start + batch_size

            x_batch = torch.from_numpy(
                x[start:end]
            ).to(device)

            logits = model(x_batch)

            probabilities = torch.sigmoid(
                logits
            )

            scores.append(
                probabilities
                .cpu()
                .numpy()
            )

    return np.concatenate(scores)