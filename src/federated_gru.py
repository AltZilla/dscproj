"""
Federated Learning-Based GRU Network
======================================
Privacy-preserving load forecasting through decentralized model training
and FedAvg parameter aggregation.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd


# ---------------------------------------------------------------------------
# GRU Model
# ---------------------------------------------------------------------------
class GRUForecaster(nn.Module):
    """GRU-based load forecasting model."""

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        gru_out, _ = self.gru(x)
        # Use last hidden state
        last_hidden = gru_out[:, -1, :]
        return self.fc(last_hidden)


# ---------------------------------------------------------------------------
# Dataset Preparation
# ---------------------------------------------------------------------------
def create_sequences(data: np.ndarray, targets: np.ndarray,
                     seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences for time-series forecasting."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(targets[i + seq_length])
    return np.array(X), np.array(y)


def prepare_client_data(home_df: pd.DataFrame, latent_features: np.ndarray,
                        seq_length: int, batch_size: int) -> DataLoader:
    """Prepare DataLoader for a single home (FL client)."""
    # Target: total power at next step
    targets = home_df["total_power_kw"].values.astype(np.float32)

    # Features: latent representations
    n = min(len(latent_features), len(targets))
    features = latent_features[:n].astype(np.float32)
    targets = targets[:n]

    X, y = create_sequences(features, targets, seq_length)
    if len(X) == 0:
        return None

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# ---------------------------------------------------------------------------
# Federated Learning (FedAvg)
# ---------------------------------------------------------------------------
class FederatedServer:
    """Central server for federated averaging."""

    def __init__(self, global_model: GRUForecaster):
        self.global_model = global_model

    def aggregate(self, client_models: List[GRUForecaster],
                  client_sizes: List[int]):
        """FedAvg: weighted average of client model parameters."""
        total_size = sum(client_sizes)
        global_dict = self.global_model.state_dict()

        # Zero out global params
        for key in global_dict:
            global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)

        # Weighted sum
        for model, size in zip(client_models, client_sizes):
            weight = size / total_size
            for key, param in model.state_dict().items():
                global_dict[key] += param.float() * weight

        self.global_model.load_state_dict(global_dict)

    def get_global_model(self) -> GRUForecaster:
        return copy.deepcopy(self.global_model)


class FederatedClient:
    """A single federated learning client (one smart home)."""

    def __init__(self, client_id: int, train_loader: DataLoader,
                 learning_rate: float, device: str = "cpu"):
        self.client_id = client_id
        self.train_loader = train_loader
        self.lr = learning_rate
        self.device = device
        self.data_size = len(train_loader.dataset) if train_loader else 0

    def train(self, model: GRUForecaster, local_epochs: int) -> GRUForecaster:
        """Train a copy of the global model on local data."""
        local_model = copy.deepcopy(model).to(self.device)
        optimizer = optim.Adam(local_model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        local_model.train()
        for epoch in range(local_epochs):
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                pred = local_model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

        return local_model


# ---------------------------------------------------------------------------
# Training Orchestration
# ---------------------------------------------------------------------------
def train_federated(config: dict, train_df: pd.DataFrame,
                    train_latent: np.ndarray,
                    val_df: pd.DataFrame,
                    val_latent: np.ndarray,
                    device: str = "cpu") -> Tuple[GRUForecaster, dict]:
    """Full federated training loop."""
    cfg = config["federated_gru"]
    seq_len = cfg["sequence_length"]
    actual_input_dim = train_latent.shape[1]

    # Initialize global model
    global_model = GRUForecaster(
        input_dim=actual_input_dim,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        output_dim=cfg["output_dim"],
    ).to(device)

    server = FederatedServer(global_model)

    # Prepare clients
    home_ids = train_df["home_id"].unique()
    clients = []
    home_latent_map = {}

    # Split latent features by home
    start_idx = 0
    for hid in sorted(home_ids):
        home_mask = train_df["home_id"] == hid
        home_size = home_mask.sum()
        home_latent = train_latent[start_idx:start_idx + home_size]
        home_df = train_df[home_mask].reset_index(drop=True)
        start_idx += home_size

        loader = prepare_client_data(home_df, home_latent, seq_len, cfg["batch_size"])
        if loader is not None:
            client = FederatedClient(hid, loader, cfg["learning_rate"], device)
            clients.append(client)
            home_latent_map[hid] = home_latent

    print(f"Federated GRU: {len(clients)} clients, {cfg['num_rounds']} rounds")

    history = {"round_loss": []}

    for round_num in range(cfg["num_rounds"]):
        # Select subset of clients
        n_selected = max(1, int(len(clients) * cfg["client_fraction"]))
        selected = np.random.choice(len(clients), n_selected, replace=False)

        # Local training
        client_models = []
        client_sizes = []
        for idx in selected:
            client = clients[idx]
            local_model = client.train(server.get_global_model(), cfg["local_epochs"])
            client_models.append(local_model)
            client_sizes.append(client.data_size)

        # Aggregate
        server.aggregate(client_models, client_sizes)

        # Evaluate on validation set
        val_loss = evaluate_model(server.global_model, val_df, val_latent,
                                  seq_len, device)
        history["round_loss"].append(val_loss)

        if (round_num + 1) % 5 == 0 or round_num == 0:
            print(f"  Round {round_num+1}/{cfg['num_rounds']}  Val Loss: {val_loss:.4f}")

    return server.global_model, history


def evaluate_model(model: GRUForecaster, df: pd.DataFrame,
                   latent: np.ndarray, seq_length: int,
                   device: str = "cpu") -> float:
    """Evaluate model on a dataset. Returns MSE."""
    model.eval()
    targets = df["total_power_kw"].values.astype(np.float32)
    n = min(len(latent), len(targets))
    features = latent[:n].astype(np.float32)
    targets = targets[:n]

    X, y = create_sequences(features, targets, seq_length)
    if len(X) == 0:
        return float("inf")

    X_tensor = torch.from_numpy(X).to(device)
    y_tensor = torch.from_numpy(y).unsqueeze(1).to(device)

    with torch.no_grad():
        pred = model(X_tensor)
        mse = nn.functional.mse_loss(pred, y_tensor).item()
    return mse


def predict(model: GRUForecaster, latent: np.ndarray,
            seq_length: int, device: str = "cpu") -> np.ndarray:
    """Generate predictions from latent features."""
    model.eval()
    features = latent.astype(np.float32)
    # Use sliding windows
    X = []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
    X = np.array(X)
    X_tensor = torch.from_numpy(X).to(device)

    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()
    return preds


def save_gru_model(model: GRUForecaster, path: Optional[str] = None):
    if path is None:
        path = Path(__file__).resolve().parent.parent / "outputs" / "models" / "federated_gru.pt"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Federated GRU model saved to {path}")


def run_federated_gru(config: dict, preprocessed: dict,
                      vae_results: dict, device: str = "cpu") -> dict:
    """Full federated GRU pipeline."""
    print("Training Federated GRU...")
    model, history = train_federated(
        config,
        preprocessed["train"], vae_results["train_latent"],
        preprocessed["val"], vae_results["val_latent"],
        device
    )

    # Predict on test set
    cfg = config["federated_gru"]
    test_preds = predict(model, vae_results["test_latent"], cfg["sequence_length"], device)

    # Compute metrics
    test_targets = preprocessed["test"]["total_power_kw"].values
    test_targets = test_targets[cfg["sequence_length"]:]
    n = min(len(test_preds), len(test_targets))
    test_preds = test_preds[:n]
    test_targets = test_targets[:n]

    mae = np.mean(np.abs(test_preds - test_targets))
    rmse = np.sqrt(np.mean((test_preds - test_targets) ** 2))
    # Filter out near-zero targets to avoid division explosion
    nonzero_mask = np.abs(test_targets) > 0.1
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((test_targets[nonzero_mask] - test_preds[nonzero_mask])
                              / test_targets[nonzero_mask])) * 100
    else:
        mape = 0.0

    print(f"\nFederated GRU Test Metrics:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    save_gru_model(model)

    return {
        "model": model,
        "history": history,
        "test_predictions": test_preds,
        "test_targets": test_targets,
        "metrics": {"mae": mae, "rmse": rmse, "mape": mape},
    }
