"""
Variational Autoencoder (VAE) Module
=====================================
Performs latent feature extraction and dimensionality reduction of IoT
energy data, capturing key behavioral patterns and usage correlations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd


# ---------------------------------------------------------------------------
# VAE Architecture
# ---------------------------------------------------------------------------
class VAE(nn.Module):
    """Variational Autoencoder for energy data representation learning."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             kl_weight: float = 0.5) -> Tuple[torch.Tensor, float, float]:
    """Combined reconstruction + KL divergence loss."""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + kl_weight * kl_loss
    return total, recon_loss.item(), kl_loss.item()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def prepare_vae_data(df: pd.DataFrame, feature_columns: list,
                     batch_size: int = 128) -> DataLoader:
    """Convert DataFrame to DataLoader for VAE training."""
    data = df[feature_columns].values.astype(np.float32)
    tensor = torch.from_numpy(data)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def train_vae(config: dict, train_loader: DataLoader,
              val_loader: DataLoader,
              device: str = "cpu") -> Tuple[VAE, dict]:
    """Train the VAE model."""
    cfg = config["vae"]

    model = VAE(cfg["input_dim"], cfg["hidden_dim"], cfg["latent_dim"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    history = {"train_loss": [], "val_loss": [], "recon_loss": [], "kl_loss": []}

    for epoch in range(cfg["epochs"]):
        # Training
        model.train()
        epoch_loss, epoch_recon, epoch_kl = 0, 0, 0
        n_batches = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, rl, kl = vae_loss(recon, batch, mu, logvar, cfg["kl_weight"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_recon += rl
            epoch_kl += kl
            n_batches += 1

        avg_train = epoch_loss / n_batches
        history["train_loss"].append(avg_train)
        history["recon_loss"].append(epoch_recon / n_batches)
        history["kl_loss"].append(epoch_kl / n_batches)

        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss, _, _ = vae_loss(recon, batch, mu, logvar, cfg["kl_weight"])
                val_loss += loss.item()
                n_val += 1
        avg_val = val_loss / max(n_val, 1)
        history["val_loss"].append(avg_val)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{cfg['epochs']}  "
                  f"Train: {avg_train:.4f}  Val: {avg_val:.4f}  "
                  f"Recon: {history['recon_loss'][-1]:.4f}  KL: {history['kl_loss'][-1]:.4f}")

    return model, history


def extract_latent_features(model: VAE, df: pd.DataFrame,
                            feature_columns: list,
                            device: str = "cpu") -> np.ndarray:
    """Extract latent representations from trained VAE."""
    model.eval()
    data = torch.from_numpy(df[feature_columns].values.astype(np.float32)).to(device)

    with torch.no_grad():
        mu, _ = model.encode(data)
    return mu.cpu().numpy()


def save_vae_model(model: VAE, path: Optional[str] = None):
    if path is None:
        path = Path(__file__).resolve().parent.parent / "outputs" / "models" / "vae.pt"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"VAE model saved to {path}")


def run_vae(config: dict, preprocessed: dict, device: str = "cpu") -> dict:
    """Full VAE pipeline: prepare data, train, extract features."""
    cfg = config["vae"]
    feature_cols = preprocessed["feature_columns"]

    # Adjust input dim to match actual features
    actual_input_dim = len(feature_cols)
    cfg["input_dim"] = actual_input_dim

    print(f"VAE: input_dim={actual_input_dim}, latent_dim={cfg['latent_dim']}")

    train_loader = prepare_vae_data(preprocessed["train"], feature_cols, cfg["batch_size"])
    val_loader = prepare_vae_data(preprocessed["val"], feature_cols, cfg["batch_size"])

    print("Training VAE...")
    model, history = train_vae(config, train_loader, val_loader, device)

    # Extract latent features for all data
    print("Extracting latent features...")
    train_latent = extract_latent_features(model, preprocessed["train"], feature_cols, device)
    val_latent = extract_latent_features(model, preprocessed["val"], feature_cols, device)
    test_latent = extract_latent_features(model, preprocessed["test"], feature_cols, device)

    save_vae_model(model)

    return {
        "model": model,
        "history": history,
        "train_latent": train_latent,
        "val_latent": val_latent,
        "test_latent": test_latent,
    }
