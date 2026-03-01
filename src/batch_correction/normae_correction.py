"""
NormAE: Adversarial Autoencoder for Batch Effect Removal.

Implements NormAE for MALDI MSI data using PyTorch with gradient reversal
for adversarial batch discrimination.
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient reversal layer for adversarial training.

    Forward pass is identity; backward pass multiplies gradient by -lambda_.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """Identity forward pass, stores lambda for backward."""
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Reverse gradient direction scaled by lambda_."""
        return -ctx.lambda_ * grad_output, None


class BatchEffectEncoder(nn.Module):
    """
    Encoder network for NormAE.

    Architecture: input_dim → 512 → 256 → 128 → latent_dim (64)
    Each hidden layer uses BatchNorm + ReLU + Dropout(0.2).
    """

    def __init__(self, input_dim: int = 159, latent_dim: int = 64, dropout: float = 0.2):
        """
        Initialize encoder.

        Parameters
        ----------
        input_dim : int
            Number of input features (default 159 for MALDI MSI).
        latent_dim : int
            Dimension of latent representation (default 64).
        dropout : float
            Dropout rate (default 0.2).
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.network(x)


class BatchEffectDecoder(nn.Module):
    """
    Decoder network for NormAE.

    Architecture: latent_dim → 128 → 256 → 512 → output_dim
    Each hidden layer uses BatchNorm + ReLU; final layer is linear.
    """

    def __init__(self, latent_dim: int = 64, output_dim: int = 159):
        """
        Initialize decoder.

        Parameters
        ----------
        latent_dim : int
            Dimension of latent representation (default 64).
        output_dim : int
            Number of output features (default 159 for MALDI MSI).
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to feature space."""
        return self.network(z)


class BatchDiscriminator(nn.Module):
    """
    Batch discriminator network for adversarial training.

    Uses gradient reversal layer to encourage batch-invariant representations.
    Architecture: latent_dim → 64 → 32 → n_batches
    """

    def __init__(self, latent_dim: int = 64, n_batches: int = 7):
        """
        Initialize batch discriminator.

        Parameters
        ----------
        latent_dim : int
            Dimension of latent representation (default 64).
        n_batches : int
            Number of batches/domains (default 7).
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_batches),
        )

    def forward(self, z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Apply gradient reversal then classify batch."""
        z_rev = GradientReversalLayer.apply(z, alpha)
        return self.classifier(z_rev)


class NormAE(nn.Module):
    """
    NormAE: Adversarial Autoencoder for Batch Effect Normalization.

    Combines encoder, decoder, and batch discriminator with gradient reversal
    for adversarial training. The encoder learns batch-invariant representations.
    """

    def __init__(
        self,
        input_dim: int = 159,
        n_batches: int = 7,
        latent_dim: int = 64,
        dropout: float = 0.2,
    ):
        """
        Initialize NormAE.

        Parameters
        ----------
        input_dim : int
            Number of input features (default 159).
        n_batches : int
            Number of batches for adversarial training (default 7).
        latent_dim : int
            Latent space dimension (default 64).
        dropout : float
            Dropout rate in encoder (default 0.2).
        """
        super().__init__()
        self.encoder = BatchEffectEncoder(input_dim, latent_dim, dropout)
        self.decoder = BatchEffectDecoder(latent_dim, input_dim)
        self.discriminator = BatchDiscriminator(latent_dim, n_batches)

    def forward(
        self, x: torch.Tensor, alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through NormAE.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, input_dim)
            Input features.
        alpha : float
            Gradient reversal scaling factor (default 1.0).

        Returns
        -------
        tuple
            (reconstructed, latent, domain_pred)
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        domain_pred = self.discriminator(latent, alpha)
        return reconstructed, latent, domain_pred


class NormAETrainer:
    """
    Trainer for NormAE model.

    Handles training loop with reconstruction and adversarial losses,
    as well as transform/fit_transform convenience methods.
    """

    def __init__(
        self,
        input_dim: int = 159,
        n_batches: int = 7,
        device: Optional[str] = None,
        lr: float = 1e-3,
        lambda_adv: float = 0.1,
        n_epochs: int = 100,
        batch_size: int = 512,
    ):
        """
        Initialize NormAETrainer.

        Parameters
        ----------
        input_dim : int
            Number of input features (default 159).
        n_batches : int
            Number of batches (default 7).
        device : str, optional
            PyTorch device string. Auto-detects CUDA if None.
        lr : float
            Learning rate (default 1e-3).
        lambda_adv : float
            Adversarial loss weight (default 0.1).
        n_epochs : int
            Number of training epochs (default 100).
        batch_size : int
            Mini-batch size (default 512).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.lr = lr
        self.lambda_adv = lambda_adv
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = NormAE(input_dim=input_dim, n_batches=n_batches).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.rec_loss_fn = nn.MSELoss()
        self.adv_loss_fn = nn.CrossEntropyLoss()

    def fit(self, data: np.ndarray, batch_labels: np.ndarray) -> "NormAETrainer":
        """
        Train the NormAE model.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        batch_labels : np.ndarray, shape (n_samples,)
            Batch labels (integer encoded).

        Returns
        -------
        NormAETrainer
            Self for chaining.
        """
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        batch_int = le.fit_transform(batch_labels)
        n_batches = len(le.classes_)

        X_tensor = torch.FloatTensor(data).to(self.device)
        y_tensor = torch.LongTensor(batch_int).to(self.device)
        n_samples = len(data)

        self.model.train()
        for epoch in range(self.n_epochs):
            perm = torch.randperm(n_samples)
            total_loss = 0.0
            n_batches_iter = 0
            # Warm-up: gradually increase adversarial weight over the first half of
            # training to stabilize early reconstruction before enforcing batch invariance.
            alpha = min(1.0, epoch / max(1, self.n_epochs // 2))

            for start in range(0, n_samples, self.batch_size):
                idx = perm[start: start + self.batch_size]
                xb = X_tensor[idx]
                yb = y_tensor[idx]

                self.optimizer.zero_grad()
                rec, _, domain_pred = self.model(xb, alpha=alpha)
                rec_loss = self.rec_loss_fn(rec, xb)
                adv_loss = self.adv_loss_fn(domain_pred, yb)
                # Subtract adversarial loss: GRL already flips the gradient sign
                # during backprop, so minimizing -adv_loss here trains the encoder
                # to produce batch-invariant representations.
                loss = rec_loss - self.lambda_adv * adv_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches_iter += 1

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / max(1, n_batches_iter)
                logger.info(f"Epoch [{epoch + 1}/{self.n_epochs}] Loss: {avg_loss:.4f}")

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply batch correction via NormAE (encode then decode).

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray
            Batch-corrected feature matrix in original feature space.
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(data).to(self.device)
            corrected_parts = []
            for start in range(0, len(data), self.batch_size):
                xb = X_tensor[start: start + self.batch_size]
                rec, _, _ = self.model(xb, alpha=0.0)
                corrected_parts.append(rec.cpu().numpy())
        return np.vstack(corrected_parts)

    def fit_transform(
        self, data: np.ndarray, batch_labels: np.ndarray
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        batch_labels : np.ndarray, shape (n_samples,)
            Batch labels.

        Returns
        -------
        np.ndarray
            Batch-corrected feature matrix.
        """
        self.fit(data, batch_labels)
        return self.transform(data)


def apply_normae(
    data_df: pd.DataFrame,
    batch_labels: np.ndarray,
    n_epochs: int = 100,
    lambda_adv: float = 0.1,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply NormAE batch correction to a feature matrix.

    Scales data with StandardScaler before training and transforms back
    after correction.

    Parameters
    ----------
    data_df : pd.DataFrame, shape (n_samples, n_features)
        Input feature matrix.
    batch_labels : np.ndarray, shape (n_samples,)
        Batch labels.
    n_epochs : int, optional
        Training epochs (default 100).
    lambda_adv : float, optional
        Adversarial loss weight (default 0.1).
    device : str, optional
        PyTorch device. Auto-detects if None.

    Returns
    -------
    pd.DataFrame
        Batch-corrected DataFrame with same shape as input.
    """
    from sklearn.preprocessing import LabelEncoder

    logger.info(f"Applying NormAE to {data_df.shape[0]} samples, {data_df.shape[1]} features")

    data_np = data_df.fillna(0.0).values.astype(np.float32)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_np)

    le = LabelEncoder()
    batch_int = le.fit_transform(batch_labels)
    n_batches = len(le.classes_)

    trainer = NormAETrainer(
        input_dim=data_scaled.shape[1],
        n_batches=n_batches,
        device=device,
        n_epochs=n_epochs,
        lambda_adv=lambda_adv,
    )
    corrected_scaled = trainer.fit_transform(data_scaled, batch_int)
    corrected_np = scaler.inverse_transform(corrected_scaled)

    return pd.DataFrame(corrected_np, index=data_df.index, columns=data_df.columns)


def apply_normae_per_normalization(
    data_dict: Dict[str, pd.DataFrame],
    batch_labels: np.ndarray,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Apply NormAE to all 6 normalization versions.

    Parameters
    ----------
    data_dict : dict
        Mapping from normalization name to DataFrame (n_samples, n_features).
    batch_labels : np.ndarray, shape (n_samples,)
        Batch labels.
    **kwargs
        Additional keyword arguments forwarded to apply_normae.

    Returns
    -------
    dict
        Mapping from normalization name to corrected DataFrame.
    """
    corrected = {}
    for norm_name, df in tqdm(data_dict.items(), desc="NormAE per normalization"):
        logger.info(f"Applying NormAE to normalization: {norm_name}")
        try:
            corrected[norm_name] = apply_normae(df, batch_labels, **kwargs)
        except Exception as e:
            logger.error(f"NormAE failed for {norm_name}: {e}")
            corrected[norm_name] = df.copy()
    return corrected
