"""Vanilla Transformer for Early Time Series Classification"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ._debug import debug_print
from .utils import normalize_input


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerTSClassifier(nn.Module):
    """Vanilla Transformer encoder for univariate time series classification.

    Architecture:
        input (batch, seq_len, 1)
        -> Linear projection to d_model
        -> Positional encoding
        -> TransformerEncoder (nhead, num_layers)
        -> Global average pooling over time
        -> Linear classification head
    """

    def __init__(
        self,
        n_classes,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        x = self.input_proj(x)          # (batch, seq_len, d_model)
        x = self.pos_encoder(x)         # (batch, seq_len, d_model)
        x = self.transformer_encoder(x) # (batch, seq_len, d_model)
        x = x.mean(dim=1)               # (batch, d_model)  — global average pooling
        return self.classifier(x)       # (batch, n_classes)


class EarlyTransformerClassifier:
    """Wrapper that trains one TransformerTSClassifier per observation percentage,
    matching the interface of EarlyTimeSeriesClassifier so it can be used with
    the existing EarlyClassificationEvaluator.
    """

    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        lr=1e-3,
        epochs=100,
        batch_size=64,
        patience=10,
        device=None,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.models = {}
        self.label_map = None   # str/int labels → 0-indexed ints
        self.n_classes = None
        self.percentages = None

    def fit(self, X_train, y_train, percentages=None):
        if percentages is None:
            percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.percentages = percentages

        X_train, y_train = normalize_input(X_train, y_train, name="train")

        # Build label mapping
        unique_labels = np.unique(y_train)
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.n_classes = len(unique_labels)
        y_mapped = np.array([self.label_map[l] for l in y_train])

        debug_print(f"Training Transformer at different observation percentages...")

        for p in percentages:
            debug_print(f"  {p}%...", end=" ", flush=True)
            X_partial = self._get_partial_series(X_train, p)
            model = self._train_single(X_partial, y_mapped)
            self.models[p] = model
            debug_print("✓")

        return self

    def predict_probabilities(self, X, percentage):
        if percentage not in self.models:
            raise ValueError(f"Model not trained for {percentage}%")

        X_partial = self._get_partial_series(X, percentage)
        model = self.models[percentage]
        model.eval()

        X_t = torch.tensor(X_partial, dtype=torch.float32).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            logits = model(X_t)
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def predict(self, X, percentage, threshold=0.0):
        probabilities = self.predict_probabilities(X, percentage)
        predictions = np.argmax(probabilities, axis=1)
        max_probs = np.max(probabilities, axis=1)

        if threshold > 0:
            predictions[max_probs < threshold] = -1

        return predictions, max_probs

    def _get_partial_series(self, X, percentage):
        n_timesteps = max(1, int(X.shape[1] * percentage / 100))
        return X[:, :n_timesteps]

    def _train_single(self, X, y):
        """Train a single transformer on the given (already-truncated) data."""
        model = TransformerTSClassifier(
            n_classes=self.n_classes,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_len=X.shape[1] + 10,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        # Build data loader
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, T, 1)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        wait = 0
        best_state = None

        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(dataset)

            # Early stopping
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(self.device)
        return model
