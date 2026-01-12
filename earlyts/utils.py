"""Utility functions for early time series classification"""

import warnings
import numpy as np


def normalize_input(X, y, name="test"):
    """Ensure input is a 2D numpy array of shape (n_samples, n_timesteps)"""
    X = np.asarray(X)
    y = np.asarray(y)

    # Handle single-sample X_test similar to fit
    if X.ndim == 1:
        if y.ndim != 0 and y.shape[0] > 1:
            raise ValueError(
                f"X_{name} appears to be a single time series with shape {X.shape}. "
                f"But y_{name} has length {y.shape[0]}.\n"
                f"Expected X_{name} to have shape (n_samples, n_timesteps). "
                f"If you have a single sample, wrap it as [X_{name}]."
            )
        X = X.reshape(1, -1)

    if X.ndim == 2 and X.shape[0] != y.shape[0]:
        if X.shape[1] == y.shape[0]:
            # common mistake: samples and timesteps were swapped
            warnings.warn(
                f"X_{name} appears to have samples and timesteps swapped. "
                f"Transposing X_{name} from {X.shape} to {(X.shape[1], X.shape[0])}. "
                f"Ensure X_{name} is shaped (n_samples, n_timesteps).",
            )
            X = X.T
        else:
            raise ValueError(
                f"Mismatch between number of samples in X_{name} and y_{name}: "
                f"X_{name}.shape={X.shape}, len(y_{name})={y.shape[0]}.\n"
                f"Ensure X_{name} is shaped (n_samples, n_timesteps) and y_{name} has length n_samples."
            )

    return X, y


def generate_synthetic_ts_data(n_samples=500, n_timesteps=200, n_classes=3):
    """Generate synthetic time series data for demonstration"""
    print("Generating synthetic time series data...")
    
    X = []
    y = []
    
    for i in range(n_samples):
        # Random class
        class_id = np.random.randint(n_classes)
        
        # Generate base pattern for this class
        if class_id == 0:
            # Class 0: Sine wave with noise
            t = np.linspace(0, 4*np.pi, n_timesteps)
            series = np.sin(t) + 0.3 * np.random.normal(size=n_timesteps)
        elif class_id == 1:
            # Class 1: Linear trend with noise
            series = np.linspace(0, 1, n_timesteps) + 0.3 * np.random.normal(size=n_timesteps)
        else:
            # Class 2: Random walk
            series = np.cumsum(np.random.normal(0, 0.1, n_timesteps))
        
        X.append(series)
        y.append(class_id)
    
    return np.array(X), np.array(y)
