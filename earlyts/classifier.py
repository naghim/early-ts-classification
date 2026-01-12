"""Early Time Series Classifier using ROCKET variants"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import Rocket, MiniRocket, MultiRocket
from sktime.datatypes._panel._convert import from_2d_array_to_nested

from .utils import normalize_input

USE_RANDOM_INTERVALS = False


class EarlyTimeSeriesClassifier:
    """
    Main class for Early Time Series Classification with ROCKET variants and confidence calibration
    """
    
    def __init__(self, rocket_variant='minirocket', calibrate=True, random_state=42):
        self.rocket_variant = rocket_variant
        self.calibrate = calibrate
        self.random_state = random_state
        self.transformers = {}
        self.classifiers = {}
        self.calibrators = {}
        self.scalers = {}
        
    def _get_transformer(self, variant):
        """Initialize the appropriate ROCKET transformer"""
        if variant.lower() == 'rocket':
            return Rocket(random_state=self.random_state)
        elif variant.lower() == 'minirocket':
            return MiniRocket(random_state=self.random_state)
        elif variant.lower() == 'multirocket':
            return MultiRocket(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown ROCKET variant: {variant}")
    
    def fit(self, X_train, y_train, percentages=None):
        """
        Fit models at different observation percentages
        
        Parameters:
        - X_train: List of time series (each shape [n_timesteps])
        - y_train: Labels
        - percentages: List of percentages to evaluate [10, 20, ..., 100]
        """
        if percentages is None:
            percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            
        self.percentages = percentages

        # Normalize inputs and validate shapes early to provide clearer errors
        X_train, y_train = normalize_input(X_train, y_train, name="train")

        print(f"Training {self.rocket_variant} at different observation percentages...")
        
        for p in percentages:
            print(f"  {p}%...", end=" ")
            
            # Get partial time series
            X_partial = self._get_partial_series(X_train, p)
            
            # Initialize and fit transformer
            transformer = self._get_transformer(self.rocket_variant)
            # sktime's ROCKET transformers expect nested data (pd.DataFrame of Series)
            # Convert from 2D numpy array (n_samples, n_timesteps) to nested format
            if isinstance(X_partial, np.ndarray) and X_partial.ndim == 2:
                X_partial_nested = from_2d_array_to_nested(X_partial)
            else:
                X_partial_nested = X_partial

            X_transformed = transformer.fit_transform(X_partial_nested)
            self.transformers[p] = transformer
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_transformed)
            self.scalers[p] = scaler
            
            # Train classifier
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            # Validate that number of samples match before calling sklearn
            if X_scaled.shape[0] != y_train.shape[0]:
                raise ValueError(
                    f"After transformation for {p}% the number of samples in X is "
                    f"{X_scaled.shape[0]} but y_train has length {y_train.shape[0]}. "
                    "This indicates an unexpected shape change during transformation."
                )
            classifier.fit(X_scaled, y_train)
            self.classifiers[p] = classifier
            
            # Calibrate probabilities if requested
            if self.calibrate:
                calibrator = CalibratedClassifierCV(classifier, method='isotonic', cv=3)
                calibrator.fit(X_scaled, y_train)
                self.calibrators[p] = calibrator
            
            print("✓")
        
        return self
    
    def _get_partial_series(self, X, percentage):
        """Extract first n% of each time series"""
        # TODO: Is it worth getting random intervals instead of the "first n%"?
        if USE_RANDOM_INTERVALS:
            n_timesteps = int(X.shape[1] * percentage / 100)
            start_idx = np.random.randint(0, X.shape[1] - n_timesteps + 1)
            print(f"    Using random interval from {start_idx} to {start_idx + n_timesteps} out of {X.shape[1]}")
            return X[:, start_idx:start_idx + n_timesteps]

        n_timesteps = int(X.shape[1] * percentage / 100)
        print(f"    Using first {n_timesteps} timesteps out of {X.shape[1]}")
        return X[:, :n_timesteps]
    
    def predict_probabilities(self, X, percentage):
        """Predict probabilities for given observation percentage"""
        if percentage not in self.transformers:
            raise ValueError(f"Model not trained for {percentage}%")
        
        X_partial = self._get_partial_series(X, percentage)

        # convert to nested if needed
        if isinstance(X_partial, np.ndarray) and X_partial.ndim == 2:
            X_partial_nested = from_2d_array_to_nested(X_partial)
        else:
            X_partial_nested = X_partial

        X_transformed = self.transformers[percentage].transform(X_partial_nested)
        X_scaled = self.scalers[percentage].transform(X_transformed)
        
        if self.calibrate and percentage in self.calibrators:
            return self.calibrators[percentage].predict_proba(X_scaled)
        else:
            # Convert decision function to probabilities
            classifier = self.classifiers[percentage]
            decision_scores = classifier.decision_function(X_scaled)

            # Softmax conversion
            exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def predict(self, X, percentage, threshold=0.0):
        """Predict classes with optional confidence threshold"""
        probabilities = self.predict_probabilities(X, percentage)
        predictions = np.argmax(probabilities, axis=1)
        max_probs = np.max(probabilities, axis=1)
        
        # Apply confidence threshold
        if threshold > 0:
            predictions[max_probs < threshold] = -1  # -1 indicates "not confident enough"
        
        return predictions, max_probs
