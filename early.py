import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sktime.transformations.panel.rocket import Rocket, MiniRocket, MultiRocket
from sktime.datatypes._panel._convert import from_2d_array_to_nested

USE_RANDOM_INTERVALS = False

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

class EarlyClassificationExperiment:
    """
    Main experiment class to compare multiple ROCKET variants
    """
    
    def __init__(self, variants=None):
        if variants is None:
            variants = ['rocket', 'minirocket', 'multirocket']

        self.variants = variants
        self.models = {}
        self.results = {}
        
    def run_experiment(self, X_train, y_train, X_test, y_test, percentages=None):
        """Run complete experiment across all variants"""
        if percentages is None:
            percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            
        self.percentages = percentages
        
        for variant in self.variants:
            print(f"\n=== Training {variant.upper()} ===")
            
            # Train with and without calibration
            model_calibrated = EarlyTimeSeriesClassifier(
                rocket_variant=variant, calibrate=True
            )
            model_calibrated.fit(X_train, y_train, percentages)
            
            model_uncalibrated = EarlyTimeSeriesClassifier(
                rocket_variant=variant, calibrate=False
            )
            model_uncalibrated.fit(X_train, y_train, percentages)
            
            # Evaluate both versions
            results_calibrated = self._evaluate_model(
                model_calibrated, X_test, y_test, f"{variant}_calibrated"
            )
            results_uncalibrated = self._evaluate_model(
                model_uncalibrated, X_test, y_test, f"{variant}_uncalibrated"
            )
            
            self.models[f"{variant}_calibrated"] = model_calibrated
            self.models[f"{variant}_uncalibrated"] = model_uncalibrated
            self.results[f"{variant}_calibrated"] = results_calibrated
            self.results[f"{variant}_uncalibrated"] = results_uncalibrated
        
        return self.results
    
    def _evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model at all observation percentages"""
        results = {}
        X_test, y_test = normalize_input(X_test, y_test, name="test")

        for p in model.percentages:
            # Get predictions and probabilities
            predictions, confidences = model.predict(X_test, p)
            probabilities = model.predict_probabilities(X_test, p)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            brier_score = self._brier_score(y_test, probabilities)
            ece = self._expected_calibration_error(y_test, probabilities, confidences)
            
            # Store results
            results[p] = {
                'accuracy': accuracy,
                'brier_score': brier_score,
                'ece': ece,
                'mean_confidence': np.mean(confidences),
                'predictions': predictions,
                'confidences': confidences,
                'probabilities': probabilities
            }
            
            print(f"  {p}% - Accuracy: {accuracy:.3f}, ECE: {ece:.3f}")
        
        return results
    
    def _brier_score(self, y_true, probabilities):
        """Calculate Brier score for probability calibration"""
        n_classes = probabilities.shape[1]
        y_true_onehot = np.eye(n_classes)[y_true]
        return np.mean(np.sum((probabilities - y_true_onehot) ** 2, axis=1))
    
    def _expected_calibration_error(self, y_true, probabilities, confidences, n_bins=10):
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin] == np.argmax(probabilities[in_bin], axis=1))
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def plot_results(self):
        """Create comprehensive visualization of results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Accuracy vs Observation Percentage
        self._plot_accuracy_curve(axes[0, 0])
        
        # Plot 2: Calibration Error vs Observation Percentage
        self._plot_calibration_curve(axes[0, 1])
        
        # Plot 3: Reliability Diagram at 30% observation
        self._plot_reliability_diagram(axes[1, 0], percentage=30)
        
        # Plot 4: Reliability Diagram at 70% observation
        self._plot_reliability_diagram(axes[1, 1], percentage=70)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_accuracy_curve(self, ax):
        """Plot accuracy vs observation percentage for all variants"""
        for model_name, results in self.results.items():
            accuracies = [results[p]['accuracy'] for p in self.percentages]
            linestyle = '-' if 'calibrated' in model_name else '--'
            label = model_name.replace('_calibrated', ' (calibrated)').replace('_uncalibrated', ' (uncalibrated)')
            ax.plot(self.percentages, accuracies, linestyle=linestyle, label=label, marker='o')
        
        ax.set_xlabel('Observation Percentage')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Observation Percentage')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_calibration_curve(self, ax):
        """Plot calibration error vs observation percentage"""
        for model_name, results in self.results.items():
            ece_values = [results[p]['ece'] for p in self.percentages]
            linestyle = '-' if 'calibrated' in model_name else '--'
            label = model_name.replace('_calibrated', ' (calibrated)').replace('_uncalibrated', ' (uncalibrated)')
            ax.plot(self.percentages, ece_values, linestyle=linestyle, label=label, marker='s')
        
        ax.set_xlabel('Observation Percentage')
        ax.set_ylabel('Expected Calibration Error (ECE)')
        ax.set_title('Calibration Error vs Observation Percentage')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_reliability_diagram(self, ax, percentage=50):
        """Plot reliability diagram for specific observation percentage"""
        for model_name, results in self.results.items():
            if percentage in results:
                y_test = None  # We need the true labels here - you'll need to store them
                prob_true, prob_pred = calibration_curve(
                    y_test,  # You'll need to pass true labels here
                    results[percentage]['confidences'],
                    n_bins=10
                )
                linestyle = '-' if 'calibrated' in model_name else '--'
                label = model_name.replace('_calibrated', ' (calibrated)').replace('_uncalibrated', ' (uncalibrated)')
                ax.plot(prob_pred, prob_true, linestyle=linestyle, label=label, marker='o')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly calibrated')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Reliability Diagram ({percentage}% observed)')
        ax.legend()
        ax.grid(True, alpha=0.3)

# =============================================================================
# SYNTHETIC DATA
# =============================================================================

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

def main():
    """Run complete demonstration"""
    # Generate synthetic data
    X, y = generate_synthetic_ts_data(n_samples=500, n_timesteps=200, n_classes=3)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Run experiment
    experiment = EarlyClassificationExperiment(
        variants=['minirocket', 'rocket']  # Start with these for speed
    )
    
    results = experiment.run_experiment(
        X_train, y_train, X_test, y_test,
        percentages=[20, 40, 60, 80, 100]  # Fewer points for demo speed
    )
    
    # Plot results
    #experiment.plot_results()
    
    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    
    summary_data = []
    for model_name, model_results in results.items():
        for percentage, metrics in model_results.items():
            summary_data.append({
                'Model': model_name,
                'Percentage': percentage,
                'Accuracy': metrics['accuracy'],
                'ECE': metrics['ece'],
                'Brier Score': metrics['brier_score']
            })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.round(4))

if __name__ == "__main__":
    main()