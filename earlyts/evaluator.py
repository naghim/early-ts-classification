"""Evaluation and benchmarking for early classification models"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve
from concurrent.futures import ProcessPoolExecutor, as_completed

from ._debug import debug_print
from .classifier import EarlyTimeSeriesClassifier
from .utils import normalize_input


def _evaluate_model_worker(args):
    """Worker function for parallel model evaluation"""
    model_name, model, X_test, y_test, percentages = args
    
    results = {}
    X_test, y_test = normalize_input(X_test, y_test, name="test")

    from sklearn.metrics import confusion_matrix
    for p in percentages:
        # Get predictions and probabilities
        predictions, confidences = model.predict(X_test, p)
        probabilities = model.predict_probabilities(X_test, p)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        brier_score = _brier_score_static(y_test, probabilities)
        ece = _expected_calibration_error_static(y_test, probabilities, confidences)
        cm = confusion_matrix(y_test, predictions)
        
        # Store results
        results[p] = {
            'accuracy': accuracy,
            'brier_score': brier_score,
            'ece': ece,
            'mean_confidence': np.mean(confidences),
            'predictions': predictions,
            'confidences': confidences,
            'probabilities': probabilities,
            'confusion_matrix': cm
        }
    
    return model_name, results


def _brier_score_static(y_true, probabilities):
    """Calculate Brier score for probability calibration (static version for multiprocessing)"""
    n_classes = probabilities.shape[1]
    _, y_true_int = np.unique(y_true, return_inverse=True)
    y_true_onehot = np.eye(n_classes)[y_true_int]
    return np.mean(np.sum((probabilities - y_true_onehot) ** 2, axis=1))


def _expected_calibration_error_static(y_true, probabilities, confidences, n_bins=10):
    """Calculate Expected Calibration Error (static version for multiprocessing)"""
    _, y_true_int = np.unique(y_true, return_inverse=True)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true_int[in_bin] == np.argmax(probabilities[in_bin], axis=1))
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


class EarlyClassificationEvaluator:
    """
    Evaluator class to evaluate and compare multiple early classification models
    """
    
    def __init__(self, n_jobs=None):
        """
        Initialize the evaluator.
        
        Parameters:
        - n_jobs: Number of parallel processes to use. If None, uses os.cpu_count().
        """
        self.models = {}
        self.results = {}
        self.percentages = None
        self.n_jobs = n_jobs if n_jobs is not None else 1#os.cpu_count()
    
    def add_model(self, name, model):
        """Add a trained model to the evaluator"""
        self.models[name] = model
        return self
    
    def evaluate(self, X_test, y_test, percentages=None):
        """Evaluate all added models on test data using parallel processing"""
        if percentages is None:
            # Use percentages from first model if available
            if self.models:
                first_model = next(iter(self.models.values()))
                percentages = first_model.percentages
            else:
                percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        self.percentages = percentages
        
        debug_print(f"\n=== Evaluating Models (using {self.n_jobs} processes) ===")
        
        # Prepare arguments for parallel processing
        eval_args = [
            (model_name, model, X_test, y_test, percentages)
            for model_name, model in self.models.items()
        ]
        
        if self.n_jobs == 1:
            # Fast path: run in-process (avoids pickling, needed for PyTorch models)
            for args in eval_args:
                try:
                    name, results = _evaluate_model_worker(args)
                    self.results[name] = results
                    debug_print(f"  ✓ {name} evaluated")
                except Exception as e:
                    debug_print(f"  ✗ {args[0]} failed: {e}")
        else:
            # Use ProcessPoolExecutor for parallel evaluation
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(_evaluate_model_worker, args): args[0]
                    for args in eval_args
                }
                
                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        name, results = future.result()
                        self.results[name] = results
                        debug_print(f"  ✓ {name} evaluated")
                    except Exception as e:
                        debug_print(f"  ✗ {model_name} failed: {e}")
        
        return self.results
    
    def _evaluate_model(self, model, X_test, y_test):
        """Evaluate model at all observation percentages"""
        results = {}
        X_test, y_test = normalize_input(X_test, y_test, name="test")

        from sklearn.metrics import confusion_matrix
        for p in model.percentages:
            # Get predictions and probabilities
            predictions, confidences = model.predict(X_test, p)
            probabilities = model.predict_probabilities(X_test, p)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            brier_score = self._brier_score(y_test, probabilities)
            ece = self._expected_calibration_error(y_test, probabilities, confidences)
            cm = confusion_matrix(y_test, predictions)
            
            # Store results
            results[p] = {
                'accuracy': accuracy,
                'brier_score': brier_score,
                'ece': ece,
                'mean_confidence': np.mean(confidences),
                'predictions': predictions,
                'confidences': confidences,
                'probabilities': probabilities,
                'confusion_matrix': cm
            }
        
        return results
    
    def _brier_score(self, y_true, probabilities):
        """Calculate Brier score for probability calibration"""
        n_classes = probabilities.shape[1]
        # Always map labels to 0-indexed integers to handle both 0-indexed and 1-indexed labels
        _, y_true_int = np.unique(y_true, return_inverse=True)
        y_true_onehot = np.eye(n_classes)[y_true_int]
        return np.mean(np.sum((probabilities - y_true_onehot) ** 2, axis=1))
    
    def _expected_calibration_error(self, y_true, probabilities, confidences, n_bins=10):
        """Calculate Expected Calibration Error"""
        # Map labels to 0-indexed integers to handle both 0-indexed and 1-indexed labels
        _, y_true_int = np.unique(y_true, return_inverse=True)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true_int[in_bin] == np.argmax(probabilities[in_bin], axis=1))
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def save_results(self, filename=None):
        """Save experiment results to CSV file"""
        if filename is None:
            random_num = random.randint(100000, 999999)
            filename = f'results_{random_num}.csv'
        
        summary_data = []
        for model_name, model_results in self.results.items():
            for percentage, metrics in model_results.items():
                summary_data.append({
                    'Model': model_name,
                    'Percentage': percentage,
                    'Accuracy': metrics['accuracy'],
                    'ECE': metrics['ece'],
                    'Brier Score': metrics['brier_score'],
                    'Mean Confidence': metrics['mean_confidence']
                })
        
        summary_df = pd.DataFrame(summary_data)

        return summary_df