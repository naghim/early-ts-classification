"""Evaluation and benchmarking for early classification models"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve

from .classifier import EarlyTimeSeriesClassifier
from .utils import normalize_input


class EarlyClassificationEvaluator:
    """
    Evaluator class to evaluate and compare multiple early classification models
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.percentages = None
    
    def add_model(self, name, model):
        """Add a trained model to the evaluator"""
        self.models[name] = model
        return self
    
    def evaluate(self, X_test, y_test, percentages=None):
        """Evaluate all added models on test data"""
        if percentages is None:
            # Use percentages from first model if available
            if self.models:
                first_model = next(iter(self.models.values()))
                percentages = first_model.percentages
            else:
                percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        self.percentages = percentages
        
        print("\n=== Evaluating Models ===")
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            results = self._evaluate_model(model, X_test, y_test, model_name)
            self.results[model_name] = results
        
        return self.results
    
    def _evaluate_model(self, model, X_test, y_test):
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
        summary_df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
        return summary_df
    
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
