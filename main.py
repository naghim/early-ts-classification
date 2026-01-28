"""Demo script for early time series classification"""

import os
import warnings
import arff
import torch
import numpy as np
import json
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split
from earlyts.utils import load_arff_dataset

from earlyts import (
    EarlyTimeSeriesClassifier,
    EarlyClassificationEvaluator,
    generate_synthetic_ts_data
)


def run_synthetic_demo():
    """Run complete demonstration"""
    # Generate synthetic data
    X, y = generate_synthetic_ts_data(n_samples=500, n_timesteps=200, n_classes=3)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Define percentages to evaluate
    percentages = [20, 40, 60, 80, 100]
    
    # Train models
    print("\n=== Training MINIROCKET ===")
    print("  [Calibrated version]")
    minirocket_cal = EarlyTimeSeriesClassifier(
        rocket_variant='minirocket', calibrate=True
    )
    minirocket_cal.fit(X_train, y_train, percentages)
    
    print("  [Uncalibrated version]")
    minirocket_uncal = EarlyTimeSeriesClassifier(
        rocket_variant='minirocket', calibrate=False
    )
    minirocket_uncal.fit(X_train, y_train, percentages)
    
    print("\n=== Training ROCKET ===")
    print("  [Calibrated version]")
    rocket_cal = EarlyTimeSeriesClassifier(
        rocket_variant='rocket', calibrate=True
    )
    rocket_cal.fit(X_train, y_train, percentages)
    
    print("  [Uncalibrated version]")
    rocket_uncal = EarlyTimeSeriesClassifier(
        rocket_variant='rocket', calibrate=False
    )
    rocket_uncal.fit(X_train, y_train, percentages)
    
    # Evaluate models
    evaluator = EarlyClassificationEvaluator()
    evaluator.add_model('minirocket_calibrated', minirocket_cal)
    evaluator.add_model('minirocket_uncalibrated', minirocket_uncal)
    evaluator.add_model('rocket_calibrated', rocket_cal)
    evaluator.add_model('rocket_uncalibrated', rocket_uncal)
    
    evaluator.evaluate(X_test, y_test)
    
    # Save results to CSV with random filename
    summary_df = evaluator.save_results()
    
    # Plot results (uncomment to show plots)
    # evaluator.plot_results()
    
    # Print summary table to console
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    print(summary_df.round(4))


def load_dataset(dataset_name):
	base_path = os.path.join('datasets', 'Univariate', dataset_name)
	train_path = os.path.join(base_path, f'{dataset_name}_TRAIN.arff')
	test_path = os.path.join(base_path, f'{dataset_name}_TEST.arff')
	return train_path, test_path


def load_arff_data(arff_path):
	with open(arff_path, 'r') as f:
		dataset = arff.load(f)
	data = np.array(dataset['data'])
	# Last column is label, rest are features
	X = np.array([row[:-1] for row in data], dtype=np.float32)
	y = np.array([row[-1] for row in data])
	# Reshape to (num_examples, 1, length)
	X = X.reshape((X.shape[0], 1, X.shape[1]))
	return torch.from_numpy(X), y

def get_cache_path(dataset_name):
    """Get the cache file path for a dataset"""
    cache_dir = os.path.join('cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'{dataset_name}_results.json')


def load_cached_results(dataset_name):
    """Load cached results for a dataset if they exist"""
    cache_path = get_cache_path(dataset_name)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  Warning: Failed to load cache for {dataset_name}: {e}")
            return None
    return None


def save_cached_results(dataset_name, results):
    """Save results for a dataset to cache"""
    cache_path = get_cache_path(dataset_name)
    try:
        with open(cache_path, 'w') as f:
            json.dump(results, f)
        print(f"  ✓ Cached results for {dataset_name}")
    except Exception as e:
        print(f"  Warning: Failed to cache results for {dataset_name}: {e}")


def main():
    current_dir = os.getcwd()
    datasets_path = os.path.join(current_dir, 'datasets', 'Univariate')
    
    # Get all dataset folders
    folders = [
        name for name in os.listdir(datasets_path)
        if os.path.isdir(os.path.join(datasets_path, name))
    ]
    
    print(f"Found {len(folders)} datasets to process")
    
    # Define percentages to evaluate
    percentages = [20, 40, 60, 80, 100]
    
    # Process datasets in batches of 20
    batch_size = 20
    all_results = []
    
    for i in range(0, len(folders), batch_size):
        batch_folders = folders[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"\n{'='*80}")
        print(f"Processing batch {batch_num} (datasets {i+1} to {min(i+batch_size, len(folders))})")
        print(f"{'='*80}\n")
        
        batch_results = []
        
        for dataset_name in batch_folders:
            try:
                print(f"\nProcessing dataset: {dataset_name}")
                
                # Check if results are already cached
                cached_results = load_cached_results(dataset_name)
                if cached_results is not None:
                    print(f"  ✓ Loading cached results for {dataset_name}")
                    batch_results.extend(cached_results)
                    continue
                
                # Construct paths to train and test files
                train_path = os.path.join(datasets_path, dataset_name, f"{dataset_name}_TRAIN.arff")
                test_path = os.path.join(datasets_path, dataset_name, f"{dataset_name}_TEST.arff")
                
                # Check if files exist
                if not os.path.exists(train_path) or not os.path.exists(test_path):
                    print(f"  Warning: Missing train or test file for {dataset_name}, skipping...")
                    continue
                
                # Load train and test sets
                X_train, y_train = load_arff_dataset(train_path)
                X_test, y_test = load_arff_dataset(test_path)
                
                print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
                
                # Train all models
                print("  Training MINIROCKET (calibrated)...")
                minirocket_cal = EarlyTimeSeriesClassifier(
                    rocket_variant='minirocket', calibrate=True
                )
                minirocket_cal.fit(X_train, y_train, percentages)
                
                print("  Training MINIROCKET (uncalibrated)...")
                minirocket_uncal = EarlyTimeSeriesClassifier(
                    rocket_variant='minirocket', calibrate=False
                )
                minirocket_uncal.fit(X_train, y_train, percentages)
                
                print("  Training ROCKET (calibrated)...")
                rocket_cal = EarlyTimeSeriesClassifier(
                    rocket_variant='rocket', calibrate=True
                )
                rocket_cal.fit(X_train, y_train, percentages)
                
                print("  Training ROCKET (uncalibrated)...")
                rocket_uncal = EarlyTimeSeriesClassifier(
                    rocket_variant='rocket', calibrate=False
                )
                rocket_uncal.fit(X_train, y_train, percentages)
                
                print("  Training MULTIROCKET (calibrated)...")
                multirocket_cal = EarlyTimeSeriesClassifier(
                    rocket_variant='multirocket', calibrate=True
                )
                multirocket_cal.fit(X_train, y_train, percentages)
                
                print("  Training MULTIROCKET (uncalibrated)...")
                multirocket_uncal = EarlyTimeSeriesClassifier(
                    rocket_variant='multirocket', calibrate=False
                )
                multirocket_uncal.fit(X_train, y_train, percentages)
                
                # Evaluate all models
                evaluator = EarlyClassificationEvaluator()
                evaluator.add_model('minirocket_calibrated', minirocket_cal)
                evaluator.add_model('minirocket_uncalibrated', minirocket_uncal)
                evaluator.add_model('rocket_calibrated', rocket_cal)
                evaluator.add_model('rocket_uncalibrated', rocket_uncal)
                evaluator.add_model('multirocket_calibrated', multirocket_cal)
                evaluator.add_model('multirocket_uncalibrated', multirocket_uncal)
                evaluator.evaluate(X_test, y_test)
                
                # Get results for this dataset
                dataset_results = []
                for model_name, model_results in evaluator.results.items():
                    for percentage, metrics in model_results.items():
                        # Convert confusion matrix to string representation
                        cm_str = str(metrics['confusion_matrix'].tolist())
                        
                        result_entry = {
                            'Dataset': dataset_name,
                            'Model': model_name,
                            'Percentage': percentage,
                            'Accuracy': metrics['accuracy'],
                            'ECE': metrics['ece'],
                            'Brier Score': metrics['brier_score'],
                            'Mean Confidence': metrics['mean_confidence'],
                            'Confusion Matrix': cm_str
                        }
                        dataset_results.append(result_entry)
                        batch_results.append(result_entry)
                
                # Cache the results for this dataset
                save_cached_results(dataset_name, dataset_results)
                
                print(f"  ✓ Successfully processed {dataset_name}")
                
            except Exception as e:
                raise Exception(f"  ✗ Error processing {dataset_name}: {str(e)}")
                continue
        
        # Save batch results to CSV
        if batch_results:
            batch_df = pd.DataFrame(batch_results)
            csv_filename = f'results_batch_{batch_num}.csv'
            batch_df.to_csv(csv_filename, index=False)
            print(f"\n{'='*80}")
            print(f"Batch {batch_num} results saved to {csv_filename}")
            print(f"{'='*80}\n")
            
            all_results.extend(batch_results)
    
    # Print final summary
    if all_results:
        print("\n" + "="*80)
        print("ALL RESULTS SUMMARY")
        print("="*80)
        final_df = pd.DataFrame(all_results)
        print(f"\nProcessed {len(final_df['Dataset'].unique())} datasets successfully")
        print(f"Total results: {len(final_df)} rows")
        print("\nAverage accuracy by percentage:")
        print(final_df.groupby('Percentage')['Accuracy'].mean().round(4))


if __name__ == "__main__":
    main()