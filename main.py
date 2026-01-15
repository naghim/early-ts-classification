"""Demo script for early time series classification"""

import warnings
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


def main():

    # Paths to your .arff files
    train_path = "datasets/Univariate/ArrowHead/ArrowHead_TRAIN.arff"
    test_path = "datasets/Univariate/ArrowHead/ArrowHead_TEST.arff"

    # Load train and test sets
    X_train, y_train = load_arff_dataset(train_path)
    X_test, y_test = load_arff_dataset(test_path)

    # Now X_train, y_train, X_test, y_test are ready for use
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train labels:", set(y_train), "Test labels:", set(y_test))
    
    # Define percentages to evaluate
    percentages = [20, 40]#, 60, 80, 100]
    
    # Train models
    print("\n=== Training MINIROCKET ===")
    print("  [Calibrated version]")
    minirocket_cal = EarlyTimeSeriesClassifier(
        rocket_variant='minirocket', calibrate=True
    )
    minirocket_cal.fit(X_train, y_train, percentages)
    
    # print("  [Uncalibrated version]")
    # minirocket_uncal = EarlyTimeSeriesClassifier(
    #     rocket_variant='minirocket', calibrate=False
    # )
    # minirocket_uncal.fit(X_train, y_train, percentages)
    
    # print("\n=== Training ROCKET ===")
    # print("  [Calibrated version]")
    # rocket_cal = EarlyTimeSeriesClassifier(
    #     rocket_variant='rocket', calibrate=True
    # )
    # rocket_cal.fit(X_train, y_train, percentages)
    
    # print("  [Uncalibrated version]")
    # rocket_uncal = EarlyTimeSeriesClassifier(
    #     rocket_variant='rocket', calibrate=False
    # )
    # rocket_uncal.fit(X_train, y_train, percentages)
    
    # Evaluate models
    evaluator = EarlyClassificationEvaluator()
    evaluator.add_model('minirocket_calibrated', minirocket_cal)
    # evaluator.add_model('minirocket_uncalibrated', minirocket_uncal)
    # evaluator.add_model('rocket_calibrated', rocket_cal)
    # evaluator.add_model('rocket_uncalibrated', rocket_uncal)
    
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


if __name__ == "__main__":
    main()