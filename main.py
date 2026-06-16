"""Main script for early time series classification"""

import os
import warnings
import json
import argparse
import builtins
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from earlyts.utils import load_arff_dataset

from earlyts import (
    EarlyTimeSeriesClassifier,
    EarlyClassificationEvaluator,
    EarlyTransformerClassifier,
    generate_synthetic_ts_data,
    enable_debug,
)

N_JOBS = 1
PERCENTAGES = [20, 40, 60, 80, 100]
_CLI_QUIET = False


def cli_print(*args, **kwargs):
    if not _CLI_QUIET:
        builtins.print(*args, **kwargs)

ROCKET_CONFIGS = [
    ('minirocket_calibrated', 'minirocket', True),
    ('minirocket_uncalibrated', 'minirocket', False),
    ('rocket_calibrated', 'rocket', True),
    ('rocket_uncalibrated', 'rocket', False),
    ('multirocket_calibrated', 'multirocket', True),
    ('multirocket_uncalibrated', 'multirocket', False),
]

SYNTHETIC_CONFIGS = [
    ('minirocket_calibrated', 'minirocket', True),
    ('minirocket_uncalibrated', 'minirocket', False),
    ('rocket_calibrated', 'rocket', True),
    ('rocket_uncalibrated', 'rocket', False),
]


# ── helpers ─────────────────────────────────────────────────────────────

def train_model_worker(args):
    model_name, rocket_variant, calibrate, X_train, y_train, percentages = args
    model = EarlyTimeSeriesClassifier(
        rocket_variant=rocket_variant, calibrate=calibrate
    )
    model.fit(X_train, y_train, percentages)
    return model_name, model


def train_rocket_models(X_train, y_train, percentages, configs, n_jobs=N_JOBS):
    train_args = [
        (name, variant, calibrate, X_train, y_train, percentages)
        for name, variant, calibrate in configs
    ]
    trained = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(train_model_worker, args): args[0]
            for args in train_args
        }
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                name, model = future.result()
                trained[name] = model
                cli_print(f"    \u2713 {name} trained")
            except Exception as e:
                cli_print(f"    \u2717 {model_name} failed: {e}")
    return trained


def collect_results(evaluator, dataset_name):
    results = []
    for model_name, model_results in evaluator.results.items():
        for percentage, metrics in model_results.items():
            results.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Percentage': percentage,
                'Accuracy': metrics['accuracy'],
                'ECE': metrics['ece'],
                'Brier Score': metrics['brier_score'],
                'Mean Confidence': metrics['mean_confidence'],
                'Confusion Matrix': str(metrics['confusion_matrix'].tolist()),
            })
    return results


def save_csv(results, filename):
    pd.DataFrame(results).to_csv(filename, index=False)
    cli_print(f"  \u2713 Results saved to {filename}")


def print_summary(results, title="SUMMARY RESULTS"):
    if not results:
        return
    df = pd.DataFrame(results)
    cli_print("\n" + "=" * 60)
    cli_print(title)
    cli_print("=" * 60)
    cli_print(f"\nProcessed {df['Dataset'].nunique()} dataset(s)")
    cli_print(f"Total results: {len(df)} rows")
    cli_print("\nAverage accuracy by percentage:")
    cli_print(df.groupby('Percentage')['Accuracy'].mean().round(4))


def get_cached_results(dataset_name, suffix=''):
    cache_dir = 'cache'
    path = os.path.join(cache_dir, f'{dataset_name}{suffix}_results.json')
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            cli_print(f"  Warning: failed to load cache: {e}")
    return None


def save_cached_results(dataset_name, results, suffix=''):
    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'{dataset_name}{suffix}_results.json')
    try:
        with open(path, 'w') as f:
            json.dump(results, f)
        cli_print(f"  \u2713 Cached results for {dataset_name}")
    except Exception as e:
        cli_print(f"  Warning: failed to cache: {e}")


def get_datasets(datasets_path):
    folders = [
        name for name in os.listdir(datasets_path)
        if os.path.isdir(os.path.join(datasets_path, name))
    ]

    def folder_size(name):
        total = 0
        try:
            for dirpath, _, filenames in os.walk(os.path.join(datasets_path, name)):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total += os.path.getsize(fp)
        except OSError:
            pass
        return total

    cli_print(f"Found {len(folders)} datasets, sorting by size...")
    folders.sort(key=folder_size)
    return folders


def process_single_dataset(dataset_name, datasets_path, configs, percentages,
                           n_jobs=N_JOBS):
    train_path = os.path.join(datasets_path, dataset_name, f"{dataset_name}_TRAIN.arff")
    test_path = os.path.join(datasets_path, dataset_name, f"{dataset_name}_TEST.arff")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing files for {dataset_name}")

    X_train, y_train = load_arff_dataset(train_path)
    X_test, y_test = load_arff_dataset(test_path)
    cli_print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    trained = train_rocket_models(X_train, y_train, percentages, configs, n_jobs)

    evaluator = EarlyClassificationEvaluator(n_jobs=n_jobs)
    for name, model in trained.items():
        evaluator.add_model(name, model)
    evaluator.evaluate(X_test, y_test)

    return collect_results(evaluator, dataset_name)


# ── mode functions ──────────────────────────────────────────────────────

def run_synthetic():
    cli_print("=== Synthetic data demo ===")
    X, y = generate_synthetic_ts_data(n_samples=500, n_timesteps=200, n_classes=3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    cli_print(f"Training set: {X_train.shape}")
    cli_print(f"Test set: {X_test.shape}")

    trained = train_rocket_models(X_train, y_train, PERCENTAGES, SYNTHETIC_CONFIGS)
    evaluator = EarlyClassificationEvaluator(n_jobs=N_JOBS)
    for name, model in trained.items():
        evaluator.add_model(name, model)
    evaluator.evaluate(X_test, y_test)

    results = collect_results(evaluator, 'synthetic')
    save_csv(results, 'results_synthetic.csv')
    print_summary(results)
    return results


def run_single(dataset_name):
    datasets_path = os.path.join(os.getcwd(), 'datasets', 'Univariate')
    cli_print(f"=== Single dataset: {dataset_name} ===")

    cached = get_cached_results(dataset_name)
    if cached is not None:
        cli_print(f"  \u2713 Loading cached results for {dataset_name}")
        df = pd.DataFrame(cached)
        save_csv(df, f'results_{dataset_name.lower()}.csv')
        print_summary(cached)
        return cached

    results = process_single_dataset(
        dataset_name, datasets_path, ROCKET_CONFIGS, PERCENTAGES
    )
    save_cached_results(dataset_name, results)
    save_csv(results, f'results_{dataset_name.lower()}.csv')
    print_summary(results)
    return results


def run_all(mode='rocket'):
    """mode: 'rocket' (6 ROCKET variants) or 'transformer'."""
    datasets_path = os.path.join(os.getcwd(), 'datasets', 'Univariate')
    cache_suffix = '_transformer' if mode == 'transformer' else ''
    csv_prefix = 'results_transformer' if mode == 'transformer' else 'results'

    cli_print(f"=== All datasets ({mode} mode) ===")
    folders = get_datasets(datasets_path)
    cli_print(f"Processing {len(folders)} datasets (smallest to largest)")

    batch_size = 10
    all_results = []

    for i in range(0, len(folders), batch_size):
        batch = folders[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        cli_print(f"\n{'=' * 80}")
        cli_print(f"Batch {batch_num} (datasets {i+1} to {min(i+batch_size, len(folders))})")
        cli_print(f"{'=' * 80}\n")

        batch_results = []

        for dataset_name in batch:
            try:
                cli_print(f"\nProcessing dataset: {dataset_name}")

                cached = get_cached_results(dataset_name, cache_suffix)
                if cached is not None:
                    cli_print(f"  \u2713 Loading cached {mode} results for {dataset_name}")
                    batch_results.extend(cached)
                    continue

                if mode == 'transformer':
                    train_path = os.path.join(datasets_path, dataset_name,
                                              f"{dataset_name}_TRAIN.arff")
                    test_path = os.path.join(datasets_path, dataset_name,
                                             f"{dataset_name}_TEST.arff")
                    if not os.path.exists(train_path) or not os.path.exists(test_path):
                        cli_print(f"  Warning: Missing files for {dataset_name}, skipping...")
                        continue

                    X_train, y_train = load_arff_dataset(train_path)
                    X_test, y_test = load_arff_dataset(test_path)
                    cli_print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")

                    model = EarlyTransformerClassifier()
                    model.fit(X_train, y_train, percentages=PERCENTAGES)

                    evaluator = EarlyClassificationEvaluator(n_jobs=1)
                    evaluator.add_model('transformer', model)
                    evaluator.evaluate(X_test, y_test)

                    dataset_results = collect_results(evaluator, dataset_name)
                else:
                    dataset_results = process_single_dataset(
                        dataset_name, datasets_path, ROCKET_CONFIGS, PERCENTAGES
                    )

                save_cached_results(dataset_name, dataset_results, cache_suffix)
                batch_results.extend(dataset_results)
                cli_print(f"  \u2713 Successfully processed {dataset_name}")

            except Exception as e:
                cli_print(f"  \u2717 Error processing {dataset_name}: {e}")
                continue

        if batch_results:
            save_csv(batch_results, f'{csv_prefix}_batch_{batch_num}.csv')
            all_results.extend(batch_results)

    print_summary(all_results, f"ALL RESULTS SUMMARY ({mode})")
    return all_results


# ── entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Early Time Series Classification experiments with ROCKET variants and Transformer'
    )
    parser.add_argument(
        '--mode', choices=['synthetic', 'single', 'all', 'transformer'],
        default='synthetic',
        help='Which experiment to run (default: synthetic)'
    )
    parser.add_argument(
        '--dataset', default='Rock',
        help='Dataset name for --mode single (default: Rock)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Show internal training and evaluation progress'
    )
    parser.add_argument(
        '--strict-silence', action='store_true',
        help='Suppress all output (overrides --verbose)'
    )
    args = parser.parse_args()

    if args.strict_silence:
        global _CLI_QUIET
        _CLI_QUIET = True
    elif args.verbose:
        enable_debug()

    if args.mode == 'synthetic':
        run_synthetic()
    elif args.mode == 'single':
        run_single(args.dataset)
    elif args.mode == 'all':
        run_all(mode='rocket')
    elif args.mode == 'transformer':
        run_all(mode='transformer')


if __name__ == "__main__":
    main()
