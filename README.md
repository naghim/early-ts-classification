# Early Time Series Classification

Early classification of time series using ROCKET variants (miniROCKET, ROCKET, multiROCKET) with and without calibrated confidence estimation, plus a vanilla transformer implementation.

## Setup

Requires Python ≥3.12 and [uv](https://docs.astral.sh/uv/) installed.

```bash
uv sync
```

Expects UCR `.arff` datasets under `datasets/Univariate/<name>/<name>_{TRAIN,TEST}.arff`.

## CLI Usage

```bash
# Synthetic data demo (default)
uv run main.py

# Single real dataset with all ROCKET variants
uv run main.py --mode single --dataset ArrowHead

# All datasets with ROCKET variants (batched)
uv run main.py --mode all

# All datasets with transformer
uv run main.py --mode transformer
```

Results are saved as CSV files (`results_synthetic.csv`, `results_{dataset}.csv`, or `results_batch_{n}.csv` / `results_transformer_batch_{n}.csv`).

## Library Usage

```python
from earlyts import EarlyTimeSeriesClassifier, EarlyClassificationEvaluator

model = EarlyTimeSeriesClassifier(rocket_variant='minirocket', calibrate=True)
model.fit(X_train, y_train, percentages=[20, 50, 100])

predictions, confidences = model.predict(X_test, percentage=50)

evaluator = EarlyClassificationEvaluator()
evaluator.add_model('my_model', model)
evaluator.evaluate(X_test, y_test)
```

## Models

- **ROCKET variants**: `minirocket`, `rocket`, `multirocket` - each with `RidgeClassifierCV` on top, optionally wrapped in `CalibratedClassifierCV(method='isotonic', cv=3)`.
- **Transformer**: vanilla `TransformerEncoder` (4 heads, 2 layers, d_model=64) trained per-percentage with early stopping.

All models train one sub-model per observation percentage (20%, 40%, 60%, 80%, 100%) by truncating each time series to the first _p%_ of timesteps.
