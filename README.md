# Early Time Series Classification

Early classification of time series using ROCKET variants with calibrated confidence estimation.

## Setup

```bash
uv sync
```

## Run

```bash
uv run main.py
```

Results are saved to `results_XXXXXX.csv` with random ID if no name is provided.

## Usage

```python
from earlyts import EarlyTimeSeriesClassifier, EarlyClassificationEvaluator

# Train a model
model = EarlyTimeSeriesClassifier(rocket_variant='minirocket', calibrate=True)
model.fit(X_train, y_train, percentages=[20, 50, 100])

# Predict at 50% observation
predictions, confidences = ű
.predict(X_test, percentage=50)

# Evaluate multiple models
evaluator = EarlyClassificationEvaluator()
evaluator.add_model('my_model', model)
evaluator.evaluate(X_test, y_test)
evaluator.save_results('results.csv')
```

## ROCKET Variants

- `minirocket`
- `rocket`
- `multirocket`
