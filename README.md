# slm-forecast

Salomon sales forecasting tool.

## Installation

```bash
poetry install
mkdir -p ./data/raw
cp ~/Downloads/sales.csv ./data/raw/sales.csv
poetry run python main.py process
```

## Evaluation

```bash
poetry run python main.py evaluate_all
```

### Ratio-based - 2024 prediction - $R^2 = 9\%$

![linear-regression](./assets/images/predictions_RATIO_BASED.png)


### Linear regression - 2024 prediction - $R^2 = 48\%$

![linear-regression](./assets/images/predictions_LINEAR.png)