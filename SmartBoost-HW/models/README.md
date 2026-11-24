# SmartBoost-HW models (placeholders)

This folder is intentionally empty. The trained models are **not** part of this repository:
the numbers in the architecture files are placeholders (`N`), and you provide your own
trained models.

Put these files here:

| File | What it is |
|---|---|
| `thermal_model.pth` | Your trained thermal forecasting model (full pickled `torch` module, class `LSTMForecaster`) |
| `performance_model.pth` | Your trained performance forecasting model (class `ForecastLSTM`) |
| `input_scaler.pkl` | Scaler fitted on the performance model inputs (`joblib`) |
| `output_scaler.pkl` | Scaler fitted on the performance model output (`joblib`) |
| `gmm.pkl` | Gaussian mixture fitted on the thermal-model embeddings (`margin_mode="gmm"` only) |
| `cluster_stats.pkl` | Per-cluster statistics with the `p90` uncertainty (`margin_mode="gmm"` only) |

The architecture of your models must match
`ardis/core/policies/dvfs/thermal_model.py` and `performance_model.py`. See the main
[README](../README.md) for how to fill in the placeholders.
