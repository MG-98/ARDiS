"""PLACEHOLDER thermal forecasting model architecture.

Every `N` below is a placeholder: replace it with the value used by YOUR trained model.
The architecture must match the one your model file was saved with, otherwise
torch.load() fails or the forecasts are wrong. See SmartBoost-HW/README.md.
"""

import torch.nn as nn

N = None  # <-- placeholder marker, never leave it as None


def _require_filled(**dims):
    missing = [name for name, value in dims.items() if value is None]
    if missing:
        raise ValueError(
            f"thermal_model.py still has placeholders (N) for: {', '.join(missing)}. "
            "Replace them with the values of your trained thermal model."
        )


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size=N,    # N: number of input features per time step
        hidden_size=N,   # N: LSTM hidden size
        num_layers=N,    # N: number of stacked LSTM layers
        output_size=N,   # N: forecast length (one temperature per core)
        fc_size=N,       # N: width of the hidden layer in the forecast head
    ):
        super().__init__()
        _require_filled(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                        output_size=output_size, fc_size=fc_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, output_size),
        )

    def forward(self, x):
        # x: [batch_size, N (window length), N (input features)]
        lstm_out, _ = self.lstm(x)

        # Take the output of the last time step
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]

        return self.fc(last_output)
