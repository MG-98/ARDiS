"""PLACEHOLDER performance forecasting model architecture.

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
            f"perfromance_model.py still has placeholders (N) for: {', '.join(missing)}. "
            "Replace them with the values of your trained performance model."
        )


class ForecastLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        lstm1_size=N,   # N: hidden size of the first LSTM
        lstm2_size=N,   # N: hidden size of the second LSTM (the latent vector)
        fc1_size=N,     # N: first dense layer width
        fc2_size=N,     # N: second dense layer width
        output_size=N,  # N: forecast size (one value: instructions)
    ):
        super().__init__()
        _require_filled(lstm1_size=lstm1_size, lstm2_size=lstm2_size, fc1_size=fc1_size,
                        fc2_size=fc2_size, output_size=output_size)

        # Encoder
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=lstm1_size, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=lstm1_size, hidden_size=lstm2_size, batch_first=True)

        # Forecast head
        self.fc1 = nn.Linear(lstm2_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        x, _ = self.lstm1(x)          # (B, T, lstm1_size)
        _, (h_n, _) = self.lstm2(x)
        latent = h_n[-1]              # (B, lstm2_size)

        x = self.relu(self.fc1(latent))
        x = self.relu(self.fc2(x))
        forecast = self.fc3(x)        # (B, output_size)

        return forecast.squeeze(1)    # (B,)
