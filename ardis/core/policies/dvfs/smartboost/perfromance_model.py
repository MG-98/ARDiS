# perfromance_model.py

import torch
import torch.nn as nn

class ForecastLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # Encoder
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=32,
            batch_first=True
        )

        # Forecast head
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)

        # First LSTM (return_sequences=True)
        x, _ = self.lstm1(x)          # (B, T, 1024)

        # Second LSTM (return_sequences=False)
        _, (h_n, _) = self.lstm2(x)
        latent = h_n[-1]              # (B, 512)

        # Forecast head
        x = self.relu(self.fc1(latent))
        x = self.relu(self.fc2(x))
        forecast = self.fc3(x)        # (B, 1)

        return forecast.squeeze(1)    # (B,)