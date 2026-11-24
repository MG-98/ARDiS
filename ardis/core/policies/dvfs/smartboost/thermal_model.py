# thermal_model.py

import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size=38,
        hidden_size=128,
        num_layers=2,
        output_size=24,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            #dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        # x: [batch_size, 5, 38]

        lstm_out, _ = self.lstm(x)

        # Take the output of the last time step
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]

        prediction = self.fc(last_output)

        return prediction