import torch
import torch.nn as nn

class ECGLSTMForecaster(nn.Module):
    """
    A PyTorch LSTM model for forecasting future ECG anomaly scores.
    It takes a history sequence of anomaly scores and predicts the trend
    for the next several beats/timeframes.
    """
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, forecast_dim=5):
        super(ECGLSTMForecaster, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        # input shape: [batch_size, seq_len, input_dim]
        # output shape: [batch_size, seq_len, hidden_dim]
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Fully Connected Layer to forecast the next N steps
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, forecast_dim) # Predicts forecast_dim (e.g. 5) future values
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): Input sequence of shape [BatchSize, SeqLen, InputDim] (e.g., [B, 10, 1])
        Returns:
            out (Tensor): Forecasted future anomaly scores of shape [BatchSize, ForecastDim] (e.g., [B, 5])
        """
        # Initialize hidden and cell states (PyTorch does this automatically to zero, but we can do it explicitly)
        # lstm_out shape: [BatchSize, SeqLen, HiddenDim]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Get the output of the last sequence step: [BatchSize, HiddenDim]
        last_step_out = lstm_out[:, -1, :]
        
        # Forecast the future values
        forecast = self.fc(last_step_out)
        
        # Anomaly scores should be non-negative, but can be higher.
        # We can apply ReLU or Clamp to make sure they are at least 0.0
        return torch.clamp(forecast, min=0.0)
