import os
import torch
import numpy as np
import scipy.signal
from backend.models.autoencoder import ECGAutoencoder
from backend.models.lstm_forecaster import ECGLSTMForecaster

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUTOENCODER_PATH = os.path.join(BASE_DIR, "backend", "models", "autoencoder.pth")
LSTM_PATH = os.path.join(BASE_DIR, "backend", "models", "lstm_forecaster.pth")

class AIService:
    """
    Service to manage AI model loading, inference, forecasting, and risk scoring.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.autoencoder = ECGAutoencoder(input_dim=180, latent_dim=8).to(self.device)
        self.lstm = ECGLSTMForecaster(input_dim=1, hidden_dim=32, num_layers=2, forecast_dim=5).to(self.device)
        
        # Load weights if available
        self.load_models()

    def load_models(self):
        """Loads weights into PyTorch models from saved state dicts."""
        # 1. Load Autoencoder
        if os.path.exists(AUTOENCODER_PATH):
            try:
                self.autoencoder.load_state_dict(torch.load(AUTOENCODER_PATH, map_location=self.device))
                self.autoencoder.eval()
                print("Successfully loaded Autoencoder model weights.")
            except Exception as e:
                print(f"Error loading Autoencoder: {e}. Running with randomized weights.")
        else:
            print(f"Autoencoder checkpoint not found at {AUTOENCODER_PATH}. Run training first.")
            self.autoencoder.eval()
            
        # 2. Load LSTM Forecaster
        if os.path.exists(LSTM_PATH):
            try:
                self.lstm.load_state_dict(torch.load(LSTM_PATH, map_location=self.device))
                self.lstm.eval()
                print("Successfully loaded LSTM Forecaster model weights.")
            except Exception as e:
                print(f"Error loading LSTM Forecaster: {e}. Running with randomized weights.")
        else:
            print(f"LSTM checkpoint not found at {LSTM_PATH}. Run training first.")
            self.lstm.eval()

    def predict_anomalies(self, windows):
        """
        Feeds heartbeat windows through the Autoencoder.
        Computes reconstruction, reconstruction error, and anomaly score.
        
        Args:
            windows (np.ndarray): Array of shape [N, 180]
        Returns:
            reconstructed (np.ndarray): Reconstructed waves of shape [N, 180]
            scores (list): Anomaly scores (MSE) for each window
        """
        if len(windows) == 0:
            return np.array([]), []
            
        # Convert to PyTorch tensor
        tensor_windows = torch.tensor(windows, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            errors, reconstructed_t = self.autoencoder.get_reconstruction_error(tensor_windows)
            
            # Convert back to numpy
            reconstructed = reconstructed_t.cpu().numpy()
            
            # Apply Savitzky-Golay smoothing to reconstructed heartbeats
            # window length 17, poly order 3
            for i in range(len(reconstructed)):
                try:
                    reconstructed[i] = scipy.signal.savgol_filter(reconstructed[i], 17, 3)
                except Exception as e:
                    print(f"Savitzky-Golay smoothing failed on reconstructed window {i}: {e}")
                    
            scores = errors.cpu().numpy().tolist()
            
        return reconstructed, scores

    def forecast_risk(self, anomaly_scores, history_len=10, forecast_len=5, peaks=None):
        """
        Takes a time series sequence of anomaly scores and predicts the future trend using LSTM.
        Adjusts the forecast mathematically based on Heart Rate Variability (HRV) and current trend.
        
        Args:
            anomaly_scores (list): list of numerical anomaly scores
            history_len (int): input sequence length for LSTM
            forecast_len (int): future prediction step size
            peaks (list): list of peak sample indices
        Returns:
            forecasted_scores (list): next 5 anomaly scores
        """
        # Padding or slicing to match history_len (10)
        if len(anomaly_scores) < history_len:
            # Pad with the first element or zeros if empty
            pad_val = anomaly_scores[0] if len(anomaly_scores) > 0 else 0.0
            history = [pad_val] * (history_len - len(anomaly_scores)) + anomaly_scores
        else:
            # Take the last history_len scores
            history = anomaly_scores[-history_len:]
            
        # Convert to tensor shape [batch_size=1, seq_len=10, input_dim=1]
        tensor_history = torch.tensor(history, dtype=torch.float32).view(1, history_len, 1).to(self.device)
        
        with torch.no_grad():
            forecast_t = self.lstm(tensor_history)
            forecasted_scores = forecast_t.cpu().numpy().flatten().tolist()
            
        # Mathematical adjustments based on HRV and trend slope
        # 1. HRV multiplier (SDRR relative to the mean)
        hrv_multiplier = 1.0
        if peaks is not None and len(peaks) >= 2:
            rr_intervals = np.diff(peaks) / 360.0
            mean_rr = np.mean(rr_intervals)
            std_rr = np.std(rr_intervals)
            hrv_metric = std_rr / (mean_rr + 1e-8)
            # High variability (arrhythmias) or extremely low variability (stress/failure)
            # can boost or scale predictions
            hrv_multiplier = 1.0 + hrv_metric

        # 2. Trend slope of the last 5 anomaly scores
        trend_slope = 0.0
        if len(anomaly_scores) >= 2:
            last_scores = anomaly_scores[-5:]
            x_vals = np.arange(len(last_scores))
            # Fit line to find slope
            try:
                slope, _ = np.polyfit(x_vals, last_scores, 1)
                trend_slope = float(slope)
            except Exception as e:
                print(f"Failed to fit trend line: {e}")
                
        # 3. Project and scale the predictions
        adjusted_scores = []
        for i, score in enumerate(forecasted_scores):
            # Scale future value and add linear projection of current trend slope
            adj_score = score * hrv_multiplier + trend_slope * (i + 1)
            # Clamp to [0, 1]
            adj_score = max(0.0, min(1.0, adj_score))
            adjusted_scores.append(float(adj_score))
            
        return adjusted_scores

    def compute_risk_profile(self, current_score, forecasted_scores):
        """
        Computes current risk, future risk, and generates medical alert details.
        Risk Categories:
            - Safe: Score < 0.05
            - Mild: 0.05 <= Score < 0.12
            - Moderate: 0.12 <= Score < 0.25
            - Critical: Score >= 0.25
        """
        max_future_score = max(forecasted_scores) if len(forecasted_scores) > 0 else current_score
        
        # Determine current level
        if current_score < 0.05:
            current_level = "Safe"
            current_color = "emerald"
        elif current_score < 0.12:
            current_level = "Mild"
            current_color = "amber"
        elif current_score < 0.25:
            current_level = "Moderate"
            current_color = "orange"
        else:
            current_level = "Critical"
            current_color = "red"
            
        # Determine future risk trend
        if max_future_score < 0.05:
            future_level = "Safe"
        elif max_future_score < 0.12:
            future_level = "Mild"
        elif max_future_score < 0.25:
            future_level = "Moderate"
        else:
            future_level = "Critical"
            
        # Generate alert/recommendation based on levels
        alert = ""
        action = ""
        
        if current_level == "Critical" or future_level == "Critical":
            alert = "High risk of cardiovascular event detected. ECG displays abnormal ventricular morphology or extreme rate instability."
            action = "Immediate physician review recommended. Trigger 12-lead ECG, establish IV access, and prepare emergency cardiac protocols."
        elif current_level == "Moderate" or future_level == "Moderate":
            alert = "Moderate anomaly score pattern. Continuous ectopic activity or borderline rhythm irregularity observed."
            action = "Schedule patient evaluation within 12-24 hours. Monitor vital signs closely and perform telemetry check."
        elif current_level == "Mild" or future_level == "Mild":
            alert = "Mild fluctuations in reconstruction integrity. Minor sinus arrhythmia or artifact noise detected."
            action = "Routine checkup advised. Maintain log of ambulatory monitoring."
        else:
            alert = "Stable cardiac output. Normal sinus rhythm detected, reconstruction matches template profile."
            action = "Continue standard patient monitoring protocols."
            
        return {
            "current_score": float(current_score),
            "current_level": current_level,
            "current_color": current_color,
            "future_level": future_level,
            "max_future_score": float(max_future_score),
            "alert": alert,
            "action": action
        }
