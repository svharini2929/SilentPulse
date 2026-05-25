import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(BASE_DIR)

from backend.models.autoencoder import ECGAutoencoder
from backend.models.lstm_forecaster import ECGLSTMForecaster
from backend.services.ecg_service import ECGService
from backend.utils.helpers import ensure_directories, download_mitbih_record

def train_models():
    """
    Trains the PyTorch Autoencoder on normal ECG signals and the LSTM
    forecaster on sequences of anomaly scores.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training models using device: {device}")
    
    # 1. Ensure datasets and upload directories are created
    ensure_directories()
    
    # 2. Load data for Autoencoder (Record 103 is normal rhythm, good for normal template training)
    print("\n--- Phase 1: Training ECG Autoencoder on Normal Heartbeats ---")
    
    # Try to load Record 103 (Normal rhythm)
    signal_normal, annotations_normal, is_synth = ECGService.load_record("103")
    print(f"Loaded normal record. Signal size: {len(signal_normal)}, annotations count: {len(annotations_normal)}, synthetic: {is_synth}")
    
    # Segment into heartbeat windows (window_size = 180)
    windows_normal, peaks_normal, labels_normal = ECGService.segment_heartbeats(signal_normal, annotations_normal)
    print(f"Segmented {len(windows_normal)} heartbeat windows.")
    
    # Keep only normal heartbeats ('N' annotation)
    normal_beats = [win for win, lbl in zip(windows_normal, labels_normal) if lbl == 'N']
    if len(normal_beats) < 10:
        print("Not enough normal beats, using all available windows.")
        normal_beats = windows_normal
    else:
        normal_beats = np.array(normal_beats)
        
    print(f"Training Autoencoder on {len(normal_beats)} normal beats...")
    
    # Prepare DataLoader
    x_train_ae = torch.tensor(normal_beats, dtype=torch.float32)
    ae_dataset = TensorDataset(x_train_ae, x_train_ae) # Autoencoder tries to reconstruct its input
    ae_loader = DataLoader(ae_dataset, batch_size=32, shuffle=True)
    
    # Define model, loss, optimizer
    autoencoder = ECGAutoencoder(input_dim=180, latent_dim=8).to(device)
    ae_criterion = nn.MSELoss()
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.005)
    
    # Train Autoencoder
    autoencoder.train()
    ae_epochs = 30
    for epoch in range(ae_epochs):
        epoch_loss = 0.0
        for inputs, targets in ae_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            ae_optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = ae_criterion(outputs, targets)
            loss.backward()
            ae_optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            
        avg_loss = epoch_loss / len(normal_beats)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{ae_epochs} - Loss: {avg_loss:.6f}")
            
    # Save Autoencoder model weights
    ae_path = os.path.join(BASE_DIR, "backend", "models", "autoencoder.pth")
    os.makedirs(os.path.dirname(ae_path), exist_ok=True)
    torch.save(autoencoder.state_dict(), ae_path)
    print(f"Autoencoder model saved to {ae_path}")
    
    # 3. Load data for LSTM Forecaster (Record 100 has some ectopic beats, causing reconstruction spikes)
    print("\n--- Phase 2: Training LSTM Forecaster on Anomaly Score Trends ---")
    
    # Try to load Record 100 (which contains PVC / arrhythmia anomalies)
    signal_anom, annotations_anom, _ = ECGService.load_record("100")
    windows_anom, peaks_anom, labels_anom = ECGService.segment_heartbeats(signal_anom, annotations_anom)
    
    # Run through the Autoencoder to calculate anomaly scores
    autoencoder.eval()
    with torch.no_grad():
        x_anom_tensor = torch.tensor(windows_anom, dtype=torch.float32).to(device)
        reconstructed = autoencoder(x_anom_tensor)
        # Compute MSE error for each window
        anomaly_scores = torch.mean((x_anom_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        
    print(f"Computed anomaly scores for {len(anomaly_scores)} beats. Min: {np.min(anomaly_scores):.4f}, Max: {np.max(anomaly_scores):.4f}, Mean: {np.mean(anomaly_scores):.4f}")
    
    # Generate sequential datasets for LSTM
    # Input sequence length: 10
    # Forecast length: 5
    seq_len = 10
    forecast_len = 5
    
    x_lstm_data = []
    y_lstm_data = []
    
    for i in range(len(anomaly_scores) - seq_len - forecast_len + 1):
        x_seq = anomaly_scores[i : i + seq_len]
        y_seq = anomaly_scores[i + seq_len : i + seq_len + forecast_len]
        x_lstm_data.append(x_seq)
        y_lstm_data.append(y_seq)
        
    x_lstm_data = np.array(x_lstm_data)
    y_lstm_data = np.array(y_lstm_data)
    
    print(f"Generated {len(x_lstm_data)} sequences for LSTM forecasting.")
    
    # Reshape input: [N, seq_len, input_dim=1]
    x_lstm_tensor = torch.tensor(x_lstm_data, dtype=torch.float32).unsqueeze(-1)
    y_lstm_tensor = torch.tensor(y_lstm_data, dtype=torch.float32)
    
    lstm_dataset = TensorDataset(x_lstm_tensor, y_lstm_tensor)
    lstm_loader = DataLoader(lstm_dataset, batch_size=16, shuffle=True)
    
    # Define LSTM model, loss, optimizer
    lstm_model = ECGLSTMForecaster(input_dim=1, hidden_dim=32, num_layers=2, forecast_dim=5).to(device)
    lstm_criterion = nn.MSELoss()
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.005)
    
    # Train LSTM
    lstm_model.train()
    lstm_epochs = 30
    for epoch in range(lstm_epochs):
        epoch_loss = 0.0
        for inputs, targets in lstm_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            lstm_optimizer.zero_grad()
            outputs = lstm_model(inputs)
            loss = lstm_criterion(outputs, targets)
            loss.backward()
            lstm_optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            
        avg_loss = epoch_loss / len(x_lstm_data)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{lstm_epochs} - Loss: {avg_loss:.6f}")
            
    # Save LSTM model weights
    lstm_path = os.path.join(BASE_DIR, "backend", "models", "lstm_forecaster.pth")
    torch.save(lstm_model.state_dict(), lstm_path)
    print(f"LSTM model saved to {lstm_path}")
    print("\n--- Training Completed Successfully! ---")

if __name__ == "__main__":
    train_models()
