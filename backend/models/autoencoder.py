import torch
import torch.nn as nn

class ECGAutoencoder(nn.Module):
    """
    A PyTorch Autoencoder model for ECG signal anomaly detection.
    It takes an ECG segment of length 180, compresses it into a low-dimensional
    bottleneck representation, and tries to reconstruct the original signal.
    """
    def __init__(self, input_dim=180, latent_dim=8):
        super(ECGAutoencoder, self).__init__()
        
        # Encoder: compress high-dimensional ECG signal to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: reconstruct original ECG signal from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim) # Linear output to reconstruct arbitrary signal values
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): Input ECG segment of shape [BatchSize, input_dim]
        Returns:
            reconstructed (Tensor): Reconstructed ECG segment of shape [BatchSize, input_dim]
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def get_reconstruction_error(self, x):
        """
        Helper method to compute reconstruction error (Mean Squared Error) for each sample.
        Args:
            x (Tensor): Input shape [BatchSize, input_dim]
        Returns:
            errors (Tensor): MSE for each window in the batch of shape [BatchSize]
            reconstructed (Tensor): Reconstructed segments of shape [BatchSize, input_dim]
        """
        reconstructed = self.forward(x)
        # Compute squared differences element-wise
        squared_diff = (x - reconstructed) ** 2
        # Mean across the features (dim=1)
        errors = torch.mean(squared_diff, dim=1)
        return errors, reconstructed
