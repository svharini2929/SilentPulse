import os
import torch
import numpy as np
import shap
import scipy.signal

class ExplainService:
    """
    Service that uses SHAP (SHapley Additive exPlanations) to explain 
    why the Autoencoder flags a heartbeat segment as anomalous.
    Optimized for fast computation via 60-feature downsampling and caching.
    """
    def __init__(self, ai_service):
        self.ai_service = ai_service
        self.device = ai_service.device
        self.background_data = None
        self.cache = {} # Cache to bypass computation for identical segments
        
    def set_background_data(self, normal_windows):
        """
        Sets the background dataset of normal ECG heartbeats (e.g. 5-10 windows).
        Used by KernelExplainer as reference normal beats.
        """
        # Limit to 5 samples for fast computational speed in Flask APIs
        if len(normal_windows) > 5:
            self.background_data = normal_windows[:5]
        else:
            self.background_data = normal_windows
            
    def predict_anomaly_score_fn(self, numpy_windows):
        """
        Model prediction wrapper function for SHAP.
        Takes numpy windows [N, 180] and returns numpy anomaly scores [N].
        """
        # Convert to tensor
        tensor_windows = torch.tensor(numpy_windows, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            errors, _ = self.ai_service.autoencoder.get_reconstruction_error(tensor_windows)
            scores = errors.cpu().numpy()
            
        return scores

    def get_fallback_explanation(self, target_window):
        """
        Generates a fast fallback explanation based on squared reconstruction error.
        Used when SHAP fails or times out.
        """
        window_1d = target_window.flatten()
        
        # Get reconstruction from Autoencoder
        tensor_window = torch.tensor(window_1d.reshape(1, -1), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, reconstructed_t = self.ai_service.autoencoder.get_reconstruction_error(tensor_window)
            reconstructed = reconstructed_t.cpu().numpy().flatten()
            
        # Apply Savitzky-Golay filter to the reconstruction as done in ai_service
        try:
            reconstructed = scipy.signal.savgol_filter(reconstructed, 17, 3)
        except Exception as e:
            print(f"Savitzky-Golay failed in fallback: {e}")
            
        # Compute squared reconstruction error at each point
        sq_err = (window_1d - reconstructed) ** 2
        shap_values = sq_err.tolist()
        
        p_wave_shap = shap_values[0:50]
        pr_interval_shap = shap_values[50:80]
        qrs_complex_shap = shap_values[80:105]
        st_segment_shap = shap_values[105:130]
        t_wave_shap = shap_values[130:180]
        
        segment_importances = {
            "p_wave": float(np.mean(p_wave_shap)),
            "pr_interval": float(np.mean(pr_interval_shap)),
            "qrs_complex": float(np.mean(qrs_complex_shap)),
            "st_segment": float(np.mean(st_segment_shap)),
            "t_wave": float(np.mean(t_wave_shap))
        }
        
        max_segment = max(segment_importances, key=segment_importances.get)
        max_val = segment_importances[max_segment]
        
        explanation_summary = ""
        if max_val < 0.005:
            explanation_summary = "Signal matches normal baseline profile. Minimal deviation across all ECG wave components (Fallback error analysis)."
        else:
            if max_segment == "qrs_complex":
                explanation_summary = "Deviation detected in the QRS complex. Indicates significant distortion in the ventricular depolarization phase, often associated with Premature Ventricular Contractions (PVC) or bundle branch blocks (Fallback error analysis)."
            elif max_segment == "t_wave":
                explanation_summary = "Deviation detected in the T-wave. Suggests anomalies in ventricular repolarization (e.g., inverted T-wave, hyperkalemia, or myocardial ischemia) (Fallback error analysis)."
            elif max_segment == "st_segment":
                explanation_summary = "Deviation detected in the ST segment. Often associated with ST elevation or depression, indicating potential acute myocardial infarction or ischemia (Fallback error analysis)."
            elif max_segment == "p_wave":
                explanation_summary = "Deviation detected in the P-wave. Points to atrial depolarization abnormalities, such as atrial enlargement or ectopic atrial rhythms (Fallback error analysis)."
            else:
                explanation_summary = "Deviation detected in the PR interval, indicating possible AV blocks or conduction delays (Fallback error analysis)."
                
        return {
            "shap_values": shap_values,
            "segment_importances": segment_importances,
            "max_contributing_segment": max_segment,
            "explanation_summary": explanation_summary
        }

    def explain_heartbeat(self, target_window):
        """
        Computes SHAP values for a target heartbeat window of length 180.
        Optimized by downsampling to 60 features for faster Kernel SHAP.
        
        Args:
            target_window (np.ndarray): Shape [180] (or [1, 180])
        Returns:
            dict containing shap_values, segment_importances, max_contributing_segment, and explanation_summary
        """
        # Reshape target window if it's 1D
        if len(target_window.shape) == 1:
            target_window = target_window.reshape(1, -1)
            
        # 1. Cache lookup: check rounded representation
        cache_key = tuple(np.round(target_window.flatten(), 4).tolist())
        if cache_key in self.cache:
            print("SHAP explanation cache hit!")
            return self.cache[cache_key]

        # Ensure we have background data. If not, generate a mock normal background
        if self.background_data is None or len(self.background_data) == 0:
            print("Warning: Background data not set for SHAP. Generating mock background.")
            # Standard normal baseline centered at 0.5
            self.background_data = np.zeros((5, 180))
            for i in range(5):
                t = np.linspace(-3, 3, 180)
                # normal QRS shape
                self.background_data[i] = 0.5 + 0.5 * np.exp(-t**2) + np.random.normal(0, 0.02, 180)

        try:
            # 2. Downsampled 60-feature Kernel SHAP speedup
            # Average every 3 samples to go from 180 to 60
            target_60 = target_window.reshape(1, 60, 3).mean(axis=2)
            background_60 = self.background_data.reshape(self.background_data.shape[0], 60, 3).mean(axis=2)
            
            # Wrapper prediction function that upsamples candidate samples back to 180 dims
            def predict_anomaly_score_fn_60(numpy_windows_60):
                numpy_windows_180 = np.repeat(numpy_windows_60, 3, axis=-1)
                return self.predict_anomaly_score_fn(numpy_windows_180)
                
            # Initialize KernelExplainer on 60 dimensions
            explainer = shap.KernelExplainer(predict_anomaly_score_fn_60, background_60)
            
            # Run Kernel SHAP with nsamples=40 (extremely fast)
            shap_values_raw = explainer.shap_values(target_60, nsamples=40)
            
            if isinstance(shap_values_raw, list):
                shap_values_60 = shap_values_raw[0].flatten()
            else:
                shap_values_60 = shap_values_raw.flatten()
                
            # Upsample SHAP values back to 180 dimensions by repeating each value 3 times
            shap_values = np.repeat(shap_values_60, 3).tolist()
            
        except Exception as e:
            print(f"Kernel SHAP failed with error: {e}. Using squared reconstruction error fallback.")
            explanation = self.get_fallback_explanation(target_window)
            self.cache[cache_key] = explanation
            return explanation

        # Segment-wise analysis of ECG features (MIT-BIH is 360Hz. 180 window is 0.5 seconds centered at sample 90)
        # Approximate sample windows for components relative to R-peak at sample 90:
        # P-wave: 0 to 50
        # PR interval / Q-wave: 50 to 80
        # QRS complex (R-peak, S-wave): 80 to 105
        # ST segment: 105 to 130
        # T-wave: 130 to 180
        
        p_wave_shap = shap_values[0:50]
        pr_interval_shap = shap_values[50:80]
        qrs_complex_shap = shap_values[80:105]
        st_segment_shap = shap_values[105:130]
        t_wave_shap = shap_values[130:180]
        
        # Compute absolute mean contribution (importance) for each wave segment
        segment_importances = {
            "p_wave": float(np.mean(np.abs(p_wave_shap))),
            "pr_interval": float(np.mean(np.abs(pr_interval_shap))),
            "qrs_complex": float(np.mean(np.abs(qrs_complex_shap))),
            "st_segment": float(np.mean(np.abs(st_segment_shap))),
            "t_wave": float(np.mean(np.abs(t_wave_shap)))
        }
        
        # Clinical explanation mapping
        max_segment = max(segment_importances, key=segment_importances.get)
        max_val = segment_importances[max_segment]
        
        explanation_summary = ""
        if max_val < 0.005:
            explanation_summary = "Signal matches normal baseline profile. Minimal deviation across all ECG wave components."
        else:
            if max_segment == "qrs_complex":
                explanation_summary = "High SHAP value in the QRS complex. Indicates significant distortion in the ventricular depolarization phase, often associated with Premature Ventricular Contractions (PVC) or bundle branch blocks."
            elif max_segment == "t_wave":
                explanation_summary = "High SHAP value in the T-wave. Suggests anomalies in ventricular repolarization (e.g., inverted T-wave, hyperkalemia, or myocardial ischemia)."
            elif max_segment == "st_segment":
                explanation_summary = "High SHAP value in the ST segment. Often associated with ST elevation or depression, indicating potential acute myocardial infarction or ischemia."
            elif max_segment == "p_wave":
                explanation_summary = "High SHAP value in the P-wave. Points to atrial depolarization abnormalities, such as atrial enlargement or ectopic atrial rhythms."
            else:
                explanation_summary = "SHAP values highlight deviations in the PR interval, indicating possible AV blocks or conduction delays."
                
        explanation = {
            "shap_values": shap_values,
            "segment_importances": segment_importances,
            "max_contributing_segment": max_segment,
            "explanation_summary": explanation_summary
        }
        
        # Save to cache
        self.cache[cache_key] = explanation
        return explanation
