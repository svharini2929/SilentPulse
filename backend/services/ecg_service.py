import os
import numpy as np
import scipy.signal
import wfdb
from backend.utils.helpers import download_mitbih_record, generate_synthetic_ecg, DATASETS_DIR

class ECGService:
    """
    Service to load, preprocess, normalize, and segment ECG signals.
    Supports MIT-BIH PhysioNet records and fallback synthetic signals.
    """
    
    @staticmethod
    def load_record(record_name, is_upload=False, uploads_dir=None):
        """
        Loads an ECG record (lead 0 signal and annotations).
        Can load from datasets directory or uploads directory.
        Supports loading digitized .npy signals and custom annotations.
        """
        if is_upload and uploads_dir:
            npy_path = os.path.join(uploads_dir, f"{record_name}.npy")
            meta_path = os.path.join(uploads_dir, f"{record_name}_meta.json")
            if os.path.exists(npy_path):
                try:
                    signal = np.load(npy_path)
                    ann = []
                    if os.path.exists(meta_path):
                        import json
                        with open(meta_path, 'r') as f:
                            meta_data = json.load(f)
                            ann = [(int(a[0]), str(a[1])) for a in meta_data.get("annotations", [])]
                    print(f"Successfully loaded digitized signal from {npy_path}")
                    return signal, ann, False
                except Exception as e:
                    print(f"Error loading digitized .npy signal: {e}")
            
            record_path = os.path.join(uploads_dir, record_name)
        else:
            record_path = os.path.join(DATASETS_DIR, record_name)
            
            # Ensure files exist in datasets folder
            # For MIT-BIH dataset, try to download if they don't exist
            if not is_upload and record_name in ["100", "103"]:
                hea_exists = os.path.exists(f"{record_path}.hea")
                if not hea_exists:
                    download_mitbih_record(record_name)
                    
        # Check if record header file exists
        if not os.path.exists(f"{record_path}.hea"):
            print(f"Header file not found for: {record_name}. Using synthetic signal.")
            # Fallback to synthetic
            anomaly_type = 'pvc' if record_name == "100" else None
            signal, ann = generate_synthetic_ecg(num_samples=10000, anomaly_type=anomaly_type)
            return signal, ann, True

        try:
            # Load wfdb record
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal[:, 0]  # Take the first lead
            signal = np.nan_to_num(signal)  # Replace NaNs with zeros
            
            # Load annotations if they exist
            ann = []
            try:
                annotation = wfdb.rdann(record_path, 'atr')
                for sample, symbol in zip(annotation.sample, annotation.symbol):
                    ann.append((int(sample), str(symbol)))
            except Exception as e:
                print(f"Annotations not loaded for {record_name}: {e}. Will detect peaks programmatically.")
                
            return signal, ann, False
            
        except Exception as e:
            print(f"wfdb load error for {record_name}: {e}. Falling back to synthetic.")
            anomaly_type = 'pvc' if record_name == "100" else None
            signal, ann = generate_synthetic_ecg(num_samples=10000, anomaly_type=anomaly_type)
            return signal, ann, True

    @staticmethod
    def detect_r_peaks(signal, sampling_rate=360):
        """
        A robust Pan-Tompkins peak detection algorithm using scipy's find_peaks
        on a bandpass-filtered signal.
        """
        # Step 1: Normalize signal
        norm_sig = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)
        
        # Step 2: Bandpass filter between 5Hz and 15Hz to focus on QRS complex energy
        try:
            nyq = 0.5 * sampling_rate
            low = 5.0 / nyq
            high = 15.0 / nyq
            b, a = scipy.signal.butter(3, [low, high], btype='band')
            filtered_sig = scipy.signal.filtfilt(b, a, norm_sig)
        except Exception as e:
            print(f"Filter design failed in detect_r_peaks: {e}. Using raw normalized signal.")
            filtered_sig = norm_sig
            
        # Step 3: Differentiate the signal
        diff_sig = np.diff(filtered_sig)
        # Pad with 0 to match original signal length
        diff_sig = np.pad(diff_sig, (1, 0), 'edge')
        
        # Step 4: Square the signal to emphasize slope changes
        squared_sig = diff_sig ** 2
        
        # Step 5: Moving window integration (approx 150ms window)
        window_width = int(0.15 * sampling_rate)
        integrated_sig = np.convolve(squared_sig, np.ones(window_width)/window_width, mode='same')
        
        # Step 6: Find threshold crossings using find_peaks
        # Min distance between beats: 0.25s (up to 240 BPM support)
        min_dist = int(0.25 * sampling_rate)
        
        # Dynamic height threshold based on 90th percentile of integrated signal
        threshold = np.percentile(integrated_sig, 90) * 0.35
        threshold = max(threshold, 1e-8)
        
        peaks, _ = scipy.signal.find_peaks(integrated_sig, distance=min_dist, height=threshold)
        
        # Step 7: Refine peak locations to the local maximum deviation in raw signal
        refined_peaks = []
        search_width = int(0.05 * sampling_rate) # +/- 50ms search window
        
        for p in peaks:
            search_start = max(0, p - search_width)
            search_end = min(len(signal), p + search_width)
            if search_end > search_start:
                sub_sig = signal[search_start:search_end]
                baseline = np.median(signal)
                dev = np.abs(sub_sig - baseline)
                exact_p = search_start + np.argmax(dev)
                refined_peaks.append(int(exact_p))
                
        # Remove potential duplicate indices and sort
        refined_peaks = sorted(list(set(refined_peaks)))
        
        # Post-validate min distance
        final_peaks = []
        last_p = -min_dist
        for p in refined_peaks:
            if p - last_p >= min_dist:
                final_peaks.append(p)
                last_p = p
                
        return final_peaks
                    
        return peaks

    @staticmethod
    def normalize_signal(signal):
        """Scale ECG signal to [0, 1] range."""
        min_val = np.min(signal)
        max_val = np.max(signal)
        denom = (max_val - min_val) if max_val != min_val else 1.0
        return (signal - min_val) / denom

    @staticmethod
    def segment_heartbeats(signal, annotations=None, window_size=180, sampling_rate=360):
        """
        Segments continuous ECG signal into heartbeat windows of fixed window_size (e.g. 180)
        centered around R-peaks.
        
        Returns:
            windows (np.ndarray): shape [num_windows, window_size]
            peak_indices (list): list of R-peak indices for each window
            labels (list): list of arrhythmia labels ('N', 'V', etc.)
        """
        # If no annotations are provided, detect peaks programmatically
        if not annotations or len(annotations) == 0:
            peaks = ECGService.detect_r_peaks(signal, sampling_rate)
            labels = ['N'] * len(peaks)
        else:
            # Use provided annotations (filter out non-beat annotations)
            # Standard beats are marked by N, L, R, V, A, etc.
            valid_symbols = {'N', 'L', 'R', 'V', 'A', 'E', 'F', 'j', 'a', 'J'}
            peaks = []
            labels = []
            for sample, symbol in annotations:
                if symbol in valid_symbols:
                    peaks.append(sample)
                    labels.append(symbol)
        
        windows = []
        final_peaks = []
        final_labels = []
        half_win = window_size // 2
        
        for r_peak, label in zip(peaks, labels):
            start = r_peak - half_win
            end = r_peak + half_win
            
            # Boundary check
            if start >= 0 and end <= len(signal):
                win = signal[start:end]
                # Normalize window individually (important for Autoencoder robustness)
                win_min = np.min(win)
                win_max = np.max(win)
                norm_win = (win - win_min) / (win_max - win_min + 1e-8)
                
                windows.append(norm_win)
                final_peaks.append(r_peak)
                final_labels.append(label)
                
        return np.array(windows), final_peaks, final_labels
