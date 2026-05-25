import os
import urllib.request
import numpy as np

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

def ensure_directories():
    """Ensure that the datasets and uploads directories exist."""
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "backend", "uploads"), exist_ok=True)

def download_mitbih_record(record_name):
    """
    Downloads MIT-BIH Arrhythmia Database record files (.hea, .dat, .atr)
    from PhysioNet to the local datasets folder.
    """
    ensure_directories()
    
    extensions = [".hea", ".dat", ".atr"]
    base_url = "https://physionet.org/files/mitdb/1.0.0/"
    
    print(f"Checking for MIT-BIH record: {record_name}")
    success = True
    
    for ext in extensions:
        filename = f"{record_name}{ext}"
        filepath = os.path.join(DATASETS_DIR, filename)
        
        # Download if it doesn't exist
        if not os.path.exists(filepath):
            url = f"{base_url}{filename}"
            print(f"Downloading {url} to {filepath}...")
            try:
                # Setup request with User-Agent to avoid blocker
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req, timeout=15) as response, open(filepath, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                success = False
                # Cleanup partial files
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            print(f"File already exists: {filename}")
            
    return success

def generate_synthetic_ecg(num_samples=10000, sampling_rate=360, noise_level=0.05, anomaly_type=None, bpm=None):
    """
    Generates a realistic synthetic ECG signal with standard waves (P, QRS, T)
    for testing when the MIT-BIH dataset cannot be downloaded.
    
    anomaly_type can be:
        - None: Normal sinus rhythm
        - 'pvc': Premature Ventricular Contraction (wide, early, distorted QRS)
        - 'bradycardia': Slow heart rate
        - 'tachycardia': Fast heart rate
    """
    t = np.arange(num_samples) / sampling_rate
    ecg = np.zeros(num_samples)
    
    # Heart rate parameters
    if bpm is None:
        bpm = 72
        if anomaly_type == 'bradycardia':
            bpm = 45
        elif anomaly_type == 'tachycardia':
            bpm = 130
        
    heart_rate_period = 60.0 / bpm  # duration of one beat in seconds
    samples_per_beat = int(heart_rate_period * sampling_rate)
    
    i = 0
    beat_count = 0
    annotations = [] # Store annotations (sample index, symbol)
    
    while i < num_samples - samples_per_beat:
        beat_start = i
        
        # Determine if this beat is an anomaly
        is_pvc = (anomaly_type == 'pvc') and (beat_count % 5 == 3) # Make every 5th beat an anomaly
        
        # Define relative widths and amplitudes of waves
        if is_pvc:
            # PVC: Early, wide, high amplitude QRS, no P wave, inverted T wave
            pvc_offset = -int(samples_per_beat * 0.15) # PVC occurs early
            current_start = max(0, beat_start + pvc_offset)
            
            # PVC QRS wave (wide, deep, and tall)
            qrs_width = int(sampling_rate * 0.18) # wider than normal (0.18s vs 0.08s)
            qrs_t = np.linspace(-3, 3, qrs_width)
            qrs_signal = -2.5 * np.exp(-qrs_t**2) + 0.8 * np.exp(-(qrs_t - 0.5)**2 / 0.1)
            
            # Place PVC QRS
            qrs_idx = current_start + int(samples_per_beat * 0.2)
            ecg[qrs_idx : qrs_idx + qrs_width] += qrs_signal
            annotations.append((qrs_idx + qrs_width // 2, 'V')) # 'V' is PVC annotation
            
            # Inverted T wave
            t_width = int(sampling_rate * 0.25)
            t_t = np.linspace(-3, 3, t_width)
            t_signal = -0.6 * np.exp(-t_t**2) # inverted
            t_idx = qrs_idx + qrs_width + int(sampling_rate * 0.05)
            ecg[t_idx : t_idx + t_width] += t_signal
            
            # Move index early for next beat
            i += int(samples_per_beat * 0.8)
        else:
            # Normal beat: P-wave, Q-wave, R-wave, S-wave, T-wave
            # 1. P-wave
            p_width = int(sampling_rate * 0.08)
            p_t = np.linspace(-3, 3, p_width)
            p_signal = 0.15 * np.exp(-p_t**2)
            p_idx = beat_start + int(samples_per_beat * 0.15)
            ecg[p_idx : p_idx + p_width] += p_signal
            
            # 2. QRS Complex
            q_width = int(sampling_rate * 0.02)
            q_t = np.linspace(-3, 3, q_width)
            q_signal = -0.15 * np.exp(-q_t**2)
            
            r_width = int(sampling_rate * 0.04)
            r_t = np.linspace(-3, 3, r_width)
            r_signal = 1.2 * np.exp(-r_t**2)
            
            s_width = int(sampling_rate * 0.03)
            s_t = np.linspace(-3, 3, s_width)
            s_signal = -0.35 * np.exp(-s_t**2)
            
            q_idx = p_idx + p_width + int(sampling_rate * 0.03)
            r_idx = q_idx + q_width
            s_idx = r_idx + r_width
            
            ecg[q_idx : q_idx + q_width] += q_signal
            ecg[r_idx : r_idx + r_width] += r_signal
            ecg[s_idx : s_idx + s_width] += s_signal
            annotations.append((r_idx + r_width // 2, 'N')) # 'N' is normal annotation
            
            # 3. T-wave
            t_width = int(sampling_rate * 0.15)
            t_t = np.linspace(-3, 3, t_width)
            t_signal = 0.35 * np.exp(-t_t**2)
            t_idx = s_idx + s_width + int(sampling_rate * 0.08)
            ecg[t_idx : t_idx + t_width] += t_signal
            
            i += samples_per_beat
            
        beat_count += 1
        
    # Add random high frequency noise and baseline wander
    noise = np.random.normal(0, noise_level, num_samples)
    baseline_wander = 0.15 * np.sin(2 * np.pi * 0.15 * t)  # slow 0.15Hz drift
    
    ecg = ecg + noise + baseline_wander
    
    # Scale between -1 and 1 or standard ranges
    ecg = (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg)) * 2.0 - 1.0
    
    return ecg, annotations

def write_synthetic_mitbih_files(record_name):
    """
    Writes synthetic ECG data in a simple plain-text/binary format or 
    mocks the structure so wfdb can read it, or creates mock files.
    Note: Creating proper binary wfdb (.hea/.dat/.atr) files from scratch
    manually can be error-prone. Instead, we can write a helper function that
    our ecg_service will call when it detects that the files are synthetic.
    We will write simple mock files (.hea, .dat, .atr) that describe the dataset,
    so that the app folder structure remains complete.
    """
    ensure_directories()
    
    # We will write a metadata description file for the synthetic record
    hea_path = os.path.join(DATASETS_DIR, f"{record_name}.hea")
    dat_path = os.path.join(DATASETS_DIR, f"{record_name}.dat")
    atr_path = os.path.join(DATASETS_DIR, f"{record_name}.atr")
    
    with open(hea_path, "w") as f:
        f.write(f"{record_name} 2 360 10000\n")
        f.write(f"{record_name}.dat 212 200/mV 12 0 0 0 0 ECG1\n")
        f.write(f"{record_name}.dat 212 200/mV 12 0 0 0 0 ECG2\n")
        f.write("# Synthetic SilentPulse ECG Record\n")
        
    with open(dat_path, "wb") as f:
        # Write dummy binary data
        f.write(b"\x00" * 30000)
        
    with open(atr_path, "w") as f:
        f.write("MOCK ANNOTATION")
        
    print(f"Created mock MIT-BIH files for record: {record_name}")
