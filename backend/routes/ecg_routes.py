import os
import json
import numpy as np
from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from backend.services.ecg_service import ECGService
from backend.services.ai_service import AIService
from backend.services.explain_service import ExplainService
from backend.services.digitizer_service import DigitizerService
from backend.utils.helpers import DATASETS_DIR

# Define blueprint
ecg_bp = Blueprint("ecg", __name__)

# Initialize services
ai_service = AIService()
explain_service = ExplainService(ai_service)

# Folder settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOADS_DIR = os.path.join(BASE_DIR, "backend", "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize background SHAP dataset using record 103 normal heartbeats if available
def init_shap_background():
    try:
        # Load Record 103 normal sinus rhythm
        sig, ann, _ = ECGService.load_record("103")
        windows, _, labels = ECGService.segment_heartbeats(sig, ann)
        normal_wins = [w for w, l in zip(windows, labels) if l == "N"]
        if len(normal_wins) > 0:
            explain_service.set_background_data(np.array(normal_wins))
            print("SHAP background data initialized successfully.")
    except Exception as e:
        print(f"SHAP background initialization delayed: {e}")


# Run background initialization
init_shap_background()

def patch_hea_file(hea_filepath, record_name):
    if not os.path.exists(hea_filepath):
        return
    try:
        with open(hea_filepath, 'r') as f:
            lines = f.readlines()
        
        patched_lines = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                patched_lines.append(line)
                continue
                
            if stripped.startswith('#'):
                patched_lines.append(line)
                continue
                
            if idx == 0:
                parts = line.split()
                if len(parts) > 0:
                    parts[0] = record_name
                    patched_lines.append(" ".join(parts) + "\n")
                else:
                    patched_lines.append(line)
            else:
                parts = line.split()
                if len(parts) > 0:
                    # Replace the first token with record_name + extension
                    _, ext = os.path.splitext(parts[0])
                    if not ext:
                        ext = ".dat"
                    parts[0] = f"{record_name}{ext}"
                    patched_lines.append(" ".join(parts) + "\n")
                else:
                    patched_lines.append(line)
                    
        with open(hea_filepath, 'w') as f:
            f.writelines(patched_lines)
        print(f"Successfully patched .hea file: {hea_filepath}")
    except Exception as e:
        print(f"Error patching .hea file {hea_filepath}: {e}")

def extract_metadata_from_hea(record_path):
    metadata = {
        "age": "N/A",
        "gender": "N/A",
        "medications": "None",
        "leads": []
    }
    
    hea_path = f"{record_path}.hea"
    if not os.path.exists(hea_path):
        return metadata
        
    try:
        with open(hea_path, 'r') as f:
            lines = f.readlines()
            
        comments = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                comments.append(stripped[1:].strip())
            elif not stripped.startswith('#') and len(stripped.split()) > 8:
                parts = stripped.split()
                lead_name = parts[-1]
                metadata["leads"].append(lead_name)
                
        for comment in comments:
            parts = comment.split()
            if len(parts) >= 2 and parts[0].replace('-', '').isdigit() and parts[1] in ['M', 'F']:
                age_val = parts[0]
                metadata["age"] = age_val if age_val != "-1" else "N/A"
                metadata["gender"] = "Male" if parts[1] == 'M' else "Female"
            elif ',' in comment or len(comment) > 3:
                if not (len(parts) >= 2 and parts[0].replace('-', '').isdigit() and parts[1] in ['M', 'F']):
                    metadata["medications"] = comment
                    
        if not metadata["leads"]:
            metadata["leads"] = ["Lead I"]
            
        return metadata
    except Exception as e:
        print(f"Error parsing metadata from .hea file: {e}")
        return metadata

@ecg_bp.route("/upload", methods=["POST"])
def upload_ecg():
    """
    POST route to upload MIT-BIH files (.hea, .dat, .atr) or ECG reports (.png, .jpg, .jpeg, .pdf)
    """
    if "files" not in request.files and len(request.files) == 0:
        return jsonify({"error": "No files uploaded"}), 400
        
    uploaded_files = request.files.getlist("files")
    saved_filenames = []
    
    # We expect files to share the same record base name
    record_name = None
    is_report = False
    
    for file in uploaded_files:
        if file.filename == '':
            continue
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOADS_DIR, filename)
        file.save(filepath)
        saved_filenames.append(filename)
        
        base, ext = os.path.splitext(filename)
        ext_lower = ext.lower()
        if ext_lower in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".pdf"]:
            is_report = True
            record_name = f"report_{base}"
        elif ext_lower in [".hea", ".dat", ".atr"]:
            record_name = base
            
    if not record_name:
        return jsonify({"error": "No valid ECG record or report files found"}), 400
        
    if is_report:
        # Digitize report using computer vision and OCR
        report_filepath = os.path.join(UPLOADS_DIR, saved_filenames[0])
        try:
            digitize_res = DigitizerService.digitize_report(report_filepath)
            
            # Save digitized trace as .npy
            npy_path = os.path.join(UPLOADS_DIR, f"{record_name}.npy")
            np.save(npy_path, digitize_res["signal"])
            
            # Save metadata and annotations as json
            meta_path = os.path.join(UPLOADS_DIR, f"{record_name}_meta.json")
            meta_data = {
                "annotations": digitize_res["annotations"],
                "metadata": digitize_res["metadata"],
                "preview_image": digitize_res["preview_image"],
                "quality_metrics": digitize_res.get("quality_metrics", None)
            }
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f)
                
            return jsonify({
                "message": "Successfully uploaded and digitized ECG report",
                "record_name": record_name,
                "files": saved_filenames,
                "is_digitized": True
            }), 200
        except Exception as e:
            print(f"Error digitizing report: {e}")
            return jsonify({"error": f"Failed to digitize uploaded report: {str(e)}"}), 500
    else:
        # Patch the hea file to point to the secure filenames saved on disk
        hea_filepath = os.path.join(UPLOADS_DIR, f"{record_name}.hea")
        patch_hea_file(hea_filepath, record_name)
            
        return jsonify({
            "message": f"Successfully uploaded {len(saved_filenames)} files",
            "record_name": record_name,
            "files": saved_filenames,
            "is_digitized": False
        }), 200

@ecg_bp.route("/predict", methods=["POST"])
def predict():
    """
    POST route to load and evaluate an ECG record.
    JSON Body:
    {
        "record_name": "100",
        "is_upload": false,
        "snippet_start": 0,
        "snippet_len": 3000
    }
    """
    data = request.get_json() or {}
    record_name = data.get("record_name", "100")
    is_upload = data.get("is_upload", False)
    snippet_start = int(data.get("snippet_start", 0))
    snippet_len = int(data.get("snippet_len", 3000))
    
    # Reload model weights in case they were updated
    ai_service.load_models()
    
    # Load raw signal
    signal, annotations, is_synthetic = ECGService.load_record(
        record_name, 
        is_upload=is_upload, 
        uploads_dir=UPLOADS_DIR
    )
    
    # Segment continuous signal into heartbeat windows
    windows, peaks, labels = ECGService.segment_heartbeats(signal, annotations)
    
    if len(windows) == 0:
        if is_upload:
            windows = np.array([])
            reconstructed = np.array([])
            scores = np.array([])
            peaks = []
            labels = []
        else:
            return jsonify({"error": "No valid heartbeats could be segmented from the record."}), 400
        
    # Run Autoencoder to get reconstruction and anomaly scores
    reconstructed, scores = ai_service.predict_anomalies(windows)
    
    # Overwrite labels for digitized reports (is_upload) based on reconstruction MSE scores
    if is_upload:
        for idx, score in enumerate(scores):
            if score >= 0.12:
                labels[idx] = 'V'
            elif score >= 0.05:
                labels[idx] = 'A'
            else:
                labels[idx] = 'N'
    
    # Extract snippet of raw signal to plot in the frontend (prevent overloading UI)
    snippet_end = min(len(signal), snippet_start + snippet_len)
    raw_signal_snippet = signal[snippet_start:snippet_end].tolist()
    
    # Filter peaks and labels within our snippet range for rendering
    snippet_peaks = []
    snippet_labels = []
    snippet_beat_indices = [] # Map local peak to its window index in full list
    
    for idx, peak in enumerate(peaks):
        if snippet_start <= peak < snippet_end:
            snippet_peaks.append(int(peak - snippet_start))
            snippet_labels.append(labels[idx])
            snippet_beat_indices.append(idx)
            
    # Format the segmented windows that belong to this snippet
    snippet_windows = []
    snippet_reconstructed = []
    snippet_scores = []
    
    for idx in snippet_beat_indices:
        snippet_windows.append(windows[idx].tolist())
        snippet_reconstructed.append(reconstructed[idx].tolist())
        snippet_scores.append(float(scores[idx]))
        
    # Load metadata
    preview_image = None
    quality_metrics = None
    if is_upload:
        meta_path = os.path.join(UPLOADS_DIR, f"{record_name}_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                    metadata = meta_data.get("metadata", {})
                    preview_image = meta_data.get("preview_image", None)
                    quality_metrics = meta_data.get("quality_metrics", None)
            except Exception as e:
                print(f"Error loading custom metadata file: {e}")
                record_path = os.path.join(UPLOADS_DIR, record_name)
                metadata = extract_metadata_from_hea(record_path)
        else:
            record_path = os.path.join(UPLOADS_DIR, record_name)
            metadata = extract_metadata_from_hea(record_path)
    else:
        record_path = os.path.join(DATASETS_DIR, record_name)
        metadata = extract_metadata_from_hea(record_path)
        if is_synthetic:
            metadata = {
                "age": "69" if record_name == "100" else "72",
                "gender": "Male" if record_name == "100" else "Female",
                "medications": "Aldomet, Inderal" if record_name == "100" else "None",
                "leads": ["Lead I (Synthetic)"]
            }
            
    # Return response
    return jsonify({
        "record_name": record_name,
        "is_synthetic": is_synthetic,
        "raw_signal": raw_signal_snippet,
        "sampling_rate": 360,
        "peaks": snippet_peaks,
        "labels": snippet_labels,
        "beat_indices": snippet_beat_indices,
        "windows": snippet_windows,
        "reconstructed": snippet_reconstructed,
        "anomaly_scores": snippet_scores,
        "all_anomaly_scores": [float(s) for s in scores], # Send full list of scores for long-term forecasting
        "metadata": metadata,
        "preview_image": preview_image,
        "quality_metrics": quality_metrics
    }), 200

@ecg_bp.route("/uploads/<filename>")
def get_upload(filename):
    """Serve files uploaded to the backend folder."""
    return send_from_directory(UPLOADS_DIR, filename)

@ecg_bp.route("/forecast", methods=["POST"])
def forecast():
    """
    POST route to forecast future anomaly scores using LSTM.
    JSON Body:
    {
        "anomaly_scores": [0.02, 0.03, 0.02, 0.05, 0.07, 0.08, 0.12, 0.11, 0.14, 0.15],
        "peaks": [100, 350, ...]
    }
    """
    data = request.get_json() or {}
    scores = data.get("anomaly_scores", [])
    peaks = data.get("peaks", [])
    
    if not scores:
        return jsonify({"error": "No anomaly scores provided for forecasting"}), 400
        
    # Run prediction
    forecasted = ai_service.forecast_risk(scores, peaks=peaks)
    
    return jsonify({
        "input_scores": scores[-10:] if len(scores) >= 10 else scores,
        "forecasted_scores": forecasted
    }), 200

@ecg_bp.route("/explain", methods=["POST"])
def explain():
    """
    POST route to explain a single heartbeat window using SHAP.
    JSON Body:
    {
        "window": [180 floats]
    }
    """
    data = request.get_json() or {}
    window_list = data.get("window", [])
    
    if len(window_list) != 180:
        return jsonify({"error": "Window must contain exactly 180 values"}), 400
        
    target_window = np.array(window_list)
    
    # If SHAP background dataset is not set, initialize it
    if explain_service.background_data is None:
        init_shap_background()
        
    # Generate explanation
    try:
        explanation = explain_service.explain_heartbeat(target_window)
        return jsonify(explanation), 200
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        return jsonify({"error": f"SHAP engine error: {str(e)}"}), 500

@ecg_bp.route("/risk", methods=["POST"])
def calculate_risk():
    """
    POST route to calculate clinical risk profile.
    JSON Body:
    {
        "current_score": 0.14,
        "forecasted_scores": [0.15, 0.18, 0.22, 0.26, 0.28]
    }
    """
    data = request.get_json() or {}
    current_score = data.get("current_score")
    forecasted_scores = data.get("forecasted_scores", [])
    
    if current_score is None:
        return jsonify({"error": "current_score is required"}), 400
        
    risk_profile = ai_service.compute_risk_profile(current_score, forecasted_scores)
    
    return jsonify(risk_profile), 200
