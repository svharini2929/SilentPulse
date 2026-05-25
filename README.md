# SilentPulse: AI Cardiovascular Anomaly Detection, Forecasting & Explainability

SilentPulse is a full-stack AI-powered cardiovascular monitoring web application. It processes electrocardiogram (ECG) signals, flags anomalies using a Deep Autoencoder (PyTorch), predicts future risk trends using an LSTM model, explains predictions using SHAP (Shapley Additive exPlanations), and presents interactive results in a futuristic, dark-themed medical dashboard.

---

## Technical Architecture & Workflow

```
ECG Input (.hea/.dat/.atr) 
    │
    ▼ (wfdb library)
Normalization & R-Peak Detection
    │
    ▼
Heartbeat Segmentation (180-sample windows)
    │
    ├──► PyTorch Autoencoder ──► Reconstruction Error ──► Anomaly Score (MSE)
    │                                                        │
    ├──► SHAP Explainability Engine (KernelExplainer)        ▼
    │         │                                      PyTorch LSTM Forecaster
    │         ▼                                              │
    │   ECG Wave Segment Impact                              ▼
    │   (P, PR, QRS, ST, T attributions)             Future Risk Trend Prognosis
    │                                                        │
    └──────────────────────┬─────────────────────────────────┘
                           ▼
                  Flask API Endpoints
                           │
                           ▼ (CORS, JSON)
                  React Vite Dashboard (Chart.js + Tailwind CSS)
```

---

## Features

1. **ECG Waveform Visualization**: Continuous ECG signal display with labeled, color-coded R-peaks (Normal vs Anomaly).
2. **AI Anomaly Detection**: A PyTorch Dense Autoencoder trained on normal heartbeats. Arrhythmia beats (such as Premature Ventricular Contractions) generate high reconstruction error.
3. **Cardiovascular Risk Forecasting**: A PyTorch LSTM model that inputs past anomaly scores and forecasts the next 5 steps to predict patient deterioration.
4. **SHAP Explainability**: Explains the anatomical driver of the anomaly score by mapping game theory-based attributions to physiological waves (P-wave, QRS complex, ST segment, T-wave).
5. **Interactive Doctor Dashboard**: Premium dark UI with automated clinical suggestions, active alert levels (Safe, Mild, Moderate, Critical), and clickable heartbeat waveforms.

---

## Folder Structure

```text
silentpulse/
├── backend/
│   ├── app.py                  # Flask Entrypoint
│   ├── routes/
│   │   └── ecg_routes.py       # API Endpoint Controllers (/upload, /predict, /forecast, /explain, /risk)
│   ├── models/
│   │   ├── autoencoder.py      # PyTorch Autoencoder Network
│   │   └── lstm_forecaster.py  # PyTorch LSTM Forecaster Network
│   ├── services/
│   │   ├── ecg_service.py      # Preprocessing, normalization, and R-Peak detection
│   │   ├── ai_service.py       # Autoencoder and LSTM inference & risk evaluation
│   │   └── explain_service.py  # SHAP explanation calculations
│   ├── utils/
│   │   └── helpers.py          # PhysioNet downloader & synthetic generator
│   └── uploads/                # Temporary directory for uploaded files
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── ECGUpload.jsx   # Drag & Drop upload & demo selection
│   │   ├── pages/
│   │   │   └── Dashboard.jsx   # Core UI container
│   │   ├── charts/
│   │   │   ├── ECGWaveformChart.jsx       # Strip charts & heartbeat overlay comparisons
│   │   │   ├── ReconstructionErrorChart.jsx # Arrhythmia Loss over time
│   │   │   ├── ForecastTrendChart.jsx      # Future risk projection curves
│   │   │   └── ShapContributionChart.jsx   # Feature importance and SHAP attributions
│   │   ├── services/
│   │   │   └── api.js          # Backend API client requests
│   │   ├── App.jsx
│   │   ├── index.css           # Global stylesheets & animations
│   │   └── main.jsx
│   ├── index.html
│   ├── tailwind.config.js      # Styling themes
│   └── package.json
├── training/
│   └── train.py                # Pipeline to download/generate and train models
└── requirements.txt            # Python dependencies
```

---

## Setup & Running Instructions

### 1. Backend Setup

Prerequisites: Python 3.10+ (Tested on Python 3.13)

1. Open a terminal in the project root:
   ```bash
   cd silentpulse
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows (Powershell):
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run model training (this will download sample records and train PyTorch models):
   ```bash
   python training/train.py
   ```
5. Run the Flask server:
   ```bash
   python backend/app.py
   ```
   The API will start running at `http://127.0.0.1:5000`.

### 2. Frontend Setup

Prerequisites: Node.js v18+

1. Open a new terminal in the `frontend` folder:
   ```bash
   cd silentpulse/frontend
   ```
2. Install Node packages:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
   The dashboard will be available at `http://localhost:5173`.

---

## API Documentation

### `POST /api/upload`
Uploads multiple ECG files.
* **Payload**: Form-data with a `files` key containing `.hea` and `.dat` files.
* **Response**: `200 OK` with uploaded base record name.

### `POST /api/predict`
Calculates ECG predictions and anomaly scores for the record.
* **Payload**: `{ "record_name": "100", "is_upload": false }`
* **Response**: Slices of raw/reconstructed waves, peak locations, and reconstruction MSE scores.

### `POST /api/forecast`
Runs LSTM forecasting.
* **Payload**: `{ "anomaly_scores": [float] }`
* **Response**: Next 5 forecasted anomaly scores.

### `POST /api/explain`
Calculates SHAP explainability for a single heartbeat window.
* **Payload**: `{ "window": [float * 180] }`
* **Response**: Pointwise SHAP values and wave segment importances.

### `POST /api/risk`
Generates risk assessments and alerts.
* **Payload**: `{ "current_score": float, "forecasted_scores": [float] }`
* **Response**: Status level (Safe/Mild/Moderate/Critical), alerts, and actionable recommendations.
