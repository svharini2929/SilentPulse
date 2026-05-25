/* ==========================================================================
   SilentPulse Frontend Telemetry Logic
   ========================================================================== */

const API_BASE_URL = window.location.port === "5000" ? "/api" : "http://127.0.0.1:5000/api";

// Application State
let state = {
  recordName: "",
  isUpload: false,
  selectedBeatIndex: 0,
  
  // Data vectors
  rawSignal: [],
  peaks: [],
  labels: [],
  windows: [],
  reconstructed: [],
  anomalyScores: [],
  allAnomalyScores: [],
  
  // Prognosis & Explainability
  forecastedScores: [],
  riskProfile: null,
  shapData: null
};

// Chart.js Instances
let charts = {
  waveform: null,
  beatComparison: null,
  reconstructionLoss: null,
  forecast: null,
  shapBar: null,
  shapLine: null
};

// DOM Cache
const dom = {
  // Navigation elements
  activePatientId: document.getElementById("active-patient-id"),
  navSessionBadge: document.getElementById("nav-session-badge"),
  sessionTag: document.getElementById("session-tag"),
  
  // Containers
  landingContainer: document.getElementById("landing-container"),
  dashboardContainer: document.getElementById("dashboard-container"),
  
  // Landing Upload elements
  landingDropzone: document.getElementById("landing-dropzone"),
  landingFileInput: document.getElementById("landing-file-input"),
  btnLandingBrowse: document.getElementById("btn-landing-browse"),
  landingUploadIdle: document.getElementById("landing-upload-idle"),
  landingUploadLoading: document.getElementById("landing-upload-loading"),
  landingUploadStatus: document.getElementById("landing-upload-status"),
  
  // Sidebar Upload elements
  sidebarDropzone: document.getElementById("sidebar-dropzone"),
  sidebarFileInput: document.getElementById("sidebar-file-input"),
  btnSidebarBrowse: document.getElementById("btn-sidebar-browse"),
  sidebarUploadIdle: document.getElementById("sidebar-upload-idle"),
  sidebarUploadLoading: document.getElementById("sidebar-upload-loading"),
  sidebarUploadStatus: document.getElementById("sidebar-upload-status"),
  
  // Preview
  cardReportPreview: document.getElementById("card-report-preview"),
  imgReportPreview: document.getElementById("img-report-preview"),
  
  // Patient Clinical Info
  patientIdVal: document.getElementById("patient-id-val"),
  patientAgeGender: document.getElementById("patient-age-gender"),
  patientLeads: document.getElementById("patient-leads"),
  patientMeds: document.getElementById("patient-meds"),
  patientDiagnosis: document.getElementById("patient-diagnosis"),
  
  // Vitals
  vitalBpm: document.getElementById("vital-bpm"),
  vitalPeaks: document.getElementById("vital-peaks"),
  txtSelectedBeatNum: document.getElementById("txt-selected-beat-num"),
  txtSelectedBeatSample: document.getElementById("txt-selected-beat-sample"),
  txtSelectedBeatLabel: document.getElementById("txt-selected-beat-label"),
  
  // Metrics
  metricAnomalyScore: document.getElementById("metric-anomaly-score"),
  metricRiskBadge: document.getElementById("metric-risk-badge"),
  metricRiskPrognosis: document.getElementById("metric-risk-prognosis"),
  metricForecastScore: document.getElementById("metric-forecast-score"),
  metricForecastTrend: document.getElementById("metric-forecast-trend"),
  riskMetricCard: document.getElementById("risk-metric-card"),
  riskIconBox: document.getElementById("risk-icon-box"),
  
  // Banners
  alertBanner: document.getElementById("alert-banner"),
  alertBannerMessage: document.getElementById("alert-banner-message"),
  alertBannerAction: document.getElementById("alert-banner-action"),
  errorBanner: document.getElementById("error-banner"),
  errorMessage: document.getElementById("error-message"),
  
  // Loaders
  loaderWaveform: document.getElementById("loader-waveform"),
  loaderComparison: document.getElementById("loader-comparison"),
  loaderLoss: document.getElementById("loader-loss"),
  loaderForecast: document.getElementById("loader-forecast"),
  loaderShap: document.getElementById("loader-shap"),
  
  // SHAP text
  txtShapSummary: document.getElementById("txt-shap-summary"),
  
  // Signal Quality Metrics
  cardSignalQuality: document.getElementById("card-signal-quality"),
  metricExtractionConfidence: document.getElementById("metric-extraction-confidence"),
  barExtractionConfidence: document.getElementById("bar-extraction-confidence"),
  metricSignalQuality: document.getElementById("metric-signal-quality"),
  barSignalQuality: document.getElementById("bar-signal-quality"),
  metricBpmConfidence: document.getElementById("metric-bpm-confidence"),
  barBpmConfidence: document.getElementById("bar-bpm-confidence"),
  metricWaveformValidity: document.getElementById("metric-waveform-validity"),
  barWaveformValidity: document.getElementById("bar-waveform-validity"),
  qualityWarningBanner: document.getElementById("quality-warning-banner"),
  qualityWarningList: document.getElementById("quality-warning-list"),
  
  // Theme Toggle
  btnThemeToggle: document.getElementById("btn-theme-toggle"),
  themeIconSun: document.getElementById("theme-icon-sun"),
  themeIconMoon: document.getElementById("theme-icon-moon")
};

/* ==========================================================================
   Initialization & Event Listeners
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  // Initialize Theme
  initTheme();
  
  // Initialize Lucide Icons
  lucide.createIcons();
  
  // Setup Event Listeners
  setupEventListeners();
});

function setupEventListeners() {
  // Theme toggle
  dom.btnThemeToggle.addEventListener("click", toggleTheme);
  
  // Landing File Browse button
  if (dom.btnLandingBrowse && dom.landingFileInput) {
    dom.btnLandingBrowse.addEventListener("click", () => dom.landingFileInput.click());
    dom.landingFileInput.addEventListener("change", (e) => handleFileSelect(e, true));
  }
  
  // Landing Drag and Drop Upload
  if (dom.landingDropzone) {
    dom.landingDropzone.addEventListener("dragenter", (e) => handleDragEnter(e, dom.landingDropzone));
    dom.landingDropzone.addEventListener("dragover", (e) => handleDragOver(e, dom.landingDropzone));
    dom.landingDropzone.addEventListener("dragleave", (e) => handleDragLeave(e, dom.landingDropzone));
    dom.landingDropzone.addEventListener("drop", (e) => handleDrop(e, true));
  }
  
  // Sidebar File Browse button
  if (dom.btnSidebarBrowse && dom.sidebarFileInput) {
    dom.btnSidebarBrowse.addEventListener("click", () => dom.sidebarFileInput.click());
    dom.sidebarFileInput.addEventListener("change", (e) => handleFileSelect(e, false));
  }
  
  // Sidebar Drag and Drop Upload
  if (dom.sidebarDropzone) {
    dom.sidebarDropzone.addEventListener("dragenter", (e) => handleDragEnter(e, dom.sidebarDropzone));
    dom.sidebarDropzone.addEventListener("dragover", (e) => handleDragOver(e, dom.sidebarDropzone));
    dom.sidebarDropzone.addEventListener("dragleave", (e) => handleDragLeave(e, dom.sidebarDropzone));
    dom.sidebarDropzone.addEventListener("drop", (e) => handleDrop(e, false));
  }
}

function initTheme() {
  // Default to light theme as requested
  if (!localStorage.getItem("theme_initialized")) {
    localStorage.setItem("theme", "light");
    localStorage.setItem("theme_initialized", "true");
  }
  
  const savedTheme = localStorage.getItem("theme") || "light";
  if (savedTheme === "light") {
    document.body.classList.add("light-theme");
    dom.themeIconSun.classList.add("hidden");
    dom.themeIconMoon.classList.remove("hidden");
  } else {
    document.body.classList.remove("light-theme");
    dom.themeIconSun.classList.remove("hidden");
    dom.themeIconMoon.classList.add("hidden");
  }
}

function toggleTheme() {
  const isLight = document.body.classList.toggle("light-theme");
  localStorage.setItem("theme", isLight ? "light" : "dark");
  
  if (isLight) {
    dom.themeIconSun.classList.add("hidden");
    dom.themeIconMoon.classList.remove("hidden");
  } else {
    dom.themeIconSun.classList.remove("hidden");
    dom.themeIconMoon.classList.add("hidden");
  }
  
  // Redraw all charts to update grid and label colors for the new theme
  redrawCharts();
  if (state.shapData) {
    renderShapCharts();
  }
}

function getChartColors() {
  const style = window.getComputedStyle(document.body);
  return {
    grid: style.getPropertyValue("--chart-grid").trim() || "rgba(255, 255, 255, 0.05)",
    gridWaveform: style.getPropertyValue("--chart-grid-waveform").trim() || "rgba(6, 182, 212, 0.035)",
    text: style.getPropertyValue("--chart-text").trim() || "#9ca3af",
    titleText: style.getPropertyValue("--chart-title-text").trim() || "#f3f4f6",
    cyan: style.getPropertyValue("--color-cyan").trim() || "#06b6d4",
    purple: style.getPropertyValue("--color-purple").trim() || "#8b5cf6",
    rose: style.getPropertyValue("--color-rose").trim() || "#f43f5e",
    emerald: style.getPropertyValue("--color-emerald").trim() || "#10b981",
    orange: style.getPropertyValue("--color-orange").trim() || "#f97316",
    amber: style.getPropertyValue("--color-amber").trim() || "#fbbf24",
    gray: style.getPropertyValue("--color-gray").trim() || "#9ca3af"
  };
}

async function loadECGRecord(name, uploadFlag) {
  showGlobalLoaders(true);
  hideError();
  
  state.recordName = name;
  state.isUpload = uploadFlag;
  if (dom.activePatientId) {
    dom.activePatientId.textContent = `ID-${name}`;
  }
  
  try {
    // 1. Fetch predictions (ECG signals, reconstructions, anomaly scores)
    const data = await apiCall("/predict", "POST", {
      record_name: name,
      is_upload: uploadFlag,
      snippet_start: 0,
      snippet_len: 3000
    });
    
    state.rawSignal = data.raw_signal;
    state.peaks = data.peaks;
    state.labels = data.labels;
    state.windows = data.windows;
    state.reconstructed = data.reconstructed;
    state.anomalyScores = data.anomaly_scores;
    state.allAnomalyScores = data.all_anomaly_scores;
    
    // Choose starting beat: first PVC if it exists, otherwise beat 0
    let defaultIdx = 0;
    const firstArrhythmia = data.labels.findIndex(l => l !== "N");
    if (firstArrhythmia !== -1) {
      defaultIdx = firstArrhythmia;
    }
    state.selectedBeatIndex = defaultIdx;
    
    // Calculate actual Heart Rate (BPM) dynamically from R-peak intervals
    let bpm = 0; // default to 0 for flatlines (0 or 1 peaks)
    if (data.peaks && data.peaks.length > 1) {
      let diffsSum = 0;
      for (let i = 1; i < data.peaks.length; i++) {
        diffsSum += (data.peaks[i] - data.peaks[i - 1]);
      }
      const avgDiffSamples = diffsSum / (data.peaks.length - 1);
      if (avgDiffSamples > 0) {
        bpm = Math.round(21600 / avgDiffSamples);
      }
    }
    if (dom.vitalBpm) dom.vitalBpm.textContent = bpm;
    if (dom.vitalPeaks) dom.vitalPeaks.textContent = data.peaks.length;
    
    // Dynamically update progress bar fills for vitals
    const vitalBpmBar = document.getElementById("vital-bpm-bar");
    if (vitalBpmBar) {
      const percentage = Math.min(100, Math.max(0, (bpm / 200) * 100));
      vitalBpmBar.style.width = `${percentage}%`;
    }
    const vitalPeaksBar = document.getElementById("vital-peaks-bar");
    if (vitalPeaksBar) {
      const percentage = Math.min(100, Math.max(0, (data.peaks.length / 30) * 100));
      vitalPeaksBar.style.width = `${percentage}%`;
    }
    
    // Inject demographic metadata
    const age = data.metadata.age || "N/A";
    const gender = data.metadata.gender || "N/A";
    if (dom.patientIdVal) {
      dom.patientIdVal.textContent = name;
    }
    if (dom.patientAgeGender) {
      dom.patientAgeGender.textContent = (age === "N/A" && gender === "N/A") ? "N/A" : `${age} / ${gender}`;
    }
    if (dom.patientLeads) {
      dom.patientLeads.textContent = data.metadata.leads && data.metadata.leads.length > 0 ? data.metadata.leads.join(", ") : "N/A";
    }
    if (dom.patientMeds) {
      dom.patientMeds.textContent = data.metadata.medications || "None";
    }
    if (dom.patientDiagnosis) {
      dom.patientDiagnosis.textContent = data.metadata.diagnosis || "Sinus Rhythm";
    }
    
    // Toggle containers to reveal dashboard and hide landing page
    if (dom.landingContainer) dom.landingContainer.classList.add("hidden");
    if (dom.dashboardContainer) dom.dashboardContainer.classList.remove("hidden");
    if (dom.navSessionBadge) dom.navSessionBadge.classList.remove("hidden");
    if (dom.sessionTag) {
      dom.sessionTag.textContent = state.isUpload ? "Digitized ECG Report" : "Clinical WFDB Record";
    }
    
    // Show report preview image if available
    if (data.preview_image && dom.cardReportPreview && dom.imgReportPreview) {
      dom.cardReportPreview.classList.remove("hidden");
      dom.imgReportPreview.src = `/api/uploads/${data.preview_image}`;
    } else if (dom.cardReportPreview) {
      dom.cardReportPreview.classList.add("hidden");
    }
    
    // Update Signal Integrity Analysis Card
    if (uploadFlag && data.quality_metrics && dom.cardSignalQuality) {
      dom.cardSignalQuality.classList.remove("hidden");
      
      const qm = data.quality_metrics;
      
      if (dom.metricExtractionConfidence) {
        dom.metricExtractionConfidence.textContent = `${qm.extraction_confidence.toFixed(1)}%`;
      }
      if (dom.barExtractionConfidence) {
        dom.barExtractionConfidence.style.width = `${qm.extraction_confidence}%`;
      }
      
      if (dom.metricSignalQuality) {
        dom.metricSignalQuality.textContent = `${qm.signal_quality.toFixed(1)}%`;
      }
      if (dom.barSignalQuality) {
        dom.barSignalQuality.style.width = `${qm.signal_quality}%`;
      }
      
      if (dom.metricBpmConfidence) {
        dom.metricBpmConfidence.textContent = `${qm.bpm_confidence.toFixed(1)}%`;
      }
      if (dom.barBpmConfidence) {
        dom.barBpmConfidence.style.width = `${qm.bpm_confidence}%`;
      }
      
      if (dom.metricWaveformValidity) {
        dom.metricWaveformValidity.textContent = `${qm.waveform_validity.toFixed(1)}%`;
      }
      if (dom.barWaveformValidity) {
        dom.barWaveformValidity.style.width = `${qm.waveform_validity}%`;
      }
      
      // Handle Quality Warnings Banner
      if (dom.qualityWarningBanner && dom.qualityWarningList) {
        if (qm.quality_warnings && qm.quality_warnings.length > 0) {
          dom.qualityWarningBanner.classList.remove("hidden");
          dom.qualityWarningList.innerHTML = qm.quality_warnings
            .map(warn => `<li>${warn}</li>`)
            .join("");
          
          const hasCritical = qm.quality_warnings.some(warn => warn.includes("CRITICAL"));
          if (hasCritical) {
            dom.qualityWarningBanner.style.borderLeft = "4px solid var(--color-rose)";
            dom.qualityWarningBanner.style.backgroundColor = "rgba(244, 63, 94, 0.06)";
            const warnTitle = dom.qualityWarningBanner.querySelector(".error-title");
            const warnIcon = dom.qualityWarningBanner.querySelector(".error-icon");
            if (warnTitle) warnTitle.style.color = "var(--color-rose)";
            if (warnIcon) warnIcon.style.color = "var(--color-rose)";
          } else {
            dom.qualityWarningBanner.style.borderLeft = "4px solid var(--color-orange)";
            dom.qualityWarningBanner.style.backgroundColor = "rgba(249, 115, 22, 0.06)";
            const warnTitle = dom.qualityWarningBanner.querySelector(".error-title");
            const warnIcon = dom.qualityWarningBanner.querySelector(".error-icon");
            if (warnTitle) warnTitle.style.color = "var(--color-orange)";
            if (warnIcon) warnIcon.style.color = "var(--color-orange)";
          }
        } else {
          dom.qualityWarningBanner.classList.add("hidden");
          dom.qualityWarningList.innerHTML = "";
        }
      }
    } else {
      if (dom.cardSignalQuality) {
        dom.cardSignalQuality.classList.add("hidden");
      }
      if (dom.qualityWarningBanner) {
        dom.qualityWarningBanner.classList.add("hidden");
      }
    }
    
    // 2. Run LSTM forecasting based on anomaly score timeseries
    if (data.all_anomaly_scores && data.all_anomaly_scores.length > 0) {
      const forecastData = await apiCall("/forecast", "POST", {
        anomaly_scores: data.all_anomaly_scores,
        peaks: data.peaks
      });
      state.forecastedScores = forecastData.forecasted_scores;
    } else {
      state.forecastedScores = [0, 0, 0, 0, 0];
    }
    
    // 3. Update active beat selection metrics & risk calculations
    await handleSelectBeat(defaultIdx, false); // skips redundant chart redraws
    
    // 4. Draw all charts for the first time
    redrawCharts();
    
  } catch (err) {
    showError(err.message || "Failed to load patient record from Flask API");
  } finally {
    showGlobalLoaders(false);
  }
}

async function handleSelectBeat(index, redraw = true) {
  if (state.windows.length === 0) {
    state.selectedBeatIndex = 0;
    
    // Update sidebar metadata texts for flatline
    if (dom.txtSelectedBeatNum) {
      dom.txtSelectedBeatNum.textContent = "0 of 0";
    }
    if (dom.txtSelectedBeatSample) {
      dom.txtSelectedBeatSample.textContent = "N/A";
    }
    if (dom.txtSelectedBeatLabel) {
      dom.txtSelectedBeatLabel.textContent = "Asystole / Cardiac Arrest";
      dom.txtSelectedBeatLabel.className = "detail-val badge font-bold border-rose";
    }
    
    // Dynamically sync vital telemetry heart rate and peaks directly to 0 to prevent contradictions
    if (dom.vitalBpm) {
      dom.vitalBpm.textContent = "0";
    }
    const vitalBpmBar = document.getElementById("vital-bpm-bar");
    if (vitalBpmBar) {
      vitalBpmBar.style.width = "0%";
    }
    if (dom.vitalPeaks) {
      dom.vitalPeaks.textContent = "0";
    }
    const vitalPeaksBar = document.getElementById("vital-peaks-bar");
    if (vitalPeaksBar) {
      vitalPeaksBar.style.width = "0%";
    }
    
    // Update Metrics values
    if (dom.metricAnomalyScore) {
      dom.metricAnomalyScore.textContent = "0.0000";
    }
    if (dom.metricRiskBadge) {
      dom.metricRiskBadge.textContent = "CRITICAL";
      dom.metricRiskBadge.className = "badge badge-risk uppercase font-black border-red";
    }
    if (dom.metricRiskPrognosis) {
      dom.metricRiskPrognosis.textContent = "Asystole";
    }
    if (dom.metricForecastScore) {
      dom.metricForecastScore.textContent = "0.0000";
    }
    if (dom.metricForecastTrend) {
      dom.metricForecastTrend.textContent = "Cardiac Arrest";
      dom.metricForecastTrend.className = "badge badge-trend up";
    }
    
    // Set Severity colors for risk cards
    if (dom.riskMetricCard) dom.riskMetricCard.className = "metric-card glass-panel border-bottom-rose";
    if (dom.riskIconBox) dom.riskIconBox.className = "metric-icon-box bg-rose-950 border-rose text-rose";
    
    // Show and update alert banner
    if (dom.alertBanner) {
      dom.alertBanner.className = "alert-banner glass-panel critical";
      dom.alertBanner.classList.remove("hidden");
    }
    if (dom.alertBannerMessage) {
      dom.alertBannerMessage.textContent = "CRITICAL ALERT: Asystole / Cardiac Arrest detected. No active heartbeats or sinus rhythm found in the digitized waveform trace.";
    }
    if (dom.alertBannerAction) {
      dom.alertBannerAction.textContent = "Assess patient immediately. Check ECG lead connection. Initiate resuscitation protocol if clinically indicated.";
    }
    
    // SHAP text
    if (dom.txtShapSummary) {
      dom.txtShapSummary.textContent = "No heartbeat windows available for attribution analysis.";
    }
    
    showShapLoader(false);
    
    // Redraw sub-charts with empty configurations
    if (redraw) {
      renderBeatComparisonChart();
      renderReconstructionLossChart();
      renderLstmForecastChart();
    }
    return;
  }
  if (index < 0 || index >= state.windows.length) return;
  state.selectedBeatIndex = index;
  
  const activeScore = state.anomalyScores[index] || 0.0;
  
  // Update sidebar metadata texts
  if (dom.txtSelectedBeatNum) {
    dom.txtSelectedBeatNum.textContent = `#${index + 1} of ${state.windows.length}`;
  }
  if (dom.txtSelectedBeatSample) {
    dom.txtSelectedBeatSample.textContent = state.peaks[index] !== undefined ? `${state.peaks[index]} sample` : "N/A";
  }
  
  const label = state.labels[index] || "N";
  let labelText = "Normal (Safe)";
  let labelClass = "border-emerald";
  
  if (activeScore >= 0.25) {
    labelText = "Critical Arrhythmia";
    labelClass = "border-rose";
  } else if (activeScore >= 0.12) {
    labelText = "Moderate Arrhythmia";
    labelClass = "border-orange";
  } else if (activeScore >= 0.05) {
    labelText = "Mild Anomaly";
    labelClass = "border-amber";
  }
  
  if (dom.txtSelectedBeatLabel) {
    dom.txtSelectedBeatLabel.textContent = labelText;
    dom.txtSelectedBeatLabel.className = `detail-val badge font-bold ${labelClass}`;
  }
  
  // Update Metrics values
  if (dom.metricAnomalyScore) {
    dom.metricAnomalyScore.textContent = activeScore.toFixed(4);
  }
  
  // Calculate Risk Profile via Flask Backend
  try {
    const riskData = await apiCall("/risk", "POST", {
      current_score: activeScore,
      forecasted_scores: state.forecastedScores
    });
    state.riskProfile = riskData;
    
    // Update risk metric card styles
    updateRiskUI(riskData);
    
  } catch (err) {
    console.error("Risk score calculation failed:", err);
  }
  
  // Load and calculate SHAP explainability values for the selected beat window
  if (state.windows[index]) {
    fetchShapExplanation(state.windows[index]);
  }
  
  // Redraw sub-charts that depend directly on the selected beat index
  if (redraw) {
    renderBeatComparisonChart();
    renderReconstructionLossChart(); // highlights active bar
    renderLstmForecastChart(); // adjusts forecast start to active score
  }
}

async function fetchShapExplanation(windowData) {
  showShapLoader(true);
  try {
    const explainData = await apiCall("/explain", "POST", {
      window: windowData
    });
    state.shapData = explainData;
    
    // Update SHAP texts and charts
    if (dom.txtShapSummary) {
      dom.txtShapSummary.textContent = `"${explainData.explanation_summary}"`;
    }
    renderShapCharts();
    
  } catch (err) {
    console.error("SHAP explanation fetch failed:", err);
    if (dom.txtShapSummary) {
      dom.txtShapSummary.textContent = "SHAP engine calculation timed out or encountered an error.";
    }
  } finally {
    showShapLoader(false);
  }
}

/* ==========================================================================
   UI Helpers & DOM Updates
   ========================================================================== */

function showGlobalLoaders(show) {
  const toggleLoader = (loader, showEl) => {
    if (!loader) return;
    if (showEl) loader.classList.remove("hidden");
    else loader.classList.add("hidden");
  };
  
  toggleLoader(dom.loaderWaveform, show);
  toggleLoader(dom.loaderComparison, show);
  toggleLoader(dom.loaderLoss, show);
  toggleLoader(dom.loaderForecast, show);
}

function showShapLoader(show) {
  if (!dom.loaderShap) return;
  if (show) {
    dom.loaderShap.classList.remove("hidden");
  } else {
    dom.loaderShap.classList.add("hidden");
  }
}

function generateDynamicClinicalFindings(activeScore, label, forecastedScores) {
  let findings = [];
  
  if (label !== "N") {
    findings.push("Irregular ventricular rhythm detected.");
  } else {
    findings.push("Sinus rhythm pattern detected.");
  }
  
  if (activeScore >= 0.12) {
    findings.push("Elevated reconstruction error near QRS peaks.");
  } else if (activeScore >= 0.05) {
    findings.push("Borderline reconstruction error near QRS/T waves.");
  } else {
    findings.push("Stable reconstruction error.");
  }
  
  const maxFuture = forecastedScores.length > 0 ? Math.max(...forecastedScores) : activeScore;
  const isUp = maxFuture > activeScore + 0.02;
  if (isUp) {
    findings.push("Forecast indicates increasing cardiovascular instability.");
  } else {
    findings.push("Forecast indicates stable cardiovascular projection.");
  }
  
  return findings.join(" ");
}

function updateRiskUI(risk) {
  if (dom.metricRiskBadge) dom.metricRiskBadge.textContent = risk.current_level;
  if (dom.metricRiskPrognosis) dom.metricRiskPrognosis.textContent = risk.future_level;
  if (dom.metricForecastScore) dom.metricForecastScore.textContent = risk.max_future_score.toFixed(4);
  
  // Update Forecast trend badge
  const activeScore = state.anomalyScores[state.selectedBeatIndex] || 0.0;
  const isUp = risk.max_future_score > activeScore;
  if (dom.metricForecastTrend) {
    dom.metricForecastTrend.textContent = isUp ? "Trending UP" : "Trending Stable";
    dom.metricForecastTrend.className = `badge badge-trend ${isUp ? "up" : "stable"}`;
  }
  
  // Set Severity colors for risk cards
  let cardBorder = "border-bottom-green";
  let iconBoxClass = "metric-icon-box bg-emerald-950 border-emerald text-emerald";
  let badgeColor = "border-emerald";
  let bannerClass = "alert-banner glass-panel safe";
  
  switch(risk.current_level) {
    case "Critical":
      cardBorder = "border-bottom-rose";
      iconBoxClass = "metric-icon-box bg-rose-950 border-rose text-rose";
      badgeColor = "border-red";
      bannerClass = "alert-banner glass-panel critical";
      break;
    case "Moderate":
      cardBorder = "border-bottom-orange";
      iconBoxClass = "metric-icon-box bg-orange-950 border-orange text-orange";
      badgeColor = "badge-risk border-red"; // Custom color fallback
      bannerClass = "alert-banner glass-panel moderate";
      break;
    case "Mild":
      cardBorder = "border-bottom-orange"; // use orange/amber variables
      iconBoxClass = "metric-icon-box bg-orange-950 border-orange text-orange";
      badgeColor = "border-red";
      bannerClass = "alert-banner glass-panel mild";
      break;
  }
  
  // Apply card borders & badges
  if (dom.riskMetricCard) dom.riskMetricCard.className = `metric-card glass-panel ${cardBorder}`;
  if (dom.riskIconBox) dom.riskIconBox.className = iconBoxClass;
  if (dom.metricRiskBadge) dom.metricRiskBadge.className = `badge badge-risk uppercase font-black ${badgeColor}`;
  
  // Show and update alert banner
  if (dom.alertBanner) {
    dom.alertBanner.className = bannerClass;
    dom.alertBanner.classList.remove("hidden");
  }
  
  // Compile dynamic clinical findings
  const label = state.labels[state.selectedBeatIndex] || "N";
  const findingsText = generateDynamicClinicalFindings(activeScore, label, state.forecastedScores);
  
  if (dom.alertBannerMessage) {
    dom.alertBannerMessage.innerHTML = `<strong>${findingsText}</strong><br><br>${risk.alert}`;
  }
  if (dom.alertBannerAction) {
    dom.alertBannerAction.textContent = risk.action;
  }
  
  // Refresh Lucide dynamically rendered icons
  lucide.createIcons();
}

function showError(msg) {
  if (dom.errorBanner) dom.errorBanner.classList.remove("hidden");
  if (dom.errorMessage) dom.errorMessage.textContent = msg;
}

function hideError() {
  if (dom.errorBanner) dom.errorBanner.classList.add("hidden");
}

/* ==========================================================================
   Chart.js Renderer Methods
   ========================================================================== */

function redrawCharts() {
  if (state.rawSignal.length === 0) return;
  renderWaveformChart();
  renderBeatComparisonChart();
  renderReconstructionLossChart();
  renderLstmForecastChart();
}

// 1. ECG Waveform Continuous Strip
function renderWaveformChart() {
  if (charts.waveform) charts.waveform.destroy();
  
  const ctx = document.getElementById("chart-waveform").getContext("2d");
  const xLabels = state.rawSignal.map((_, idx) => (idx / 360).toFixed(2) + 's');
  const colors = getChartColors();
  
  // Map R-Peak dot coordinates
  const peakDataPoints = state.rawSignal.map((val, idx) => {
    const peakIdx = state.peaks.indexOf(idx);
    return peakIdx !== -1 ? val : null;
  });
  
  charts.waveform = new Chart(ctx, {
    type: "line",
    data: {
      labels: xLabels,
      datasets: [
        {
          label: "ECG Signal (Lead I)",
          data: state.rawSignal,
          borderColor: colors.cyan,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.1,
          fill: false,
        },
        {
          label: "Detected R-Peaks",
          data: peakDataPoints,
          borderColor: colors.rose,
          showLine: false,
          pointStyle: 'circle',
          pointRadius: (context) => {
            const index = context.dataIndex;
            const peakIdx = state.peaks.indexOf(index);
            if (peakIdx !== -1) {
              return peakIdx === state.selectedBeatIndex ? 9 : 5.5;
            }
            return 0;
          },
          pointBackgroundColor: (context) => {
            const index = context.dataIndex;
            const peakIdx = state.peaks.indexOf(index);
            if (peakIdx !== -1) {
              const score = state.anomalyScores[peakIdx] || 0.0;
              if (score >= 0.25) return colors.rose;
              if (score >= 0.12) return colors.orange;
              if (score >= 0.05) return colors.amber;
              return colors.emerald;
            }
            return colors.rose;
          },
          pointBorderColor: (context) => {
            const index = context.dataIndex;
            const peakIdx = state.peaks.indexOf(index);
            if (peakIdx !== -1) {
              const score = state.anomalyScores[peakIdx] || 0.0;
              if (score >= 0.25) return colors.rose;
              if (score >= 0.12) return colors.orange;
              if (score >= 0.05) return colors.amber;
              return colors.emerald;
            }
            return colors.rose;
          },
          pointHoverRadius: 8
        }
      ]
    },
    options: {
      layout: {
        padding: { left: 10, right: 15, top: 10, bottom: 5 }
      },
      responsive: true,
      maintainAspectRatio: false,
      onClick: (event, elements) => {
        if (elements.length > 0) {
          // Check if they clicked the R-Peaks dataset (index 1)
          const peakClick = elements.find(el => el.datasetIndex === 1);
          if (peakClick) {
            const clickedSampleIdx = peakClick.index;
            const peakIdx = state.peaks.indexOf(clickedSampleIdx);
            if (peakIdx !== -1) {
              handleSelectBeat(peakIdx);
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: colors.gridWaveform },
          ticks: { color: colors.text, maxTicksLimit: 12 }
        },
        y: {
          grid: { color: colors.gridWaveform },
          ticks: { color: colors.text }
        }
      },
      plugins: {
        legend: { labels: { color: colors.titleText } },
        tooltip: {
          callbacks: {
            label: (context) => {
              if (context.datasetIndex === 1) {
                const peakIdx = state.peaks.indexOf(context.dataIndex);
                const label = state.labels[peakIdx];
                const labelName = label === "N" ? "Normal (N)" : `Arrhythmia (${label})`;
                return `R-Peak: ${labelName} at sample index ${context.dataIndex}`;
              }
              return `Lead I: ${context.parsed.y.toFixed(3)} mV`;
            }
          }
        }
      }
    }
  });
}

// 2. Single Heartbeat comparison (original vs AE reconstruction)
function renderBeatComparisonChart() {
  if (charts.beatComparison) charts.beatComparison.destroy();
  
  const ctx = document.getElementById("chart-beat-comparison").getContext("2d");
  const originalBeat = state.windows[state.selectedBeatIndex];
  const reconstructedBeat = state.reconstructed[state.selectedBeatIndex];
  
  if (!originalBeat || !reconstructedBeat) return;
  
  const colors = getChartColors();
  const xLabels = originalBeat.map((_, idx) => idx);
  const errorDiff = originalBeat.map((val, idx) => Math.pow(val - reconstructedBeat[idx], 2));
  
  const roseBorder = colors.rose.startsWith("#") ? colors.rose + "59" : "rgba(244, 63, 94, 0.35)";
  const roseBg = colors.rose.startsWith("#") ? colors.rose + "1f" : "rgba(244, 63, 94, 0.12)";
  
  charts.beatComparison = new Chart(ctx, {
    type: "line",
    data: {
      labels: xLabels,
      datasets: [
        {
          label: "Original Wave",
          data: originalBeat,
          borderColor: colors.cyan,
          borderWidth: 2,
          pointRadius: 0,
          fill: false
        },
        {
          label: "Reconstructed",
          data: reconstructedBeat,
          borderColor: colors.purple,
          borderWidth: 2,
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false
        },
        {
          label: "Pointwise MSE Error",
          data: errorDiff,
          borderColor: roseBorder,
          backgroundColor: roseBg,
          borderWidth: 1,
          pointRadius: 0,
          fill: true
        }
      ]
    },
    options: {
      layout: {
        padding: { left: 10, right: 15, top: 10, bottom: 5 }
      },
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { display: false },
        y: {
          grid: { color: colors.grid },
          ticks: { color: colors.text }
        }
      },
      plugins: {
        legend: {
          position: "top",
          labels: { color: colors.titleText, boxWidth: 12, font: { size: 10 } }
        }
      }
    }
  });
}

// 3. Reconstruction error bar chart (anomaly scores per beat)
function renderReconstructionLossChart() {
  if (charts.reconstructionLoss) charts.reconstructionLoss.destroy();
  
  const ctx = document.getElementById("chart-reconstruction-loss").getContext("2d");
  const xLabels = state.anomalyScores.map((_, idx) => `Beat ${idx + 1}`);
  const colors = getChartColors();
  
  // Highlight active beat bar
  const bgColors = state.anomalyScores.map((score, idx) => {
    if (idx === state.selectedBeatIndex) return colors.purple; // Highlight purple
    if (score < 0.05) return colors.emerald; // Safe green
    if (score < 0.12) return colors.amber; // Mild amber
    if (score < 0.25) return colors.orange; // Moderate orange
    return colors.rose; // Critical red
  });
  
  charts.reconstructionLoss = new Chart(ctx, {
    type: "bar",
    data: {
      labels: xLabels,
      datasets: [{
        label: "Reconstruction Error",
        data: state.anomalyScores,
        backgroundColor: bgColors,
        borderRadius: 4,
        barPercentage: 0.7
      }]
    },
    options: {
      layout: {
        padding: { left: 10, right: 15, top: 10, bottom: 5 }
      },
      responsive: true,
      maintainAspectRatio: false,
      onClick: (event, elements) => {
        if (elements.length > 0) {
          const clickedIdx = elements[0].index;
          handleSelectBeat(clickedIdx);
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: colors.text, maxTicksLimit: 12, font: { size: 9 } }
        },
        y: {
          grid: { color: colors.grid },
          ticks: { color: colors.text, font: { size: 9 } }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterBody: (context) => {
              const idx = context[0].dataIndex;
              const score = state.anomalyScores[idx] || 0.0;
              let severity = "Safe";
              if (score >= 0.25) severity = "Critical Arrhythmia";
              else if (score >= 0.12) severity = "Moderate Arrhythmia";
              else if (score >= 0.05) severity = "Mild Anomaly";
              return `Severity: ${severity}`;
            }
          }
        }
      }
    }
  });
}

// 4. LSTM Future Risk Score Trend Chart
function renderLstmForecastChart() {
  if (charts.forecast) charts.forecast.destroy();
  
  const ctx = document.getElementById("chart-lstm-forecast").getContext("2d");
  const colors = getChartColors();
  
  // Cut scores relative to current selected beat for visual continuity
  const history = state.anomalyScores.slice(0, state.selectedBeatIndex + 1);
  const maxHistoryLength = 10;
  const historySegment = history.slice(-maxHistoryLength);
  
  // Generate labels
  const historyLabels = historySegment.map((_, idx) => `T - ${historySegment.length - 1 - idx}`);
  const forecastLabels = state.forecastedScores.map((_, idx) => `T + ${idx + 1}`);
  const labels = [...historyLabels, ...forecastLabels];
  
  // History data vector (stops at T-0)
  const historyData = [...historySegment, ...Array(state.forecastedScores.length).fill(null)];
  
  // Forecast data vector (starts at T-0)
  const lastHistoryVal = historySegment[historySegment.length - 1] ?? 0.0;
  const fillLength = Math.max(0, historySegment.length - 1);
  const forecastData = [
    ...Array(fillLength).fill(null),
    lastHistoryVal,
    ...state.forecastedScores
  ];
  
  charts.forecast = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "History Trend",
          data: historyData,
          borderColor: colors.cyan,
          borderWidth: 2,
          pointRadius: 4,
          fill: false,
          spanGaps: false
        },
        {
          label: "LSTM Prognosis",
          data: forecastData,
          borderColor: colors.rose,
          borderWidth: 2,
          borderDash: [5, 5],
          pointBackgroundColor: colors.rose,
          pointRadius: 5,
          fill: false,
          spanGaps: true
        }
      ]
    },
    options: {
      layout: {
        padding: { left: 10, right: 15, top: 10, bottom: 5 }
      },
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: colors.text, font: { size: 9 } }
        },
        y: {
          grid: { color: colors.grid },
          ticks: { color: colors.text, font: { size: 9 } },
          min: 0,
          max: 0.35
        }
      },
      plugins: {
        legend: {
          position: "top",
          labels: { color: colors.titleText, boxWidth: 10, font: { size: 9 } }
        }
      }
    }
  });
}

// 5. SHAP attribution sub-charts (Bar and Line)
function renderShapCharts() {
  if (charts.shapBar) charts.shapBar.destroy();
  if (charts.shapLine) charts.shapLine.destroy();
  
  const barCtx = document.getElementById("chart-shap-bar").getContext("2d");
  const lineCtx = document.getElementById("chart-shap-line").getContext("2d");
  
  const shap = state.shapData;
  if (!shap) return;
  
  const colors = getChartColors();
  
  // A. Horizontal Bar Chart (Attribution Impact per wave segment)
  const segments = ["P-Wave", "PR Interval", "QRS Complex", "ST Segment", "T-Wave"];
  const segmentAttributions = [
    shap.segment_importances.p_wave || 0.0,
    shap.segment_importances.pr_interval || 0.0,
    shap.segment_importances.qrs_complex || 0.0,
    shap.segment_importances.st_segment || 0.0,
    shap.segment_importances.t_wave || 0.0,
  ];
  
  const bgOpacityColors = [
    colors.cyan.startsWith("#") ? colors.cyan + "c0" : "rgba(6, 182, 212, 0.75)",
    colors.purple.startsWith("#") ? colors.purple + "c0" : "rgba(139, 92, 246, 0.75)",
    colors.rose.startsWith("#") ? colors.rose + "cc" : "rgba(239, 68, 68, 0.8)",
    colors.orange.startsWith("#") ? colors.orange + "c0" : "rgba(249, 115, 22, 0.75)",
    colors.emerald.startsWith("#") ? colors.emerald + "c0" : "rgba(16, 185, 129, 0.75)"
  ];
  
  charts.shapBar = new Chart(barCtx, {
    type: "bar",
    data: {
      labels: segments,
      datasets: [{
        label: "SHAP Impact Score",
        data: segmentAttributions,
        backgroundColor: bgOpacityColors,
        borderColor: [colors.cyan, colors.purple, colors.rose, colors.orange, colors.emerald],
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      layout: {
        padding: { left: 10, right: 15, top: 10, bottom: 5 }
      },
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: colors.grid },
          ticks: { color: colors.text, font: { size: 8 } }
        },
        y: {
          grid: { display: false },
          ticks: { color: colors.titleText, font: { size: 9, weight: "bold" } }
        }
      },
      plugins: { legend: { display: false } }
    }
  });
  
  // B. Pointwise Line Chart (Point-by-point feature attributions across 180 samples)
  const samplesLabels = shap.shap_values.map((_, idx) => {
    const ms = Math.round((idx - 90) * (1000 / 360));
    return (ms >= 0 ? "+" : "") + ms + "ms";
  });
  const purpleLineBg = colors.purple.startsWith("#") ? colors.purple + "26" : "rgba(139, 92, 246, 0.15)";
  const purpleLineBorder = colors.purple.startsWith("#") ? colors.purple + "cc" : "rgba(139, 92, 246, 0.8)";
  
  charts.shapLine = new Chart(lineCtx, {
    type: "line",
    data: {
      labels: samplesLabels,
      datasets: [{
        label: "Pointwise Attribution",
        data: shap.shap_values,
        borderColor: purpleLineBorder,
        backgroundColor: purpleLineBg,
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.2
      }]
    },
    options: {
      layout: {
        padding: { left: 10, right: 15, top: 10, bottom: 5 }
      },
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: colors.text, maxTicksLimit: 6, font: { size: 8 } }
        },
        y: {
          grid: { color: colors.grid },
          ticks: { color: colors.text, font: { size: 8 } }
        }
      },
      plugins: { legend: { display: false } }
    }
  });
}

/* ==========================================================================
   File Drag & Drop Upload
   ========================================================================== */

function handleDragEnter(e, element) {
  e.preventDefault();
  e.stopPropagation();
  if (element) element.classList.add("drag-active");
}

function handleDragOver(e, element) {
  e.preventDefault();
  e.stopPropagation();
  if (element) element.classList.add("drag-active");
}

function handleDragLeave(e, element) {
  e.preventDefault();
  e.stopPropagation();
  if (element) element.classList.remove("drag-active");
}

async function handleDrop(e, isLanding) {
  e.preventDefault();
  e.stopPropagation();
  
  const dropzone = isLanding ? dom.landingDropzone : dom.sidebarDropzone;
  if (dropzone) dropzone.classList.remove("drag-active");
  
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    await uploadECGFiles(e.dataTransfer.files, isLanding);
  }
}

async function handleFileSelect(e, isLanding) {
  if (e.target.files && e.target.files.length > 0) {
    await uploadECGFiles(e.target.files, isLanding);
  }
}

async function uploadECGFiles(files, isLanding) {
  const uploadIdle = isLanding ? dom.landingUploadIdle : dom.sidebarUploadIdle;
  const uploadLoading = isLanding ? dom.landingUploadLoading : dom.sidebarUploadLoading;
  
  // Show upload UI loading state
  if (uploadIdle) uploadIdle.classList.add("hidden");
  if (uploadLoading) uploadLoading.classList.remove("hidden");
  hideUploadStatus(isLanding);
  
  try {
    const fileNames = Array.from(files).map(f => f.name.toLowerCase());
    const hasImageOrPdf = fileNames.some(name => 
      name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg") || 
      name.endsWith(".bmp") || name.endsWith(".tiff") || name.endsWith(".pdf")
    );
    const hasHea = fileNames.some(name => name.endsWith(".hea"));
    const hasDat = fileNames.some(name => name.endsWith(".dat"));
    
    if (!hasImageOrPdf && (!hasHea || !hasDat)) {
      throw new Error("Invalid files. Please upload an ECG report (PNG, JPG, PDF) or a WFDB record (.hea and .dat pair).");
    }
    
    // Prepare multi-part request body
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }
    
    // Upload files to API
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: "POST",
      body: formData
    });
    
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || "Failed to upload files to Flask server");
    }
    
    const res = await response.json();
    showUploadStatus(`Successfully uploaded record: ${res.record_name}`, "success", isLanding);
    
    // Load the custom ECG file
    await loadECGRecord(res.record_name, true);
    
  } catch (err) {
    showUploadStatus(err.message || "Failed to process files", "error", isLanding);
  } finally {
    if (uploadIdle) uploadIdle.classList.remove("hidden");
    if (uploadLoading) uploadLoading.classList.add("hidden");
  }
}

function showUploadStatus(msg, type, isLanding) {
  const uploadStatus = isLanding ? dom.landingUploadStatus : dom.sidebarUploadStatus;
  if (!uploadStatus) return;
  uploadStatus.className = `alert-box ${type}`;
  uploadStatus.classList.remove("hidden");
  uploadStatus.textContent = msg;
}

function hideUploadStatus(isLanding) {
  const uploadStatus = isLanding ? dom.landingUploadStatus : dom.sidebarUploadStatus;
  if (uploadStatus) {
    uploadStatus.classList.add("hidden");
  }
}

/* ==========================================================================
   REST API Helper Client
   ========================================================================== */

async function apiCall(endpoint, method = "GET", bodyData = null) {
  const url = `${API_BASE_URL}${endpoint}`;
  const options = {
    method: method,
    headers: {
      "Content-Type": "application/json"
    }
  };
  
  if (bodyData) {
    options.body = JSON.stringify(bodyData);
  }
  
  const response = await fetch(url, options);
  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.error || `HTTP error! Status: ${response.status}`);
  }
  return response.json();
}
