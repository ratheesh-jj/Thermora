const API_BASE = (window.location.protocol === 'file:' || !window.location.port) ? 'http://localhost:8000' : '';
let ws;

// ==========================================
// THEME TOGGLE (Light / Dark)
// ==========================================
function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme');
    const next = current === 'light' ? 'dark' : 'light';
    html.setAttribute('data-theme', next);
    localStorage.setItem('thermora-theme', next);
    document.getElementById('theme-icon').innerText = next === 'light' ? '🌙' : '☀️';
}

// Apply saved theme on load
(function initTheme() {
    const saved = localStorage.getItem('thermora-theme');
    if (saved === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
        document.addEventListener('DOMContentLoaded', function() {
            const icon = document.getElementById('theme-icon');
            if (icon) icon.innerText = '🌙';
        });
    }
})();

function connect() {
    ws = new WebSocket("ws://localhost:8000/ws");

    ws.onopen = function() {
        console.log("Connected to server");
    };

    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        updateDashboard(data);
    };

    ws.onclose = function() {
        console.log("Disconnected from server, retrying in 2s...");
        setTimeout(connect, 2000);
    };
    
    ws.onerror = function(err) {
        console.error("WebSocket error", err);
    };
}

function updateDashboard(data) {
    // Topbar
    document.getElementById("last-packet-time").innerText = data.timestamp;
    
    const wristDot = document.getElementById("wrist-dot");
    const wristText = document.getElementById("wrist-text");
    if (data.connection.wristband) {
        wristDot.classList.add("connected");
        wristText.innerText = "CONNECTED";
        wristText.style.color = "var(--status-ok)";
    } else {
        wristDot.classList.remove("connected");
        wristText.innerText = "DISCONNECTED";
        wristText.style.color = "var(--status-err)";
    }
    
    const envDot = document.getElementById("env-dot");
    const envText = document.getElementById("env-text");
    if (data.connection.environment) {
        envDot.classList.add("connected");
        envText.innerText = "CONNECTED";
        envText.style.color = "var(--status-ok)";
    } else {
        envDot.classList.remove("connected");
        envText.innerText = "DISCONNECTED";
        envText.style.color = "var(--status-err)";
    }

    // Vitals
    document.getElementById("val-hr").innerText = data.vitals.hr !== null ? data.vitals.hr : "--";
    document.getElementById("val-spo2").innerText = data.vitals.spo2 !== null ? data.vitals.spo2 : "--";
    document.getElementById("val-hrv").innerText = data.vitals.hrv !== null ? data.vitals.hrv : "--";
    document.getElementById("val-ppg-qual").innerText = data.vitals.ppg_quality || "--";
    
    // SpO2 Forecast
    document.getElementById("val-spo2-30").innerText = data.spo2_forecast.plus30 !== null ? data.spo2_forecast.plus30 : "--";
    document.getElementById("val-spo2-60").innerText = data.spo2_forecast.plus60 !== null ? data.spo2_forecast.plus60 : "--";
    
    // Thermal
    document.getElementById("val-env-t").innerText = data.thermal.ambient_temp_c !== null ? Number(data.thermal.ambient_temp_c).toFixed(1) : "--";
    document.getElementById("val-env-h").innerText = data.thermal.humidity_pct !== null ? Number(data.thermal.humidity_pct).toFixed(1) : "--";
    document.getElementById("val-core-est").innerText = data.thermal.estimated_core_temp_c !== null ? Number(data.thermal.estimated_core_temp_c).toFixed(2) : "--";
    document.getElementById("val-core-pred").innerText = data.thermal.predicted_core_temp_c !== null ? Number(data.thermal.predicted_core_temp_c).toFixed(2) : "--";
    document.getElementById("val-core-rise").innerText = data.thermal.predicted_rise_c !== null ? ((Number(data.thermal.predicted_rise_c) > 0 ? "+" : "") + Number(data.thermal.predicted_rise_c).toFixed(2)) : "--";
    updateStatusBadge("thermal-status", data.thermal.status);

    // Fatigue
    document.getElementById("val-fatigue").innerText = data.fatigue.score !== null ? data.fatigue.score.toFixed(1) : "--";
    document.getElementById("val-fatigue-trend").innerText = "Trend: " + (data.fatigue.trend || "unknown").toUpperCase();
    updateStatusBadge("fatigue-status", data.fatigue.status);

    // Fall
    document.getElementById("val-fall-prob").innerText = data.fall.probability !== null ? (data.fall.probability * 100).toFixed(1) : "--";
    const fallAlert = document.getElementById("val-fall-detected");
    if (data.fall.detected) {
        fallAlert.innerText = "FALL DETECTED!";
        fallAlert.classList.add("danger");
    } else {
        fallAlert.innerText = "NO FALL DETECTED";
        fallAlert.classList.remove("danger");
    }
    updateStatusBadge("fall-status", data.fall.status);

    // Vital Anomaly
    const vitalStatus = data.vital_anomaly.status || "--";
    document.getElementById("val-anomaly-status").innerText = vitalStatus;

    // Severity mapping
    const severityMap = {
        "CALIBRATING_BASELINE": { icon: "⚪", level: "INFO",     color: "#94a3b8", desc: "Learning worker baseline",       cls: "severity--info" },
        "NORMAL":               { icon: "🟢", level: "NORMAL",   color: "#10b981", desc: "Vitals stable and within bounds", cls: "severity--normal" },
        "MEASUREMENT_UNRELIABLE":{ icon: "🟡", level: "WARNING", color: "#f59e0b", desc: "Poor PPG sensor contact",         cls: "severity--warning" },
        "PHYSIOLOGICAL_DEVIATION":{ icon: "🟡", level: "LOW",    color: "#f59e0b", desc: "Mild vital shift detected",       cls: "severity--low" },
        "ML_ANOMALY_SUPPORT":   { icon: "🟡", level: "LOW-MED",  color: "#fb923c", desc: "ML model flagged anomaly pattern",cls: "severity--low-med" },
        "HR_DEVIATION":         { icon: "🟠", level: "MEDIUM",   color: "#f97316", desc: "Heart rate deviation detected",   cls: "severity--medium" },
        "SPO2_DEVIATION":       { icon: "🟠", level: "MEDIUM",   color: "#f97316", desc: "Oxygen saturation drop",          cls: "severity--medium" },
        "UNCONFIRMED_VITAL_ANOMALY":{ icon: "🟠", level: "HIGH", color: "#ef4444", desc: "Anomaly detected — confirming",   cls: "severity--high" },
        "VITAL_ANOMALY":        { icon: "🔴", level: "CRITICAL", color: "#ef4444", desc: "Confirmed vital anomaly active",  cls: "severity--critical" },
    };
    const sev = severityMap[vitalStatus] || { icon: "⚪", level: "--", color: "#94a3b8", desc: "Waiting for data", cls: "severity--info" };

    document.getElementById("val-anomaly-status").style.color = sev.color;
    document.getElementById("severity-icon").innerText = sev.icon;
    document.getElementById("severity-level").innerText = sev.level;
    document.getElementById("severity-level").style.color = sev.color;
    document.getElementById("severity-desc").innerText = sev.desc;

    const sevBar = document.getElementById("severity-bar");
    sevBar.className = "severity-bar " + sev.cls;

    document.getElementById("val-anomaly-hr").innerText = data.vital_anomaly.hr_deviation !== null ? data.vital_anomaly.hr_deviation : "--";
    document.getElementById("val-anomaly-spo2").innerText = data.vital_anomaly.spo2_deviation !== null ? data.vital_anomaly.spo2_deviation : "--";
    document.getElementById("val-anomaly-ml").innerText = data.vital_anomaly.ml_support ? "YES" : "NO";
    
    if (data.vital_anomaly.episode_id) {
        document.getElementById("val-episode").innerText = "Active Episode: " + data.vital_anomaly.episode_id;
        document.getElementById("val-episode").style.display = "block";
    } else {
        document.getElementById("val-episode").style.display = "none";
    }

    // Movement
    document.getElementById("val-mag").innerText = data.movement.mag !== null ? data.movement.mag : "--";
    document.getElementById("val-ax").innerText = data.movement.ax !== null ? data.movement.ax.toFixed(2) : "--";
    document.getElementById("val-ay").innerText = data.movement.ay !== null ? data.movement.ay.toFixed(2) : "--";
    document.getElementById("val-az").innerText = data.movement.az !== null ? data.movement.az.toFixed(2) : "--";
    document.getElementById("val-gx").innerText = data.movement.gx !== null ? data.movement.gx.toFixed(2) : "--";
    document.getElementById("val-gy").innerText = data.movement.gy !== null ? data.movement.gy.toFixed(2) : "--";
    document.getElementById("val-gz").innerText = data.movement.gz !== null ? data.movement.gz.toFixed(2) : "--";

    // GPS Location
    if (data.gps) {
        const lat = data.gps.lat;
        const lng = data.gps.lng;
        document.getElementById("val-gps-lat").innerText = lat !== null ? lat.toFixed(6) : "--";
        document.getElementById("val-gps-lng").innerText = lng !== null ? lng.toFixed(6) : "--";
        document.getElementById("val-gps-acc").innerText = data.gps.accuracy_m !== null ? data.gps.accuracy_m + " m" : "--";
        document.getElementById("val-gps-fix").innerText = data.gps.last_fix || "--";

        if (lat !== null && lng !== null) {
            updateMap(lat, lng);
        }
    }

    // Check for alerts
    checkAlerts(data);
}

// ---- Leaflet Map ----
let gpsMap = null;
let gpsMarker = null;
let gpsTrail = [];

function initMap(lat, lng) {
    if (gpsMap) return;
    gpsMap = L.map("location-map", {
        zoomControl: true,
        attributionControl: false
    }).setView([lat, lng], 17);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19
    }).addTo(gpsMap);
    gpsMarker = L.circleMarker([lat, lng], {
        radius: 8,
        fillColor: "#6366f1",
        color: "#818cf8",
        weight: 2,
        fillOpacity: 0.9
    }).addTo(gpsMap);
}

function updateMap(lat, lng) {
    if (!gpsMap) {
        initMap(lat, lng);
        return;
    }
    gpsMarker.setLatLng([lat, lng]);
    gpsMap.setView([lat, lng], gpsMap.getZoom(), { animate: true, duration: 0.5 });

    // Trail breadcrumb
    gpsTrail.push([lat, lng]);
    if (gpsTrail.length > 100) gpsTrail.shift();
    if (window._gpsTrailLine) gpsMap.removeLayer(window._gpsTrailLine);
    if (gpsTrail.length > 1) {
        window._gpsTrailLine = L.polyline(gpsTrail, { color: "#6366f1", weight: 2, opacity: 0.5, dashArray: "4 6" }).addTo(gpsMap);
    }
}

// ==========================================
// ALERT SYSTEM
// ==========================================
let alertVisible = false;
let lastAlertKey = null;
let lastAlertTime = 0;
const ALERT_COOLDOWN_MS = 10000; // 10 seconds between alerts

function checkAlerts(data) {
    let alert = null;

    const wid = (data.worker && data.worker.id) ? data.worker.id : "--";
    const locStr = (data.gps && data.gps.lat !== null) ? data.gps.lat.toFixed(5) + ", " + data.gps.lng.toFixed(5) : "Unknown";
    const footer = "\nWorker: " + wid + " | Location: " + locStr;

    if (data.fall && data.fall.detected === true) {
        alert = { icon: "🚨", level: "SOS — FALL DETECTED", title: "Worker " + wid + " has fallen!", desc: "Immediate response required. Fall probability: " + ((data.fall.probability || 0) * 100).toFixed(0) + "%" + footer, key: "fall" };
    } else if (data.vital_anomaly && data.vital_anomaly.status === "VITAL_ANOMALY") {
        alert = { icon: "🔴", level: "SOS — VITAL ANOMALY", title: "Critical vital signs — Worker " + wid, desc: "HR deviation: " + data.vital_anomaly.hr_deviation + " | SpO2 deviation: " + data.vital_anomaly.spo2_deviation + footer, key: "vital" };
    } else if (data.thermal && data.thermal.estimated_core_temp_c >= 39.0) {
        alert = { icon: "🌡️", level: "ALERT — HEAT STRESS", title: "Core temperature dangerously high — Worker " + wid, desc: "Estimated: " + Number(data.thermal.estimated_core_temp_c).toFixed(1) + "°C | Predicted: " + Number(data.thermal.predicted_core_temp_c).toFixed(1) + "°C" + footer, key: "heat" };
    } else if (data.fatigue && data.fatigue.score !== null && data.fatigue.score >= 7.5) {
        alert = { icon: "😴", level: "ALERT — SEVERE FATIGUE", title: "Worker " + wid + " fatigue critical", desc: "Fatigue score: " + data.fatigue.score.toFixed(1) + " / 10 | Trend: " + (data.fatigue.trend || "unknown").toUpperCase() + footer, key: "fatigue" };
    } else if (data.vitals && data.vitals.spo2 !== null && data.vitals.spo2 < 90) {
        alert = { icon: "💨", level: "ALERT — SpO2 DROP", title: "Blood oxygen critically low — Worker " + wid, desc: "Current SpO2: " + data.vitals.spo2 + "% — Below safe threshold of 90%" + footer, key: "spo2" };
    }

    const now = Date.now();

    if (alert && (now - lastAlertTime >= ALERT_COOLDOWN_MS)) {
        showAlert(alert);
        lastAlertTime = now;
    } else if (!alert && alertVisible) {
        dismissAlert();
    }
}

function showAlert(a) {
    document.getElementById("alert-modal-icon").innerText = a.icon;
    document.getElementById("alert-modal-level").innerText = a.level;
    document.getElementById("alert-modal-title").innerText = a.title;
    document.getElementById("alert-modal-desc").innerText = a.desc;
    document.getElementById("alert-overlay").style.display = "flex";
    alertVisible = true;
    lastAlertKey = a.key;
}

function dismissAlert() {
    document.getElementById("alert-overlay").style.display = "none";
    alertVisible = false;
    lastAlertKey = null;
}

// ==========================================
// SCENARIO TRIGGERS
// ==========================================
async function triggerScenario(scenario) {
    try {
        await fetch(API_BASE + '/api/simulation/scenario', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scenario: scenario, duration: 10 })
        });
    } catch (e) {
        console.error('Failed to trigger scenario:', e);
    }
}

// Hidden Keyboard Shortcuts for Scenarios
document.addEventListener('keydown', (e) => {
    if (e.altKey) {
        switch(e.key.toLowerCase()) {
            case 'f': triggerScenario('fall'); break;
            case 'v': triggerScenario('vital_anomaly'); break;
            case 'h': triggerScenario('heat_stress'); break;
            case 'e': triggerScenario('fatigue'); break;
            case 'o': triggerScenario('spo2_drop'); break;
            case 'r': triggerScenario('reset'); break;
        }
    }
});

function updateStatusBadge(id, status) {
    const el = document.getElementById(id);
    if (!status) return;
    el.innerText = status;
    if (status === "ACTIVE") {
        el.classList.add("active");
    } else {
        el.classList.remove("active");
    }
}

function saveProfile() {
    const p = {
        id: document.getElementById("w-id").value,
        age: parseFloat(document.getElementById("w-age").value),
        sex: parseInt(document.getElementById("w-sex").value),
        height_cm: parseFloat(document.getElementById("w-height").value),
        weight_kg: parseFloat(document.getElementById("w-weight").value),
        job: parseInt(document.getElementById("w-job").value),
        work_rest: parseInt(document.getElementById("w-wr").value)
    };
    
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ worker: p }));
        alert("Profile saved and sent to ML models!");
    } else {
        alert("Not connected to server.");
    }
}

window.onload = connect;

// ==========================================
// AI MODEL TEST LAB — Self-contained module
// ==========================================
(function TestLab() {
    let models = [];
    let selectedModel = null;
    let currentInputs = {};
    let activeScenario = null;

    // DOM refs
    const modelsGrid = document.getElementById('lab-models-grid');
    const inputsCard = document.getElementById('lab-inputs-card');
    const inputsGrid = document.getElementById('lab-inputs-grid');
    const scenariosCard = document.getElementById('lab-scenarios-card');
    const scenariosGrid = document.getElementById('lab-scenarios-grid');
    const runArea = document.getElementById('lab-run-area');
    const runBtn = document.getElementById('lab-run-btn');
    const resetBtn = document.getElementById('lab-reset-btn');
    const resultCard = document.getElementById('lab-result-card');
    const resultBody = document.getElementById('lab-result-body');
    const explainCard = document.getElementById('lab-explain-card');
    const explainBody = document.getElementById('lab-explain-body');

    // Flow steps
    const flowSteps = {
        inputs: document.getElementById('flow-inputs'),
        model: document.getElementById('flow-model'),
        prediction: document.getElementById('flow-prediction'),
        risk: document.getElementById('flow-risk')
    };

    function setFlowActive(...ids) {
        Object.entries(flowSteps).forEach(([id, el]) => {
            if (el) el.classList.toggle('lab__flow-step--active', ids.includes(id));
        });
    }

    // ---- Load Models ----
    async function loadModels() {
        try {
            const res = await fetch(API_BASE + '/api/models');
            const data = await res.json();
            models = data.models || [];
            renderModelSelector();
        } catch (e) {
            modelsGrid.innerHTML = '<p class="lab__loading">⚠️ Could not connect to THERMORA server. Start the server with <strong>python src/thermora_live_server.py</strong></p>';
        }
    }

    function renderModelSelector() {
        if (models.length === 0) {
            modelsGrid.innerHTML = '<p class="lab__loading">No ML models available.</p>';
            return;
        }
        modelsGrid.innerHTML = models.map(m => `
            <div class="lab__model-btn" data-model-id="${m.id}" id="lab-model-${m.id}">
                <span class="lab__model-type">${m.type}</span>
                <span class="lab__model-name">${m.name}</span>
                <span class="lab__model-algo">${m.algorithm}</span>
                <span class="lab__model-desc">${m.description}</span>
            </div>
        `).join('');

        modelsGrid.querySelectorAll('.lab__model-btn').forEach(btn => {
            btn.addEventListener('click', () => selectModel(btn.dataset.modelId));
        });
    }

    function selectModel(modelId) {
        selectedModel = models.find(m => m.id === modelId);
        if (!selectedModel) return;

        // Update selector visuals
        modelsGrid.querySelectorAll('.lab__model-btn').forEach(b => b.classList.remove('lab__model-btn--selected'));
        document.getElementById('lab-model-' + modelId)?.classList.add('lab__model-btn--selected');

        // Reset results
        resultCard.style.display = 'none';
        explainCard.style.display = 'none';
        activeScenario = null;

        // Show panels
        inputsCard.style.display = '';
        scenariosCard.style.display = '';
        runArea.style.display = '';

        setFlowActive('inputs', 'model');
        renderInputs();
        renderScenarios();
    }

    // ---- Render Inputs ----
    function renderInputs() {
        currentInputs = {};
        inputsGrid.innerHTML = '';

        selectedModel.inputs.forEach(inp => {
            currentInputs[inp.key] = inp.default;

            if (inp.options) {
                // Select input
                const optionsHtml = inp.options.map((opt, i) =>
                    `<option value="${i}" ${i === inp.default ? 'selected' : ''}>${opt}</option>`
                ).join('');
                inputsGrid.innerHTML += `
                    <div class="lab__input-item" id="lab-input-${inp.key}">
                        <div class="lab__input-header">
                            <span class="lab__input-label">${inp.label}</span>
                            <div class="lab__input-value">
                                <span class="lab__input-num" id="lab-val-${inp.key}">${inp.options[inp.default]}</span>
                            </div>
                        </div>
                        <select class="lab__input-select" data-key="${inp.key}">
                            ${optionsHtml}
                        </select>
                    </div>
                `;
            } else {
                // Slider input
                const step = inp.step;
                const decimals = step < 1 ? (step < 0.1 ? 2 : 1) : 0;
                inputsGrid.innerHTML += `
                    <div class="lab__input-item" id="lab-input-${inp.key}">
                        <div class="lab__input-header">
                            <span class="lab__input-label">${inp.label}</span>
                            <div class="lab__input-value">
                                <span class="lab__input-num" id="lab-val-${inp.key}">${Number(inp.default).toFixed(decimals)}</span>
                                <span class="lab__input-unit">${inp.unit}</span>
                            </div>
                        </div>
                        <input type="range" class="lab__input-slider" data-key="${inp.key}" data-decimals="${decimals}"
                               min="${inp.min}" max="${inp.max}" step="${inp.step}" value="${inp.default}">
                    </div>
                `;
            }
        });

        // Attach listeners
        inputsGrid.querySelectorAll('.lab__input-slider').forEach(slider => {
            slider.addEventListener('input', (e) => {
                const key = e.target.dataset.key;
                const decimals = parseInt(e.target.dataset.decimals);
                const val = parseFloat(e.target.value);
                currentInputs[key] = val;
                document.getElementById('lab-val-' + key).textContent = val.toFixed(decimals);
                clearActiveScenario();
            });
        });

        inputsGrid.querySelectorAll('.lab__input-select').forEach(sel => {
            sel.addEventListener('change', (e) => {
                const key = e.target.dataset.key;
                const val = parseInt(e.target.value);
                currentInputs[key] = val;
                const inp = selectedModel.inputs.find(i => i.key === key);
                document.getElementById('lab-val-' + key).textContent = inp.options[val];
                clearActiveScenario();
            });
        });
    }

    function setInputValue(key, value) {
        currentInputs[key] = value;
        const inp = selectedModel.inputs.find(i => i.key === key);
        if (!inp) return;

        if (inp.options) {
            const sel = inputsGrid.querySelector(`select[data-key="${key}"]`);
            if (sel) sel.value = value;
            document.getElementById('lab-val-' + key).textContent = inp.options[value] || value;
        } else {
            const slider = inputsGrid.querySelector(`input[data-key="${key}"]`);
            if (slider) slider.value = value;
            const decimals = inp.step < 1 ? (inp.step < 0.1 ? 2 : 1) : 0;
            document.getElementById('lab-val-' + key).textContent = Number(value).toFixed(decimals);
        }
    }

    // ---- Scenarios ----
    function renderScenarios() {
        if (!selectedModel.scenarios || selectedModel.scenarios.length === 0) {
            scenariosCard.style.display = 'none';
            return;
        }
        scenariosGrid.innerHTML = selectedModel.scenarios.map((s, i) => `
            <button class="lab__scenario-btn" data-scenario-idx="${i}" id="lab-scenario-${i}">
                <span class="lab__scenario-icon">${s.icon}</span>
                <span>${s.name}</span>
            </button>
        `).join('');

        scenariosGrid.querySelectorAll('.lab__scenario-btn').forEach(btn => {
            btn.addEventListener('click', () => applyScenario(parseInt(btn.dataset.scenarioIdx)));
        });
    }

    function applyScenario(idx) {
        const scenario = selectedModel.scenarios[idx];
        if (!scenario) return;

        activeScenario = idx;
        scenariosGrid.querySelectorAll('.lab__scenario-btn').forEach(b => b.classList.remove('lab__scenario-btn--active'));
        document.getElementById('lab-scenario-' + idx)?.classList.add('lab__scenario-btn--active');

        Object.entries(scenario.values).forEach(([key, val]) => setInputValue(key, val));
    }

    function clearActiveScenario() {
        activeScenario = null;
        scenariosGrid.querySelectorAll('.lab__scenario-btn').forEach(b => b.classList.remove('lab__scenario-btn--active'));
    }

    // ---- Run Prediction ----
    async function runPrediction() {
        if (!selectedModel) return;

        runBtn.disabled = true;
        runBtn.innerHTML = '<span class="lab__run-icon">⏳</span><span>Running...</span>';
        setFlowActive('inputs', 'model', 'prediction');

        try {
            const res = await fetch(API_BASE + '/api/demo-predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: selectedModel.id, inputs: currentInputs })
            });
            const data = await res.json();

            if (data.error) {
                resultBody.innerHTML = `<div class="lab__loading">⚠️ ${data.error}</div>`;
                resultCard.style.display = '';
                explainCard.style.display = 'none';
                setFlowActive('inputs', 'model');
            } else {
                renderResult(data);
                renderExplainability(data.feature_importance);
                setFlowActive('inputs', 'model', 'prediction', 'risk');
            }
        } catch (e) {
            resultBody.innerHTML = '<div class="lab__loading">⚠️ Failed to connect to THERMORA server.</div>';
            resultCard.style.display = '';
            explainCard.style.display = 'none';
        }

        runBtn.disabled = false;
        runBtn.innerHTML = '<span class="lab__run-icon">🚀</span><span>Run AI Prediction</span>';
    }

    // ---- Render Result ----
    function renderResult(data) {
        const pred = data.prediction;
        const info = data.model_info || {};
        const risk = (pred.risk_level || '').toLowerCase().replace(/\s+/g, '-');

        // Determine badge class
        let badgeClass = 'lab__result-risk-badge--activity';
        if (['normal', 'low', 'no-fall'].includes(risk)) badgeClass = 'lab__result-risk-badge--low';
        else if (['elevated', 'moderate'].includes(risk)) badgeClass = 'lab__result-risk-badge--moderate';
        else if (risk === 'high') badgeClass = 'lab__result-risk-badge--high';
        else if (['critical', 'fall-detected'].includes(risk)) badgeClass = 'lab__result-risk-badge--critical';

        let detailsHtml = '';
        if (data.model === 'thermal') {
            detailsHtml = `
                <div class="lab__result-detail"><span class="lab__result-detail-label">Est. Core Temp</span><span class="lab__result-detail-value">${pred.estimated_core_temp} °C</span></div>
                <div class="lab__result-detail"><span class="lab__result-detail-label">Predicted +10min</span><span class="lab__result-detail-value">${pred.predicted_core_temp_10min} °C</span></div>
                <div class="lab__result-detail"><span class="lab__result-detail-label">Predicted Rise</span><span class="lab__result-detail-value">${pred.predicted_rise > 0 ? '+' : ''}${pred.predicted_rise} °C</span></div>
            `;
        } else if (data.model === 'fatigue') {
            detailsHtml = `
                <div class="lab__result-detail"><span class="lab__result-detail-label">Fatigue Score</span><span class="lab__result-detail-value">${pred.fatigue_score} / 10</span></div>
            `;
        } else if (data.model === 'fall') {
            detailsHtml = `
                <div class="lab__result-detail"><span class="lab__result-detail-label">Fall Probability</span><span class="lab__result-detail-value">${pred.fall_probability}%</span></div>
                <div class="lab__result-detail"><span class="lab__result-detail-label">Threshold</span><span class="lab__result-detail-value">${pred.threshold}</span></div>
            `;
        } else if (data.model === 'activity') {
            detailsHtml = `
                <div class="lab__result-detail"><span class="lab__result-detail-label">Activity</span><span class="lab__result-detail-value">${pred.predicted_activity}</span></div>
            `;
            if (pred.top_probabilities && pred.top_probabilities.length > 0) {
                detailsHtml += '<div class="lab__result-detail" style="grid-column: span 2;"><span class="lab__result-detail-label">Top Predictions</span><div class="lab__probs-grid">';
                pred.top_probabilities.forEach(p => {
                    detailsHtml += `<span class="lab__prob-name">${p.activity}</span><span class="lab__prob-val">${p.probability}%</span>`;
                });
                detailsHtml += '</div></div>';
            }
        }

        resultBody.innerHTML = `
            <div class="lab__result-main">
                <div class="lab__result-risk">
                    <span class="lab__result-risk-badge ${badgeClass}">${pred.risk_level}</span>
                    <span class="lab__result-risk-label">Risk Level</span>
                </div>
                <div class="lab__result-details">${detailsHtml}</div>
            </div>
            <div class="lab__result-meta">
                <div class="lab__result-meta-item">🧠 <strong>Model:</strong> ${info.model_name || selectedModel.name}</div>
                <div class="lab__result-meta-item">⚙️ <strong>Algorithm:</strong> ${info.algorithm || selectedModel.algorithm}</div>
                <div class="lab__result-meta-item">📥 <strong>Input:</strong> CUSTOM DEMO INPUT</div>
                ${info.validated_mae ? `<div class="lab__result-meta-item">📏 <strong>Validated MAE:</strong> ${info.validated_mae}</div>` : ''}
                ${info.accuracy ? `<div class="lab__result-meta-item">🎯 <strong>Accuracy:</strong> ${(info.accuracy * 100).toFixed(1)}%</div>` : ''}
            </div>
        `;
        resultCard.style.display = '';
    }

    // ---- Render Explainability ----
    function renderExplainability(fi) {
        if (!fi || fi.length === 0) {
            explainBody.innerHTML = '<p class="lab__explain-unavailable">Model explanation unavailable for this model.</p>';
            explainCard.style.display = '';
            return;
        }

        const maxVal = Math.max(...fi.map(f => f.importance));

        explainBody.innerHTML = fi.map(f => {
            const pct = maxVal > 0 ? (f.importance / maxVal * 100) : 0;
            return `
                <div class="lab__explain-row">
                    <span class="lab__explain-feature">${f.feature}</span>
                    <div class="lab__explain-bar-track">
                        <div class="lab__explain-bar" style="width: ${pct.toFixed(1)}%"></div>
                    </div>
                    <span class="lab__explain-val">${f.importance}</span>
                </div>
            `;
        }).join('');

        explainCard.style.display = '';
    }

    // ---- Reset ----
    function resetInputs() {
        if (!selectedModel) return;
        selectedModel.inputs.forEach(inp => setInputValue(inp.key, inp.default));
        clearActiveScenario();
        resultCard.style.display = 'none';
        explainCard.style.display = 'none';
        setFlowActive('inputs', 'model');
    }

    // ---- Event Bindings ----
    if (runBtn) runBtn.addEventListener('click', runPrediction);
    if (resetBtn) resetBtn.addEventListener('click', resetInputs);

    // Load models on page load
    loadModels();
})();

// ==========================================
// SIMULATION TOGGLE — Controls built-in sensor simulation
// ==========================================
(function SimulationToggle() {
    const toggle = document.getElementById('sim-switch');
    const bar = document.getElementById('sim-bar');
    const statusText = document.getElementById('sim-status-text');
    const desc = document.getElementById('sim-desc');

    if (!toggle || !bar) return;

    function updateUI(active) {
        toggle.checked = active;
        if (active) {
            bar.classList.add('sim-bar--active');
            statusText.textContent = 'ACTIVE';
            desc.textContent = 'Synthetic sensor data is being injected — dashboard is in demo mode';
        } else {
            bar.classList.remove('sim-bar--active');
            statusText.textContent = 'OFF';
            desc.textContent = 'Toggle to inject synthetic sensor data for demonstration';
        }
    }

    // Check server status on page load
    async function checkStatus() {
        try {
            const res = await fetch(API_BASE + '/api/simulation/status');
            const data = await res.json();
            updateUI(data.simulation);
        } catch (e) {
            // Server not reachable — leave as OFF
        }
    }

    // Toggle handler
    toggle.addEventListener('change', async function () {
        toggle.disabled = true;
        try {
            const res = await fetch(API_BASE + '/api/simulation/toggle', { method: 'POST' });
            const data = await res.json();
            updateUI(data.simulation);
        } catch (e) {
            console.error('Failed to toggle simulation:', e);
            // Revert the checkbox
            toggle.checked = !toggle.checked;
        }
        toggle.disabled = false;
    });

    checkStatus();
})();
