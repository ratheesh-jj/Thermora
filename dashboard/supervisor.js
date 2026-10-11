// ==========================================
// THERMORA SUPERVISOR DASHBOARD
// ==========================================
const API_BASE = (window.location.protocol === 'file:' || !window.location.port) ? 'http://localhost:8000' : '';

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

let supMap = null;
let workerMarkers = {};
let lastWorkerData = [];

// ---- Clock ----
function updateClock() {
    const now = new Date();
    document.getElementById('sup-clock').textContent = now.toLocaleTimeString('en-IN', { hour12: false });
}
setInterval(updateClock, 1000);
updateClock();

// ---- Map ----
function initSupMap() {
    supMap = L.map('sup-map', {
        zoomControl: true,
        attributionControl: false
    }).setView([11.6780, 78.1243], 16);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19
    }).addTo(supMap);
}

function getMarkerColor(status) {
    if (status === 'CRITICAL') return '#ef4444';
    if (status === 'ATTENTION') return '#f59e0b';
    return '#10b981';
}

function updateMapMarkers(workers) {
    workers.forEach(w => {
        if (!w.gps || w.gps.lat === null) return;
        const color = getMarkerColor(w.status);
        if (workerMarkers[w.id]) {
            workerMarkers[w.id].setLatLng([w.gps.lat, w.gps.lng]);
            workerMarkers[w.id].setStyle({ fillColor: color, color: color });
        } else {
            const marker = L.circleMarker([w.gps.lat, w.gps.lng], {
                radius: 10,
                fillColor: color,
                color: color,
                weight: 2,
                fillOpacity: 0.85
            }).addTo(supMap);
            marker.bindPopup(`
                <div style="font-family: Inter, sans-serif; min-width: 160px;">
                    <strong>${w.id} — ${w.name}</strong><br>
                    <span style="color:#888">${w.role}</span><br>
                    <hr style="border:none;border-top:1px solid #ddd;margin:6px 0">
                    ❤️ HR: <strong>${w.hr !== null ? Math.round(w.hr) : '--'}</strong> bpm<br>
                    💨 SpO2: <strong>${w.spo2 !== null ? Math.round(w.spo2) : '--'}</strong>%<br>
                    🌡️ Core: <strong>${w.core_temp !== null ? w.core_temp.toFixed(1) : '--'}</strong>°C<br>
                    😴 Fatigue: <strong>${w.fatigue !== null ? w.fatigue.toFixed(1) : '--'}</strong>/10
                </div>
            `);
            workerMarkers[w.id] = marker;
        }
    });
}

// ---- Summary Stats ----
function updateSummary(workers) {
    const total = workers.filter(w => w.connected).length;
    const normal = workers.filter(w => w.status === 'NORMAL').length;
    const attention = workers.filter(w => w.status === 'ATTENTION').length;
    const critical = workers.filter(w => w.status === 'CRITICAL').length;

    document.getElementById('sup-total').textContent = total;
    document.getElementById('sup-normal').textContent = normal;
    document.getElementById('sup-attention').textContent = attention;
    document.getElementById('sup-critical').textContent = critical;
}

// ---- Alerts ----
function updateAlerts(workers) {
    const list = document.getElementById('sup-alerts-list');
    const alertWorkers = workers.filter(w =>
        w.status === 'CRITICAL' || w.status === 'ATTENTION' || w.fall_detected
    );

    if (alertWorkers.length === 0) {
        list.innerHTML = '<div class="sup-alert-empty">✅ No active alerts — all workers safe</div>';
        return;
    }

    list.innerHTML = alertWorkers.map(w => {
        const isCrit = w.status === 'CRITICAL' || w.fall_detected;
        const cls = isCrit ? 'sup-alert-item--critical' : 'sup-alert-item--attention';
        const icon = isCrit ? '🚨' : '⚠️';
        let reason = w.vital_status || w.status;
        if (w.fall_detected) reason = 'FALL DETECTED';
        const loc = (w.gps && w.gps.lat) ? w.gps.lat.toFixed(5) + ', ' + w.gps.lng.toFixed(5) : 'Unknown';
        return `
            <div class="sup-alert-item ${cls}">
                <span class="sup-alert-item__icon">${icon}</span>
                <div class="sup-alert-item__body">
                    <div class="sup-alert-item__title">${w.id} — ${w.name} <span class="sup-alert-item__role">${w.role}</span></div>
                    <div class="sup-alert-item__reason">${reason}</div>
                    <div class="sup-alert-item__meta">📍 ${loc} · 🕐 ${w.last_update}</div>
                </div>
            </div>
        `;
    }).join('');
}

// ---- Worker Grid ----
function updateWorkerGrid(workers) {
    const grid = document.getElementById('sup-workers-grid');
    grid.innerHTML = workers.map(w => {
        const statusCls = w.status === 'CRITICAL' ? 'sup-wcard--critical' :
                          w.status === 'ATTENTION' ? 'sup-wcard--attention' : 'sup-wcard--normal';
        const badgeCls = w.status === 'CRITICAL' ? 'sup-wcard__badge--critical' :
                         w.status === 'ATTENTION' ? 'sup-wcard__badge--attention' : 'sup-wcard__badge--normal';
        const connDot = w.connected ? '<span class="sup-wcard__conn sup-wcard__conn--on"></span>' : '<span class="sup-wcard__conn"></span>';
        const loc = (w.gps && w.gps.lat) ? w.gps.lat.toFixed(5) + ', ' + w.gps.lng.toFixed(5) : '--';

        return `
            <div class="sup-wcard ${statusCls}">
                <div class="sup-wcard__header">
                    <div class="sup-wcard__id-row">
                        ${connDot}
                        <span class="sup-wcard__id">${w.id}</span>
                        <span class="sup-wcard__badge ${badgeCls}">${w.status}</span>
                    </div>
                    <div class="sup-wcard__name">${w.name}</div>
                    <div class="sup-wcard__role">${w.role}</div>
                </div>
                <div class="sup-wcard__vitals">
                    <div class="sup-wcard__vital">
                        <span class="sup-wcard__vital-icon">❤️</span>
                        <span class="sup-wcard__vital-val">${w.hr !== null ? Math.round(w.hr) : '--'}</span>
                        <span class="sup-wcard__vital-unit">bpm</span>
                    </div>
                    <div class="sup-wcard__vital">
                        <span class="sup-wcard__vital-icon">💨</span>
                        <span class="sup-wcard__vital-val">${w.spo2 !== null ? Math.round(w.spo2) : '--'}</span>
                        <span class="sup-wcard__vital-unit">%</span>
                    </div>
                    <div class="sup-wcard__vital">
                        <span class="sup-wcard__vital-icon">🌡️</span>
                        <span class="sup-wcard__vital-val">${w.core_temp !== null ? w.core_temp.toFixed(1) : '--'}</span>
                        <span class="sup-wcard__vital-unit">°C</span>
                    </div>
                    <div class="sup-wcard__vital">
                        <span class="sup-wcard__vital-icon">😴</span>
                        <span class="sup-wcard__vital-val">${w.fatigue !== null ? w.fatigue.toFixed(1) : '--'}</span>
                        <span class="sup-wcard__vital-unit">/10</span>
                    </div>
                </div>
                <div class="sup-wcard__footer">
                    <span class="sup-wcard__loc">📍 ${loc}</span>
                    <span class="sup-wcard__time">🕐 ${w.last_update}</span>
                </div>
                <div class="sup-wcard__vital-status">
                    <span class="sup-wcard__vs-label">Vital:</span>
                    <span class="sup-wcard__vs-val">${w.vital_status}</span>
                </div>
            </div>
        `;
    }).join('');
}

// ---- Polling ----
async function fetchWorkers() {
    try {
        const res = await fetch(API_BASE + '/api/workers');
        const data = await res.json();
        lastWorkerData = data.workers || [];
        updateSummary(lastWorkerData);
        updateAlerts(lastWorkerData);
        updateWorkerGrid(lastWorkerData);
        updateMapMarkers(lastWorkerData);
    } catch (e) {
        console.error('Failed to fetch workers:', e);
    }
}

// ---- Init ----
window.addEventListener('load', () => {
    initSupMap();
    fetchWorkers();
    setInterval(fetchWorkers, 2000);

    // SOS system init
    seedDemoSOSEvents();
    initSOSSection();
});

// ==========================================
// SOS & EMERGENCY CALL LOG
// ==========================================

let sosActiveFilter = 'all';
let sosSearchQuery = '';

function initSOSSection() {
    renderSOSTable();
    updateSOSCounters();
    bindSOSFilters();
    bindSOSSearch();
    bindSOSSimulate();
    bindSOSModal();
}

// ---- Render SOS Table ----
function renderSOSTable() {
    const tbody = document.getElementById('sos-table-body');
    let events = getAllSOSEvents();

    // Apply filter
    const todayStr = new Date().toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
    if (sosActiveFilter === 'today') {
        events = events.filter(e => e.date === todayStr);
    } else if (sosActiveFilter !== 'all') {
        events = events.filter(e => e.emergencyType === sosActiveFilter);
    }

    // Apply search
    if (sosSearchQuery) {
        const q = sosSearchQuery.toLowerCase();
        events = events.filter(e =>
            e.eventId.toLowerCase().includes(q) ||
            e.workerId.toLowerCase().includes(q) ||
            e.workerName.toLowerCase().includes(q)
        );
    }

    if (events.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11" class="sos-table__empty">No SOS events match your criteria</td></tr>';
        return;
    }

    tbody.innerHTML = events.map(evt => {
        const callBadgeCls = getCallStatusClass(evt.callStatus);
        const respBadgeCls = getResponseStatusClass(evt.responseStatus);
        const typeBadgeCls = getEmergencyTypeClass(evt.emergencyType);
        return `
            <tr class="sos-table__row" data-event-id="${evt.eventId}">
                <td class="sos-table__cell--id">${evt.eventId}</td>
                <td class="sos-table__cell--mono">${evt.workerId}</td>
                <td>${evt.workerName}</td>
                <td>${evt.date}</td>
                <td class="sos-table__cell--mono">${evt.time}</td>
                <td><span class="sos-badge ${typeBadgeCls}">${evt.emergencyType}</span></td>
                <td>${evt.triggerType}</td>
                <td>${evt.contactedParty}</td>
                <td><span class="sos-badge ${callBadgeCls}">${evt.callStatus}</span></td>
                <td><span class="sos-badge ${respBadgeCls}">${evt.responseStatus}</span></td>
                <td><button class="sos-details-btn" onclick="openSOSDetail('${evt.eventId}')">View Details</button></td>
            </tr>
        `;
    }).join('');
}

// ---- Badge CSS class helpers ----
function getCallStatusClass(status) {
    switch (status) {
        case 'Connected': return 'sos-badge--connected';
        case 'Completed': return 'sos-badge--completed';
        case 'Pending':   return 'sos-badge--pending';
        case 'Failed':    return 'sos-badge--failed';
        default:          return '';
    }
}

function getResponseStatusClass(status) {
    switch (status) {
        case 'Acknowledged': return 'sos-badge--acknowledged';
        case 'Responding':   return 'sos-badge--responding';
        case 'Dispatched':   return 'sos-badge--dispatched';
        case 'Pending':      return 'sos-badge--pending';
        default:             return '';
    }
}

function getEmergencyTypeClass(type) {
    switch (type) {
        case 'Fall Detected': return 'sos-badge--fall';
        case 'Heat Stress':   return 'sos-badge--heat';
        case 'Vital Anomaly': return 'sos-badge--vital';
        case 'Severe Fatigue':return 'sos-badge--fatigue';
        case 'Manual SOS':    return 'sos-badge--manual';
        default:              return '';
    }
}

// ---- Update SOS Counters ----
function updateSOSCounters() {
    const summary = getSOSSummary();
    document.getElementById('sos-count-events').textContent = String(summary.totalToday).padStart(2, '0');
    document.getElementById('sos-count-calls').textContent = String(summary.callsMade).padStart(2, '0');
    document.getElementById('sos-count-ack').textContent = String(summary.acknowledged).padStart(2, '0');
    document.getElementById('sos-count-pending').textContent = String(summary.pending).padStart(2, '0');
}

// ---- Filter Buttons ----
function bindSOSFilters() {
    const container = document.getElementById('sos-filters');
    container.addEventListener('click', (e) => {
        const btn = e.target.closest('.sos-filter-btn');
        if (!btn) return;
        container.querySelectorAll('.sos-filter-btn').forEach(b => b.classList.remove('sos-filter-btn--active'));
        btn.classList.add('sos-filter-btn--active');
        sosActiveFilter = btn.dataset.filter;
        renderSOSTable();
    });
}

// ---- Search ----
function bindSOSSearch() {
    const input = document.getElementById('sos-search');
    input.addEventListener('input', () => {
        sosSearchQuery = input.value.trim();
        renderSOSTable();
    });
}

// ---- Simulate SOS ----
function bindSOSSimulate() {
    const btn = document.getElementById('sos-simulate-btn');
    if (btn) btn.addEventListener('click', simulateSOS);
}

function simulateSOS() {
    // Pick a random realistic worker
    const workerPool = [
        { id: 'WRK-003', name: 'Worker 03' },
        { id: 'WRK-005', name: 'Worker 05' },
        { id: 'WRK-009', name: 'Worker 09' },
        { id: 'WRK-012', name: 'Worker 12' },
        { id: 'WRK-017', name: 'Worker 17' },
        { id: 'WRK-019', name: 'Worker 19' },
        { id: 'WRK-022', name: 'Worker 22' },
        { id: 'WRK-025', name: 'Worker 25' },
    ];
    const worker = workerPool[Math.floor(Math.random() * workerPool.length)];
    const emergencyType = EMERGENCY_TYPES[Math.floor(Math.random() * EMERGENCY_TYPES.length)];
    const contactedParty = CONTACTED_PARTIES[Math.floor(Math.random() * CONTACTED_PARTIES.length)];

    const evt = registerSOSEvent({
        workerId: worker.id,
        workerName: worker.name,
        emergencyType: emergencyType,
        contactedParty: contactedParty,
        callStatus: 'Connected',
        responseStatus: 'Acknowledged'
    });

    // Update UI
    renderSOSTable();
    updateSOSCounters();
    showSOSToast(evt);

    // Briefly highlight the new row
    setTimeout(() => {
        const row = document.querySelector(`tr[data-event-id="${evt.eventId}"]`);
        if (row) {
            row.classList.add('sos-table__row--new');
            setTimeout(() => row.classList.remove('sos-table__row--new'), 2500);
        }
    }, 50);
}

// ---- Toast Notification ----
function showSOSToast(evt) {
    const toast = document.getElementById('sos-toast');
    document.getElementById('sos-toast-title').textContent = 'SOS ALERT REGISTERED — ' + evt.eventId;
    document.getElementById('sos-toast-msg').textContent = `${evt.workerName} (${evt.workerId}) · ${evt.emergencyType} · ${evt.contactedParty} contacted · ${evt.callStatus}`;
    toast.style.display = 'flex';
    toast.classList.add('sos-toast--visible');

    setTimeout(() => {
        toast.classList.remove('sos-toast--visible');
        setTimeout(() => { toast.style.display = 'none'; }, 400);
    }, 5000);
}

// ---- Detail Modal ----
function bindSOSModal() {
    document.getElementById('sos-modal-close').addEventListener('click', closeSOSDetail);
    document.getElementById('sos-modal-overlay').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) closeSOSDetail();
    });
}

function openSOSDetail(eventId) {
    const events = getAllSOSEvents();
    const evt = events.find(e => e.eventId === eventId);
    if (!evt) return;

    const body = document.getElementById('sos-modal-body');
    const riskCls = evt.riskLevel === 'Critical' ? 'sos-risk--critical' : evt.riskLevel === 'High' ? 'sos-risk--high' : 'sos-risk--moderate';

    body.innerHTML = `
        <div class="sos-detail-grid">
            <div class="sos-detail-row">
                <span class="sos-detail-label">Event ID</span>
                <span class="sos-detail-value sos-detail-value--mono">${evt.eventId}</span>
            </div>
            <div class="sos-detail-row">
                <span class="sos-detail-label">Worker</span>
                <span class="sos-detail-value">${evt.workerName} (${evt.workerId})</span>
            </div>
            <div class="sos-detail-row">
                <span class="sos-detail-label">Trigger</span>
                <span class="sos-detail-value">${evt.triggerType}</span>
            </div>
            <div class="sos-detail-row">
                <span class="sos-detail-label">Detection Time</span>
                <span class="sos-detail-value sos-detail-value--mono">${evt.detectionTime}</span>
            </div>
            <div class="sos-detail-row">
                <span class="sos-detail-label">Emergency Type</span>
                <span class="sos-detail-value"><span class="sos-badge ${getEmergencyTypeClass(evt.emergencyType)}">${evt.emergencyType}</span></span>
            </div>
            <div class="sos-detail-row">
                <span class="sos-detail-label">Risk Level</span>
                <span class="sos-detail-value"><span class="sos-risk ${riskCls}">${evt.riskLevel}</span></span>
            </div>
            <div class="sos-detail-row">
                <span class="sos-detail-label">Contacted</span>
                <span class="sos-detail-value">${evt.contactedParty}</span>
            </div>
            <div class="sos-detail-row">
                <span class="sos-detail-label">Call Initiated</span>
                <span class="sos-detail-value sos-detail-value--mono">${evt.callInitTime}</span>
            </div>
            <div class="sos-detail-row">
                <span class="sos-detail-label">Call Connected</span>
                <span class="sos-detail-value sos-detail-value--mono">${evt.callConnTime}</span>
            </div>
            <div class="sos-detail-row">
                <span class="sos-detail-label">Response Status</span>
                <span class="sos-detail-value"><span class="sos-badge ${getResponseStatusClass(evt.responseStatus)}">${evt.responseStatus}</span></span>
            </div>
            <div class="sos-detail-row sos-detail-row--full">
                <span class="sos-detail-label">Alert Message</span>
                <span class="sos-detail-value sos-detail-value--msg">${evt.alertMessage}</span>
            </div>
        </div>

        <div class="sos-timeline">
            <div class="sos-timeline__title">Event Timeline</div>
            <div class="sos-timeline__steps">
                <div class="sos-timeline__step sos-timeline__step--done">
                    <div class="sos-timeline__dot"></div>
                    <div class="sos-timeline__info">
                        <span class="sos-timeline__step-label">Emergency Detected</span>
                        <span class="sos-timeline__step-time">${evt.detectionTime}</span>
                    </div>
                </div>
                <div class="sos-timeline__connector"></div>
                <div class="sos-timeline__step sos-timeline__step--done">
                    <div class="sos-timeline__dot"></div>
                    <div class="sos-timeline__info">
                        <span class="sos-timeline__step-label">SOS Triggered</span>
                        <span class="sos-timeline__step-time">${evt.triggerTime}</span>
                    </div>
                </div>
                <div class="sos-timeline__connector"></div>
                <div class="sos-timeline__step sos-timeline__step--done">
                    <div class="sos-timeline__dot"></div>
                    <div class="sos-timeline__info">
                        <span class="sos-timeline__step-label">Call Initiated</span>
                        <span class="sos-timeline__step-time">${evt.callInitTime}</span>
                    </div>
                </div>
                <div class="sos-timeline__connector"></div>
                <div class="sos-timeline__step ${evt.callStatus !== 'Pending' && evt.callStatus !== 'Failed' ? 'sos-timeline__step--done' : ''}">
                    <div class="sos-timeline__dot"></div>
                    <div class="sos-timeline__info">
                        <span class="sos-timeline__step-label">Call Connected</span>
                        <span class="sos-timeline__step-time">${evt.callConnTime}</span>
                    </div>
                </div>
                <div class="sos-timeline__connector"></div>
                <div class="sos-timeline__step ${evt.responseStatus !== 'Pending' ? 'sos-timeline__step--done' : ''}">
                    <div class="sos-timeline__dot"></div>
                    <div class="sos-timeline__info">
                        <span class="sos-timeline__step-label">Response ${evt.responseStatus}</span>
                        <span class="sos-timeline__step-time">${evt.date} ${evt.time}</span>
                    </div>
                </div>
            </div>
        </div>
    `;

    document.getElementById('sos-modal-overlay').style.display = 'flex';
}

function closeSOSDetail() {
    document.getElementById('sos-modal-overlay').style.display = 'none';
}

// ==========================================
// PROGRESSIVE SOS DEMO SIMULATION
// Keyboard Shortcuts:
//   Shift+Alt+D  →  Full progressive vital anomaly demo
//   Shift+Alt+F  →  Fall progressive demo
//   Shift+Alt+Q  →  Instant SOS event
//   Shift+Alt+R  →  Reset demo worker
// ==========================================

let demoRunning = false;
let demoTimers = [];
let demoWorkerInjected = false;

const DEMO_WORKER = {
    id: 'WRK-007',
    name: 'Rajesh Kumar',
    role: 'Welder — Zone B',
    connected: true,
    status: 'NORMAL',
    vital_status: 'NORMAL',
    hr: 78,
    spo2: 97,
    core_temp: 37.2,
    fatigue: 3.1,
    fall_detected: false,
    gps: { lat: 11.6785, lng: 78.1248 },
    last_update: new Date().toLocaleTimeString('en-IN', { hour12: false })
};

function injectDemoWorker() {
    // Add or replace demo worker in the worker list
    const idx = lastWorkerData.findIndex(w => w.id === DEMO_WORKER.id);
    const worker = JSON.parse(JSON.stringify(DEMO_WORKER));
    worker.last_update = new Date().toLocaleTimeString('en-IN', { hour12: false });
    if (idx >= 0) {
        lastWorkerData[idx] = worker;
    } else {
        lastWorkerData.unshift(worker);
    }
    demoWorkerInjected = true;
    refreshSupervisorUI();
}

function updateDemoWorker(updates) {
    const idx = lastWorkerData.findIndex(w => w.id === DEMO_WORKER.id);
    if (idx < 0) return;
    Object.assign(lastWorkerData[idx], updates, {
        last_update: new Date().toLocaleTimeString('en-IN', { hour12: false })
    });
    refreshSupervisorUI();
    highlightWorkerCard(DEMO_WORKER.id);
}

function refreshSupervisorUI() {
    updateSummary(lastWorkerData);
    updateAlerts(lastWorkerData);
    updateWorkerGrid(lastWorkerData);
    updateMapMarkers(lastWorkerData);
}

function highlightWorkerCard(workerId) {
    // Flash the worker card briefly
    setTimeout(() => {
        const cards = document.querySelectorAll('.sup-wcard');
        cards.forEach(card => {
            if (card.textContent.includes(workerId)) {
                card.style.transition = 'box-shadow 0.3s, transform 0.3s';
                card.style.boxShadow = '0 0 20px rgba(239,68,68,0.5)';
                card.style.transform = 'scale(1.02)';
                setTimeout(() => {
                    card.style.boxShadow = '';
                    card.style.transform = '';
                }, 1500);
            }
        });
    }, 50);
}

function showDemoPhaseNotification(phase, message) {
    // Reuse the SOS toast for phase notifications
    const toast = document.getElementById('sos-toast');
    const icons = { detect: '📡', warn: '⚠️', critical: '🚨', sos: '🆘', log: '📋' };
    document.getElementById('sos-toast-title').textContent = phase;
    document.getElementById('sos-toast-msg').textContent = message;
    toast.style.display = 'flex';
    toast.classList.add('sos-toast--visible');

    clearTimeout(window._demoToastTimer);
    window._demoToastTimer = setTimeout(() => {
        toast.classList.remove('sos-toast--visible');
        setTimeout(() => { toast.style.display = 'none'; }, 400);
    }, 3000);
}

function clearDemoTimers() {
    demoTimers.forEach(t => clearTimeout(t));
    demoTimers = [];
}

function resetDemoWorker() {
    clearDemoTimers();
    demoRunning = false;
    const idx = lastWorkerData.findIndex(w => w.id === DEMO_WORKER.id);
    if (idx >= 0) {
        lastWorkerData[idx] = JSON.parse(JSON.stringify(DEMO_WORKER));
        lastWorkerData[idx].last_update = new Date().toLocaleTimeString('en-IN', { hour12: false });
    }
    refreshSupervisorUI();
    showDemoPhaseNotification('🔄 DEMO RESET', 'Worker WRK-007 vitals restored to normal.');
}

// ---- Full Progressive Demo ----
function runProgressiveDemo() {
    if (demoRunning) {
        showDemoPhaseNotification('⏳ DEMO IN PROGRESS', 'A simulation is already running. Press Alt+R to reset.');
        return;
    }
    demoRunning = true;
    clearDemoTimers();

    // PHASE 0 — Inject healthy worker
    injectDemoWorker();
    showDemoPhaseNotification('📡 PHASE 1 — MONITORING', 'WRK-007 Rajesh Kumar — Vitals normal. Monitoring active.');

    // PHASE 1 (2s) — Slight elevation, ATTENTION
    demoTimers.push(setTimeout(() => {
        updateDemoWorker({
            status: 'ATTENTION',
            vital_status: 'PHYSIOLOGICAL_DEVIATION',
            hr: 102,
            spo2: 94,
            core_temp: 38.1,
            fatigue: 5.8
        });
        showDemoPhaseNotification('⚠️ PHASE 2 — VITALS RISING', 'HR: 102 bpm | SpO2: 94% | Core Temp: 38.1°C — Worker under heat stress.');
    }, 2500));

    // PHASE 2 (5s) — Worsening, still ATTENTION
    demoTimers.push(setTimeout(() => {
        updateDemoWorker({
            status: 'ATTENTION',
            vital_status: 'UNCONFIRMED_VITAL_ANOMALY',
            hr: 118,
            spo2: 91,
            core_temp: 38.7,
            fatigue: 7.2
        });
        showDemoPhaseNotification('⚠️ PHASE 3 — ANOMALY DETECTED', 'HR: 118 bpm | SpO2: 91% | Fatigue: 7.2 — Vital anomaly unconfirmed.');
    }, 5000));

    // PHASE 3 (7.5s) — CRITICAL
    demoTimers.push(setTimeout(() => {
        updateDemoWorker({
            status: 'CRITICAL',
            vital_status: 'VITAL_ANOMALY',
            hr: 138,
            spo2: 86,
            core_temp: 39.4,
            fatigue: 8.9,
            fall_detected: false
        });
        showDemoPhaseNotification('🚨 PHASE 4 — CRITICAL ALERT', 'VITAL ANOMALY CONFIRMED — HR: 138 bpm | SpO2: 86% | Core: 39.4°C');
    }, 7500));

    // PHASE 4 (10s) — SOS Triggered + Event Registered
    demoTimers.push(setTimeout(() => {
        // Register the SOS event
        const evt = registerSOSEvent({
            workerId: 'WRK-007',
            workerName: 'Rajesh Kumar',
            emergencyType: 'Vital Anomaly',
            triggerType: 'Automatic SOS',
            contactedParty: 'Supervisor',
            callStatus: 'Connected',
            responseStatus: 'Responding',
            riskLevel: 'Critical',
            alertMessage: 'Confirmed vital anomaly — HR 138 bpm, SpO2 86%, Core Temp 39.4°C. Automatic SOS triggered. Supervisor contacted and responding.'
        });

        // Update table & counters
        renderSOSTable();
        updateSOSCounters();

        // Show SOS toast
        showSOSToast(evt);

        // Highlight new row
        setTimeout(() => {
            const row = document.querySelector(`tr[data-event-id="${evt.eventId}"]`);
            if (row) {
                row.classList.add('sos-table__row--new');
                setTimeout(() => row.classList.remove('sos-table__row--new'), 3000);
            }
            // Scroll to SOS section
            document.getElementById('sos-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);

        demoRunning = false;
    }, 10000));

    // Also try to trigger the backend scenario (if server is running)
    fetch(API_BASE + '/api/simulation/scenario', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: 'vital_anomaly', duration: 12 })
    }).catch(() => { /* Server may not be running — that's fine for demo */ });
}

// ---- Keyboard Shortcuts (Shift+Alt) ----
console.log('[THERMORA] SOS Demo shortcuts loaded: Shift+Alt+D / F / Q / R');
document.addEventListener('keydown', (e) => {
    // Require both Shift and Alt held down
    if (!e.shiftKey || !e.altKey) return;
    // Prevent Ctrl from interfering
    if (e.ctrlKey) return;

    const code = e.code; // physical key, reliable on all keyboards

    switch (code) {
        // Shift+Alt+D → Full progressive vital anomaly demo
        case 'KeyD':
            e.preventDefault();
            runProgressiveDemo();
            break;
        // Shift+Alt+Q → Instant SOS (same as Simulate button)
        case 'KeyQ':
            e.preventDefault();
            simulateSOS();
            break;
        // Shift+Alt+R → Reset demo worker
        case 'KeyR':
            e.preventDefault();
            resetDemoWorker();
            break;
        // Shift+Alt+F → Fall scenario (progressive)
        case 'KeyF':
            e.preventDefault();
            runFallDemo();
            break;
    }
});

// ---- Fall Progressive Demo (Alt+F) ----
function runFallDemo() {
    if (demoRunning) return;
    demoRunning = true;
    clearDemoTimers();

    injectDemoWorker();
    showDemoPhaseNotification('📡 MONITORING', 'WRK-007 Rajesh Kumar — Normal activity detected.');

    demoTimers.push(setTimeout(() => {
        updateDemoWorker({
            status: 'ATTENTION',
            vital_status: 'PHYSIOLOGICAL_DEVIATION',
            hr: 95,
            fatigue: 6.5
        });
        showDemoPhaseNotification('⚠️ FATIGUE RISING', 'Fatigue score increasing — Worker may be impaired.');
    }, 2500));

    demoTimers.push(setTimeout(() => {
        updateDemoWorker({
            status: 'CRITICAL',
            vital_status: 'VITAL_ANOMALY',
            hr: 145,
            spo2: 89,
            fall_detected: true
        });
        showDemoPhaseNotification('🚨 FALL DETECTED', 'Accelerometer impact detected — WRK-007 has fallen!');
    }, 5500));

    demoTimers.push(setTimeout(() => {
        const evt = registerSOSEvent({
            workerId: 'WRK-007',
            workerName: 'Rajesh Kumar',
            emergencyType: 'Fall Detected',
            triggerType: 'Automatic SOS',
            contactedParty: 'Emergency Services',
            callStatus: 'Connected',
            responseStatus: 'Dispatched',
            riskLevel: 'Critical',
            alertMessage: 'Fall impact detected by accelerometer. Worker may be injured. Emergency services contacted and dispatched.'
        });

        renderSOSTable();
        updateSOSCounters();
        showSOSToast(evt);

        setTimeout(() => {
            const row = document.querySelector(`tr[data-event-id="${evt.eventId}"]`);
            if (row) {
                row.classList.add('sos-table__row--new');
                setTimeout(() => row.classList.remove('sos-table__row--new'), 3000);
            }
            document.getElementById('sos-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);

        // Also trigger backend
        fetch(API_BASE + '/api/simulation/scenario', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scenario: 'fall', duration: 10 })
        }).catch(() => {});

        demoRunning = false;
    }, 8000));
}
