// ============================================================
// THERMORA — SOS Event Data Layer
// Reusable data model & store for SOS / Emergency Call events.
// Uses localStorage for persistence. Replace with backend API
// calls when real GSM/SMS/call integration is available.
// ============================================================

const SOS_STORAGE_KEY = 'thermora_sos_events';

// ---- Emergency type & status enums ----
const EMERGENCY_TYPES = ['Fall Detected', 'Heat Stress', 'Vital Anomaly', 'Severe Fatigue', 'Manual SOS'];
const TRIGGER_TYPES   = ['Automatic SOS', 'Manual SOS'];
const CONTACTED_PARTIES = ['Supervisor', 'Caretaker', 'Emergency Services'];
const CALL_STATUSES   = ['Connected', 'Failed', 'Pending', 'Completed'];
const RESPONSE_STATUSES = ['Acknowledged', 'Responding', 'Dispatched', 'Pending'];
const RISK_LEVELS     = ['Critical', 'High', 'Moderate'];

// ---- Alert messages per emergency type ----
const ALERT_MESSAGES = {
    'Fall Detected':  'Fall impact detected by accelerometer. Worker may be injured. Immediate assistance required.',
    'Heat Stress':    'Core body temperature exceeds safe threshold. Worker showing signs of heat exhaustion.',
    'Vital Anomaly':  'Abnormal vital signs detected — heart rate and SpO2 outside safe range. Medical evaluation needed.',
    'Severe Fatigue': 'Fatigue index critically high. Worker reaction time impaired. Risk of accident elevated.',
    'Manual SOS':     'Worker manually triggered SOS emergency button. Immediate response required.'
};

// ---- Event ID counter (auto-increments) ----
function _getNextEventId() {
    let counter = parseInt(localStorage.getItem('thermora_sos_counter') || '1024', 10);
    counter++;
    localStorage.setItem('thermora_sos_counter', counter.toString());
    return 'EVT-' + String(counter).padStart(4, '0');
}

// ---- SOS Event Model ----
// Creates a new SOS event object.
// @param {Object} opts — partial event data; missing fields are auto-filled.
// @returns {Object} complete SOS event record.
function createSOSEvent(opts = {}) {
    const now = opts.timestamp ? new Date(opts.timestamp) : new Date();
    const emergencyType = opts.emergencyType || EMERGENCY_TYPES[Math.floor(Math.random() * EMERGENCY_TYPES.length)];
    const triggerType = opts.triggerType || (emergencyType === 'Manual SOS' ? 'Manual SOS' : 'Automatic SOS');
    const contactedParty = opts.contactedParty || CONTACTED_PARTIES[Math.floor(Math.random() * CONTACTED_PARTIES.length)];
    const riskLevel = opts.riskLevel || (emergencyType === 'Fall Detected' || emergencyType === 'Vital Anomaly' ? 'Critical' : emergencyType === 'Severe Fatigue' ? 'High' : 'Moderate');

    // Call timeline: detection → trigger → initiate → connect
    const detectionTime = new Date(now.getTime());
    const triggerTime   = new Date(now.getTime() + 1200);   // +1.2s
    const callInitTime  = new Date(now.getTime() + 3500);   // +3.5s
    const callConnTime  = new Date(now.getTime() + 8200);   // +8.2s

    return {
        eventId:         opts.eventId || _getNextEventId(),
        workerId:        opts.workerId || 'WRK-001',
        workerName:      opts.workerName || 'Worker 01',
        timestamp:       now.toISOString(),
        date:            opts.date || now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }),
        time:            opts.time || now.toLocaleTimeString('en-GB', { hour12: false }),
        emergencyType:   emergencyType,
        triggerType:     triggerType,
        contactedParty:  contactedParty,
        callStatus:      opts.callStatus || 'Connected',
        responseStatus:  opts.responseStatus || 'Acknowledged',
        riskLevel:       riskLevel,
        alertMessage:    opts.alertMessage || ALERT_MESSAGES[emergencyType] || 'Emergency alert triggered.',
        // Timeline timestamps
        detectionTime:   opts.detectionTime || detectionTime.toLocaleTimeString('en-GB', { hour12: false }),
        triggerTime:     opts.triggerTime   || triggerTime.toLocaleTimeString('en-GB', { hour12: false }),
        callInitTime:    opts.callInitTime  || callInitTime.toLocaleTimeString('en-GB', { hour12: false }),
        callConnTime:    opts.callConnTime  || callConnTime.toLocaleTimeString('en-GB', { hour12: false })
    };
}

// ---- Persistence helpers ----
function loadSOSEvents() {
    try {
        const raw = localStorage.getItem(SOS_STORAGE_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch { return []; }
}

function saveSOSEvents(events) {
    localStorage.setItem(SOS_STORAGE_KEY, JSON.stringify(events));
}

// ---- Public API ----

// Register a new SOS event (prepend to log, persist).
function registerSOSEvent(opts = {}) {
    const evt = createSOSEvent(opts);
    const events = loadSOSEvents();
    events.unshift(evt); // newest first
    saveSOSEvents(events);
    return evt;
}

// Get all SOS events (newest first).
function getAllSOSEvents() {
    return loadSOSEvents();
}

// Get SOS counter summary for today.
function getSOSSummary() {
    const events = loadSOSEvents();
    const todayStr = new Date().toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
    const todayEvents = events.filter(e => e.date === todayStr);
    return {
        totalToday:   todayEvents.length,
        callsMade:    todayEvents.length,
        acknowledged: todayEvents.filter(e => e.responseStatus === 'Acknowledged' || e.responseStatus === 'Responding' || e.responseStatus === 'Dispatched').length,
        pending:      todayEvents.filter(e => e.responseStatus === 'Pending' || e.callStatus === 'Pending').length
    };
}

// Clear all events (for testing).
function clearSOSEvents() {
    localStorage.removeItem(SOS_STORAGE_KEY);
    localStorage.removeItem('thermora_sos_counter');
}

// ---- Seed initial demo data (only once) ----
function seedDemoSOSEvents() {
    if (loadSOSEvents().length > 0) return; // already seeded

    // Reset counter to 1021 so first seeded event = EVT-1022
    localStorage.setItem('thermora_sos_counter', '1021');

    const demoEvents = [
        {
            workerId: 'WRK-008', workerName: 'Worker 08',
            date: '24 Sep 2026', time: '12:18:41',
            emergencyType: 'Vital Anomaly', triggerType: 'Automatic SOS',
            contactedParty: 'Emergency Services',
            callStatus: 'Completed', responseStatus: 'Dispatched',
            riskLevel: 'Critical',
            alertMessage: 'Abnormal vital signs detected — heart rate and SpO2 outside safe range. Medical evaluation needed.',
            detectionTime: '12:18:38', triggerTime: '12:18:39',
            callInitTime: '12:18:41', callConnTime: '12:18:49',
            timestamp: '2026-09-24T12:18:41'
        },
        {
            workerId: 'WRK-014', workerName: 'Worker 14',
            date: '24 Sep 2026', time: '13:47:05',
            emergencyType: 'Heat Stress', triggerType: 'Manual SOS',
            contactedParty: 'Caretaker',
            callStatus: 'Connected', responseStatus: 'Acknowledged',
            riskLevel: 'High',
            alertMessage: 'Core body temperature exceeds safe threshold. Worker showing signs of heat exhaustion.',
            detectionTime: '13:46:58', triggerTime: '13:47:02',
            callInitTime: '13:47:05', callConnTime: '13:47:12',
            timestamp: '2026-09-24T13:47:05'
        },
        {
            workerId: 'WRK-021', workerName: 'Worker 21',
            date: '24 Sep 2026', time: '14:32:18',
            emergencyType: 'Fall Detected', triggerType: 'Automatic SOS',
            contactedParty: 'Supervisor',
            callStatus: 'Connected', responseStatus: 'Responding',
            riskLevel: 'Critical',
            alertMessage: 'Fall impact detected by accelerometer. Worker may be injured. Immediate assistance required.',
            detectionTime: '14:32:15', triggerTime: '14:32:16',
            callInitTime: '14:32:18', callConnTime: '14:32:25',
            timestamp: '2026-09-24T14:32:18'
        }
    ];

    // Register in chronological order so newest ends up first
    demoEvents.forEach(d => registerSOSEvent(d));
}
