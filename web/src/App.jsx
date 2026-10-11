import { useState } from 'react'
import './App.css'

const INITIAL_FORM = {
  age_years: '',
  body_mass_kg: '',
  ambient_temp_c: '',
  humidity_pct: '',
  heart_rate_bpm: '',
  insulated_skin_temp_c: '',
  sex: 'Male',
  current_core_temp_c: '',
}

const FIELD_CONFIG = [
  {
    key: 'age_years',
    label: 'Age',
    unit: 'years',
    placeholder: 'e.g. 28',
    icon: '👤',
    min: 16,
    max: 80,
    step: 1,
  },
  {
    key: 'body_mass_kg',
    label: 'Body Mass',
    unit: 'kg',
    placeholder: 'e.g. 75',
    icon: '⚖️',
    min: 40,
    max: 200,
    step: 0.1,
  },
  {
    key: 'ambient_temp_c',
    label: 'Ambient Temperature',
    unit: '°C',
    placeholder: 'e.g. 35',
    icon: '🌡️',
    min: 10,
    max: 55,
    step: 0.1,
  },
  {
    key: 'humidity_pct',
    label: 'Humidity',
    unit: '%',
    placeholder: 'e.g. 60',
    icon: '💧',
    min: 0,
    max: 100,
    step: 1,
  },
  {
    key: 'heart_rate_bpm',
    label: 'Heart Rate',
    unit: 'bpm',
    placeholder: 'e.g. 120',
    icon: '❤️',
    min: 30,
    max: 220,
    step: 1,
  },
  {
    key: 'insulated_skin_temp_c',
    label: 'Skin Temperature',
    unit: '°C',
    placeholder: 'e.g. 36.5',
    icon: '🔥',
    min: 28,
    max: 42,
    step: 0.1,
  },
  {
    key: 'current_core_temp_c',
    label: 'Current Core Temperature',
    unit: '°C',
    placeholder: 'e.g. 37.2',
    icon: '🎯',
    min: 35,
    max: 42,
    step: 0.01,
  },
]

function getRiskGradient(level) {
  switch (level) {
    case 'Normal':
      return 'linear-gradient(135deg, #059669, #22c55e)'
    case 'Elevated':
      return 'linear-gradient(135deg, #d97706, #f59e0b)'
    case 'High':
      return 'linear-gradient(135deg, #ea580c, #f97316)'
    case 'Critical':
      return 'linear-gradient(135deg, #dc2626, #ef4444)'
    default:
      return 'linear-gradient(135deg, #6366f1, #8b5cf6)'
  }
}

function getDirectionArrow(direction) {
  switch (direction) {
    case 'Heating':
      return '↑'
    case 'Cooling':
      return '↓'
    case 'Stable':
      return '→'
    default:
      return ''
  }
}

function App() {
  const [form, setForm] = useState(INITIAL_FORM)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })

      const data = await response.json()

      if (!data.success) {
        throw new Error(data.error || 'Prediction failed')
      }

      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setForm(INITIAL_FORM)
    setResult(null)
    setError(null)
  }

  const isFormValid = FIELD_CONFIG.every(
    (f) => form[f.key] !== ''
  )

  return (
    <div className="app">
      {/* Background effects */}
      <div className="bg-glow bg-glow-1" />
      <div className="bg-glow bg-glow-2" />
      <div className="bg-glow bg-glow-3" />

      <header className="header">
        <div className="logo">
          <span className="logo-icon">🌡️</span>
          <div>
            <h1 className="logo-title">ThermoSentinel</h1>
            <p className="logo-subtitle">
              Core Temperature Prediction — LightGBM ΔT Model
            </p>
          </div>
        </div>
        <div className="badge">
          <span className="badge-dot" />
          MAE: 0.111 °C
        </div>
      </header>

      <main className="main">
        <form className="form-card" onSubmit={handleSubmit}>
          <div className="form-header">
            <h2>Input Parameters</h2>
            <p>
              Enter the subject's current readings to predict core temperature
              10 minutes into the future.
            </p>
          </div>

          <div className="fields-grid">
            {FIELD_CONFIG.map((field) => (
              <div className="field" key={field.key}>
                <label className="field-label" htmlFor={field.key}>
                  <span className="field-icon">{field.icon}</span>
                  {field.label}
                  <span className="field-unit">{field.unit}</span>
                </label>
                <input
                  id={field.key}
                  type="number"
                  className="field-input"
                  placeholder={field.placeholder}
                  value={form[field.key]}
                  onChange={(e) => handleChange(field.key, e.target.value)}
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  required
                />
              </div>
            ))}

            {/* Sex selector */}
            <div className="field">
              <label className="field-label" htmlFor="sex">
                <span className="field-icon">⚧</span>
                Sex
              </label>
              <select
                id="sex"
                className="field-input field-select"
                value={form.sex}
                onChange={(e) => handleChange('sex', e.target.value)}
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>
          </div>

          <div className="form-actions">
            <button
              type="submit"
              className="btn btn-primary"
              disabled={!isFormValid || loading}
            >
              {loading ? (
                <span className="spinner" />
              ) : (
                <>
                  <span>Predict</span>
                  <span className="btn-arrow">→</span>
                </>
              )}
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleReset}
            >
              Reset
            </button>
          </div>
        </form>

        {/* Error display */}
        {error && (
          <div className="error-card">
            <span className="error-icon">⚠️</span>
            <div>
              <strong>Prediction Error</strong>
              <p>{error}</p>
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="result-card fade-in">
            <div className="result-header">
              <h2>Prediction Result</h2>
              <div
                className="risk-badge"
                style={{ background: getRiskGradient(result.risk_level) }}
              >
                {result.risk_level}
              </div>
            </div>

            <div className="result-grid">
              {/* Main temperature gauge */}
              <div className="gauge-container">
                <div
                  className="gauge-ring"
                  style={{
                    background: getRiskGradient(result.risk_level),
                  }}
                >
                  <div className="gauge-inner">
                    <span className="gauge-value">
                      {result.predicted_future_temp.toFixed(2)}
                    </span>
                    <span className="gauge-unit">°C</span>
                    <span className="gauge-label">Predicted</span>
                  </div>
                </div>
              </div>

              {/* Detail cards */}
              <div className="detail-cards">
                <div className="detail-card">
                  <span className="detail-label">Current Core Temp</span>
                  <span className="detail-value">
                    {result.current_core_temp.toFixed(2)} °C
                  </span>
                </div>
                <div className="detail-card">
                  <span className="detail-label">Predicted Change (ΔT)</span>
                  <span
                    className="detail-value"
                    style={{
                      color:
                        result.predicted_delta > 0
                          ? '#f97316'
                          : result.predicted_delta < 0
                          ? '#3b82f6'
                          : '#a1a1aa',
                    }}
                  >
                    {result.predicted_delta > 0 ? '+' : ''}
                    {result.predicted_delta.toFixed(4)} °C
                  </span>
                </div>
                <div className="detail-card">
                  <span className="detail-label">Direction</span>
                  <span className="detail-value direction-value">
                    <span className="direction-arrow">
                      {getDirectionArrow(result.direction)}
                    </span>
                    {result.direction}
                  </span>
                </div>
                <div className="detail-card">
                  <span className="detail-label">Predicted Future Temp</span>
                  <span
                    className="detail-value"
                    style={{ color: result.risk_color }}
                  >
                    {result.predicted_future_temp.toFixed(2)} °C
                  </span>
                </div>
              </div>
            </div>

            {/* Risk scale */}
            <div className="risk-scale">
              <div className="risk-scale-bar">
                <div className="risk-zone risk-normal" />
                <div className="risk-zone risk-elevated" />
                <div className="risk-zone risk-high" />
                <div className="risk-zone risk-critical" />
                <div
                  className="risk-marker"
                  style={{
                    left: `${Math.min(
                      Math.max(
                        ((result.predicted_future_temp - 36) / (41 - 36)) * 100,
                        2
                      ),
                      98
                    )}%`,
                  }}
                />
              </div>
              <div className="risk-labels">
                <span>36 °C</span>
                <span>38 °C</span>
                <span>38.5 °C</span>
                <span>39 °C</span>
                <span>41 °C</span>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>
          ThermoSentinel — LightGBM ΔT Model · Predicts core body temperature
          10 minutes ahead
        </p>
      </footer>
    </div>
  )
}

export default App
