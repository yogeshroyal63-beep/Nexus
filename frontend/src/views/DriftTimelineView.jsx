import React from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts'
import { useTheme } from '../theme/ThemeContext.jsx'
import { CHART_COLORS } from '../theme/chartColors.js'
import './DriftTimelineView.css'

const SEVERITY_COLOR = {
  none: 'var(--signal-ok)',
  low: 'var(--signal-ok)',
  moderate: 'var(--signal-trace)',
  high: 'var(--signal-critical)',
  critical: 'var(--signal-critical)',
}

export default function DriftTimelineView({ result }) {
  const { theme } = useTheme()
  const c = CHART_COLORS[theme]

  if (!result) {
    return (
      <div className="empty-state mono">
        <p>No drift evidence yet.</p>
        <p className="empty-sub">Run the pipeline to see per-node statistical drift results.</p>
      </div>
    )
  }

  const { trace } = result
  const pred = trace.prediction_drift

  return (
    <div className="drift-view">
      <section className="drift-card">
        <header className="drift-card-header">
          <h3>Prediction output drift — {trace.model_urn.split(',')[1]}</h3>
          <span
            className="severity-badge mono"
            style={{ color: SEVERITY_COLOR[pred.severity], borderColor: SEVERITY_COLOR[pred.severity] }}
          >
            {pred.severity}
          </span>
        </header>
        <div className="drift-stats mono">
          <Stat label="method" value={pred.method} />
          <Stat label="KS statistic" value={pred.statistic.toFixed(4)} />
          <Stat label="p-value" value={pred.p_value < 0.0001 ? pred.p_value.toExponential(2) : pred.p_value.toFixed(4)} />
        </div>
      </section>

      <section className="drift-card">
        <header className="drift-card-header">
          <h3>Upstream candidates examined</h3>
          <span className="drift-card-sub mono">{trace.candidates_examined.length} node(s) tested</span>
        </header>

        <ResponsiveContainer width="100%" height={Math.max(220, trace.candidates_examined.length * 70)}>
          <BarChart
            layout="vertical"
            data={trace.candidates_examined.map((cand) => ({
              name: cand.node_name,
              'KS statistic': cand.drift_result.statistic,
              'Intervention Δ': cand.intervention_delta,
              genuine: cand.is_genuine_cause,
            }))}
            margin={{ left: 24, right: 24 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke={c.hairline} horizontal={false} />
            <XAxis type="number" stroke={c.textLo} fontSize={11} domain={[0, 1]} />
            <YAxis type="category" dataKey="name" stroke={c.textLo} fontSize={11} width={170} />
            <Tooltip
              contentStyle={{ background: c.bgPanelRaised, border: `1px solid ${c.hairline}`, fontSize: 12 }}
              labelStyle={{ color: theme === 'dark' ? '#e8ecf4' : '#141922' }}
              cursor={{ fill: c.hairline, opacity: 0.3 }}
            />
            <Bar dataKey="KS statistic" fill={c.textLo} radius={[0, 3, 3, 0]} />
            <Bar dataKey="Intervention Δ" fill={c.critical} radius={[0, 3, 3, 0]}>
              {trace.candidates_examined.map((cand, i) => (
                <Cell key={i} fill={cand.is_genuine_cause ? c.critical : c.trace} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <ChartLegend colors={c} />

        <p className="drift-note mono">
          Intervention Δ = how much of the downstream (model output) drift disappears when this
          node's contribution is counterfactually held at its baseline behavior. High raw drift
          with low Δ means the node co-drifted but isn't the cause — see confounded_with in the report.
        </p>
      </section>
    </div>
  )
}

// Recharts' auto-legend can only show one swatch per Bar, but the
// "Intervention Δ" bar is two-toned (red = genuine cause, amber = not) via
// per-Cell fills — a single swatch would misrepresent it. This spells out
// what each color actually means instead.
function ChartLegend({ colors }) {
  return (
    <div className="chart-legend mono">
      <span><i className="chart-legend-dot" style={{ background: colors.textLo }} />KS statistic</span>
      <span><i className="chart-legend-dot" style={{ background: colors.critical }} />Intervention Δ · genuine cause</span>
      <span><i className="chart-legend-dot" style={{ background: colors.trace }} />Intervention Δ · not genuine</span>
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  )
}
