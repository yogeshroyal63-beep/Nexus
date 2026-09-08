import React from 'react'
import './RootCauseReportView.css'

const CONFIDENCE_COLOR = {
  low: 'var(--signal-trace)',
  moderate: 'var(--signal-trace)',
  high: 'var(--signal-critical)',
}

export default function RootCauseReportView({ result }) {
  const report = result?.report

  if (!report) {
    return (
      <div className="empty-state mono">
        <p>No report generated.</p>
        <p className="empty-sub">
          A report appears once the causal engine isolates at least one genuine root cause.
          Run “Replay a failure” with drift injected to see one.
        </p>
      </div>
    )
  }

  return (
    <div className="report-view">
      <section className="report-card report-summary">
        <div className="report-summary-head">
          <h2>Diagnosis</h2>
          <span
            className="severity-badge mono"
            style={{ color: CONFIDENCE_COLOR[report.confidence], borderColor: CONFIDENCE_COLOR[report.confidence] }}
          >
            {report.confidence} confidence
          </span>
        </div>
        <p className="report-lede">{report.summary}</p>
        <p className="report-body">{report.detailed_explanation}</p>
      </section>

      <div className="report-grid">
        <section className="report-card">
          <h3>Isolated root cause(s)</h3>
          <ul className="root-cause-list">
            {report.root_causes.map((rc) => (
              <li key={rc} className="mono">{rc}</li>
            ))}
          </ul>
        </section>

        <section className="report-card">
          <h3>Suggested fixes</h3>
          <ul className="fix-list">
            {report.suggested_fixes.map((fix, i) => (
              <li key={i}>
                <div className="fix-action">{fix.action}</div>
                <div className="fix-target mono">{fix.target_urn}</div>
                <div className="fix-rationale">{fix.rationale}</div>
              </li>
            ))}
          </ul>
        </section>
      </div>

      <section className="report-card">
        <h3>Evidence trail</h3>
        <table className="evidence-table mono">
          <thead>
            <tr>
              <th>node</th>
              <th>hops</th>
              <th>method</th>
              <th>statistic</th>
              <th>intervention Δ</th>
              <th>genuine cause</th>
            </tr>
          </thead>
          <tbody>
            {report.raw_trace.candidates_examined.map((c) => (
              <tr key={c.node_urn} className={c.is_genuine_cause ? 'row-genuine' : ''}>
                <td>{c.node_name}</td>
                <td>{c.hops_from_model}</td>
                <td>{c.drift_result.method}</td>
                <td>{c.drift_result.statistic.toFixed(4)}</td>
                <td>{c.intervention_delta.toFixed(4)}</td>
                <td>{c.is_genuine_cause ? '✓' : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}
