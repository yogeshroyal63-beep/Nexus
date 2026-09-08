import React, { useEffect, useState } from 'react'
import { listIncidents } from '../api/client'

const ACTION_ICON = {
  trigger_retrain: '🔁',
  rollback_model_version: '⏪',
  quarantine_data_source: '🔒',
  open_incident_ticket: '🎫',
  no_action: '✓',
}

const RISK_COLOR = {
  low: 'var(--signal-ok)',
  medium: 'var(--signal-warn)',
  high: 'var(--signal-critical)',
}

function timeAgo(dateStr) {
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

function IncidentRow({ record, expanded, onToggle }) {
  const plan = record.plan
  const outcome = record.outcome
  const verified = outcome?.verified
  const riskColor = plan ? RISK_COLOR[plan.risk_level] : 'var(--text-lo)'
  const verifiedIcon = verified === true ? '✓' : verified === false ? '✗' : '—'
  const verifiedColor = verified === true
    ? 'var(--signal-ok)'
    : verified === false
    ? 'var(--signal-critical)'
    : 'var(--text-lo)'

  return (
    <>
      <tr
        style={{ cursor: 'pointer', transition: 'background var(--t-fast)' }}
        onClick={onToggle}
      >
        <td>
          <span className="mono" style={{ color: 'var(--accent)', fontSize: '0.72rem' }}>
            {record.incident_id.slice(0, 16)}…
          </span>
        </td>
        <td style={{ color: 'var(--text-mid)', fontSize: '0.8rem' }}>
          {timeAgo(record.created_at)}
        </td>
        <td>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--text-mid)' }}>
            {record.model_urn.split(',')[1] || record.model_urn}
          </span>
        </td>
        <td>
          {plan ? (
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <span>{ACTION_ICON[plan.action_type] || '⚡'}</span>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-mid)' }}>
                {plan.action_type.replace(/_/g, ' ')}
              </span>
            </span>
          ) : (
            <span style={{ color: 'var(--text-lo)', fontSize: '0.78rem' }}>—</span>
          )}
        </td>
        <td>
          {plan && (
            <span className="badge" style={{ color: riskColor, borderColor: riskColor, background: `${riskColor}18` }}>
              {plan.risk_level}
            </span>
          )}
        </td>
        <td style={{ color: verifiedColor, fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}>
          {verifiedIcon}
          {outcome?.follow_up && (
            <span
              title={
                outcome.follow_up.executed
                  ? 'Automatic follow-up action succeeded'
                  : 'Automatic follow-up action did not complete — needs review'
              }
              style={{
                marginLeft: '0.3rem',
                color: outcome.follow_up.executed ? 'var(--signal-warn)' : 'var(--signal-critical)',
              }}
            >
              ↳
            </span>
          )}
        </td>
        <td style={{ color: 'var(--text-lo)', fontSize: '0.75rem' }}>
          {expanded ? '▲' : '▼'}
        </td>
      </tr>

      {expanded && (
        <tr>
          <td colSpan={7} style={{ padding: 0, background: 'var(--bg-elevated)' }}>
            <div style={{ padding: '1rem 1.2rem', borderTop: '1px solid var(--border)' }}>
              {record.report && (
                <div style={{ marginBottom: '0.8rem' }}>
                  <div style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-lo)', marginBottom: '0.3rem', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                    Summary
                  </div>
                  <p style={{ fontSize: '0.83rem', color: 'var(--text-mid)', lineHeight: 1.6 }}>
                    {record.report.summary}
                  </p>
                </div>
              )}
              {plan && (
                <div style={{ marginBottom: '0.6rem' }}>
                  <div style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-lo)', marginBottom: '0.3rem', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                    Rationale
                  </div>
                  <p style={{ fontSize: '0.83rem', color: 'var(--text-mid)', lineHeight: 1.6 }}>
                    {plan.rationale}
                  </p>
                </div>
              )}
              {outcome && (
                <div>
                  <div style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-lo)', marginBottom: '0.3rem', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                    Execution Detail
                  </div>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-lo)', fontFamily: 'var(--font-mono)' }}>
                    {outcome.execution_detail}
                  </p>
                </div>
              )}
              {outcome?.follow_up && (
                <div style={{ marginTop: '0.8rem', paddingTop: '0.8rem', borderTop: '1px dashed var(--border)' }}>
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.3rem',
                    fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--signal-warn)',
                    letterSpacing: '0.06em', textTransform: 'uppercase',
                  }}>
                    <span>↳</span>
                    <span>Automatic Follow-Up (verification failed)</span>
                  </div>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-lo)', fontFamily: 'var(--font-mono)' }}>
                    {outcome.follow_up.plan.action_type.replace(/_/g, ' ')}
                    {' — '}
                    {outcome.follow_up.executed ? '✓ succeeded' : '✗ did not complete'}
                    {': '}
                    {outcome.follow_up.execution_detail}
                  </p>
                </div>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

export default function IncidentHistoryView() {
  const [incidents, setIncidents] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [expanded, setExpanded] = useState(null)

  useEffect(() => {
    listIncidents(50)
      .then(setIncidents)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', padding: '1rem' }}>
        {[1, 2, 3].map(i => (
          <div key={i} className="skeleton" style={{ height: 48, borderRadius: 'var(--r-md)' }} />
        ))}
      </div>
    )
  }

  if (error) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">⚠</div>
        <p className="empty-title">Failed to load incidents</p>
        <p className="empty-sub mono" style={{ color: 'var(--signal-critical)' }}>{error}</p>
      </div>
    )
  }

  if (!incidents.length) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📋</div>
        <p className="empty-title">No incidents yet</p>
        <p className="empty-sub">Run Nexus to create your first incident record.</p>
      </div>
    )
  }

  return (
    <div className="animate-in">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', fontWeight: 700 }}>
            Incident History
          </h2>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-mid)', marginTop: '0.2rem' }}>
            {incidents.length} incident{incidents.length !== 1 ? 's' : ''} stored in memory
          </p>
        </div>
        <button
          className="btn btn-secondary"
          onClick={() => listIncidents(50).then(setIncidents).catch(e => setError(e.message))}
          style={{ fontSize: '0.78rem' }}
        >
          ↻ Refresh
        </button>
      </div>

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Incident ID</th>
              <th>Time</th>
              <th>Model</th>
              <th>Action</th>
              <th>Risk</th>
              <th>Verified</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {incidents.map(record => (
              <IncidentRow
                key={record.incident_id}
                record={record}
                expanded={expanded === record.incident_id}
                onToggle={() => setExpanded(e => e === record.incident_id ? null : record.incident_id)}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
