import React, { useState } from 'react'
import { approveIncident } from '../api/client'

const RISK_COLOR = {
  low: 'var(--signal-ok)',
  medium: 'var(--signal-warn)',
  high: 'var(--signal-critical)',
}

const RISK_CARD = {
  low: 'card-ok',
  medium: 'card-warn',
  high: 'card-critical',
}

const ACTION_META = {
  trigger_retrain: { label: 'Trigger Retrain', icon: '🔁', desc: 'Pipeline re-trains the model on fresh data.' },
  rollback_model_version: { label: 'Rollback Version', icon: '⏪', desc: 'Reverts to last known-good model version.' },
  quarantine_data_source: { label: 'Quarantine Data', icon: '🔒', desc: 'Isolates the corrupted upstream dataset.' },
  open_incident_ticket: { label: 'Open Ticket', icon: '🎫', desc: 'Creates a GitHub issue for human review.' },
  no_action: { label: 'No Action', icon: '✓', desc: 'Confidence or evidence too low to act.' },
}

function ConfidenceBar({ value }) {
  const pct = Math.round(value * 100)
  const color = value >= 0.75
    ? 'var(--signal-ok)'
    : value >= 0.4
    ? 'var(--signal-warn)'
    : 'var(--signal-critical)'
  return (
    <div className="confidence-bar">
      <div className="confidence-track">
        <div
          className="confidence-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="confidence-label mono" style={{ color }}>{pct}%</span>
    </div>
  )
}

function StepIndicator({ steps, current }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 0, marginBottom: '1.5rem' }}>
      {steps.map((step, i) => {
        const done = i < current
        const active = i === current
        return (
          <React.Fragment key={step}>
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '4px',
              flex: i < steps.length - 1 ? 0 : undefined,
            }}>
              <div style={{
                width: 28,
                height: 28,
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '0.7rem',
                fontFamily: 'var(--font-mono)',
                fontWeight: 600,
                background: done
                  ? 'var(--signal-ok-glow)'
                  : active
                  ? 'var(--accent-glow)'
                  : 'var(--bg-elevated)',
                border: `2px solid ${done ? 'var(--signal-ok)' : active ? 'var(--accent)' : 'var(--border)'}`,
                color: done ? 'var(--signal-ok)' : active ? 'var(--accent)' : 'var(--text-lo)',
                transition: 'all 0.3s',
              }}>
                {done ? '✓' : i + 1}
              </div>
              <span style={{
                fontSize: '0.6rem',
                fontFamily: 'var(--font-mono)',
                color: active ? 'var(--accent)' : done ? 'var(--signal-ok)' : 'var(--text-lo)',
                whiteSpace: 'nowrap',
              }}>{step}</span>
            </div>
            {i < steps.length - 1 && (
              <div style={{
                flex: 1,
                height: 2,
                background: done ? 'var(--signal-ok)' : 'var(--border)',
                marginBottom: '18px',
                transition: 'background 0.3s',
                minWidth: 20,
              }} />
            )}
          </React.Fragment>
        )
      })}
    </div>
  )
}

function PlanCard({ plan }) {
  if (!plan) return null
  const meta = ACTION_META[plan.action_type] || { label: plan.action_type, icon: '⚡', desc: '' }
  const riskColor = RISK_COLOR[plan.risk_level]
  const cardClass = RISK_CARD[plan.risk_level] || ''

  return (
    <div className={`card ${cardClass} animate-in`} style={{ marginBottom: '1rem' }}>
      <div className="section-header">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.3rem' }}>
            <span style={{ fontSize: '1.2rem' }}>{meta.icon}</span>
            <span className="section-title">{meta.label}</span>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-mid)' }}>{meta.desc}</p>
        </div>
        <span className="badge" style={{ color: riskColor, borderColor: riskColor, background: `${riskColor}18` }}>
          {plan.risk_level} risk
        </span>
      </div>

      <div style={{
        background: 'var(--bg-elevated)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--r-md)',
        padding: '0.9rem 1rem',
        marginBottom: '1rem',
        fontStyle: 'italic',
        color: 'var(--text-mid)',
        fontSize: '0.87rem',
        lineHeight: 1.6,
      }}>
        "{plan.rationale}"
      </div>

      <div style={{ marginBottom: '0.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-lo)', fontFamily: 'var(--font-mono)' }}>
            STRANDS CONFIDENCE
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-lo)', fontFamily: 'var(--font-mono)' }}>
            threshold: {(0.75 * 100).toFixed(0)}%
          </span>
        </div>
        <ConfidenceBar value={plan.confidence} />
      </div>

      <div className="sep" />

      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
        <span className="badge badge-neutral">
          <span style={{ opacity: 0.6 }}>target</span>&nbsp;
          {plan.target_urn.split(',')[1] || plan.target_urn}
        </span>
        <span className="badge badge-neutral">
          <span style={{ opacity: 0.6 }}>auto-exec</span>&nbsp;
          {plan.requires_human_approval ? 'blocked' : 'approved'}
        </span>
      </div>
    </div>
  )
}

function OutcomeCard({ outcome }) {
  if (!outcome) return null
  const verified = outcome.verified
  const color = verified === true
    ? 'var(--signal-ok)'
    : verified === false
    ? 'var(--signal-critical)'
    : 'var(--text-lo)'
  const label = verified === true
    ? '✓ Verified — issue resolved'
    : verified === false
    ? '✗ Still drifting — action inconclusive'
    : '⟳ Not yet verified'

  return (
    <div className={`card animate-in ${verified === true ? 'card-ok' : verified === false ? 'card-critical' : ''}`}
      style={{ marginBottom: '1rem' }}>
      <div className="section-header">
        <span className="section-title">Execution Result</span>
        <span style={{ fontSize: '0.82rem', color, fontFamily: 'var(--font-mono)' }}>{label}</span>
      </div>
      <p style={{ fontSize: '0.85rem', color: 'var(--text-mid)', lineHeight: 1.7 }}>
        {outcome.execution_detail}
      </p>
      {outcome.verification_detail && (
        <p style={{
          marginTop: '0.6rem',
          fontSize: '0.82rem',
          color: 'var(--text-lo)',
          fontFamily: 'var(--font-mono)',
          padding: '0.6rem',
          background: 'var(--bg-elevated)',
          borderRadius: 'var(--r-sm)',
        }}>
          {outcome.verification_detail}
        </p>
      )}
      {outcome.reversible && outcome.rollback_reference && (
        <div style={{ marginTop: '0.8rem' }}>
          <span className="badge badge-neutral">rollback ref: {outcome.rollback_reference}</span>
        </div>
      )}
    </div>
  )
}

function EscalationCard({ result, onApproved }) {
  const [approving, setApproving] = useState(false)
  const [error, setError] = useState(null)
  const [approved, setApproved] = useState(null)

  if (!result?.escalated) return null

  const handleApprove = async () => {
    setApproving(true)
    setError(null)
    try {
      const updated = await approveIncident(result.incident_id)
      setApproved(updated)
      onApproved?.(updated)
    } catch (e) {
      setError(e.message || String(e))
    } finally {
      setApproving(false)
    }
  }

  return (
    <div className="card card-warn animate-in" style={{ marginBottom: '1rem' }}>
      <div className="section-header">
        <div>
          <span className="section-title">⚠ Human Approval Required</span>
          <p className="section-sub">
            Confidence ({Math.round((result.plan?.confidence ?? 0) * 100)}%) or risk level (
            {result.plan?.risk_level}) blocked auto-execution.
          </p>
        </div>
      </div>

      {!approved ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-mid)' }}>
            Review the plan above. If you approve, Nexus will execute the action and verify the result.
          </p>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              className="btn btn-primary"
              onClick={handleApprove}
              disabled={approving}
            >
              {approving ? '⟳ Approving…' : '✓ Approve & Execute'}
            </button>
          </div>
          {error && (
            <div className="error-banner" style={{ borderRadius: 'var(--r-sm)', marginTop: 0 }}>
              {error}
            </div>
          )}
        </div>
      ) : (
        <div style={{
          padding: '0.8rem',
          background: 'var(--signal-ok-glow)',
          border: '1px solid rgba(104, 211, 145, 0.3)',
          borderRadius: 'var(--r-md)',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.8rem',
          color: 'var(--signal-ok)',
        }}>
          ✓ Approved and executed — {approved.outcome?.execution_detail}
        </div>
      )}
    </div>
  )
}

function MemoryCard({ similarCount }) {
  if (!similarCount) return null
  return (
    <div className="card animate-in" style={{ marginBottom: '1rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
        <div style={{
          width: 36,
          height: 36,
          borderRadius: 'var(--r-sm)',
          background: 'var(--accent-2-glow)',
          border: '1px solid rgba(183, 148, 244, 0.2)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '1rem',
          flexShrink: 0,
        }}>
          🧠
        </div>
        <div>
          <div style={{ fontSize: '0.87rem', fontWeight: 600, color: 'var(--text-hi)' }}>
            Informed by Memory
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-mid)' }}>
            Strands considered {similarCount} past incident{similarCount !== 1 ? 's' : ''} when planning this action.
          </div>
        </div>
      </div>
    </div>
  )
}

const LOOP_STEPS = ['Detect', 'Diagnose', 'Plan', 'Execute', 'Verify', 'Remember']

export default function AgentActionView({ result }) {
  const [liveResult, setLiveResult] = useState(null)
  const active = liveResult || result

  if (!active?.plan) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">⚡</div>
        <p className="empty-title">No agent decision yet</p>
        <p className="empty-sub">
          Click "Run Nexus" to trigger the full autonomous loop: Strands SDK + Bedrock
          detects drift, plans a remediation action, executes it, and verifies the result.
        </p>
      </div>
    )
  }

  const currentStep = active.outcome
    ? 5
    : active.escalated
    ? 3
    : active.plan
    ? 3
    : 2

  return (
    <div style={{ maxWidth: 760, margin: '0 auto' }}>
      <StepIndicator steps={LOOP_STEPS} current={currentStep} />

      <div className="stagger">
        <MemoryCard similarCount={active.similar_past_incidents?.length} />
        <PlanCard plan={active.plan} />
        <EscalationCard result={active} onApproved={setLiveResult} />
        <OutcomeCard outcome={active.outcome} />
      </div>
    </div>
  )
}
