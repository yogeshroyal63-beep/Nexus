import React from 'react'

export default function StatusFooter({ result, stage }) {
  const incidentId = result?.incident_id
  const escalated = result?.escalated

  return (
    <footer className="status-footer">
      <span className="status-dot" />
      <span>Nexus · Autonomous Developer Agent</span>
      <span style={{ color: 'var(--border)' }}>·</span>
      <span>AWS Strands SDK · Amazon Bedrock</span>
      {incidentId && (
        <>
          <span style={{ color: 'var(--border)' }}>·</span>
          <span className="mono" style={{ color: 'var(--accent)' }}>{incidentId}</span>
        </>
      )}
      {escalated && (
        <>
          <span style={{ color: 'var(--border)' }}>·</span>
          <span style={{ color: 'var(--signal-warn)' }}>⚠ awaiting approval</span>
        </>
      )}
      <span style={{ flex: 1 }} />
      <span style={{ color: 'var(--text-lo)' }}>
        {stage === 'idle' ? 'Idle' : stage === 'done' ? 'Run complete' : 'Processing…'}
      </span>
    </footer>
  )
}
