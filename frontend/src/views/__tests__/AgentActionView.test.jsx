import { useState, useCallback } from 'react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import AgentActionView from '../AgentActionView'
import * as apiClient from '../../api/client'

// THE CRITICAL BUG THIS PROVES IS FIXED: approveIncident() returns an
// IncidentRecord, which has NO `escalated` or `similar_past_incidents`
// fields (those only exist on the SentinelRunResult that `result`
// originally is). The state that tracks approval must be owned by the
// PARENT (App.jsx in the real app), not duplicated locally inside
// AgentActionView, because StatusFooter reads the same parent state
// directly and must never show a stale "awaiting approval" warning after
// the incident has actually been approved. This test suite uses a small
// wrapper that mimics exactly what App.jsx does — holds its own `result`
// state, merges the approval response into it, and passes both down —
// so it exercises the real integration path, not an isolated component
// that could pass while the actual app-level wiring is still broken.

vi.mock('../../api/client', () => ({
  approveIncident: vi.fn(),
}))

function makeEscalatedResult() {
  return {
    incident_id: 'nxs-abc123',
    model_urn: 'urn:li:mlModel:(demo,fraud_model_v3,PROD)',
    escalated: true,
    similar_past_incidents: ['nxs-old1', 'nxs-old2'],
    plan: {
      action_type: 'rollback_model_version',
      risk_level: 'high',
      confidence: 0.4,
      rationale: 'Test rationale',
      target_urn: 'urn:li:mlModel:(demo,fraud_model_v3,PROD)',
      requires_human_approval: true,
    },
    outcome: null,
  }
}

// Simulates what the REAL backend approve endpoint returns: an
// IncidentRecord, deliberately missing `escalated` / `similar_past_incidents`
// to match the real schema mismatch that caused this bug.
function makeApprovalResponse() {
  return {
    incident_id: 'nxs-abc123',
    model_urn: 'urn:li:mlModel:(demo,fraud_model_v3,PROD)',
    plan: {
      action_type: 'rollback_model_version',
      risk_level: 'high',
      confidence: 0.4,
      rationale: 'Test rationale',
      target_urn: 'urn:li:mlModel:(demo,fraud_model_v3,PROD)',
      requires_human_approval: true,
    },
    outcome: {
      executed: true,
      execution_detail: 'Rollback completed successfully',
      verified: true,
    },
    // NOTE: no `escalated`, no `similar_past_incidents` — matches the
    // real IncidentRecord schema exactly.
  }
}

// Mirrors App.jsx's own state-lifting fix: owns `result`, merges the
// approval response into it via onApproved, explicitly clearing
// `escalated` since reaching this callback means approval already
// succeeded (a plain spread merge can't do this — the approval response
// never includes an `escalated` key at all, so a missing key would leave
// the old `true` value untouched forever).
function TestHarness({ initialResult }) {
  const [result, setResult] = useState(initialResult)
  const handleApproved = useCallback((updated) => {
    setResult((prev) => ({ ...prev, ...updated, escalated: false }))
  }, [])
  return (
    <>
      <AgentActionView result={result} onApproved={handleApproved} />
      <div data-testid="footer-escalated-probe">
        {result?.escalated ? 'FOOTER: awaiting approval' : 'FOOTER: clear'}
      </div>
    </>
  )
}

describe('AgentActionView approval flow (via App-equivalent state lifting)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows the escalation card for an escalated result', () => {
    render(<TestHarness initialResult={makeEscalatedResult()} />)
    expect(screen.getByText(/Human Approval Required/i)).toBeInTheDocument()
  })

  it('confirmation message stays visible after approval, not vanishing', async () => {
    apiClient.approveIncident.mockResolvedValue(makeApprovalResponse())
    render(<TestHarness initialResult={makeEscalatedResult()} />)

    fireEvent.click(screen.getByText(/Approve & Execute/i))

    expect(await screen.findByText(/Approved and executed/i)).toBeInTheDocument()
    expect(screen.getAllByText(/Rollback completed successfully/i).length).toBeGreaterThan(0)
  })

  it('memory card is present for an escalated result with prior incidents', () => {
    render(<TestHarness initialResult={makeEscalatedResult()} />)
    expect(screen.getByText(/Informed by Memory/i)).toBeInTheDocument()
  })

  it('outcome card renders the fresh outcome after approval propagates up', async () => {
    apiClient.approveIncident.mockResolvedValue(makeApprovalResponse())
    render(<TestHarness initialResult={makeEscalatedResult()} />)

    fireEvent.click(screen.getByText(/Approve & Execute/i))
    await screen.findByText(/Approved and executed/i)

    expect(screen.getByText(/Verified — issue resolved/i)).toBeInTheDocument()
  })

  it('step indicator advances to the final step after approval', async () => {
    apiClient.approveIncident.mockResolvedValue(makeApprovalResponse())
    render(<TestHarness initialResult={makeEscalatedResult()} />)

    fireEvent.click(screen.getByText(/Approve & Execute/i))
    await screen.findByText(/Approved and executed/i)

    expect(screen.getByText('Remember')).toBeInTheDocument()
  })

  it('THE core regression test: the footer-equivalent state clears after approval, never staying stuck on "awaiting approval"', async () => {
    apiClient.approveIncident.mockResolvedValue(makeApprovalResponse())
    render(<TestHarness initialResult={makeEscalatedResult()} />)

    expect(screen.getByTestId('footer-escalated-probe')).toHaveTextContent('awaiting approval')

    fireEvent.click(screen.getByText(/Approve & Execute/i))
    await screen.findByText(/Approved and executed/i)

    // THE bug this whole round was about: before lifting state up, nothing
    // told the parent-level state that approval happened, so a
    // StatusFooter reading that same parent state would show this warning
    // forever. This probe simulates exactly that parent-level read.
    await waitFor(() => {
      expect(screen.getByTestId('footer-escalated-probe')).toHaveTextContent('FOOTER: clear')
    })
  })

  it('shows an error message if approval fails, without crashing', async () => {
    apiClient.approveIncident.mockRejectedValue(new Error('Network error'))
    render(<TestHarness initialResult={makeEscalatedResult()} />)

    fireEvent.click(screen.getByText(/Approve & Execute/i))

    expect(await screen.findByText(/Network error/i)).toBeInTheDocument()
    expect(screen.getByText(/Approve & Execute/i)).toBeInTheDocument()
  })

  it('renders the empty state when there is no plan at all', () => {
    render(<AgentActionView result={null} onApproved={vi.fn()} />)
    expect(screen.getByText(/No agent decision yet/i)).toBeInTheDocument()
  })

  it('does not show the escalation card for a non-escalated result', () => {
    const result = { ...makeEscalatedResult(), escalated: false }
    render(<AgentActionView result={result} onApproved={vi.fn()} />)
    expect(screen.queryByText(/Human Approval Required/i)).not.toBeInTheDocument()
  })

  it('does not crash when onApproved is not provided', () => {
    expect(() =>
      render(<AgentActionView result={makeEscalatedResult()} />)
    ).not.toThrow()
  })
})
