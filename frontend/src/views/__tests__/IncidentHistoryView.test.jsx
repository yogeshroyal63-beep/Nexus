import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import IncidentHistoryView from '../IncidentHistoryView'
import * as apiClient from '../../api/client'

// THE GAP THIS PROVES IS CLOSED: outcome.follow_up (added when fixing the
// "verify has no consequence" backend gap) was displayed in the live
// AgentActionView but completely absent from this persistent history view.
// Anyone reviewing history later would see a failed action's original
// detail with zero indication that Nexus automatically rolled it back or
// escalated it -- undermining the entire point of the backend fix.

vi.mock('../../api/client', () => ({
  listIncidents: vi.fn(),
}))

function makeIncident({ withFollowUp = false, followUpSucceeded = true } = {}) {
  const base = {
    incident_id: 'nxs-abc123def456',
    model_urn: 'urn:li:mlModel:(demo,fraud_model_v3,PROD)',
    created_at: new Date().toISOString(),
    report: { summary: 'Test summary' },
    plan: {
      action_type: 'trigger_retrain',
      risk_level: 'medium',
      rationale: 'Test rationale',
    },
    outcome: {
      executed: true,
      execution_detail: 'Original action detail',
      verified: false,
    },
  }
  if (withFollowUp) {
    base.outcome.follow_up = {
      executed: followUpSucceeded,
      execution_detail: followUpSucceeded ? 'Rolled back successfully' : 'Rollback failed',
      plan: { action_type: 'rollback_model_version' },
    }
  }
  return base
}

describe('IncidentHistoryView follow_up display', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows the follow-up indicator icon when follow_up exists', async () => {
    apiClient.listIncidents.mockResolvedValue([makeIncident({ withFollowUp: true })])
    render(<IncidentHistoryView />)
    await waitFor(() => screen.getByText('↳'))
    expect(screen.getByText('↳')).toBeInTheDocument()
  })

  it('does not show the follow-up indicator when there is no follow_up', async () => {
    apiClient.listIncidents.mockResolvedValue([makeIncident({ withFollowUp: false })])
    render(<IncidentHistoryView />)
    await waitFor(() => screen.getByText(/nxs-abc123def456/))
    expect(screen.queryByText('↳')).not.toBeInTheDocument()
  })

  it('reveals the full follow-up detail when the row is expanded', async () => {
    apiClient.listIncidents.mockResolvedValue([makeIncident({ withFollowUp: true, followUpSucceeded: true })])
    render(<IncidentHistoryView />)
    const idCell = await screen.findByText(/nxs-abc123def456/)
    fireEvent.click(idCell.closest('tr'))
    expect(await screen.findByText(/Automatic Follow-Up/i)).toBeInTheDocument()
    expect(screen.getByText(/rollback model version/i)).toBeInTheDocument()
    expect(screen.getByText(/Rolled back successfully/i)).toBeInTheDocument()
  })

  it('distinguishes a failed follow-up from a succeeded one in the expanded detail', async () => {
    apiClient.listIncidents.mockResolvedValue([makeIncident({ withFollowUp: true, followUpSucceeded: false })])
    render(<IncidentHistoryView />)
    const idCell = await screen.findByText(/nxs-abc123def456/)
    fireEvent.click(idCell.closest('tr'))
    expect(await screen.findByText(/did not complete/i)).toBeInTheDocument()
  })

  it('shows the original execution detail even when a follow-up also exists', async () => {
    apiClient.listIncidents.mockResolvedValue([makeIncident({ withFollowUp: true })])
    render(<IncidentHistoryView />)
    const idCell = await screen.findByText(/nxs-abc123def456/)
    fireEvent.click(idCell.closest('tr'))
    expect(await screen.findByText('Original action detail')).toBeInTheDocument()
  })

  it('renders normally for an incident with no outcome at all', async () => {
    const incident = makeIncident()
    delete incident.outcome
    apiClient.listIncidents.mockResolvedValue([incident])
    render(<IncidentHistoryView />)
    expect(await screen.findByText(/nxs-abc123def456/)).toBeInTheDocument()
  })
})
