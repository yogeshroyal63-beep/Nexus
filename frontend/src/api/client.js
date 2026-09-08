const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const getLineage = (modelUrn) =>
  request(`/api/lineage/${encodeURIComponent(modelUrn)}`)

export const investigate = ({ modelUrn, injectDrift = true }) =>
  request(`/api/investigate?model_urn=${encodeURIComponent(modelUrn)}&inject_drift=${injectDrift}`, { method: 'POST' })

export const runAgent = ({ modelUrn, injectDrift = true, writeBack = true }) =>
  request(`/api/run?model_urn=${encodeURIComponent(modelUrn)}&inject_drift=${injectDrift}&write_back=${writeBack}`, { method: 'POST' })

export const listIncidents = (limit = 50) =>
  request(`/api/incidents?limit=${limit}`)

export const getIncident = (id) =>
  request(`/api/incidents/${encodeURIComponent(id)}`)

export const approveIncident = (id) =>
  request(`/api/incidents/${encodeURIComponent(id)}/approve`, { method: 'POST' })
