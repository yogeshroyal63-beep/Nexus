import React, { Suspense, lazy, useCallback, useState } from 'react'
import { investigate, runAgent } from './api/client'
import TopBar from './components/TopBar'
import StatusFooter from './components/StatusFooter'
import { ErrorBoundary } from './components/ErrorBoundary'
import { ThemeProvider } from './theme/ThemeContext'
import './index.css'

const LineageGraphView = lazy(() => import('./views/LineageGraphView'))
const DriftTimelineView = lazy(() => import('./views/DriftTimelineView'))
const RootCauseReportView = lazy(() => import('./views/RootCauseReportView'))
const AgentActionView = lazy(() => import('./views/AgentActionView'))
const IncidentHistoryView = lazy(() => import('./views/IncidentHistoryView'))

const MODEL_URN = 'urn:li:mlModel:(demo,fraud_model_v3,PROD)'

const TABS = [
  { id: 'graph', label: 'Lineage Graph', num: '01' },
  { id: 'timeline', label: 'Drift Timeline', num: '02' },
  { id: 'report', label: 'Root-Cause', num: '03', requiresResult: true, requiresReport: true },
  { id: 'agent', label: 'Agent Decision', num: '04', requiresAgentic: true },
  { id: 'history', label: 'Incident History', num: '05' },
]

function ViewLoader() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', padding: '1rem' }}>
      {[1, 2, 3].map(i => (
        <div key={i} className="skeleton" style={{ height: 80, borderRadius: 'var(--r-md)' }} />
      ))}
    </div>
  )
}

export default function App() {
  const [result, setResult] = useState(null)
  const [stage, setStage] = useState('idle')
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('graph')
  const [mode, setMode] = useState('idle')

  const runReport = useCallback(async (injectDrift) => {
    setError(null)
    setResult(null)
    setMode('report')
    setStage('ingesting')
    try {
      const s1 = setTimeout(() => setStage('detecting'), 400)
      const s2 = setTimeout(() => setStage('tracing'), 900)
      const s3 = setTimeout(() => setStage('reasoning'), 1500)
      const data = await investigate({ modelUrn: MODEL_URN, injectDrift })
      clearTimeout(s1); clearTimeout(s2); clearTimeout(s3)
      setStage('writing_back')
      await new Promise(r => setTimeout(r, 350))
      setResult(data)
      setStage('done')
      setActiveTab(data.report ? 'report' : 'graph')
    } catch (e) {
      setError(e.message || String(e))
      setStage('idle')
    }
  }, [])

  const runAgentic = useCallback(async (injectDrift) => {
    setError(null)
    setResult(null)
    setMode('agentic')
    setStage('ingesting')
    try {
      const s1 = setTimeout(() => setStage('detecting'), 400)
      const s2 = setTimeout(() => setStage('tracing'), 900)
      const s3 = setTimeout(() => setStage('reasoning'), 1400)
      const s4 = setTimeout(() => setStage('planning'), 1900)
      const s5 = setTimeout(() => setStage('acting'), 2500)
      const data = await runAgent({ modelUrn: MODEL_URN, injectDrift })
      ;[s1, s2, s3, s4, s5].forEach(clearTimeout)
      setStage('writing_back')
      await new Promise(r => setTimeout(r, 350))
      setResult(data)
      setStage('done')
      setActiveTab('agent')
    } catch (e) {
      setError(e.message || String(e))
      setStage('idle')
    }
  }, [])

  const isTabDisabled = (tab) => {
    if (tab.requiresReport && !result?.report) return true
    if (tab.requiresAgentic && (mode !== 'agentic' || !result?.plan)) return true
    return false
  }

  // FIXED — a real bug found via testing: AgentActionView previously kept
  // its OWN local copy of the post-approval result (liveResult), completely
  // disconnected from this component's `result` state. StatusFooter reads
  // `result` directly and is visible across every tab, so after a human
  // approved an escalated plan, the footer's "⚠ awaiting approval" warning
  // would keep showing FOREVER — nothing ever told App.jsx's own state that
  // the approval happened. Lifting the merge up here means both
  // AgentActionView and StatusFooter now read from the SAME updated state,
  // so there's only one source of truth instead of two that can drift apart.
  //
  // Merge (not replace) because approveIncident() returns an IncidentRecord,
  // which lacks the `escalated` / `similar_past_incidents` fields that only
  // exist on the SentinelRunResult `result` already holds — a naive replace
  // would silently drop them.
  //
  // Explicitly setting `escalated: false` is NOT optional here: object
  // spread only overwrites keys that EXIST on the source object. Since the
  // approval response never includes an `escalated` key at all, a plain
  // `{...prev, ...updated}` merge would leave `escalated: true` untouched
  // forever, since a missing key never resets an existing one. Reaching
  // this callback at all means the approve endpoint already succeeded, so
  // "still needs approval" is definitionally false from this point on.
  const handleIncidentApproved = useCallback((updated) => {
    setResult((prev) => ({ ...prev, ...updated, escalated: false }))
  }, [])

  return (
    <ThemeProvider>
      <div className="app-shell">
        <TopBar stage={stage} onReplay={runReport} onReplayAgentic={runAgentic} />

        {error && (
          <div className="error-banner">
            <span>⚠</span>
            <strong>Pipeline failed.</strong>
            <span>{error}</span>
          </div>
        )}

        <nav className="tab-bar">
          {TABS.map(tab => (
            <button
              key={tab.id}
              className={`tab${activeTab === tab.id ? ' tab-active' : ''}`}
              disabled={isTabDisabled(tab)}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="tab-num">{tab.num}</span>
              {tab.label}
            </button>
          ))}
        </nav>

        <main className="main-panel">
          <ErrorBoundary
            key={activeTab}
            onReset={() => { setResult(null); setError(null); setStage('idle'); setActiveTab('graph') }}
          >
            <Suspense fallback={<ViewLoader />}>
              {activeTab === 'graph' && <LineageGraphView result={result} stage={stage} />}
              {activeTab === 'timeline' && <DriftTimelineView result={result} />}
              {activeTab === 'report' && <RootCauseReportView result={result} />}
              {activeTab === 'agent' && <AgentActionView result={result} onApproved={handleIncidentApproved} />}
              {activeTab === 'history' && <IncidentHistoryView />}
            </Suspense>
          </ErrorBoundary>
        </main>

        <StatusFooter result={result} stage={stage} />
      </div>
    </ThemeProvider>
  )
}
