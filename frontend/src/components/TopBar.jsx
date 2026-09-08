import React from 'react'
import { useTheme } from '../theme/ThemeContext'

const STAGE_LABELS = {
  idle: 'Ready',
  ingesting: 'Ingesting lineage',
  detecting: 'Detecting drift',
  tracing: 'Tracing causality',
  reasoning: 'LLM reasoning',
  planning: 'Strands planning',
  acting: 'Executing action',
  writing_back: 'Writing back',
  done: 'Complete',
}

function SunIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
      <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
    </svg>
  )
}

function PlayIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
      <polygon points="5,3 19,12 5,21"/>
    </svg>
  )
}

function BoltIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
      <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
    </svg>
  )
}

export default function TopBar({ stage, onReplay, onReplayAgentic }) {
  const { theme, toggle } = useTheme()
  const isRunning = stage !== 'idle' && stage !== 'done'
  const isDone = stage === 'done'

  return (
    <header className="topbar">
      <div className="topbar-brand">
        <div className="topbar-logo">N</div>
        <span className="topbar-name">Nexus</span>
        <span className="topbar-tag">v1.0 · AWS Strands</span>
      </div>

      <div className="topbar-spacer" />

      <div className="topbar-actions">
        <div className={`stage-pill ${isRunning ? 'active' : isDone ? 'done' : ''}`}>
          <span className="stage-dot" />
          {STAGE_LABELS[stage] || stage}
        </div>

        <button
          className="btn btn-secondary"
          onClick={() => onReplay(true)}
          disabled={isRunning}
          title="Run pipeline without autonomous action"
        >
          <PlayIcon />
          Report only
        </button>

        <button
          className="btn btn-primary"
          onClick={() => onReplayAgentic(true)}
          disabled={isRunning}
          title="Run full Nexus agentic loop with Strands+Bedrock"
        >
          <BoltIcon />
          {isRunning ? 'Running…' : 'Run Nexus'}
        </button>

        <button className="btn-icon" onClick={toggle} title="Toggle theme">
          {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
        </button>
      </div>
    </header>
  )
}
