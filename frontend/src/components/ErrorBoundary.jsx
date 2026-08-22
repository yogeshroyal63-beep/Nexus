import React from 'react'

export class ErrorBoundary extends React.Component {
  state = { error: null }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('Nexus view error:', error, info)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="empty-state">
          <div className="empty-state-icon">⚠</div>
          <p className="empty-title">View failed to render</p>
          <p className="empty-sub mono" style={{ color: 'var(--signal-critical)' }}>
            {this.state.error.message}
          </p>
          <button
            className="btn btn-secondary"
            style={{ marginTop: '1rem' }}
            onClick={() => {
              this.setState({ error: null })
              this.props.onReset?.()
            }}
          >
            Reset view
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
