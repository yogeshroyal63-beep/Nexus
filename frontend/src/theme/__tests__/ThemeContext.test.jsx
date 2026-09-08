import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ThemeProvider, useTheme } from '../ThemeContext'

// THE GAP THIS PROVES IS CLOSED: ThemeProvider wraps the entire app,
// INCLUDING the app's only ErrorBoundary. The original code called
// localStorage.getItem and window.matchMedia with no guard -- either can
// legitimately throw (corporate storage policies, privacy-hardened
// browsers), and since this runs above the only error boundary in the
// tree, a throw here would white-screen the ENTIRE app with zero recovery
// path, not just fail to remember a theme preference.

function Probe() {
  const { theme, toggle } = useTheme()
  return (
    <div>
      <span data-testid="theme-value">{theme}</span>
      <button onClick={toggle}>toggle</button>
    </div>
  )
}

describe('ThemeContext resilience', () => {
  let originalGetItem, originalSetItem, originalMatchMedia

  beforeEach(() => {
    originalGetItem = Storage.prototype.getItem
    originalSetItem = Storage.prototype.setItem
    originalMatchMedia = window.matchMedia
  })

  afterEach(() => {
    Storage.prototype.getItem = originalGetItem
    Storage.prototype.setItem = originalSetItem
    window.matchMedia = originalMatchMedia
    vi.restoreAllMocks()
  })

  it('renders normally when localStorage and matchMedia both work', () => {
    render(<ThemeProvider><Probe /></ThemeProvider>)
    expect(screen.getByTestId('theme-value')).toBeInTheDocument()
  })

  it('does not crash the app when localStorage.getItem throws', () => {
    Storage.prototype.getItem = () => {
      throw new Error('SecurityError: storage disabled by policy')
    }
    expect(() =>
      render(<ThemeProvider><Probe /></ThemeProvider>)
    ).not.toThrow()
  })

  it('falls back to a valid theme when localStorage.getItem throws', () => {
    Storage.prototype.getItem = () => {
      throw new Error('blocked')
    }
    render(<ThemeProvider><Probe /></ThemeProvider>)
    const value = screen.getByTestId('theme-value').textContent
    expect(['dark', 'light']).toContain(value)
  })

  it('does not crash when window.matchMedia throws', () => {
    Storage.prototype.getItem = () => null
    window.matchMedia = () => {
      throw new Error('matchMedia unavailable')
    }
    expect(() =>
      render(<ThemeProvider><Probe /></ThemeProvider>)
    ).not.toThrow()
  })

  it('falls back to the default theme when both APIs throw', () => {
    Storage.prototype.getItem = () => {
      throw new Error('blocked')
    }
    window.matchMedia = () => {
      throw new Error('blocked')
    }
    render(<ThemeProvider><Probe /></ThemeProvider>)
    expect(screen.getByTestId('theme-value')).toHaveTextContent('dark')
  })

  it('does not crash when localStorage.setItem throws on toggle', () => {
    Storage.prototype.setItem = () => {
      throw new Error('QuotaExceededError')
    }
    render(<ThemeProvider><Probe /></ThemeProvider>)
    expect(() => fireEvent.click(screen.getByText('toggle'))).not.toThrow()
  })

  it('theme still applies for the session even when persistence fails', () => {
    Storage.prototype.setItem = () => {
      throw new Error('blocked')
    }
    render(<ThemeProvider><Probe /></ThemeProvider>)
    const initial = screen.getByTestId('theme-value').textContent
    fireEvent.click(screen.getByText('toggle'))
    const after = screen.getByTestId('theme-value').textContent
    expect(after).not.toBe(initial)
  })

  it('respects a valid stored preference when localStorage works normally', () => {
    Storage.prototype.getItem = () => 'light'
    render(<ThemeProvider><Probe /></ThemeProvider>)
    expect(screen.getByTestId('theme-value')).toHaveTextContent('light')
  })
})
