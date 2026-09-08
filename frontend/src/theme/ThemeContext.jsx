import React, { createContext, useContext, useEffect, useState } from 'react'

const ThemeContext = createContext({ theme: 'dark', toggle: () => {} })

const DEFAULT_THEME = 'dark'

// FIXED — a real single-point-of-failure found via review: ThemeProvider
// wraps the ENTIRE app, including the only ErrorBoundary in the tree (see
// App.jsx — <ThemeProvider><ErrorBoundary>...</ErrorBoundary></ThemeProvider>).
// The original initializer called localStorage.getItem and
// window.matchMedia with no guard at all. Either can legitimately throw —
// some corporate-managed browsers disable localStorage via policy
// (throwing SecurityError on access), and privacy-hardened browser
// configurations do the same — and since this runs ABOVE the app's only
// error boundary, a throw here would white-screen the entire app with
// zero recovery path, not just fail to remember a theme preference.
// Wrapped both the initial read and the persistence write in try/catch so
// a blocked storage API degrades to "theme doesn't persist across
// reloads" instead of "the app doesn't render at all".
function detectInitialTheme() {
  try {
    const stored = localStorage.getItem('nexus-theme')
    if (stored) return stored
  } catch {
    // localStorage blocked (policy, privacy mode, etc.) — fall through to
    // system-preference detection instead of failing here.
  }
  try {
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : DEFAULT_THEME
  } catch {
    return DEFAULT_THEME
  }
}

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(detectInitialTheme)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      localStorage.setItem('nexus-theme', theme)
    } catch {
      // Persistence is a nice-to-have — the theme still applies for this
      // session via the DOM attribute above even if it can't be saved.
    }
  }, [theme])

  const toggle = () => setTheme(t => t === 'dark' ? 'light' : 'dark')

  return (
    <ThemeContext.Provider value={{ theme, toggle }}>
      {children}
    </ThemeContext.Provider>
  )
}

export const useTheme = () => useContext(ThemeContext)
