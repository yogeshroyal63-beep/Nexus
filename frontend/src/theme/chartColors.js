// Recharts and React Flow both render some props (axis/grid stroke, bar
// fill, marker color, background dot color) as raw SVG presentation
// attributes rather than inline styles, so CSS custom properties don't
// reliably resolve there. This table mirrors the --signal-*/--hairline/
// --text-* tokens in index.css so charts stay theme-correct — keep the two
// in sync if you change one.
export const CHART_COLORS = {
  dark: {
    hairline: '#232b3a',
    textLo: '#7c8698',
    textFaint: '#4b5468',
    bgPanelRaised: '#171d29',
    ink: '#0a0d12',
    critical: '#ff5d5d',
    trace: '#ffb454',
    ok: '#35d0a0',
    model: '#7fa8ff',
    gridDot: '#1a2030',
  },
  light: {
    hairline: '#dde2ea',
    textLo: '#5b6472',
    textFaint: '#98a1b0',
    bgPanelRaised: '#f4f6fa',
    ink: '#0a0d12',
    critical: '#c9303f',
    trace: '#a8660a',
    ok: '#0e9a70',
    model: '#3568d4',
    gridDot: '#dfe3ea',
  },
}
