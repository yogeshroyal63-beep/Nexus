import '@testing-library/jest-dom/vitest'

// jsdom does not implement ResizeObserver, which recharts' ResponsiveContainer
// depends on to measure its container. Without this polyfill, every test that
// renders a chart-containing view fails with "ResizeObserver is not defined"
// regardless of whether the component itself has a real bug.
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
