import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import DriftTimelineView from '../DriftTimelineView'

// THE CRASH THIS TEST PROVES IS FIXED:
// pred.p_value can now legitimately be null when severity is
// 'insufficient_data' (a real backend state introduced this session).
// The original code did `pred.p_value < 0.0001 ? ... .toExponential(2) : ...`
// -- in JS, `null < 0.0001` evaluates to true (null coerces to 0), so it
// took the .toExponential() branch and crashed with
// "Cannot read properties of null (reading 'toExponential')".

function makeResult({ severity = 'critical', pValue = 0.001, statistic = 0.5 } = {}) {
  return {
    trace: {
      model_urn: 'urn:li:mlModel:(demo,fraud_model_v3,PROD)',
      prediction_drift: {
        method: 'ks_test',
        statistic,
        p_value: pValue,
        severity,
      },
      candidates_examined: [],
    },
  }
}

describe('DriftTimelineView p_value null-safety', () => {
  it('renders normally with a real p_value', () => {
    const result = makeResult({ pValue: 0.0123 })
    render(<DriftTimelineView result={result} />)
    expect(screen.getByText('0.0123')).toBeInTheDocument()
  })

  it('does not crash when p_value is null (insufficient_data)', () => {
    const result = makeResult({ severity: 'insufficient_data', pValue: null })
    expect(() => render(<DriftTimelineView result={result} />)).not.toThrow()
  })

  it('renders "n/a" for a null p_value instead of crashing', () => {
    const result = makeResult({ severity: 'insufficient_data', pValue: null })
    render(<DriftTimelineView result={result} />)
    expect(screen.getByText('n/a')).toBeInTheDocument()
  })

  it('shows the insufficient-data warning message', () => {
    const result = makeResult({ severity: 'insufficient_data', pValue: null })
    render(<DriftTimelineView result={result} />)
    expect(screen.getByText(/Not enough samples/i)).toBeInTheDocument()
  })

  it('does not show the insufficient-data warning for a normal critical result', () => {
    const result = makeResult({ severity: 'critical', pValue: 0.001 })
    render(<DriftTimelineView result={result} />)
    expect(screen.queryByText(/Not enough samples/i)).not.toBeInTheDocument()
  })

  it('still uses exponential notation for very small real p_values', () => {
    const result = makeResult({ pValue: 0.000001 })
    render(<DriftTimelineView result={result} />)
    expect(screen.getByText(/1\.00e-6/i)).toBeInTheDocument()
  })

  it('handles a missing p_value key the same as null', () => {
    // Built manually (not via makeResult's destructured defaults) because
    // JS destructuring defaults trigger on `undefined`, which would
    // silently substitute makeResult's own default pValue instead of
    // actually testing the missing-key case.
    const result = {
      trace: {
        model_urn: 'urn:li:mlModel:(demo,fraud_model_v3,PROD)',
        prediction_drift: {
          method: 'ks_test',
          statistic: 0.5,
          severity: 'insufficient_data',
          // p_value intentionally omitted entirely
        },
        candidates_examined: [],
      },
    }
    expect(() => render(<DriftTimelineView result={result} />)).not.toThrow()
    expect(screen.getByText('n/a')).toBeInTheDocument()
  })

  it('renders empty state when result is null', () => {
    render(<DriftTimelineView result={null} />)
    expect(screen.getByText(/No drift evidence yet/i)).toBeInTheDocument()
  })
})
