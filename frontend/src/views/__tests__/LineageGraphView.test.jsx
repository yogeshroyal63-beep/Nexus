import { describe, it, expect } from 'vitest'
import { computeAutoLayout } from '../LineageGraphView'

// THE GAP THIS PROVES IS CLOSED: unmatched URNs (any model_urn other than
// the one hardcoded demo model) used to all fall back to the identical
// {x:0, y:0} position, rendering as a single unreadable overlapping pile.
// computeAutoLayout gives each node a distinct, readable position based on
// its distance from a source node, fanning out into columns/rows the same
// way the hand-placed LAYOUT does for the demo graph.

describe('computeAutoLayout', () => {
  it('gives every node a distinct position when none overlap in the graph', () => {
    const nodeUrns = ['a', 'b', 'c']
    const edges = [
      { upstream_urn: 'a', downstream_urn: 'b' },
      { upstream_urn: 'b', downstream_urn: 'c' },
    ]
    const positions = computeAutoLayout(nodeUrns, edges)
    const values = [...positions.values()].map((p) => `${p.x},${p.y}`)
    const unique = new Set(values)
    expect(unique.size).toBe(3)
  })

  it('places a straight chain left-to-right in increasing columns', () => {
    const nodeUrns = ['a', 'b', 'c']
    const edges = [
      { upstream_urn: 'a', downstream_urn: 'b' },
      { upstream_urn: 'b', downstream_urn: 'c' },
    ]
    const positions = computeAutoLayout(nodeUrns, edges)
    expect(positions.get('a').x).toBeLessThan(positions.get('b').x)
    expect(positions.get('b').x).toBeLessThan(positions.get('c').x)
  })

  it('places sibling nodes at the same column but different rows', () => {
    // Two independent sources both feeding the same sink: a and b are
    // both "distance 1 from the sink", should share a column, different rows.
    const nodeUrns = ['a', 'b', 'sink']
    const edges = [
      { upstream_urn: 'a', downstream_urn: 'sink' },
      { upstream_urn: 'b', downstream_urn: 'sink' },
    ]
    const positions = computeAutoLayout(nodeUrns, edges)
    expect(positions.get('a').x).toBe(positions.get('b').x)
    expect(positions.get('a').y).not.toBe(positions.get('b').y)
  })

  it('handles a single isolated node with no edges at all', () => {
    const positions = computeAutoLayout(['solo'], [])
    expect(positions.get('solo')).toEqual({ x: 0, y: 0 })
  })

  it('handles an empty graph without crashing', () => {
    const positions = computeAutoLayout([], [])
    expect(positions.size).toBe(0)
  })

  it('does not infinite-loop on a cyclic graph (malformed input)', () => {
    // Should never occur given build_dag's backend cycle check, but this
    // frontend function must still degrade safely if it ever did.
    const nodeUrns = ['a', 'b']
    const edges = [
      { upstream_urn: 'a', downstream_urn: 'b' },
      { upstream_urn: 'b', downstream_urn: 'a' },
    ]
    expect(() => computeAutoLayout(nodeUrns, edges)).not.toThrow()
  })

  it('handles a diamond shape (two paths converging) without overlap', () => {
    const nodeUrns = ['a', 'b', 'c', 'd']
    const edges = [
      { upstream_urn: 'a', downstream_urn: 'b' },
      { upstream_urn: 'a', downstream_urn: 'c' },
      { upstream_urn: 'b', downstream_urn: 'd' },
      { upstream_urn: 'c', downstream_urn: 'd' },
    ]
    const positions = computeAutoLayout(nodeUrns, edges)
    // b and c are both between a and d -> same column, different rows
    expect(positions.get('b').x).toBe(positions.get('c').x)
    expect(positions.get('b').y).not.toBe(positions.get('c').y)
    // a is upstream of everything, d is downstream of everything
    expect(positions.get('a').x).toBeLessThan(positions.get('b').x)
    expect(positions.get('d').x).toBeGreaterThan(positions.get('b').x)
  })

  it('ignores edges referencing nodes not in nodeUrns', () => {
    const nodeUrns = ['a', 'b']
    const edges = [
      { upstream_urn: 'a', downstream_urn: 'b' },
      { upstream_urn: 'ghost', downstream_urn: 'a' },  // 'ghost' not in nodeUrns
    ]
    expect(() => computeAutoLayout(nodeUrns, edges)).not.toThrow()
    const positions = computeAutoLayout(nodeUrns, edges)
    expect(positions.size).toBe(2)
  })
})
