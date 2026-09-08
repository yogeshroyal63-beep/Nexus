import React, { useMemo } from 'react'
import ReactFlow, { Background, Controls, Handle, MarkerType, Position } from 'reactflow'
import 'reactflow/dist/style.css'
import { useTheme } from '../theme/ThemeContext.jsx'
import { CHART_COLORS } from '../theme/chartColors.js'
import './LineageGraphView.css'

// Fixed layout for the demo lineage graph (3 raw datasets -> 3 features -> model -> deployment)
const LAYOUT = {
  'urn:li:dataset:(demo,raw_transactions,PROD)': { x: 0, y: 0 },
  'urn:li:dataset:(demo,raw_user_profiles,PROD)': { x: 0, y: 130 },
  'urn:li:dataset:(demo,raw_device_signals,PROD)': { x: 0, y: 260 },
  'urn:li:mlFeatureTable:(demo,feature_txn_velocity)': { x: 300, y: 0 },
  'urn:li:mlFeatureTable:(demo,feature_user_risk_score)': { x: 300, y: 130 },
  'urn:li:mlFeatureTable:(demo,feature_device_trust)': { x: 300, y: 260 },
  'urn:li:mlModel:(demo,fraud_model_v3,PROD)': { x: 600, y: 130 },
  'urn:li:mlModelDeployment:(demo,fraud_model_v3_prod)': { x: 880, y: 130 },
}

const COLUMN_WIDTH = 300
const ROW_HEIGHT = 130

/**
 * Auto-layout fallback for any node not in the hand-placed LAYOUT map above.
 *
 * THE GAP THIS CLOSES: the original code fell back to a bare `{ x: 0, y: 0 }`
 * for any URN not in LAYOUT — meaning every unmatched node rendered at the
 * exact same point, completely overlapping and unreadable. LAYOUT is keyed
 * to the ONE demo model, but the backend API fully supports an arbitrary
 * model_urn via /api/investigate?model_urn=... — this frontend is currently
 * only ever driven with the one hardcoded demo URN (see App.jsx's MODEL_URN
 * constant), so the overlap can't be hit through the deployed demo UI today,
 * but it's a real, foreseeable break the moment that changes, and silently
 * producing an unreadable graph instead of a clear layout is exactly the
 * class of "only works for the one happy path" gap this whole audit has
 * been about.
 *
 * Computes each node's column as its longest-path distance (in hops) from
 * any source node (no incoming edges) — i.e. the same "how many hops from
 * the root" concept the backend already uses for hops_from_model — and its
 * row as its index within that column, so nodes fan out into a readable
 * grid instead of stacking. Column/row spacing matches LAYOUT's own
 * conventions (300px / 130px) so a graph mixing hand-placed and
 * auto-computed nodes still looks visually consistent.
 *
 * Pure function, no React/ReactFlow dependency, so it's directly unit
 * testable without mounting any component.
 */
export function computeAutoLayout(nodeUrns, edges) {
  const children = new Map()
  const hasIncoming = new Set()
  for (const urn of nodeUrns) children.set(urn, [])
  for (const e of edges) {
    if (children.has(e.upstream_urn)) children.get(e.upstream_urn).push(e.downstream_urn)
    hasIncoming.add(e.downstream_urn)
  }

  const depth = new Map()
  const visiting = new Set()

  function longestPathFrom(urn) {
    if (depth.has(urn)) return depth.get(urn)
    if (visiting.has(urn)) return 0 // cycle guard — never infinite-loop on malformed input
    visiting.add(urn)
    const kids = children.get(urn) || []
    const result = kids.length === 0 ? 0 : 1 + Math.max(...kids.map(longestPathFrom))
    visiting.delete(urn)
    depth.set(urn, result)
    return result
  }

  const sources = nodeUrns.filter((urn) => !hasIncoming.has(urn))
  const roots = sources.length > 0 ? sources : nodeUrns // degenerate: no clear source, treat all as roots
  for (const urn of roots) longestPathFrom(urn)
  for (const urn of nodeUrns) if (!depth.has(urn)) longestPathFrom(urn)

  // Column = distance from a source (root), NOT distance-to-sink, so a
  // longer causal chain reads left-to-right the same direction as the
  // hand-placed LAYOUT (raw data -> features -> model -> deployment).
  const maxDepth = Math.max(0, ...depth.values())
  const columnOf = new Map()
  for (const urn of nodeUrns) columnOf.set(urn, maxDepth - depth.get(urn))

  const rowCounters = new Map()
  const positions = new Map()
  for (const urn of nodeUrns) {
    const col = columnOf.get(urn)
    const row = rowCounters.get(col) || 0
    rowCounters.set(col, row + 1)
    positions.set(urn, { x: col * COLUMN_WIDTH, y: row * ROW_HEIGHT })
  }
  return positions
}

export function nodeStatus(urn, result) {
  if (!result) return 'idle'
  const trace = result.trace
  const rootCause = trace?.isolated_root_causes?.some((c) => c.node_urn === urn)
  if (rootCause) return 'root-cause'
  const onPath = trace?.graph_path?.includes(urn)
  if (onPath) return 'on-path'
  const candidate = trace?.candidates_examined?.find((c) => c.node_urn === urn)
  if (candidate && !candidate.is_genuine_cause) return 'confounded'
  return 'idle'
}

export default function LineageGraphView({ result, stage }) {
  const graph = result?.graph
  const { theme } = useTheme()
  const c = CHART_COLORS[theme]

  const { nodes, edges } = useMemo(() => {
    if (!graph) return { nodes: [], edges: [] }

    const allUrns = graph.nodes.map((n) => n.urn)
    const unknownUrns = allUrns.filter((urn) => !LAYOUT[urn])
    const autoPositions = unknownUrns.length > 0 ? computeAutoLayout(allUrns, graph.edges) : new Map()

    const rfNodes = graph.nodes.map((n) => {
      const pos = LAYOUT[n.urn] ?? autoPositions.get(n.urn) ?? { x: 0, y: 0 }
      const status = nodeStatus(n.urn, result)
      return {
        id: n.urn,
        position: pos,
        data: { label: n.name, type: n.node_type, status, description: n.description },
        type: 'nexusNode',
      }
    })

    const rfEdges = graph.edges.map((e, i) => {
      const onPath =
        result?.trace?.graph_path &&
        result.trace.graph_path.includes(e.upstream_urn) &&
        result.trace.graph_path.includes(e.downstream_urn)
      return {
        id: `e${i}`,
        source: e.upstream_urn,
        target: e.downstream_urn,
        animated: !!onPath,
        style: {
          stroke: onPath ? 'var(--signal-critical)' : 'var(--hairline)',
          strokeWidth: onPath ? 2.5 : 1.5,
        },
        markerEnd: { type: MarkerType.ArrowClosed, color: onPath ? c.critical : c.hairline },
      }
    })

    return { nodes: rfNodes, edges: rfEdges }
  }, [graph, result, c])

  if (!graph) {
    return (
      <div className="empty-state mono">
        <p>No lineage loaded yet.</p>
        <p className="empty-sub">Trigger “Replay a failure” above to pull the ML lineage graph and run an investigation.</p>
      </div>
    )
  }

  return (
    <div className="graph-shell">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        panOnScroll
        zoomOnScroll={false}
      >
        <Background color={c.gridDot} gap={20} />
        <Controls showInteractive={false} />
      </ReactFlow>
      <Legend />
    </div>
  )
}

function NexusNode({ data }) {
  return (
    <div className={`nexus-node nexus-node-${data.status}`}>
      <Handle type="target" position={Position.Left} className="nexus-handle" />
      <div className="nexus-node-type mono">{data.type}</div>
      <div className="nexus-node-label">{data.label}</div>
      {data.status === 'root-cause' && <div className="nexus-node-tag mono">ROOT CAUSE</div>}
      {data.status === 'confounded' && <div className="nexus-node-tag mono">confounded</div>}
      <Handle type="source" position={Position.Right} className="nexus-handle" />
    </div>
  )
}

const nodeTypes = { nexusNode: NexusNode }

function Legend() {
  return (
    <div className="graph-legend mono">
      <span><i className="dot dot-idle" />unaffected</span>
      <span><i className="dot dot-confounded" />drifted, not causal</span>
      <span><i className="dot dot-path" />causal path</span>
      <span><i className="dot dot-root" />isolated root cause</span>
    </div>
  )
}
