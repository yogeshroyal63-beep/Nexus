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

    const rfNodes = graph.nodes.map((n) => {
      const pos = LAYOUT[n.urn] ?? { x: 0, y: 0 }
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
