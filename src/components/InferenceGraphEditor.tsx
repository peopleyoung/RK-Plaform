import { useEffect, useMemo, useState } from 'react'
import { AlertCircle, Boxes, CheckCircle2, Cpu, Plus, RadioTower, Send, Trash2 } from 'lucide-react'
import type {
  InferenceGraph,
  InferenceGraphLayout,
  InferenceGraphNode,
  InferenceGraphValidation,
  InferenceOperator,
  InferenceOperatorCatalog,
  MediaGateway,
  ModelRelease,
} from '../types'

const OPERATOR_ORDER: Record<string, number> = {
  'capture.opencv': 0,
  'capture.rkmpp': 0,
  'inference.primary': 10,
  'processing.bytetrack': 20,
  'inference.secondary': 30,
  'processing.analytics': 40,
  'processing.events': 50,
  'output.json': 100,
  'output.kafka': 101,
  'output.zlm_sei': 102,
}

const categoryIcons = {
  capture: RadioTower,
  inference: Cpu,
  processing: Boxes,
  output: Send,
}

function cloneConfig(value: Record<string, unknown>): Record<string, unknown> {
  return JSON.parse(JSON.stringify(value)) as Record<string, unknown>
}

function makeNode(operator: InferenceOperator, suffix = 1): InferenceGraphNode {
  return {
    id: `${operator.operatorId.replaceAll('.', '-')}-${suffix}`,
    operator: operator.operatorId,
    config: cloneConfig(operator.defaults),
  }
}

function orderedNodes(nodes: InferenceGraphNode[]): InferenceGraphNode[] {
  return [...nodes].sort((left, right) => {
    const stage = (OPERATOR_ORDER[left.operator] ?? 999) - (OPERATOR_ORDER[right.operator] ?? 999)
    return stage || left.id.localeCompare(right.id)
  })
}

export function graphWithRebuiltEdges(graph: InferenceGraph): InferenceGraph {
  const nodes = orderedNodes(graph.nodes)
  const main = nodes.filter((node) => !node.operator.startsWith('output.'))
  const outputs = nodes.filter((node) => node.operator.startsWith('output.'))
  const edges = main.slice(1).map((node, index) => ({
    source: main[index].id,
    sourcePort: 'frame',
    target: node.id,
    targetPort: 'frame',
  }))
  const terminal = main.at(-1)
  if (terminal) {
    edges.push(...outputs.map((node) => ({
      source: terminal.id,
      sourcePort: 'frame',
      target: node.id,
      targetPort: 'frame',
    })))
  }
  return { ...graph, nodes, edges }
}

export function createDefaultInferenceGraph(
  catalog: InferenceOperatorCatalog,
  releaseId: string,
): { graph: InferenceGraph; layout: InferenceGraphLayout } {
  const required = ['capture.opencv', 'inference.primary', 'output.json'].map((operatorId) => {
    const operator = catalog.operators.find((item) => item.operatorId === operatorId)
    if (!operator) throw new Error(`算子目录缺少 ${operatorId}`)
    return makeNode(operator)
  })
  required[1].config.releaseId = releaseId
  const graph = graphWithRebuiltEdges({
    schemaVersion: catalog.schemaVersion,
    catalogVersion: catalog.catalogVersion,
    nodes: required,
    edges: [],
  })
  return { graph, layout: layoutFor(graph) }
}

function layoutFor(graph: InferenceGraph): InferenceGraphLayout {
  const positions: InferenceGraphLayout['positions'] = {}
  const mainNodes = graph.nodes.filter((node) => !node.operator.startsWith('output.'))
  const outputNodes = graph.nodes.filter((node) => node.operator.startsWith('output.'))
  mainNodes.forEach((node, index) => {
    positions[node.id] = { x: 36 + index * 168, y: 164 }
  })
  const outputStartY = Math.max(36, 164 - ((outputNodes.length - 1) * 84) / 2)
  outputNodes.forEach((node, index) => {
    positions[node.id] = { x: 36 + mainNodes.length * 168, y: outputStartY + index * 84 }
  })
  return { positions }
}

function ConfigField({
  field,
  value,
  releases,
  gateways,
  onChange,
}: {
  field: string
  value: unknown
  releases: ModelRelease[]
  gateways: MediaGateway[]
  onChange: (value: unknown) => void
}) {
  if (field === 'releaseId') {
    return <label className="field"><span>模型版本</span><select value={typeof value === 'string' ? value : ''} onChange={(event) => onChange(event.target.value)}><option value="">请选择已发布版本</option>{releases.map((release) => <option key={release.id} value={release.id}>{release.name} · {release.version}</option>)}</select></label>
  }
  if (field === 'gatewayId') {
    return <label className="field"><span>媒体网关</span><select value={typeof value === 'string' ? value : ''} onChange={(event) => onChange(event.target.value)}><option value="">请选择媒体网关</option>{gateways.map((gateway) => <option key={gateway.id} value={gateway.id} disabled={!gateway.enabled || gateway.status !== 'online'}>{gateway.name} · {gateway.status === 'online' ? '在线' : '不可用'}</option>)}</select></label>
  }
  if (field === 'type') {
    return <label className="field"><span>输出方式</span><select value={typeof value === 'string' ? value : 'jsonl'} onChange={(event) => onChange(event.target.value)}><option value="jsonl">JSONL 文件</option><option value="http">HTTP(S)</option></select></label>
  }
  if (typeof value === 'boolean') {
    return <label className="graph-boolean-field"><input type="checkbox" checked={value} onChange={(event) => onChange(event.target.checked)} /><span>{field}</span></label>
  }
  if (typeof value === 'number') {
    return <label className="field"><span>{field}</span><input type="number" value={value} onChange={(event) => onChange(Number(event.target.value))} /></label>
  }
  if (Array.isArray(value) || (typeof value === 'object' && value !== null)) {
    return <JsonConfigField field={field} value={value} onChange={onChange} />
  }
  return <label className="field"><span>{field}</span><input value={typeof value === 'string' ? value : ''} onChange={(event) => onChange(event.target.value)} /></label>
}

function JsonConfigField({ field, value, onChange }: { field: string; value: unknown; onChange: (value: unknown) => void }) {
  const serialized = JSON.stringify(value, null, 2)
  const [draft, setDraft] = useState(serialized)
  const [invalid, setInvalid] = useState(false)
  useEffect(() => {
    setDraft(serialized)
    setInvalid(false)
  }, [serialized])
  return <label className={`field full-span ${invalid ? 'invalid' : ''}`}><span>{field}</span><textarea rows={5} value={draft} aria-invalid={invalid} onChange={(event) => {
    const next = event.target.value
    setDraft(next)
    try {
      onChange(JSON.parse(next) as unknown)
      setInvalid(false)
    } catch {
      setInvalid(true)
    }
  }} /></label>
}

export function InferenceGraphEditor({
  catalog,
  releases,
  gateways,
  graph,
  validation,
  onChange,
}: {
  catalog: InferenceOperatorCatalog
  releases: ModelRelease[]
  gateways: MediaGateway[]
  graph: InferenceGraph
  validation: InferenceGraphValidation | null
  onChange: (graph: InferenceGraph, layout: InferenceGraphLayout) => void
}) {
  const [selectedId, setSelectedId] = useState(graph.nodes[0]?.id ?? '')
  const selected = graph.nodes.find((node) => node.id === selectedId) ?? graph.nodes[0]
  const operators = useMemo(() => new Map(catalog.operators.map((item) => [item.operatorId, item])), [catalog])
  const layout = layoutFor(graph)
  const canvasWidth = Math.max(560, ...Object.values(layout.positions).map((position) => position.x + 178))

  useEffect(() => {
    if (!graph.nodes.some((node) => node.id === selectedId)) setSelectedId(graph.nodes[0]?.id ?? '')
  }, [graph.nodes, selectedId])

  const commit = (nodes: InferenceGraphNode[]) => {
    const next = graphWithRebuiltEdges({ ...graph, nodes })
    onChange(next, layoutFor(next))
  }
  const add = (operator: InferenceOperator) => {
    if (operator.operatorId.startsWith('capture.')) {
      const current = graph.nodes.find((node) => node.operator.startsWith('capture.'))
      const replacement = makeNode(operator)
      replacement.id = current?.id ?? replacement.id
      setSelectedId(replacement.id)
      commit(graph.nodes.map((node) => node.id === current?.id ? replacement : node))
      return
    }
    const count = graph.nodes.filter((node) => node.operator === operator.operatorId).length
    if (count >= operator.maxInstances) return
    const node = makeNode(operator, count + 1)
    setSelectedId(node.id)
    commit([...graph.nodes, node])
  }
  const remove = (node: InferenceGraphNode) => {
    const operator = operators.get(node.operator)
    if (!operator || operator.minInstances > 0) return
    commit(graph.nodes.filter((item) => item.id !== node.id))
  }
  const updateConfig = (field: string, value: unknown) => {
    commit(graph.nodes.map((node) => node.id === selected.id
      ? { ...node, config: { ...node.config, [field]: value } }
      : node))
  }

  return <div className="graph-editor">
    <aside className="graph-palette" aria-label="算子目录">
      {(['capture', 'inference', 'processing', 'output'] as const).map((category) => {
        const Icon = categoryIcons[category]
        return <section key={category}><h4><Icon size={15} />{category}</h4>{catalog.operators.filter((item) => item.category === category).map((operator) => {
          const count = graph.nodes.filter((node) => node.operator === operator.operatorId).length
          const disabled = count >= operator.maxInstances
          return <button key={operator.operatorId} type="button" className="graph-palette-item" disabled={disabled && !operator.operatorId.startsWith('capture.')} onClick={() => add(operator)} title={operator.description}><span><strong>{operator.title}</strong><small>{operator.runtimeNode}</small></span>{disabled ? <CheckCircle2 size={15} /> : <Plus size={15} />}</button>
        })}</section>
      })}
    </aside>
    <section className="graph-workspace" aria-label="推理图画布">
      <div className="graph-canvas">
        <svg className="graph-edge-layer" width={canvasWidth} height="390" aria-hidden="true">
          {graph.edges.map((edge) => {
            const source = layout.positions[edge.source]
            const target = layout.positions[edge.target]
            if (!source || !target) return null
            const sourceX = source.x + 142
            const sourceY = source.y + 31
            const targetX = target.x
            const targetY = target.y + 31
            const middleX = sourceX + Math.max(13, (targetX - sourceX) / 2)
            return <path key={`${edge.source}:${edge.target}`} d={`M ${sourceX} ${sourceY} H ${middleX} V ${targetY} H ${targetX}`} />
          })}
        </svg>
        {graph.nodes.map((node, index) => {
          const operator = operators.get(node.operator)
          const position = layout.positions[node.id]
          return <div key={node.id} className={`graph-node ${selected?.id === node.id ? 'selected' : ''}`} style={{ left: position.x, top: position.y }} onClick={() => setSelectedId(node.id)}>
            <span className="graph-node-index">{index + 1}</span>
            <strong>{operator?.title ?? node.operator}</strong>
            <small>{operator?.runtimeNode}</small>
          </div>
        })}
      </div>
      <ol className="graph-mobile-steps">
        {graph.nodes.map((node, index) => <li key={node.id}><button type="button" onClick={() => setSelectedId(node.id)} className={selected?.id === node.id ? 'selected' : ''}><span>{index + 1}</span><strong>{operators.get(node.operator)?.title ?? node.operator}</strong></button></li>)}
      </ol>
      <div className={`graph-validation ${validation?.valid ? 'valid' : ''}`}>
        {validation?.valid ? <CheckCircle2 size={15} /> : <AlertCircle size={15} />}
        <span>{validation?.valid ? `图契约有效 · ${validation.requiredContexts} 个 RKNN 上下文` : validation?.issues[0]?.message ?? '正在校验图契约'}</span>
      </div>
    </section>
    <aside className="graph-inspector">
      {selected && <>
        <header><div><strong>{operators.get(selected.operator)?.title}</strong><small>{selected.operator}</small></div><button type="button" className="icon-button ghost danger-action" title="移除算子" disabled={(operators.get(selected.operator)?.minInstances ?? 0) > 0} onClick={() => remove(selected)}><Trash2 size={16} /></button></header>
        <div className="graph-config-fields">{(operators.get(selected.operator)?.configurableFields ?? []).map((field) => <ConfigField key={field} field={field} value={selected.config[field]} releases={releases.filter((release) => (operators.get(selected.operator)?.supportedAdapters.length ?? 0) === 0 || operators.get(selected.operator)?.supportedAdapters.includes(release.adapter))} gateways={gateways} onChange={(value) => updateConfig(field, value)} />)}</div>
      </>}
    </aside>
  </div>
}
