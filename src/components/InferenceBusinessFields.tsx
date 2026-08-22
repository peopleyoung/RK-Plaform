import { Plus, Trash2 } from 'lucide-react'
import { Button } from '../components'
import type { ModelRelease } from '../types'

export interface NormalizedPoint { x: number; y: number }
export interface AreaRule {
  id: string
  name: string
  polygon: NormalizedPoint[]
  classIds: number[]
  minCount: number
  holdFrames: number
}
export interface LineRule {
  id: string
  name: string
  start: NormalizedPoint
  end: NormalizedPoint
  direction: 'both' | 'a_to_b' | 'b_to_a'
  classIds: number[]
}
export interface SecondaryModelRule {
  releaseId: string
  sourceClassIds: number[]
  confidenceThreshold: number
}
export interface InferenceAnalyticsConfig extends Record<string, unknown> {
  areas: AreaRule[]
  lines: LineRule[]
  osd: {
    enabled: boolean
    showLabels: boolean
    showConfidence: boolean
    showTrackId: boolean
    showAreas: boolean
    showLines: boolean
  }
  events: {
    enabled: boolean
    snapshot: boolean
    record: boolean
    preSeconds: number
    postSeconds: number
    retentionDays: number
  }
  secondaryModels: SecondaryModelRule[]
}

export const emptyInferenceAnalytics = (): InferenceAnalyticsConfig => ({
  areas: [],
  lines: [],
  osd: {
    enabled: true,
    showLabels: true,
    showConfidence: true,
    showTrackId: true,
    showAreas: true,
    showLines: true,
  },
  events: {
    enabled: false,
    snapshot: true,
    record: false,
    preSeconds: 3,
    postSeconds: 5,
    retentionDays: 30,
  },
  secondaryModels: [],
})

function objectValue(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function numberValue(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function classIds(value: unknown): number[] {
  return Array.isArray(value)
    ? value.filter((item): item is number => Number.isInteger(item) && item >= 0)
    : []
}

function point(value: unknown, fallback: NormalizedPoint): NormalizedPoint {
  const raw = objectValue(value)
  return { x: numberValue(raw.x, fallback.x), y: numberValue(raw.y, fallback.y) }
}

export function normalizeInferenceAnalytics(value: unknown): InferenceAnalyticsConfig {
  const raw = objectValue(value)
  const defaults = emptyInferenceAnalytics()
  const osd = objectValue(raw.osd)
  const events = objectValue(raw.events)
  const areas = Array.isArray(raw.areas) ? raw.areas.map((item, index) => {
    const area = objectValue(item)
    const polygon = Array.isArray(area.polygon)
      ? area.polygon.map((item) => point(item, { x: 0, y: 0 }))
      : []
    return {
      id: typeof area.id === 'string' ? area.id : `area-${index + 1}`,
      name: typeof area.name === 'string' ? area.name : `区域 ${index + 1}`,
      polygon,
      classIds: classIds(area.classIds),
      minCount: numberValue(area.minCount, 1),
      holdFrames: numberValue(area.holdFrames, 1),
    }
  }) : []
  const lines = Array.isArray(raw.lines) ? raw.lines.map((item, index) => {
    const line = objectValue(item)
    const direction = ['both', 'a_to_b', 'b_to_a'].includes(String(line.direction))
      ? line.direction as LineRule['direction']
      : 'both'
    return {
      id: typeof line.id === 'string' ? line.id : `line-${index + 1}`,
      name: typeof line.name === 'string' ? line.name : `越线 ${index + 1}`,
      start: point(line.start, { x: 0.5, y: 0.1 }),
      end: point(line.end, { x: 0.5, y: 0.9 }),
      direction,
      classIds: classIds(line.classIds),
    }
  }) : []
  const secondaryModels = Array.isArray(raw.secondaryModels)
    ? raw.secondaryModels.map((item) => {
      const secondary = objectValue(item)
      return {
        releaseId: typeof secondary.releaseId === 'string' ? secondary.releaseId : '',
        sourceClassIds: classIds(secondary.sourceClassIds),
        confidenceThreshold: numberValue(secondary.confidenceThreshold, 0.25),
      }
    })
    : []
  return {
    areas,
    lines,
    osd: {
      enabled: typeof osd.enabled === 'boolean' ? osd.enabled : defaults.osd.enabled,
      showLabels: typeof osd.showLabels === 'boolean' ? osd.showLabels : defaults.osd.showLabels,
      showConfidence: typeof osd.showConfidence === 'boolean' ? osd.showConfidence : defaults.osd.showConfidence,
      showTrackId: typeof osd.showTrackId === 'boolean' ? osd.showTrackId : defaults.osd.showTrackId,
      showAreas: typeof osd.showAreas === 'boolean' ? osd.showAreas : defaults.osd.showAreas,
      showLines: typeof osd.showLines === 'boolean' ? osd.showLines : defaults.osd.showLines,
    },
    events: {
      enabled: typeof events.enabled === 'boolean' ? events.enabled : defaults.events.enabled,
      snapshot: typeof events.snapshot === 'boolean' ? events.snapshot : defaults.events.snapshot,
      record: typeof events.record === 'boolean' ? events.record : defaults.events.record,
      preSeconds: numberValue(events.preSeconds, defaults.events.preSeconds),
      postSeconds: numberValue(events.postSeconds, defaults.events.postSeconds),
      retentionDays: numberValue(events.retentionDays, defaults.events.retentionDays),
    },
    secondaryModels,
  }
}

export function inferenceAnalyticsError(
  value: InferenceAnalyticsConfig,
  detectionSupported: boolean,
  trackingEnabled: boolean,
  decoder: 'opencv' | 'rkmpp',
): string {
  const hasDetectionBusiness = value.areas.length > 0 || value.lines.length > 0 || value.secondaryModels.length > 0
  if (hasDetectionBusiness && !detectionSupported) return '区域、越线和二级推理仅支持 YOLO 检测模型。'
  if ((value.areas.length > 0 || value.lines.length > 0) && !trackingEnabled) return '区域和越线分析需要启用 ByteTrack。'
  const ids = [...value.areas.map((item) => item.id.trim()), ...value.lines.map((item) => item.id.trim())]
  if (ids.some((item) => !item) || new Set(ids).size !== ids.length) return '区域和越线标识不能为空或重复。'
  if (value.areas.some((area) => area.polygon.length < 3)) return '每个区域至少需要三个边界点。'
  if (value.areas.some((area) => !Number.isInteger(area.minCount) || area.minCount < 1 || area.minCount > 100000 || !Number.isInteger(area.holdFrames) || area.holdFrames < 1 || area.holdFrames > 10000)) return '区域触发数量或稳定帧数超出允许范围。'
  const points = [
    ...value.areas.flatMap((area) => area.polygon),
    ...value.lines.flatMap((line) => [line.start, line.end]),
  ]
  if (points.some((item) => !Number.isFinite(item.x) || !Number.isFinite(item.y) || item.x < 0 || item.x > 1 || item.y < 0 || item.y > 1)) return '区域和越线坐标必须在 0 到 1 之间。'
  if (value.lines.some((line) => line.start.x === line.end.x && line.start.y === line.end.y)) return '越线的起点和终点不能相同。'
  if (value.events.enabled && value.areas.length === 0 && value.lines.length === 0) return '事件输出至少需要一个区域或越线规则。'
  if (value.events.enabled && !value.events.snapshot && !value.events.record) return '事件输出至少需要抓拍或录像。'
  if (value.events.record && decoder !== 'rkmpp') return '事件录像需要使用 RKMPP 硬解码。'
  if (!Number.isInteger(value.events.preSeconds) || value.events.preSeconds < 0 || value.events.preSeconds > 60 || !Number.isInteger(value.events.postSeconds) || value.events.postSeconds < 0 || value.events.postSeconds > 300 || !Number.isInteger(value.events.retentionDays) || value.events.retentionDays < 1 || value.events.retentionDays > 3650) return '事件前后录时长或留存天数超出允许范围。'
  const secondaryIds = value.secondaryModels.map((item) => item.releaseId)
  if (secondaryIds.some((item) => !item) || new Set(secondaryIds).size !== secondaryIds.length) return '二级模型不能为空或重复。'
  if (value.secondaryModels.some((item) => !Number.isFinite(item.confidenceThreshold) || item.confidenceThreshold < 0 || item.confidenceThreshold > 1)) return '二级模型置信度必须在 0 到 1 之间。'
  return ''
}

export function inferenceAnalyticsSummary(value: Record<string, unknown>): string[] {
  const analytics = normalizeInferenceAnalytics(value)
  const result: string[] = []
  if (analytics.areas.length) result.push(`${analytics.areas.length} 个区域`)
  if (analytics.lines.length) result.push(`${analytics.lines.length} 条越线`)
  if (analytics.secondaryModels.length) result.push(`${analytics.secondaryModels.length} 个二级模型`)
  if (analytics.events.enabled) result.push(analytics.events.record ? '事件录像' : '事件抓拍')
  return result
}

function parseClassIds(value: string): number[] {
  return [...new Set(value.split(',').map((item) => Number(item.trim())).filter((item) => Number.isInteger(item) && item >= 0))]
}

interface Props {
  value: InferenceAnalyticsConfig
  releases: ModelRelease[]
  currentReleaseId: string
  detectionSupported: boolean
  trackingEnabled: boolean
  decoder: 'opencv' | 'rkmpp'
  error: string
  onChange: (value: InferenceAnalyticsConfig) => void
  onTrackingChange: (enabled: boolean) => void
  onDecoderChange: (decoder: 'opencv' | 'rkmpp') => void
}

export function InferenceBusinessFields({ value, releases, currentReleaseId, detectionSupported, trackingEnabled, decoder, error, onChange, onTrackingChange, onDecoderChange }: Props) {
  const updateArea = (index: number, area: AreaRule) => onChange({ ...value, areas: value.areas.map((item, itemIndex) => itemIndex === index ? area : item) })
  const updateLine = (index: number, line: LineRule) => onChange({ ...value, lines: value.lines.map((item, itemIndex) => itemIndex === index ? line : item) })
  const updateSecondary = (index: number, secondary: SecondaryModelRule) => onChange({ ...value, secondaryModels: value.secondaryModels.map((item, itemIndex) => itemIndex === index ? secondary : item) })
  const secondaryReleases = releases.filter((release) => release.id !== currentReleaseId && release.adapter.startsWith('yolo_'))
  const enableTracking = () => { if (!trackingEnabled) onTrackingChange(true) }

  return <>
    <section className="form-section"><h3>区域与越线</h3>
      <div className="business-toolbar"><span>{detectionSupported ? '基于跟踪 ID 计数并去重' : '当前模型不支持检测业务规则'}</span><div className="row-actions"><Button variant="secondary" icon={<Plus size={14} />} disabled={!detectionSupported || value.areas.length >= 32} onClick={() => { enableTracking(); onChange({ ...value, areas: [...value.areas, { id: `area-${value.areas.length + 1}`, name: `区域 ${value.areas.length + 1}`, polygon: [{ x: 0.1, y: 0.1 }, { x: 0.9, y: 0.1 }, { x: 0.9, y: 0.9 }, { x: 0.1, y: 0.9 }], classIds: [], minCount: 1, holdFrames: 1 }] }) }}>新增区域</Button><Button variant="secondary" icon={<Plus size={14} />} disabled={!detectionSupported || value.lines.length >= 32} onClick={() => { enableTracking(); onChange({ ...value, lines: [...value.lines, { id: `line-${value.lines.length + 1}`, name: `越线 ${value.lines.length + 1}`, start: { x: 0.5, y: 0.1 }, end: { x: 0.5, y: 0.9 }, direction: 'both', classIds: [] }] }) }}>新增越线</Button></div></div>
      <div className="business-rule-list">{value.areas.map((area, index) => <div className="business-rule" key={`area-${index}`}><header><strong>区域规则 {index + 1}</strong><button className="icon-button ghost danger-action" type="button" title="删除区域" aria-label={`删除区域 ${index + 1}`} onClick={() => onChange({ ...value, areas: value.areas.filter((_, itemIndex) => itemIndex !== index) })}><Trash2 size={15} /></button></header><div className="form-grid three-columns"><label className="field"><span>规则标识</span><input value={area.id} onChange={(event) => updateArea(index, { ...area, id: event.target.value })} /></label><label className="field"><span>显示名称</span><input value={area.name} onChange={(event) => updateArea(index, { ...area, name: event.target.value })} /></label><label className="field"><span>类别 ID</span><input value={area.classIds.join(',')} placeholder="留空表示全部类别" onChange={(event) => updateArea(index, { ...area, classIds: parseClassIds(event.target.value) })} /></label><label className="field"><span>触发数量</span><input type="number" min="1" max="100000" value={area.minCount} onChange={(event) => updateArea(index, { ...area, minCount: Number(event.target.value) })} /></label><label className="field"><span>稳定帧数</span><input type="number" min="1" max="10000" value={area.holdFrames} onChange={(event) => updateArea(index, { ...area, holdFrames: Number(event.target.value) })} /></label></div><div className="point-editor"><span>归一化边界点</span>{area.polygon.map((item, pointIndex) => <div key={pointIndex}><input type="number" min="0" max="1" step="0.01" aria-label={`区域 ${index + 1} 点 ${pointIndex + 1} X`} value={item.x} onChange={(event) => updateArea(index, { ...area, polygon: area.polygon.map((pointItem, itemIndex) => itemIndex === pointIndex ? { ...pointItem, x: Number(event.target.value) } : pointItem) })} /><input type="number" min="0" max="1" step="0.01" aria-label={`区域 ${index + 1} 点 ${pointIndex + 1} Y`} value={item.y} onChange={(event) => updateArea(index, { ...area, polygon: area.polygon.map((pointItem, itemIndex) => itemIndex === pointIndex ? { ...pointItem, y: Number(event.target.value) } : pointItem) })} /><button className="icon-button ghost" type="button" disabled={area.polygon.length <= 3} title="删除边界点" aria-label={`删除区域 ${index + 1} 点 ${pointIndex + 1}`} onClick={() => updateArea(index, { ...area, polygon: area.polygon.filter((_, itemIndex) => itemIndex !== pointIndex) })}><Trash2 size={14} /></button></div>)}<button className="point-add" type="button" onClick={() => updateArea(index, { ...area, polygon: [...area.polygon, { x: 0.5, y: 0.5 }] })}><Plus size={14} />边界点</button></div></div>)}
      {value.lines.map((line, index) => <div className="business-rule" key={`line-${index}`}><header><strong>越线规则 {index + 1}</strong><button className="icon-button ghost danger-action" type="button" title="删除越线" aria-label={`删除越线 ${index + 1}`} onClick={() => onChange({ ...value, lines: value.lines.filter((_, itemIndex) => itemIndex !== index) })}><Trash2 size={15} /></button></header><div className="form-grid three-columns"><label className="field"><span>规则标识</span><input value={line.id} onChange={(event) => updateLine(index, { ...line, id: event.target.value })} /></label><label className="field"><span>显示名称</span><input value={line.name} onChange={(event) => updateLine(index, { ...line, name: event.target.value })} /></label><label className="field"><span>统计方向</span><select value={line.direction} onChange={(event) => updateLine(index, { ...line, direction: event.target.value as LineRule['direction'] })}><option value="both">双向</option><option value="a_to_b">A 到 B</option><option value="b_to_a">B 到 A</option></select></label><label className="field"><span>类别 ID</span><input value={line.classIds.join(',')} placeholder="留空表示全部类别" onChange={(event) => updateLine(index, { ...line, classIds: parseClassIds(event.target.value) })} /></label><label className="field"><span>起点 X / Y</span><div className="inline-fields"><input type="number" min="0" max="1" step="0.01" value={line.start.x} onChange={(event) => updateLine(index, { ...line, start: { ...line.start, x: Number(event.target.value) } })} /><input type="number" min="0" max="1" step="0.01" value={line.start.y} onChange={(event) => updateLine(index, { ...line, start: { ...line.start, y: Number(event.target.value) } })} /></div></label><label className="field"><span>终点 X / Y</span><div className="inline-fields"><input type="number" min="0" max="1" step="0.01" value={line.end.x} onChange={(event) => updateLine(index, { ...line, end: { ...line.end, x: Number(event.target.value) } })} /><input type="number" min="0" max="1" step="0.01" value={line.end.y} onChange={(event) => updateLine(index, { ...line, end: { ...line.end, y: Number(event.target.value) } })} /></div></label></div></div>)}
      </div>
      {!value.areas.length && !value.lines.length && <div className="business-empty">未配置区域或越线规则</div>}
    </section>
    <section className="form-section"><h3>OSD 与事件媒体</h3><div className="business-toggle-grid">{([['enabled', '启用 OSD'], ['showLabels', '类别'], ['showConfidence', '置信度'], ['showTrackId', '跟踪 ID'], ['showAreas', '区域'], ['showLines', '越线']] as const).map(([key, label]) => <label className="toggle-card" key={key}><input type="checkbox" checked={value.osd[key]} onChange={(event) => onChange({ ...value, osd: { ...value.osd, [key]: event.target.checked } })} /><span><strong>{label}</strong></span></label>)}</div><div className="form-grid three-columns business-row"><label className="toggle-card"><input type="checkbox" checked={value.events.enabled} onChange={(event) => onChange({ ...value, events: { ...value.events, enabled: event.target.checked } })} /><span><strong>事件输出</strong></span></label><label className="toggle-card"><input type="checkbox" checked={value.events.snapshot} disabled={!value.events.enabled} onChange={(event) => onChange({ ...value, events: { ...value.events, snapshot: event.target.checked } })} /><span><strong>JPEG 抓拍</strong></span></label><label className="toggle-card"><input type="checkbox" checked={value.events.record} disabled={!value.events.enabled} onChange={(event) => { const record = event.target.checked; if (record && decoder !== 'rkmpp') onDecoderChange('rkmpp'); onChange({ ...value, events: { ...value.events, record } }) }} /><span><strong>事件录像</strong></span></label>{value.events.enabled && <><label className="field"><span>前录秒数</span><input type="number" min="0" max="60" value={value.events.preSeconds} onChange={(event) => onChange({ ...value, events: { ...value.events, preSeconds: Number(event.target.value) } })} /></label><label className="field"><span>后录秒数</span><input type="number" min="0" max="300" value={value.events.postSeconds} onChange={(event) => onChange({ ...value, events: { ...value.events, postSeconds: Number(event.target.value) } })} /></label><label className="field"><span>留存天数</span><input type="number" min="1" max="3650" value={value.events.retentionDays} onChange={(event) => onChange({ ...value, events: { ...value.events, retentionDays: Number(event.target.value) } })} /></label></>}</div></section>
    <section className="form-section"><h3>非人脸二级推理</h3><div className="business-toolbar"><span>按主模型目标裁剪后调用另一个已发布 YOLO 模型</span><Button variant="secondary" icon={<Plus size={14} />} disabled={!detectionSupported || !secondaryReleases.length || value.secondaryModels.length >= 4} onClick={() => onChange({ ...value, secondaryModels: [...value.secondaryModels, { releaseId: secondaryReleases.find((item) => !value.secondaryModels.some((secondary) => secondary.releaseId === item.id))?.id ?? '', sourceClassIds: [], confidenceThreshold: 0.25 }] })}>新增二级模型</Button></div><div className="business-rule-list">{value.secondaryModels.map((secondary, index) => <div className="business-rule compact" key={index}><header><strong>二级推理 {index + 1}</strong><button className="icon-button ghost danger-action" type="button" title="删除二级模型" aria-label={`删除二级模型 ${index + 1}`} onClick={() => onChange({ ...value, secondaryModels: value.secondaryModels.filter((_, itemIndex) => itemIndex !== index) })}><Trash2 size={15} /></button></header><div className="form-grid three-columns"><label className="field"><span>模型版本</span><select value={secondary.releaseId} onChange={(event) => updateSecondary(index, { ...secondary, releaseId: event.target.value })}><option value="">请选择模型</option>{secondaryReleases.map((release) => <option key={release.id} value={release.id} disabled={value.secondaryModels.some((item, itemIndex) => itemIndex !== index && item.releaseId === release.id)}>{release.name} · {release.version}</option>)}</select></label><label className="field"><span>主模型类别 ID</span><input value={secondary.sourceClassIds.join(',')} placeholder="留空表示全部类别" onChange={(event) => updateSecondary(index, { ...secondary, sourceClassIds: parseClassIds(event.target.value) })} /></label><label className="field"><span>置信度阈值</span><input type="number" min="0" max="1" step="0.01" value={secondary.confidenceThreshold} onChange={(event) => updateSecondary(index, { ...secondary, confidenceThreshold: Number(event.target.value) })} /></label></div></div>)}</div></section>
    {error && <div className="api-banner danger" role="alert">{error}</div>}
  </>
}
