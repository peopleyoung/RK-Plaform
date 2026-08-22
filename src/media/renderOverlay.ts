import type {
  InferenceDetection,
  InferenceEnvelope,
  OcrDetectionResult,
  OcrRecognitionResult,
  StructuredResult,
} from './contracts'
import { computeContainTransform, mapPoint, type ContainTransform } from './geometry'

export interface NormalizedPoint {
  readonly x: number
  readonly y: number
}

export interface OverlayArea {
  readonly id: string
  readonly name: string
  readonly polygon: readonly NormalizedPoint[]
}

export interface OverlayLine {
  readonly id: string
  readonly name: string
  readonly start: NormalizedPoint
  readonly end: NormalizedPoint
}

export interface OverlayRenderOptions {
  readonly segmentation?: CanvasImageSource | null
  readonly areas?: readonly OverlayArea[]
  readonly lines?: readonly OverlayLine[]
}

const COLORS = {
  geometry: '#facc15',
  primary: '#22c55e',
  secondary: '#38bdf8',
  ocr: '#f472b6',
  linkage: '#a78bfa',
  text: '#ffffff',
  textBackground: 'rgba(0, 0, 0, 0.72)',
} as const

export function renderOverlay(
  context: CanvasRenderingContext2D,
  envelope: InferenceEnvelope,
  options: OverlayRenderOptions,
): void {
  context.save()
  try {
    context.clearRect(0, 0, context.canvas.width, context.canvas.height)
    const transform = computeContainTransform(
      envelope.sourceWidth,
      envelope.sourceHeight,
      context.canvas.width,
      context.canvas.height,
    )
    context.lineWidth = Math.min(4, Math.max(1, transform.scale * 1.5))
    context.font = `${Math.min(20, Math.max(11, 12 * transform.scale))}px sans-serif`
    context.textBaseline = 'top'
    context.textAlign = 'left'

    drawSegmentation(context, transform, options.segmentation)
    drawGeometry(context, transform, envelope, options)
    drawDetections(context, transform, envelope.detections, COLORS.primary)
    drawSecondaryDetections(context, transform, envelope)
    drawOcr(context, transform, envelope)
    drawAnalytics(context, transform, envelope, options)
  } finally {
    context.restore()
  }
}

function drawSegmentation(
  context: CanvasRenderingContext2D,
  transform: ContainTransform,
  segmentation: CanvasImageSource | null | undefined,
): void {
  if (!segmentation) return
  context.globalAlpha = 0.5
  context.drawImage(
    segmentation,
    transform.offsetX,
    transform.offsetY,
    transform.width,
    transform.height,
  )
  context.globalAlpha = 1
}

function drawGeometry(
  context: CanvasRenderingContext2D,
  transform: ContainTransform,
  envelope: InferenceEnvelope,
  options: OverlayRenderOptions,
): void {
  context.strokeStyle = COLORS.geometry
  for (const area of options.areas ?? []) {
    if (area.polygon.length < 3) continue
    context.beginPath()
    area.polygon.forEach((point, index) => {
      const mapped = mapNormalized(transform, envelope, point)
      if (index === 0) context.moveTo(mapped.x, mapped.y)
      else context.lineTo(mapped.x, mapped.y)
    })
    context.closePath()
    context.stroke()
  }
  for (const line of options.lines ?? []) {
    const start = mapNormalized(transform, envelope, line.start)
    const end = mapNormalized(transform, envelope, line.end)
    context.beginPath()
    context.moveTo(start.x, start.y)
    context.lineTo(end.x, end.y)
    context.stroke()
  }
}

function drawDetections(
  context: CanvasRenderingContext2D,
  transform: ContainTransform,
  detections: readonly InferenceDetection[],
  color: string,
): void {
  context.strokeStyle = color
  for (const detection of detections) {
    const origin = mapPoint(transform, detection.x, detection.y)
    const width = detection.width * transform.scale
    const height = detection.height * transform.scale
    context.strokeRect(origin.x, origin.y, width, height)
    const track = detection.trackId === undefined ? '' : ` #${detection.trackId}`
    drawLabel(
      context,
      transform,
      `${detection.label} ${(detection.confidence * 100).toFixed(0)}%${track}`,
      origin.x,
      origin.y,
      color,
    )
  }
}

function drawSecondaryDetections(
  context: CanvasRenderingContext2D,
  transform: ContainTransform,
  envelope: InferenceEnvelope,
): void {
  for (const [instance, detections] of Object.entries(envelope.detectionsByInstance)) {
    if (instance === envelope.primaryInstance) continue
    for (const detection of detections) {
      drawParentLink(context, transform, envelope.detections, detection)
    }
    drawDetections(context, transform, detections, COLORS.secondary)
  }
}

function drawParentLink(
  context: CanvasRenderingContext2D,
  transform: ContainTransform,
  parents: readonly InferenceDetection[],
  child: InferenceDetection,
): void {
  if (child.parentDetectionIndex === undefined) return
  const parent = parents[child.parentDetectionIndex]
  if (!parent) return
  const parentCenter = mapPoint(
    transform,
    parent.x + parent.width / 2,
    parent.y + parent.height / 2,
  )
  const childCenter = mapPoint(
    transform,
    child.x + child.width / 2,
    child.y + child.height / 2,
  )
  context.strokeStyle = COLORS.linkage
  context.beginPath()
  context.moveTo(parentCenter.x, parentCenter.y)
  context.lineTo(childCenter.x, childCenter.y)
  context.stroke()
}

function drawOcr(
  context: CanvasRenderingContext2D,
  transform: ContainTransform,
  envelope: InferenceEnvelope,
): void {
  const results: StructuredResult[] = [
    ...(envelope.primaryResult ? [envelope.primaryResult] : []),
    ...Object.values(envelope.structuredResults),
  ]
  for (const result of results) {
    if (result.type === 'ocr_detection') drawOcrRegions(context, transform, result)
    if (result.type === 'ocr_recognition') drawRecognition(context, transform, result)
  }
}

function drawOcrRegions(
  context: CanvasRenderingContext2D,
  transform: ContainTransform,
  result: OcrDetectionResult,
): void {
  context.strokeStyle = COLORS.ocr
  for (const region of result.regions) {
    context.beginPath()
    region.points.forEach((point, index) => {
      const mapped = mapPoint(transform, point.x, point.y)
      if (index === 0) context.moveTo(mapped.x, mapped.y)
      else context.lineTo(mapped.x, mapped.y)
    })
    context.closePath()
    context.stroke()
    if (region.text && region.points[0]) {
      const origin = mapPoint(transform, region.points[0].x, region.points[0].y)
      drawLabel(context, transform, region.text, origin.x, origin.y, COLORS.ocr)
    }
  }
}

function drawRecognition(
  context: CanvasRenderingContext2D,
  transform: ContainTransform,
  result: OcrRecognitionResult,
): void {
  drawLabel(
    context,
    transform,
    `${result.text} ${(result.confidence * 100).toFixed(0)}%`,
    transform.offsetX + 8,
    transform.offsetY + 8,
    COLORS.ocr,
  )
}

function drawAnalytics(
  context: CanvasRenderingContext2D,
  transform: ContainTransform,
  envelope: InferenceEnvelope,
  options: OverlayRenderOptions,
): void {
  const analyticsAreas = indexedAnalytics(envelope.analytics.areas)
  const analyticsLines = indexedAnalytics(envelope.analytics.lines)
  let row = 0
  for (const area of options.areas ?? []) {
    const value = analyticsAreas.get(area.id)
    const count = numberField(value, 'count')
    if (count === undefined) continue
    drawLabel(
      context,
      transform,
      `${area.name}: ${count}`,
      transform.offsetX + 8,
      transform.offsetY + 8 + row * 24,
      COLORS.geometry,
    )
    row += 1
  }
  for (const line of options.lines ?? []) {
    const value = analyticsLines.get(line.id)
    const forward = numberField(value, 'a_to_b_count') ?? 0
    const reverse = numberField(value, 'b_to_a_count') ?? 0
    if (!value) continue
    drawLabel(
      context,
      transform,
      `${line.name}: ${forward}/${reverse}`,
      transform.offsetX + 8,
      transform.offsetY + 8 + row * 24,
      COLORS.geometry,
    )
    row += 1
  }
}

function drawLabel(
  context: CanvasRenderingContext2D,
  transform: ContainTransform,
  text: string,
  x: number,
  y: number,
  accent: string,
): void {
  const padding = 4
  const height = 20
  const width = Math.min(transform.width, context.measureText(text).width + padding * 2)
  const left = Math.max(transform.offsetX, Math.min(x, transform.offsetX + transform.width - width))
  const top = Math.max(transform.offsetY, Math.min(y - height, transform.offsetY + transform.height - height))
  context.fillStyle = COLORS.textBackground
  context.fillRect(left, top, width, height)
  context.fillStyle = accent
  context.fillRect(left, top, 3, height)
  context.fillStyle = COLORS.text
  context.fillText(text, left + padding, top + 3, Math.max(0, width - padding * 2))
}

function mapNormalized(
  transform: ContainTransform,
  envelope: InferenceEnvelope,
  point: NormalizedPoint,
): { x: number; y: number } {
  return mapPoint(
    transform,
    Math.max(0, Math.min(1, point.x)) * envelope.sourceWidth,
    Math.max(0, Math.min(1, point.y)) * envelope.sourceHeight,
  )
}

function indexedAnalytics(value: unknown): Map<string, Record<string, unknown>> {
  const result = new Map<string, Record<string, unknown>>()
  if (!Array.isArray(value)) return result
  for (const item of value) {
    if (typeof item !== 'object' || item === null || Array.isArray(item)) continue
    const record = item as Record<string, unknown>
    if (typeof record.id === 'string') result.set(record.id, record)
  }
  return result
}

function numberField(value: Record<string, unknown> | undefined, name: string): number | undefined {
  const candidate = value?.[name]
  return typeof candidate === 'number' && Number.isFinite(candidate) ? candidate : undefined
}
