export const RKNODE_SEI_UUID = new Uint8Array([
  0x94, 0x51, 0xef, 0x8f, 0xd2, 0x41, 0x49, 0x6a,
  0x80, 0xba, 0x68, 0x18, 0xe2, 0x4d, 0xc0, 0x4e,
])

export const MAX_SEI_PAYLOAD_BYTES = 1024 * 1024
export const MAX_RESULT_ITEMS = 10_000
export const MAX_SEGMENTATION_RUNS = 262_144
export const MAX_SOURCE_WIDTH = 3840
export const MAX_SOURCE_HEIGHT = 2160
const MAX_TEXT_BYTES = 256

export type MetadataDiagnostic =
  | 'unsupported_sei_type'
  | 'foreign_uuid'
  | 'payload_too_large'
  | 'invalid_pts'
  | 'invalid_utf8'
  | 'invalid_json'
  | 'invalid_envelope'
  | 'unsupported_schema'
  | 'task_mismatch'
  | 'revision_mismatch'
  | 'invalid_dimensions'
  | 'too_many_results'
  | 'text_too_long'
  | 'unsupported_result'
  | 'invalid_segmentation'
  | 'too_many_runs'

export interface RawSeiPayload {
  readonly type: number
  readonly uuid: Uint8Array
  readonly userData: Uint8Array
  readonly ptsSeconds: number
}

export interface InferenceDetection {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
  readonly confidence: number
  readonly classId: number
  readonly label: string
  readonly trackId?: number
  readonly parentTrackId?: number
  readonly parentDetectionIndex?: number
  readonly parentInstance?: string
}

export interface OcrPoint {
  readonly x: number
  readonly y: number
}

export interface OcrRegion {
  readonly points: readonly OcrPoint[]
  readonly confidence: number
  readonly text?: string
}

export interface SegmentationResult {
  readonly type: 'segmentation'
  readonly width: number
  readonly height: number
  readonly sourceWidth: number
  readonly sourceHeight: number
  readonly encoding: 'class-rle-v1'
  readonly labels: readonly string[]
  readonly runs: readonly (readonly [number, number])[]
}

export interface OcrDetectionResult {
  readonly type: 'ocr_detection'
  readonly regions: readonly OcrRegion[]
}

export interface OcrRecognitionResult {
  readonly type: 'ocr_recognition'
  readonly text: string
  readonly confidence: number
}

export type StructuredResult = SegmentationResult | OcrDetectionResult | OcrRecognitionResult

export interface InferenceEnvelope {
  readonly schemaVersion: 2
  readonly taskId: string
  readonly revision: number
  readonly frameIndex: number
  readonly sourceWidth: number
  readonly sourceHeight: number
  readonly primaryInstance?: string
  readonly detections: readonly InferenceDetection[]
  readonly detectionsByInstance: Readonly<Record<string, readonly InferenceDetection[]>>
  readonly primaryResult?: StructuredResult
  readonly structuredResults: Readonly<Record<string, StructuredResult>>
  readonly analytics: Readonly<Record<string, unknown>>
  readonly media: Readonly<Record<string, unknown>>
  readonly ptsSeconds: number
}

export type DecodeResult =
  | { readonly ok: true; readonly value: InferenceEnvelope }
  | { readonly ok: false; readonly code: MetadataDiagnostic }

class ContractFailure extends Error {
  constructor(readonly code: MetadataDiagnostic) {
    super(code)
  }
}

const textEncoder = new TextEncoder()

export function decodeInferenceSei(
  payload: RawSeiPayload,
  expectedTaskId: string,
  expectedRevision: number,
): DecodeResult {
  if (payload.type !== 5) return failure('unsupported_sei_type')
  if (!equalBytes(payload.uuid, RKNODE_SEI_UUID)) return failure('foreign_uuid')
  if (payload.userData.byteLength > MAX_SEI_PAYLOAD_BYTES) return failure('payload_too_large')
  if (!Number.isFinite(payload.ptsSeconds) || payload.ptsSeconds < 0) return failure('invalid_pts')

  let text: string
  try {
    let end = payload.userData.byteLength
    while (end > 0 && payload.userData[end - 1] === 0) end -= 1
    text = new TextDecoder('utf-8', { fatal: true }).decode(payload.userData.subarray(0, end))
  } catch {
    return failure('invalid_utf8')
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    return failure('invalid_json')
  }
  try {
    return {
      ok: true,
      value: parseEnvelope(parsed, expectedTaskId, expectedRevision, payload.ptsSeconds),
    }
  } catch (reason) {
    return failure(reason instanceof ContractFailure ? reason.code : 'invalid_envelope')
  }
}

function parseEnvelope(
  value: unknown,
  expectedTaskId: string,
  expectedRevision: number,
  ptsSeconds: number,
): InferenceEnvelope {
  const root = objectValue(value)
  if (root.schema_version !== 2) throw new ContractFailure('unsupported_schema')
  if (root.task_id !== expectedTaskId) throw new ContractFailure('task_mismatch')
  const revision = integer(root.revision)
  if (revision !== expectedRevision) throw new ContractFailure('revision_mismatch')
  const sourceWidth = boundedDimension(root.width, MAX_SOURCE_WIDTH)
  const sourceHeight = boundedDimension(root.height, MAX_SOURCE_HEIGHT)
  const detections = parseDetections(root.detections ?? [])
  const detectionsByInstance: Record<string, readonly InferenceDetection[]> = {}
  const detectionResults = optionalObject(root.detection_results)
  for (const [name, result] of Object.entries(detectionResults)) {
    detectionsByInstance[name] = parseDetections(result)
  }
  const structuredResults: Record<string, StructuredResult> = {}
  for (const [name, result] of Object.entries(optionalObject(root.structured_results))) {
    const entry = objectValue(result)
    structuredResults[name] = parseStructuredResult(entry.type, entry.result)
  }
  const primaryResult = root.result_type === undefined
    ? undefined
    : parseStructuredResult(root.result_type, root.result)
  const primaryInstance = optionalBoundedText(root.primary_instance)
  const envelope: InferenceEnvelope = {
    schemaVersion: 2,
    taskId: expectedTaskId,
    revision,
    frameIndex: nonNegativeInteger(root.frame_index ?? root.index ?? 0),
    sourceWidth,
    sourceHeight,
    ...(primaryInstance === undefined ? {} : { primaryInstance }),
    detections: Object.freeze(detections),
    detectionsByInstance: Object.freeze(detectionsByInstance),
    ...(primaryResult === undefined ? {} : { primaryResult }),
    structuredResults: Object.freeze(structuredResults),
    analytics: Object.freeze(optionalObject(root.analytics)),
    media: Object.freeze(optionalObject(root.media)),
    ptsSeconds,
  }
  return Object.freeze(envelope)
}

function parseDetections(value: unknown): InferenceDetection[] {
  const items = boundedArray(value)
  return items.map((raw) => {
    const item = objectValue(raw)
    const detection: InferenceDetection = {
      x: finiteNumber(item.x),
      y: finiteNumber(item.y),
      width: nonNegativeNumber(item.w),
      height: nonNegativeNumber(item.h),
      confidence: finiteNumber(item.confidence),
      classId: nonNegativeInteger(item.class_id),
      label: boundedText(item.label ?? ''),
      ...optionalIntegerField(item.track_id, 'trackId'),
      ...optionalIntegerField(item.parent_track_id, 'parentTrackId'),
      ...optionalIntegerField(item.parent_detection_index, 'parentDetectionIndex'),
      ...(item.parent_instance === undefined
        ? {}
        : { parentInstance: boundedText(item.parent_instance) }),
    }
    return Object.freeze(detection)
  })
}

function parseStructuredResult(type: unknown, raw: unknown): StructuredResult {
  const result = objectValue(raw)
  if (type === 'segmentation') return parseSegmentation(result)
  if (type === 'ocr_detection') {
    const regions = boundedArray(result.regions ?? []).map(parseOcrRegion)
    return Object.freeze({ type, regions: Object.freeze(regions) })
  }
  if (type === 'ocr_recognition') {
    return Object.freeze({
      type,
      text: boundedText(result.text),
      confidence: finiteNumber(result.confidence),
    })
  }
  throw new ContractFailure('unsupported_result')
}

function parseSegmentation(result: Record<string, unknown>): SegmentationResult {
  try {
    const width = boundedDimension(result.width, MAX_SOURCE_WIDTH)
    const height = boundedDimension(result.height, MAX_SOURCE_HEIGHT)
    const sourceWidth = boundedDimension(result.source_width, MAX_SOURCE_WIDTH)
    const sourceHeight = boundedDimension(result.source_height, MAX_SOURCE_HEIGHT)
    if (result.encoding !== 'class-rle-v1') throw new ContractFailure('invalid_segmentation')
    const labels = boundedArray(result.labels).map(boundedText)
    const rawRuns = arrayValue(result.runs)
    if (rawRuns.length > MAX_SEGMENTATION_RUNS) throw new ContractFailure('too_many_runs')
    let pixels = 0
    const runs = rawRuns.map((raw): readonly [number, number] => {
      if (!Array.isArray(raw) || raw.length !== 2) throw new ContractFailure('invalid_segmentation')
      const classId = nonNegativeInteger(raw[0])
      const count = integer(raw[1])
      if (count <= 0) throw new ContractFailure('invalid_segmentation')
      pixels += count
      if (!Number.isSafeInteger(pixels) || pixels > width * height) {
        throw new ContractFailure('invalid_segmentation')
      }
      return Object.freeze([classId, count] as const)
    })
    if (pixels !== width * height || width !== sourceWidth || height !== sourceHeight) {
      throw new ContractFailure('invalid_segmentation')
    }
    return Object.freeze({
      type: 'segmentation',
      width,
      height,
      sourceWidth,
      sourceHeight,
      encoding: 'class-rle-v1',
      labels: Object.freeze(labels),
      runs: Object.freeze(runs),
    })
  } catch (reason) {
    if (reason instanceof ContractFailure && reason.code === 'too_many_runs') throw reason
    if (reason instanceof ContractFailure && reason.code === 'text_too_long') throw reason
    throw new ContractFailure('invalid_segmentation')
  }
}

function parseOcrRegion(raw: unknown): OcrRegion {
  const value = objectValue(raw)
  const points = boundedArray(value.points).map((rawPoint) => {
    if (!Array.isArray(rawPoint) || rawPoint.length !== 2) throw new ContractFailure('invalid_envelope')
    return Object.freeze({ x: finiteNumber(rawPoint[0]), y: finiteNumber(rawPoint[1]) })
  })
  if (points.length < 3 || points.length > 16) throw new ContractFailure('invalid_envelope')
  const text = optionalBoundedText(value.text)
  return Object.freeze({
    points: Object.freeze(points),
    confidence: finiteNumber(value.confidence),
    ...(text === undefined ? {} : { text }),
  })
}

function failure(code: MetadataDiagnostic): DecodeResult {
  return { ok: false, code }
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  return left.byteLength === right.byteLength && left.every((value, index) => value === right[index])
}

function objectValue(value: unknown): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new ContractFailure('invalid_envelope')
  }
  return value as Record<string, unknown>
}

function optionalObject(value: unknown): Record<string, unknown> {
  return value === undefined ? {} : objectValue(value)
}

function arrayValue(value: unknown): unknown[] {
  if (!Array.isArray(value)) throw new ContractFailure('invalid_envelope')
  return value
}

function boundedArray(value: unknown): unknown[] {
  const items = arrayValue(value)
  if (items.length > MAX_RESULT_ITEMS) throw new ContractFailure('too_many_results')
  return items
}

function finiteNumber(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new ContractFailure('invalid_envelope')
  return value
}

function nonNegativeNumber(value: unknown): number {
  const result = finiteNumber(value)
  if (result < 0) throw new ContractFailure('invalid_envelope')
  return result
}

function integer(value: unknown): number {
  const result = finiteNumber(value)
  if (!Number.isSafeInteger(result)) throw new ContractFailure('invalid_envelope')
  return result
}

function nonNegativeInteger(value: unknown): number {
  const result = integer(value)
  if (result < 0) throw new ContractFailure('invalid_envelope')
  return result
}

function boundedDimension(value: unknown, maximum: number): number {
  const result = integer(value)
  if (result <= 0 || result > maximum) throw new ContractFailure('invalid_dimensions')
  return result
}

function boundedText(value: unknown): string {
  if (typeof value !== 'string') throw new ContractFailure('invalid_envelope')
  if (textEncoder.encode(value).byteLength > MAX_TEXT_BYTES) throw new ContractFailure('text_too_long')
  return value
}

function optionalBoundedText(value: unknown): string | undefined {
  return value === undefined ? undefined : boundedText(value)
}

function optionalIntegerField(
  value: unknown,
  name: 'trackId' | 'parentTrackId' | 'parentDetectionIndex',
): Partial<Record<typeof name, number>> {
  return value === undefined ? {} : { [name]: integer(value) } as Partial<Record<typeof name, number>>
}
