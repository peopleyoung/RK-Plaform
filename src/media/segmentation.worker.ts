import {
  MAX_SEGMENTATION_RUNS,
  MAX_SOURCE_HEIGHT,
  MAX_SOURCE_WIDTH,
  type SegmentationResult,
} from './contracts'
import { pascalColor } from './pascalPalette'

export type SegmentationDiagnostic =
  | 'invalid_dimensions'
  | 'too_many_runs'
  | 'invalid_class_id'
  | 'invalid_segmentation'

export type SegmentationRasterResult =
  | {
      readonly ok: true
      readonly width: number
      readonly height: number
      readonly rgba: Uint8ClampedArray<ArrayBuffer>
    }
  | { readonly ok: false; readonly code: SegmentationDiagnostic }

interface WorkerRequest {
  readonly id: number
  readonly segmentation: SegmentationResult
}

export function decodeSegmentationRgba(
  segmentation: SegmentationResult,
): SegmentationRasterResult {
  const { width, height, sourceWidth, sourceHeight, runs } = segmentation
  if (!Number.isInteger(width) || !Number.isInteger(height)
      || width <= 0 || height <= 0 || width > MAX_SOURCE_WIDTH
      || height > MAX_SOURCE_HEIGHT || width !== sourceWidth || height !== sourceHeight) {
    return { ok: false, code: 'invalid_dimensions' }
  }
  if (runs.length > MAX_SEGMENTATION_RUNS) return { ok: false, code: 'too_many_runs' }
  const rgba = new Uint8ClampedArray(width * height * 4)
  let pixel = 0
  for (const run of runs) {
    if (!Array.isArray(run) || run.length !== 2) return { ok: false, code: 'invalid_segmentation' }
    const [classId, count] = run
    if (!Number.isInteger(classId) || classId < 0 || classId > 255) {
      return { ok: false, code: 'invalid_class_id' }
    }
    if (!Number.isInteger(count) || count <= 0 || pixel + count > width * height) {
      return { ok: false, code: 'invalid_segmentation' }
    }
    const [red, green, blue] = pascalColor(classId)
    for (let index = 0; index < count; index += 1) {
      const offset = (pixel + index) * 4
      rgba[offset] = red
      rgba[offset + 1] = green
      rgba[offset + 2] = blue
      rgba[offset + 3] = classId === 0 ? 0 : 255
    }
    pixel += count
  }
  if (pixel !== width * height) return { ok: false, code: 'invalid_segmentation' }
  return { ok: true, width, height, rgba }
}

const scope = globalThis as typeof globalThis & {
  postMessage?: (message: unknown, transfer?: Transferable[]) => void
  onmessage?: ((event: MessageEvent<WorkerRequest>) => void) | null
}

if (typeof document === 'undefined' && typeof scope.postMessage === 'function') {
  scope.onmessage = (event) => {
    const result = decodeSegmentationRgba(event.data.segmentation)
    if (!result.ok) {
      scope.postMessage?.({ id: event.data.id, ...result })
      return
    }
    const imageData = new ImageData(result.rgba, result.width, result.height)
    if (typeof createImageBitmap === 'function') {
      void createImageBitmap(imageData).then((bitmap) => {
        scope.postMessage?.({ id: event.data.id, ok: true, bitmap }, [bitmap])
      })
      return
    }
    scope.postMessage?.(
      {
        id: event.data.id,
        ok: true,
        width: result.width,
        height: result.height,
        rgba: result.rgba.buffer,
      },
      [result.rgba.buffer],
    )
  }
}
