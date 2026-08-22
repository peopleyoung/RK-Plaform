import type { SegmentationResult } from './contracts'
import type { SegmentationDiagnostic } from './segmentation.worker'

interface WorkerSuccess {
  readonly id: number
  readonly ok: true
  readonly bitmap?: ImageBitmap
  readonly width?: number
  readonly height?: number
  readonly rgba?: ArrayBuffer
}

interface WorkerFailure {
  readonly id: number
  readonly ok: false
  readonly code: SegmentationDiagnostic
}

type WorkerResponse = WorkerSuccess | WorkerFailure

interface PendingRequest {
  readonly resolve: (value: CanvasImageSource) => void
  readonly reject: (reason: Error) => void
}

export class SegmentationWorkerClient {
  private readonly worker = new Worker(
    new URL('./segmentation.worker.ts', import.meta.url),
    { type: 'module' },
  )
  private readonly pending = new Map<number, PendingRequest>()
  private nextId = 1

  constructor() {
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const request = this.pending.get(event.data.id)
      if (!request) return
      this.pending.delete(event.data.id)
      if (!event.data.ok) {
        request.reject(new Error(event.data.code))
        return
      }
      if (event.data.bitmap) {
        request.resolve(event.data.bitmap)
        return
      }
      if (!event.data.rgba || !event.data.width || !event.data.height) {
        request.reject(new Error('invalid_worker_response'))
        return
      }
      const imageData = new ImageData(
        new Uint8ClampedArray(event.data.rgba),
        event.data.width,
        event.data.height,
      )
      if (typeof createImageBitmap === 'function') {
        void createImageBitmap(imageData).then(request.resolve, request.reject)
        return
      }
      const canvas = document.createElement('canvas')
      canvas.width = event.data.width
      canvas.height = event.data.height
      canvas.getContext('2d')?.putImageData(imageData, 0, 0)
      request.resolve(canvas)
    }
  }

  decode(segmentation: SegmentationResult): Promise<CanvasImageSource> {
    const id = this.nextId
    this.nextId += 1
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      this.worker.postMessage({ id, segmentation })
    })
  }

  destroy(): void {
    this.worker.terminate()
    this.pending.forEach(({ reject }) => reject(new Error('worker_destroyed')))
    this.pending.clear()
  }
}
