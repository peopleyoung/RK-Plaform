import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.hoisted(() => {
  Object.defineProperty(globalThis, 'self', { configurable: true, value: globalThis })
})

import type { InferencePlaybackSession, PreviewCapability } from '../types'
import type { RawSeiPayload } from '../media/contracts'
import {
  InferenceStreamController,
  inferenceStreamOptionsKey,
  type InferenceStreamDependencies,
  type StreamAdapter,
  type VideoFrameTarget,
} from '../media/useInferenceStreamPlayer'

const available: PreviewCapability = { state: 'available', reason: null }

function descriptor(overrides: Partial<InferencePlaybackSession> = {}): InferencePlaybackSession {
  return {
    streamUrl: 'ws://media.lan:8081/live/task.flv?playToken=secret',
    expiresAt: '2026-08-21T12:01:00Z',
    taskId: 'task-1',
    revision: 7,
    gatewayId: 'gateway-1',
    app: 'live',
    streamName: 'task-1',
    codec: 'h264',
    reconnectMs: 1000,
    ...overrides,
  }
}

class FakeAdapter implements StreamAdapter {
  readonly attach = vi.fn()
  readonly load = vi.fn()
  readonly play = vi.fn(async () => undefined)
  readonly destroy = vi.fn()
  private sei?: (payload: RawSeiPayload) => void
  private error?: (error: { kind: 'network' | 'media' | 'other'; code: string }) => void
  private mediaInfo?: (info: { videoCodec: string }) => void

  onSei(listener: (payload: RawSeiPayload) => void) {
    this.sei = listener
    return () => { this.sei = undefined }
  }

  onError(listener: (error: { kind: 'network' | 'media' | 'other'; code: string }) => void) {
    this.error = listener
    return () => { this.error = undefined }
  }

  onMediaInfo(listener: (info: { videoCodec: string }) => void) {
    this.mediaInfo = listener
    return () => { this.mediaInfo = undefined }
  }

  emitError(code = 'network') {
    this.error?.({ kind: 'network', code })
  }

  emitMediaInfo(videoCodec: string) {
    this.mediaInfo?.({ videoCodec })
  }

  emitSei(payload: RawSeiPayload) {
    this.sei?.(payload)
  }
}

function videoTarget() {
  let callback: ((now: number, metadata: { mediaTime: number }) => void) | undefined
  const target: VideoFrameTarget = {
    requestVideoFrameCallback: vi.fn((next) => {
      callback = next
      return 1
    }),
    cancelVideoFrameCallback: vi.fn(),
  }
  return {
    target,
    frame(mediaTime: number) {
      const current = callback
      callback = undefined
      current?.(0, { mediaTime })
    },
  }
}

function harness(options: {
  protocol?: 'http:' | 'https:'
  capability?: PreviewCapability
  sessions?: Array<InferencePlaybackSession | Error>
  h265Supported?: boolean
} = {}) {
  const sessions = [...(options.sessions ?? [descriptor()])]
  const adapters: FakeAdapter[] = []
  const fetchPlaybackSession = vi.fn(async () => {
    const next = sessions.shift() ?? descriptor()
    if (next instanceof Error) throw next
    return next
  })
  const dependencies: InferenceStreamDependencies = {
    fetchPlaybackSession,
    createAdapter: vi.fn(() => {
      const adapter = new FakeAdapter()
      adapters.push(adapter)
      return adapter
    }),
    isPlaybackSupported: () => true,
    isCodecSupported: () => options.h265Supported ?? false,
    pageProtocol: options.protocol ?? 'http:',
    decodeSei: vi.fn(() => ({ ok: false as const, code: 'invalid_json' as const })),
    render: vi.fn(),
    clear: vi.fn(),
    decodeSegmentation: vi.fn(async () => null),
    destroySegmentationDecoder: vi.fn(),
  }
  const states: string[] = []
  const controller = new InferenceStreamController({
    taskId: 'task-1',
    revision: 7,
    capability: options.capability ?? available,
    areas: [],
    lines: [],
  }, dependencies, (snapshot) => states.push(snapshot.state))
  const video = videoTarget()
  controller.attach(video.target, null)
  return { adapters, controller, dependencies, fetchPlaybackSession, states, video }
}

async function flush(): Promise<void> {
  await Promise.resolve()
  await Promise.resolve()
}

describe('InferenceStreamPlayer state machine', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('rejects HTTPS to plain WebSocket before constructing mpegts', async () => {
    const test = harness({ protocol: 'https:' })
    await test.controller.start()
    expect(test.controller.snapshot.state).toBe('unsupported')
    expect(test.controller.snapshot.diagnostic).toBe('mixed_content')
    expect(test.dependencies.createAdapter).not.toHaveBeenCalled()
  })

  it('maps task capability to a deterministic non-playing state', async () => {
    const test = harness({ capability: { state: 'migration_required', reason: 'media_migration_required' } })
    await test.controller.start()
    expect(test.controller.snapshot).toEqual({
      state: 'unsupported',
      diagnostic: 'media_migration_required',
    })
    expect(test.fetchPlaybackSession).not.toHaveBeenCalled()
  })

  it('gets a fresh descriptor for every 1, 2, 4, 4 second reconnect', async () => {
    const test = harness({ sessions: [descriptor(), descriptor(), descriptor(), descriptor(), descriptor()] })
    await test.controller.start()
    expect(test.fetchPlaybackSession).toHaveBeenCalledTimes(1)

    for (const [index, delay] of [1000, 2000, 4000, 4000].entries()) {
      test.adapters[index].emitError()
      await vi.advanceTimersByTimeAsync(delay - 1)
      expect(test.fetchPlaybackSession).toHaveBeenCalledTimes(index + 1)
      await vi.advanceTimersByTimeAsync(1)
      await flush()
      expect(test.fetchPlaybackSession).toHaveBeenCalledTimes(index + 2)
    }

    test.controller.stop()
    test.adapters.at(-1)?.emitError()
    await vi.advanceTimersByTimeAsync(8000)
    expect(test.fetchPlaybackSession).toHaveBeenCalledTimes(5)
  })

  it('maps authorization, missing publication, and unsupported H.265 explicitly', async () => {
    const unauthorized = Object.assign(new Error('denied'), { status: 403, code: 'forbidden' })
    const authTest = harness({ sessions: [unauthorized] })
    await authTest.controller.start()
    expect(authTest.controller.snapshot.state).toBe('unauthorized')

    const missing = Object.assign(new Error('not published'), { status: 409, code: 'stream_not_published' })
    const waitingTest = harness({ sessions: [missing] })
    await waitingTest.controller.start()
    expect(waitingTest.controller.snapshot.state).toBe('waiting_publish')

    const codecTest = harness({ sessions: [descriptor({ codec: 'h265' })] })
    await codecTest.controller.start()
    expect(codecTest.controller.snapshot.state).toBe('codec_unsupported')
    expect(codecTest.dependencies.createAdapter).not.toHaveBeenCalled()
  })

  it('uses video-frame mediaTime for synchronization and clears stale metadata', async () => {
    const test = harness()
    const envelope = {
      schemaVersion: 2 as const,
      taskId: 'task-1',
      revision: 7,
      frameIndex: 1,
      sourceWidth: 640,
      sourceHeight: 360,
      detections: [],
      detectionsByInstance: {},
      structuredResults: {},
      analytics: {},
      media: {},
      ptsSeconds: 3,
    }
    vi.mocked(test.dependencies.decodeSei).mockReturnValue({ ok: true, value: envelope })
    await test.controller.start()
    test.adapters[0].emitMediaInfo('avc1.640028')
    test.adapters[0].emitSei({ type: 5, uuid: new Uint8Array(), userData: new Uint8Array(), ptsSeconds: 3 })

    test.video.frame(3)
    expect(test.dependencies.render).toHaveBeenCalledWith(envelope, null)
    expect(test.controller.snapshot.state).toBe('live')
    test.video.frame(4.01)
    expect(test.dependencies.clear).toHaveBeenCalled()
    expect(test.controller.snapshot.state).toBe('metadata_degraded')
  })

  it('keeps independent tile controllers isolated', async () => {
    const first = harness()
    const second = harness()
    await Promise.all([first.controller.start(), second.controller.start()])
    first.adapters[0].emitError()
    expect(first.controller.snapshot.state).toBe('reconnecting')
    expect(second.adapters[0].destroy).not.toHaveBeenCalled()
    expect(second.fetchPlaybackSession).toHaveBeenCalledTimes(1)
  })

  it('keeps the playback signature stable when polling replaces equivalent task objects', () => {
    const first: Parameters<typeof inferenceStreamOptionsKey>[0] = {
      taskId: 'task-1',
      revision: 7,
      capability: available,
      areas: [],
      lines: [],
    }
    const second = JSON.parse(JSON.stringify(first)) as typeof first

    expect(inferenceStreamOptionsKey(second)).toBe(inferenceStreamOptionsKey(first))
  })
})
