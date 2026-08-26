import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from 'react'

import { api, ApiError } from '../api/client'
import type { InferencePlaybackSession, PreviewCapability } from '../types'
import { decodeInferenceSei, type DecodeResult, type InferenceEnvelope, type RawSeiPayload, type SegmentationResult } from './contracts'
import { createMpegtsAdapter, isMpegtsSupported, type MediaCodecInfo, type NormalizedMediaError } from './mpegtsAdapter'
import { renderOverlay, type OverlayArea, type OverlayLine } from './renderOverlay'
import { SegmentationWorkerClient } from './segmentationWorkerClient'
import { MetadataSyncQueue } from './syncQueue'

export type InferencePlayerState =
  | 'unsupported'
  | 'waiting_publish'
  | 'connecting'
  | 'live'
  | 'metadata_degraded'
  | 'reconnecting'
  | 'unauthorized'
  | 'codec_unsupported'
  | 'stopped'

export interface InferencePlayerSnapshot {
  readonly state: InferencePlayerState
  readonly diagnostic: string | null
}

export interface VideoFrameTarget {
  requestVideoFrameCallback(callback: (now: number, metadata: { mediaTime: number }) => void): number
  cancelVideoFrameCallback(handle: number): void
}

export interface StreamAdapter {
  onSei(listener: (payload: RawSeiPayload) => void): () => void
  onError(listener: (error: NormalizedMediaError) => void): () => void
  onMediaInfo(listener: (info: MediaCodecInfo) => void): () => void
  attach(video: VideoFrameTarget): void
  load(): void
  play(): Promise<void>
  destroy(): void
}

export interface InferenceStreamDependencies {
  readonly fetchPlaybackSession: (taskId: string) => Promise<InferencePlaybackSession>
  readonly createAdapter: (streamUrl: string) => StreamAdapter
  readonly isPlaybackSupported: () => boolean
  readonly isCodecSupported: (codec: 'h264' | 'h265') => boolean
  readonly pageProtocol: 'http:' | 'https:'
  readonly decodeSei: (payload: RawSeiPayload, taskId: string, revision: number) => DecodeResult
  readonly render: (envelope: InferenceEnvelope, segmentation: CanvasImageSource | null) => void
  readonly clear: () => void
  readonly decodeSegmentation: (segmentation: SegmentationResult) => Promise<CanvasImageSource | null>
  readonly destroySegmentationDecoder: () => void
}

export interface InferenceStreamOptions {
  readonly taskId: string
  readonly revision: number
  readonly capability: PreviewCapability
  readonly areas: readonly OverlayArea[]
  readonly lines: readonly OverlayLine[]
}

/**
 * Task polling replaces the option object even when playback inputs are unchanged.
 * Keep the player mounted until a playback-relevant value actually changes.
 */
export function inferenceStreamOptionsKey(options: InferenceStreamOptions): string {
  return JSON.stringify({
    taskId: options.taskId,
    revision: options.revision,
    capability: options.capability,
    areas: options.areas,
    lines: options.lines,
  })
}

const RETRY_DELAYS_MS = [1000, 2000, 4000] as const
const FRAME_TOLERANCE_SECONDS = 1 / 30
const METADATA_STALE_SECONDS = 1

export class InferenceStreamController {
  private currentSnapshot: InferencePlayerSnapshot = { state: 'connecting', diagnostic: null }
  private video: VideoFrameTarget | null = null
  private adapter: StreamAdapter | null = null
  private readonly queue = new MetadataSyncQueue<InferenceEnvelope>()
  private retryTimer: ReturnType<typeof setTimeout> | null = null
  private retryIndex = 0
  private frameHandle: number | null = null
  private active = false
  private generation = 0
  private firstFrameMediaTime: number | null = null
  private lastMetadataMediaTime: number | null = null

  constructor(
    private readonly options: InferenceStreamOptions,
    private readonly dependencies: InferenceStreamDependencies,
    private readonly onSnapshot: (snapshot: InferencePlayerSnapshot) => void,
  ) {}

  get snapshot(): InferencePlayerSnapshot {
    return this.currentSnapshot
  }

  attach(video: VideoFrameTarget | null, _canvas: HTMLCanvasElement | null): void {
    this.video = video
  }

  async start(): Promise<void> {
    if (this.active) return
    const unavailable = capabilitySnapshot(this.options.capability)
    if (unavailable) {
      this.update(unavailable.state, unavailable.diagnostic)
      return
    }
    if (!this.video || !this.dependencies.isPlaybackSupported()) {
      this.update('unsupported', 'browser_unsupported')
      return
    }
    this.active = true
    await this.connect(false)
  }

  retryNow(): void {
    if (!this.active) return
    this.clearRetry()
    this.retryIndex = 0
    this.destroyAdapter()
    void this.connect(true)
  }

  stop(): void {
    this.active = false
    this.generation += 1
    this.clearRetry()
    this.cancelFrame()
    this.destroyAdapter()
    this.queue.clear()
    this.dependencies.clear()
    this.dependencies.destroySegmentationDecoder()
    this.update('stopped', null)
  }

  private async connect(reconnecting: boolean): Promise<void> {
    if (!this.active) return
    const generation = this.generation + 1
    this.generation = generation
    this.update(reconnecting ? 'reconnecting' : 'connecting', null)
    let descriptor: InferencePlaybackSession
    try {
      descriptor = await this.dependencies.fetchPlaybackSession(this.options.taskId)
    } catch (reason) {
      if (!this.isCurrent(generation)) return
      this.handleSessionError(reason)
      return
    }
    if (!this.isCurrent(generation)) return
    if (descriptor.taskId !== this.options.taskId || descriptor.revision !== this.options.revision) {
      this.update('unsupported', 'descriptor_mismatch')
      this.active = false
      return
    }
    if (this.dependencies.pageProtocol === 'https:' && descriptor.streamUrl.startsWith('ws://')) {
      this.update('unsupported', 'mixed_content')
      this.active = false
      return
    }
    if (descriptor.codec === 'h265' && !this.dependencies.isCodecSupported('h265')) {
      this.update('codec_unsupported', 'h265_unsupported')
      this.active = false
      return
    }
    this.destroyAdapter()
    const adapter = this.dependencies.createAdapter(descriptor.streamUrl)
    this.adapter = adapter
    adapter.onSei((payload) => this.receiveSei(payload, generation))
    adapter.onError((error) => this.handleAdapterError(error, generation))
    adapter.onMediaInfo((info) => this.handleMediaInfo(info, generation))
    adapter.attach(this.video as VideoFrameTarget)
    adapter.load()
    this.firstFrameMediaTime = null
    this.lastMetadataMediaTime = null
    this.scheduleFrame(generation)
    try {
      await adapter.play()
    } catch {
      if (this.isCurrent(generation)) this.handleDisconnect('play_failed')
    }
  }

  private receiveSei(payload: RawSeiPayload, generation: number): void {
    if (!this.isCurrent(generation)) return
    const decoded = this.dependencies.decodeSei(payload, this.options.taskId, this.options.revision)
    if (!decoded.ok) {
      if (this.currentSnapshot.state !== 'live') this.update('metadata_degraded', decoded.code)
      return
    }
    this.queue.enqueue(decoded.value.ptsSeconds, decoded.value)
    this.update('live', null)
    this.retryIndex = 0
  }

  private handleMediaInfo(info: MediaCodecInfo, generation: number): void {
    if (!this.isCurrent(generation)) return
    const codec = info.videoCodec.toLowerCase()
    if ((codec.includes('hvc1') || codec.includes('hev1') || codec.includes('h265') || codec.includes('hevc'))
        && !this.dependencies.isCodecSupported('h265')) {
      this.destroyAdapter()
      this.active = false
      this.update('codec_unsupported', 'h265_unsupported')
    }
  }

  private handleAdapterError(error: NormalizedMediaError, generation: number): void {
    if (!this.isCurrent(generation)) return
    this.handleDisconnect(`${error.kind}_${error.code}`)
  }

  private handleDisconnect(diagnostic: string): void {
    this.cancelFrame()
    this.destroyAdapter()
    this.dependencies.clear()
    this.update('reconnecting', diagnostic)
    this.scheduleRetry()
  }

  private handleSessionError(reason: unknown): void {
    const status = errorNumber(reason, 'status')
    const code = errorString(reason, 'code') ?? 'playback_session_failed'
    if (status === 401 || status === 403) {
      this.active = false
      this.update('unauthorized', code)
      return
    }
    if (status === 409 || code === 'stream_not_published') {
      this.update('waiting_publish', code)
    } else {
      this.update('reconnecting', code)
    }
    this.scheduleRetry()
  }

  private scheduleRetry(): void {
    if (!this.active || this.retryTimer !== null) return
    const delay = RETRY_DELAYS_MS[Math.min(this.retryIndex, RETRY_DELAYS_MS.length - 1)]
    this.retryIndex += 1
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null
      void this.connect(true)
    }, delay)
  }

  private scheduleFrame(generation: number): void {
    if (!this.video || !this.isCurrent(generation)) return
    this.frameHandle = this.video.requestVideoFrameCallback((_now, metadata) => {
      this.frameHandle = null
      this.onVideoFrame(metadata.mediaTime, generation)
      this.scheduleFrame(generation)
    })
  }

  private onVideoFrame(mediaTime: number, generation: number): void {
    if (!this.isCurrent(generation) || !Number.isFinite(mediaTime)) return
    if (this.firstFrameMediaTime === null) this.firstFrameMediaTime = mediaTime
    const selection = this.queue.select(mediaTime, FRAME_TOLERANCE_SECONDS)
    if (selection.value) {
      this.lastMetadataMediaTime = mediaTime
      this.renderEnvelope(selection.value, generation)
      this.update('live', null)
      return
    }
    const reference = this.lastMetadataMediaTime ?? this.firstFrameMediaTime
    if (reference !== null && mediaTime - reference > METADATA_STALE_SECONDS) {
      this.dependencies.clear()
      this.update('metadata_degraded', 'metadata_stale')
    }
  }

  private renderEnvelope(envelope: InferenceEnvelope, generation: number): void {
    const segmentation = segmentationResult(envelope)
    if (!segmentation) {
      this.dependencies.render(envelope, null)
      return
    }
    void this.dependencies.decodeSegmentation(segmentation).then((raster) => {
      if (this.isCurrent(generation)) this.dependencies.render(envelope, raster)
    }).catch(() => {
      if (this.isCurrent(generation)) this.update('metadata_degraded', 'segmentation_decode_failed')
    })
  }

  private destroyAdapter(): void {
    this.adapter?.destroy()
    this.adapter = null
  }

  private cancelFrame(): void {
    if (this.video && this.frameHandle !== null) this.video.cancelVideoFrameCallback(this.frameHandle)
    this.frameHandle = null
  }

  private clearRetry(): void {
    if (this.retryTimer !== null) clearTimeout(this.retryTimer)
    this.retryTimer = null
  }

  private isCurrent(generation: number): boolean {
    return this.active && generation === this.generation
  }

  private update(state: InferencePlayerState, diagnostic: string | null): void {
    if (this.currentSnapshot.state === state && this.currentSnapshot.diagnostic === diagnostic) return
    this.currentSnapshot = { state, diagnostic }
    this.onSnapshot(this.currentSnapshot)
  }
}

export interface UseInferenceStreamPlayerResult extends InferencePlayerSnapshot {
  readonly videoRef: RefObject<HTMLVideoElement | null>
  readonly canvasRef: RefObject<HTMLCanvasElement | null>
  readonly retry: () => void
  readonly stop: () => void
}

export function useInferenceStreamPlayer(options: InferenceStreamOptions): UseInferenceStreamPlayerResult {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const controllerRef = useRef<InferenceStreamController | null>(null)
  const [snapshot, setSnapshot] = useState<InferencePlayerSnapshot>({ state: 'connecting', diagnostic: null })

  const geometry = useMemo(() => ({ areas: options.areas, lines: options.lines }), [options.areas, options.lines])
  const optionsKey = useMemo(() => inferenceStreamOptionsKey(options), [options])

  useEffect(() => {
    let mounted = true
    let segmentationDecoder: SegmentationWorkerClient | null = null
    const clear = () => {
      const context = canvasRef.current?.getContext('2d')
      if (context) context.clearRect(0, 0, context.canvas.width, context.canvas.height)
    }
    const dependencies: InferenceStreamDependencies = {
      fetchPlaybackSession: api.inferencePlaybackSession,
      createAdapter: (streamUrl) => browserAdapter(streamUrl),
      isPlaybackSupported: isMpegtsSupported,
      isCodecSupported: browserCodecSupported,
      pageProtocol: window.location.protocol === 'https:' ? 'https:' : 'http:',
      decodeSei: decodeInferenceSei,
      render: (envelope, segmentation) => {
        const context = canvasRef.current?.getContext('2d')
        if (context) renderOverlay(context, envelope, { segmentation, ...geometry })
      },
      clear,
      decodeSegmentation: async (segmentation) => {
        segmentationDecoder ??= new SegmentationWorkerClient()
        return await segmentationDecoder.decode(segmentation)
      },
      destroySegmentationDecoder: () => {
        segmentationDecoder?.destroy()
        segmentationDecoder = null
      },
    }
    const controller = new InferenceStreamController(options, dependencies, (next) => {
      if (mounted) setSnapshot(next)
    })
    controllerRef.current = controller
    controller.attach(videoRef.current, canvasRef.current)
    void controller.start()
    return () => {
      mounted = false
      controller.stop()
      if (controllerRef.current === controller) controllerRef.current = null
    }
  }, [optionsKey])

  const retry = useCallback(() => controllerRef.current?.retryNow(), [])
  const stop = useCallback(() => controllerRef.current?.stop(), [])
  return { ...snapshot, videoRef, canvasRef, retry, stop }
}

function browserAdapter(streamUrl: string): StreamAdapter {
  const adapter = createMpegtsAdapter(streamUrl)
  return {
    onSei: (listener) => adapter.onSei(listener),
    onError: (listener) => adapter.onError(listener),
    onMediaInfo: (listener) => adapter.onMediaInfo(listener),
    attach: (target) => {
      if (!(target instanceof HTMLVideoElement)) throw new Error('invalid_video_target')
      adapter.attach(target)
    },
    load: () => adapter.load(),
    play: () => adapter.play(),
    destroy: () => adapter.destroy(),
  }
}

function browserCodecSupported(codec: 'h264' | 'h265'): boolean {
  if (codec === 'h264') return true
  if (typeof MediaSource === 'undefined') return false
  return [
    'video/mp4; codecs="hvc1.1.6.L120.B0"',
    'video/mp4; codecs="hev1.1.6.L120.B0"',
  ].some((mime) => MediaSource.isTypeSupported(mime))
}

function capabilitySnapshot(capability: PreviewCapability): InferencePlayerSnapshot | null {
  if (capability.state === 'available') return null
  if (capability.state === 'gateway_offline') {
    return { state: 'waiting_publish', diagnostic: capability.reason ?? 'media_gateway_offline' }
  }
  return { state: 'unsupported', diagnostic: capability.reason ?? capability.state }
}

function segmentationResult(envelope: InferenceEnvelope): SegmentationResult | null {
  if (envelope.primaryResult?.type === 'segmentation') return envelope.primaryResult
  return Object.values(envelope.structuredResults).find(
    (result): result is SegmentationResult => result.type === 'segmentation',
  ) ?? null
}

function errorNumber(reason: unknown, field: string): number | null {
  if (reason instanceof ApiError && field === 'status') return reason.status
  if (typeof reason !== 'object' || reason === null) return null
  const value = Reflect.get(reason, field)
  return typeof value === 'number' ? value : null
}

function errorString(reason: unknown, field: string): string | null {
  if (reason instanceof ApiError && field === 'code') return reason.code
  if (typeof reason !== 'object' || reason === null) return null
  const value = Reflect.get(reason, field)
  return typeof value === 'string' && /^[A-Za-z0-9_.-]{1,80}$/.test(value) ? value : null
}
