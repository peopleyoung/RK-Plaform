import mpegts from 'mpegts.js'

import type { RawSeiPayload } from './contracts'

interface UpstreamSei {
  readonly type?: unknown
  readonly uuid?: unknown
  readonly user_data?: unknown
  readonly pts?: unknown
}

export type MediaErrorKind = 'network' | 'media' | 'other'

export interface NormalizedMediaError {
  readonly kind: MediaErrorKind
  readonly code: string
}

export interface MediaCodecInfo {
  readonly videoCodec: string
}

type SeiListener = (payload: RawSeiPayload) => void
type ErrorListener = (error: NormalizedMediaError) => void
type MediaInfoListener = (info: MediaCodecInfo) => void

const events = mpegts.Events as typeof mpegts.Events & { readonly SEI_ARRIVED: string }

export function normalizeMpegtsSei(value: unknown): RawSeiPayload | null {
  if (typeof value !== 'object' || value === null) return null
  const raw = value as UpstreamSei
  if (typeof raw.type !== 'number' || !(raw.uuid instanceof Uint8Array)
      || !(raw.user_data instanceof Uint8Array) || typeof raw.pts !== 'number'
      || !Number.isFinite(raw.pts)) return null
  return {
    type: raw.type,
    uuid: raw.uuid,
    userData: raw.user_data,
    ptsSeconds: raw.pts / 1000,
  }
}

export function isMpegtsSupported(): boolean {
  return mpegts.isSupported()
}

export class MpegtsAdapter {
  private readonly player: ReturnType<typeof mpegts.createPlayer>
  private readonly seiListeners = new Set<SeiListener>()
  private readonly errorListeners = new Set<ErrorListener>()
  private readonly mediaInfoListeners = new Set<MediaInfoListener>()
  private destroyed = false

  private readonly handleSei = (value: unknown) => {
    const payload = normalizeMpegtsSei(value)
    if (payload) this.seiListeners.forEach((listener) => listener(payload))
  }

  private readonly handleError = (type: unknown, detail: unknown) => {
    const kind: MediaErrorKind = type === mpegts.ErrorTypes.NETWORK_ERROR
      ? 'network'
      : type === mpegts.ErrorTypes.MEDIA_ERROR ? 'media' : 'other'
    const code = typeof detail === 'string' && /^[A-Za-z0-9_.-]{1,80}$/.test(detail)
      ? detail
      : 'unknown'
    this.errorListeners.forEach((listener) => listener({ kind, code }))
  }

  private readonly handleMediaInfo = (value: unknown) => {
    if (typeof value !== 'object' || value === null) return
    const codec = (value as { videoCodec?: unknown }).videoCodec
    if (typeof codec === 'string' && codec.length <= 80) {
      this.mediaInfoListeners.forEach((listener) => listener({ videoCodec: codec }))
    }
  }

  constructor(streamUrl: string) {
    this.player = mpegts.createPlayer(
      { type: 'flv', isLive: true, url: streamUrl },
      {
        enableWorker: typeof window === 'undefined'
          || window.__RKNODE_CONFIG__?.mediaWorkerEnabled !== false,
        enableStashBuffer: false,
        lazyLoad: false,
        liveBufferLatencyChasing: true,
      },
    )
    this.player.on(events.SEI_ARRIVED, this.handleSei)
    this.player.on(mpegts.Events.ERROR, this.handleError)
    this.player.on(mpegts.Events.MEDIA_INFO, this.handleMediaInfo)
  }

  onSei(listener: SeiListener): () => void {
    this.seiListeners.add(listener)
    return () => this.seiListeners.delete(listener)
  }

  onError(listener: ErrorListener): () => void {
    this.errorListeners.add(listener)
    return () => this.errorListeners.delete(listener)
  }

  onMediaInfo(listener: MediaInfoListener): () => void {
    this.mediaInfoListeners.add(listener)
    return () => this.mediaInfoListeners.delete(listener)
  }

  attach(video: HTMLVideoElement): void {
    this.player.attachMediaElement(video)
  }

  load(): void {
    this.player.load()
  }

  async play(): Promise<void> {
    await this.player.play()
  }

  destroy(): void {
    if (this.destroyed) return
    this.destroyed = true
    this.player.off(events.SEI_ARRIVED, this.handleSei)
    this.player.off(mpegts.Events.ERROR, this.handleError)
    this.player.off(mpegts.Events.MEDIA_INFO, this.handleMediaInfo)
    this.player.pause()
    this.player.unload()
    this.player.detachMediaElement()
    this.player.destroy()
    this.seiListeners.clear()
    this.errorListeners.clear()
    this.mediaInfoListeners.clear()
  }
}

export function createMpegtsAdapter(streamUrl: string): MpegtsAdapter {
  return new MpegtsAdapter(streamUrl)
}
