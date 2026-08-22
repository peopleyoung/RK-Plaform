import { beforeEach, describe, expect, it, vi } from 'vitest'

const { player, createPlayer } = vi.hoisted(() => {
  const value = {
    on: vi.fn(),
    off: vi.fn(),
    attachMediaElement: vi.fn(),
    detachMediaElement: vi.fn(),
    load: vi.fn(),
    unload: vi.fn(),
    play: vi.fn(() => Promise.resolve()),
    pause: vi.fn(),
    destroy: vi.fn(),
    mediaInfo: { videoCodec: 'avc1.640028' },
  }
  return { player: value, createPlayer: vi.fn(() => value) }
})

vi.mock('mpegts.js', () => ({
  default: {
    createPlayer,
    isSupported: () => true,
    Events: { ERROR: 'error', MEDIA_INFO: 'media_info', SEI_ARRIVED: 'sei_arrived' },
    ErrorTypes: { NETWORK_ERROR: 'NetworkError', MEDIA_ERROR: 'MediaError', OTHER_ERROR: 'OtherError' },
  },
}))

import { createMpegtsAdapter, normalizeMpegtsSei } from './mpegtsAdapter'

describe('mpegts adapter', () => {
  beforeEach(() => vi.clearAllMocks())

  it('normalizes upstream millisecond PTS to seconds', () => {
    expect(normalizeMpegtsSei({
      type: 5,
      uuid: new Uint8Array([1]),
      user_data: new Uint8Array([2]),
      pts: 1250,
    })).toEqual({ type: 5, uuid: new Uint8Array([1]), userData: new Uint8Array([2]), ptsSeconds: 1.25 })
  })

  it('constructs one low-latency live FLV player and owns its lifecycle', async () => {
    const adapter = createMpegtsAdapter('ws://media.test/live/a.live.flv?playToken=opaque')
    const video = {} as HTMLVideoElement
    adapter.attach(video)
    adapter.load()
    await adapter.play()
    adapter.destroy()

    expect(createPlayer).toHaveBeenCalledWith(
      { type: 'flv', isLive: true, url: 'ws://media.test/live/a.live.flv?playToken=opaque' },
      { enableWorker: true, enableStashBuffer: false, lazyLoad: false, liveBufferLatencyChasing: true },
    )
    expect(player.attachMediaElement).toHaveBeenCalledWith(video)
    expect(player.load).toHaveBeenCalledOnce()
    expect(player.destroy).toHaveBeenCalledOnce()
  })
})
