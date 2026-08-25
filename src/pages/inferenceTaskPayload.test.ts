import { describe, expect, it } from 'vitest'

import { buildInferenceTaskMedia } from './inferenceTaskPayload'

const defaults = {
  decoder: 'opencv' as const,
  trackingEnabled: false,
  trackBuffer: 30,
  kafkaEnabled: false,
  kafkaBrokers: '',
  kafkaTopic: 'sei_msg',
  kafkaKey: '',
  zlmSeiEnabled: false,
  zlmGatewayId: '',
  zlmStreamName: '',
}

describe('buildInferenceTaskMedia', () => {
  it('builds a valid disabled ZLM config without the removed outputUri field', () => {
    const media = buildInferenceTaskMedia(defaults)
    expect(media).toEqual({
      decoder: 'opencv',
      tracking: { enabled: false, trackBuffer: 30 },
      kafka: {
        enabled: false,
        brokers: '',
        topic: 'sei_msg',
        key: '',
        queueMessages: 10000,
        messageTimeoutMs: 3000,
      },
      zlmSei: { enabled: false, reconnectMs: 1000 },
    })
    expect(media).not.toHaveProperty('zlmSei.outputUri')
  })

  it('uses the media gateway binding for enabled ZLM output', () => {
    const media = buildInferenceTaskMedia({
      ...defaults,
      decoder: 'rkmpp',
      zlmSeiEnabled: true,
      zlmGatewayId: ' gateway_builtin ',
      zlmStreamName: ' camera_01 ',
    })
    expect(media.zlmSei).toEqual({
      enabled: true,
      gatewayId: 'gateway_builtin',
      streamName: 'camera_01',
      reconnectMs: 1000,
    })
  })
})
