export interface InferenceTaskMediaForm {
  decoder: 'opencv' | 'rkmpp'
  trackingEnabled: boolean
  trackBuffer: number
  kafkaEnabled: boolean
  kafkaBrokers: string
  kafkaTopic: string
  kafkaKey: string
  zlmSeiEnabled: boolean
  zlmGatewayId: string
  zlmStreamName: string
}

export function buildInferenceTaskMedia(
  form: InferenceTaskMediaForm,
): Record<string, unknown> {
  const zlmSei: Record<string, unknown> = {
    enabled: form.zlmSeiEnabled,
    reconnectMs: 1000,
  }
  if (form.zlmSeiEnabled) {
    zlmSei.gatewayId = form.zlmGatewayId.trim()
    zlmSei.streamName = form.zlmStreamName.trim()
  }
  return {
    decoder: form.decoder,
    tracking: { enabled: form.trackingEnabled, trackBuffer: form.trackBuffer },
    kafka: {
      enabled: form.kafkaEnabled,
      brokers: form.kafkaBrokers.trim(),
      topic: form.kafkaTopic.trim(),
      key: form.kafkaKey.trim(),
      queueMessages: 10000,
      messageTimeoutMs: 3000,
    },
    zlmSei,
  }
}
