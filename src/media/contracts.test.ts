import { describe, expect, it } from 'vitest'
import {
  RKNODE_SEI_UUID,
  decodeInferenceSei,
  type RawSeiPayload,
} from './contracts'

const encoder = new TextEncoder()

function raw(value: unknown, overrides: Partial<RawSeiPayload> = {}): RawSeiPayload {
  return {
    type: 5,
    uuid: RKNODE_SEI_UUID,
    userData: encoder.encode(JSON.stringify(value)),
    ptsSeconds: 1.25,
    ...overrides,
  }
}

function envelope(overrides: Record<string, unknown> = {}) {
  return {
    schema_version: 2,
    task_id: 'task-a',
    revision: 7,
    frame_index: 12,
    width: 8,
    height: 4,
    detections: [],
    detection_results: {},
    structured_results: {},
    analytics: {},
    ...overrides,
  }
}

describe('decodeInferenceSei', () => {
  it('accepts the exact UUID and a valid schema-v2 envelope', () => {
    const result = decodeInferenceSei(raw(envelope()), 'task-a', 7)

    expect(result.ok).toBe(true)
    if (result.ok) {
      expect(result.value.taskId).toBe('task-a')
      expect(result.value.revision).toBe(7)
      expect(result.value.ptsSeconds).toBe(1.25)
      expect(Object.isFrozen(result.value)).toBe(true)
    }
  })

  it('accepts a C-string terminator appended by FFmpeg SEI metadata', () => {
    const encoded = new Uint8Array([...new TextEncoder().encode(JSON.stringify(envelope())), 0])
    const result = decodeInferenceSei(raw(envelope(), { userData: encoded }), 'task-a', 7)

    expect(result.ok).toBe(true)
  })

  it.each([
    [raw(envelope(), { type: 4 }), 'unsupported_sei_type'],
    [raw(envelope(), { uuid: new Uint8Array(16) }), 'foreign_uuid'],
    [raw(envelope({ schema_version: 1 })), 'unsupported_schema'],
    [raw(envelope({ task_id: 'task-b' })), 'task_mismatch'],
    [raw(envelope({ revision: 8 })), 'revision_mismatch'],
    [raw(envelope({ width: 0 })), 'invalid_dimensions'],
    [raw(envelope({ structured_results: { x: { type: 'unknown', result: {} } } })), 'unsupported_result'],
  ] as const)('rejects invalid input with %s', (input, code) => {
    expect(decodeInferenceSei(input, 'task-a', 7)).toEqual({ ok: false, code })
  })

  it('rejects malformed UTF-8 and payloads above one MiB without exposing data', () => {
    expect(decodeInferenceSei(raw(envelope(), { userData: new Uint8Array([0xc3, 0x28]) }), 'task-a', 7))
      .toEqual({ ok: false, code: 'invalid_utf8' })
    expect(decodeInferenceSei(raw(envelope(), { userData: new Uint8Array(1024 * 1024 + 1) }), 'task-a', 7))
      .toEqual({ ok: false, code: 'payload_too_large' })
  })

  it('validates segmentation run count and exact pixel sum', () => {
    const valid = envelope({
      result_type: 'segmentation',
      result: {
        width: 8,
        height: 4,
        source_width: 8,
        source_height: 4,
        encoding: 'class-rle-v1',
        labels: ['background', 'defect'],
        runs: [[0, 16], [1, 16]],
      },
    })
    expect(decodeInferenceSei(raw(valid), 'task-a', 7).ok).toBe(true)

    const wrongSum = structuredSegmentation([[0, 31]])
    expect(decodeInferenceSei(raw(wrongSum), 'task-a', 7)).toEqual({ ok: false, code: 'invalid_segmentation' })

    const tooManyRuns = structuredSegmentation(Array.from({ length: 262145 }, () => []))
    expect(decodeInferenceSei(raw(tooManyRuns), 'task-a', 7)).toEqual({ ok: false, code: 'too_many_runs' })
  })

  it('bounds result arrays and UTF-8 label length', () => {
    expect(decodeInferenceSei(raw(envelope({ detections: Array.from({ length: 10001 }, () => ({})) })), 'task-a', 7))
      .toEqual({ ok: false, code: 'too_many_results' })
    expect(decodeInferenceSei(raw(envelope({ detections: [{ x: 0, y: 0, w: 1, h: 1, confidence: 1, class_id: 0, label: '中'.repeat(86) }] })), 'task-a', 7))
      .toEqual({ ok: false, code: 'text_too_long' })
  })
})

function structuredSegmentation(runs: number[][]) {
  return envelope({
    structured_results: {
      deeplab: {
        type: 'segmentation',
        result: {
          width: 8,
          height: 4,
          source_width: 8,
          source_height: 4,
          encoding: 'class-rle-v1',
          labels: ['background'],
          runs,
        },
      },
    },
  })
}
