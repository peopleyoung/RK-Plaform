import { describe, expect, it } from 'vitest'
import fixture from './fixtures/schema-v2-deeplab.json'
import { decodeInferenceSei, RKNODE_SEI_UUID } from './contracts'
import { decodeSegmentationRgba } from './segmentation.worker'
import { pascalColor } from './pascalPalette'

const encoder = new TextEncoder()

describe('segmentation raster worker', () => {
  it.each([
    [0, [0, 0, 0]],
    [1, [128, 0, 0]],
    [2, [0, 128, 0]],
    [15, [192, 128, 128]],
    [255, [224, 224, 192]],
  ] as const)('uses stable Pascal VOC color for class %i', (classId, color) => {
    expect(pascalColor(classId)).toEqual(color)
  })

  it('expands an 8x4 RLE once and makes class zero transparent', () => {
    const decoded = decodeInferenceSei({
      type: 5,
      uuid: RKNODE_SEI_UUID,
      userData: encoder.encode(JSON.stringify(fixture)),
      ptsSeconds: 2,
    }, 'task-deeplab', 5)
    expect(decoded.ok).toBe(true)
    if (!decoded.ok || decoded.value.primaryResult?.type !== 'segmentation') return

    const raster = decodeSegmentationRgba(decoded.value.primaryResult)
    expect(raster.ok).toBe(true)
    if (!raster.ok) return
    expect(raster.rgba.buffer).toBeInstanceOf(ArrayBuffer)
    expect(raster.rgba).toHaveLength(8 * 4 * 4)
    expect([...raster.rgba.slice(0, 4)]).toEqual([0, 0, 0, 0])
    expect([...raster.rgba.slice(8 * 4, 8 * 4 + 4)]).toEqual([128, 0, 0, 255])
  })

  it('rejects invalid sums, class IDs, dimensions, and excessive runs defensively', () => {
    const base = {
      type: 'segmentation' as const,
      width: 2,
      height: 2,
      sourceWidth: 2,
      sourceHeight: 2,
      encoding: 'class-rle-v1' as const,
      labels: ['background'],
    }
    expect(decodeSegmentationRgba({ ...base, runs: [[0, 3]] })).toEqual({ ok: false, code: 'invalid_segmentation' })
    expect(decodeSegmentationRgba({ ...base, runs: [[256, 4]] })).toEqual({ ok: false, code: 'invalid_class_id' })
    expect(decodeSegmentationRgba({ ...base, width: 0, runs: [] })).toEqual({ ok: false, code: 'invalid_dimensions' })
    expect(decodeSegmentationRgba({ ...base, runs: Array.from({ length: 262145 }, () => [0, 1] as const) }))
      .toEqual({ ok: false, code: 'too_many_runs' })
  })
})
