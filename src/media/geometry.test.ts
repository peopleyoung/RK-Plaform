import { describe, expect, it } from 'vitest'
import { computeContainTransform, mapPoint } from './geometry'

describe('object-fit contain geometry', () => {
  it('centers pillarboxed video', () => {
    const transform = computeContainTransform(640, 480, 1280, 720)
    expect(transform).toEqual({ scale: 1.5, offsetX: 160, offsetY: 0, width: 960, height: 720 })
    expect(mapPoint(transform, 0, 0)).toEqual({ x: 160, y: 0 })
  })

  it('centers letterboxed video', () => {
    const transform = computeContainTransform(1920, 1080, 600, 600)
    expect(transform.scale).toBeCloseTo(0.3125)
    expect(transform.offsetX).toBe(0)
    expect(transform.offsetY).toBeCloseTo(131.25)
    expect(mapPoint(transform, 1920, 1080)).toEqual({ x: 600, y: 468.75 })
  })

  it('rejects non-positive dimensions', () => {
    expect(() => computeContainTransform(0, 1080, 600, 600)).toThrow('positive')
  })
})
