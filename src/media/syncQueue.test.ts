import { describe, expect, it } from 'vitest'
import { MetadataSyncQueue } from './syncQueue'

describe('MetadataSyncQueue', () => {
  it('keeps at most 120 entries and two seconds of PTS history', () => {
    const queue = new MetadataSyncQueue<number>()
    for (let index = 0; index < 150; index += 1) queue.enqueue(index / 50, index)

    expect(queue.size).toBeLessThanOrEqual(120)
    expect(queue.oldestPts).toBeGreaterThanOrEqual(queue.newestPts - 2)
  })

  it('selects the newest entry not later than video time plus one-frame tolerance', () => {
    const queue = new MetadataSyncQueue<string>()
    queue.enqueue(1.02, 'later')
    queue.enqueue(0.98, 'older')
    queue.enqueue(1.0, 'current')

    expect(queue.select(1, 1 / 30)).toEqual({ value: 'later', ptsSeconds: 1.02, clearOverlay: false })
  })

  it('clears only after media time advances one second without valid metadata', () => {
    const queue = new MetadataSyncQueue<string>()
    queue.enqueue(2, 'frame')
    expect(queue.select(2, 1 / 30).clearOverlay).toBe(false)
    expect(queue.select(2.5, 1 / 30).clearOverlay).toBe(false)
    expect(queue.select(3.01, 1 / 30).clearOverlay).toBe(true)
    expect(queue.select(3.01, 1 / 30).clearOverlay).toBe(true)
  })
})
