export interface MetadataSelection<T> {
  readonly value?: T
  readonly ptsSeconds?: number
  readonly clearOverlay: boolean
}

interface QueueEntry<T> {
  readonly ptsSeconds: number
  readonly value: T
}

export class MetadataSyncQueue<T> {
  private readonly entries: QueueEntry<T>[] = []
  private lastMatchedMediaTime: number | null = null

  constructor(
    private readonly maxAgeSeconds = 2,
    private readonly maxEntries = 120,
    private readonly staleAfterSeconds = 1,
  ) {}

  get size(): number {
    return this.entries.length
  }

  get oldestPts(): number {
    return this.entries[0]?.ptsSeconds ?? Number.NaN
  }

  get newestPts(): number {
    return this.entries.at(-1)?.ptsSeconds ?? Number.NaN
  }

  enqueue(ptsSeconds: number, value: T): void {
    if (!Number.isFinite(ptsSeconds) || ptsSeconds < 0) return
    const index = this.entries.findIndex((entry) => entry.ptsSeconds > ptsSeconds)
    const entry = { ptsSeconds, value }
    if (index === -1) this.entries.push(entry)
    else this.entries.splice(index, 0, entry)

    const newest = this.entries.at(-1)?.ptsSeconds ?? ptsSeconds
    while (this.entries.length && this.entries[0].ptsSeconds < newest - this.maxAgeSeconds) {
      this.entries.shift()
    }
    while (this.entries.length > this.maxEntries) this.entries.shift()
  }

  select(mediaTime: number, frameTolerance: number): MetadataSelection<T> {
    if (!Number.isFinite(mediaTime) || !Number.isFinite(frameTolerance) || frameTolerance < 0) {
      return { clearOverlay: false }
    }
    const latestAllowed = mediaTime + frameTolerance
    let selectedIndex = -1
    for (let index = 0; index < this.entries.length; index += 1) {
      if (this.entries[index].ptsSeconds > latestAllowed) break
      selectedIndex = index
    }
    if (selectedIndex >= 0) {
      const selected = this.entries[selectedIndex]
      this.entries.splice(0, selectedIndex + 1)
      this.lastMatchedMediaTime = mediaTime
      return { value: selected.value, ptsSeconds: selected.ptsSeconds, clearOverlay: false }
    }
    return {
      clearOverlay: this.lastMatchedMediaTime !== null
        && mediaTime - this.lastMatchedMediaTime > this.staleAfterSeconds,
    }
  }

  clear(): void {
    this.entries.length = 0
    this.lastMatchedMediaTime = null
  }
}
