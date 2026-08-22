import { describe, expect, it } from 'vitest'
import fixture from './fixtures/schema-v2-yolo.json'
import ocrFixture from './fixtures/schema-v2-ocr.json'
import { decodeInferenceSei, RKNODE_SEI_UUID, type InferenceEnvelope } from './contracts'
import { renderOverlay } from './renderOverlay'

const encoder = new TextEncoder()

function decode(value: unknown, taskId: string, revision: number): InferenceEnvelope {
  const result = decodeInferenceSei({
    type: 5,
    uuid: RKNODE_SEI_UUID,
    userData: encoder.encode(JSON.stringify(value)),
    ptsSeconds: 1,
  }, taskId, revision)
  if (!result.ok) throw new Error(result.code)
  return result.value
}

class FakeContext {
  readonly events: string[] = []
  canvas = { width: 1280, height: 720 }
  globalAlpha = 1
  lineWidth = 1
  font = ''
  textBaseline = ''
  textAlign = ''
  private currentStroke = ''
  private currentFill = ''

  set strokeStyle(value: string | CanvasGradient | CanvasPattern) {
    this.currentStroke = String(value)
    this.events.push(`strokeStyle:${value}`)
  }
  set fillStyle(value: string | CanvasGradient | CanvasPattern) {
    this.currentFill = String(value)
    this.events.push(`fillStyle:${value}`)
  }
  save() { this.events.push('save') }
  restore() { this.events.push('restore') }
  clearRect() { this.events.push('clear') }
  drawImage() { this.events.push(`mask:${this.globalAlpha}`) }
  beginPath() { this.events.push('begin') }
  moveTo(x: number, y: number) { this.events.push(`move:${x.toFixed(1)},${y.toFixed(1)}`) }
  lineTo(x: number, y: number) { this.events.push(`line:${x.toFixed(1)},${y.toFixed(1)}`) }
  closePath() { this.events.push('close') }
  stroke() { this.events.push(`stroke:${this.currentStroke}`) }
  strokeRect(x: number, y: number, w: number, h: number) { this.events.push(`rect:${this.currentStroke}:${x},${y},${w},${h}`) }
  fillRect() { this.events.push(`fillRect:${this.currentFill}`) }
  fillText(text: string) { this.events.push(`text:${this.currentFill}:${text}`) }
  measureText(text: string) { return { width: text.length * 7 } as TextMetrics }
}

describe('renderOverlay', () => {
  it('draws contain-mapped layers in stable order with alpha 0.5 and isolated state', () => {
    const context = new FakeContext()
    const envelope = decode(fixture, 'task-yolo', 3)
    renderOverlay(context as unknown as CanvasRenderingContext2D, envelope, {
      segmentation: {} as CanvasImageSource,
      areas: [{ id: 'zone-a', name: 'Zone A', polygon: [{ x: 0, y: 0 }, { x: 0.5, y: 0 }, { x: 0.5, y: 0.5 }] }],
      lines: [{ id: 'gate-a', name: 'Gate A', start: { x: 0, y: 0.5 }, end: { x: 1, y: 0.5 } }],
    })

    expect(context.events[0]).toBe('save')
    expect(context.events.at(-1)).toBe('restore')
    expect(context.events).toContain('mask:0.5')
    expect(context.events).toContain('rect:#22c55e:128,72,320,180')
    expect(context.events).toContain('rect:#38bdf8:192,108,128,72')
    const area = context.events.indexOf('strokeStyle:#facc15')
    const primary = context.events.indexOf('strokeStyle:#22c55e')
    const secondary = context.events.indexOf('strokeStyle:#38bdf8')
    const analytics = context.events.findIndex((entry) => entry.includes('Zone A: 1'))
    expect(area).toBeLessThan(primary)
    expect(primary).toBeLessThan(secondary)
    expect(secondary).toBeLessThan(analytics)
  })

  it('draws OCR polygons and recognition text into Canvas', () => {
    const context = new FakeContext()
    renderOverlay(context as unknown as CanvasRenderingContext2D, decode(ocrFixture, 'task-ocr', 4), {})

    expect(context.events).toContain('strokeStyle:#f472b6')
    expect(context.events.some((entry) => entry.includes('A12'))).toBe(true)
    expect(context.events.at(-1)).toBe('restore')
  })
})
