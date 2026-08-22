import { decodeInferenceSei, type InferenceEnvelope } from '../../src/media/contracts'
import { createMpegtsAdapter } from '../../src/media/mpegtsAdapter'
import { renderOverlay } from '../../src/media/renderOverlay'
import { SegmentationWorkerClient } from '../../src/media/segmentationWorkerClient'

interface MediaFixtureResult {
  passed: boolean
  directWsFlv: boolean
  h264VideoNonblank: boolean
  seiArrived: boolean
  overlayCoordinates: boolean
  segmentationRendered: boolean
  diagnostics: string[]
  videoFrameCount: number
  seiCount: number
  latencySamplesMs: number[]
  overlaySkewFrames: number[]
  reconnectDurationMs: number
  maxQueueDepth: number
  maxQueueAgeMs: number
}

declare global {
  interface Window {
    __RKNODE_MEDIA_RESULT__?: MediaFixtureResult
  }
}

const parameters = new URLSearchParams(window.location.search)
const streamUrl = parameters.get('streamUrl') ?? ''
const taskId = parameters.get('taskId') ?? 'media-e2e-task'
const revision = Number(parameters.get('revision') ?? '1')
const video = document.querySelector<HTMLVideoElement>('#video')
const canvas = document.querySelector<HTMLCanvasElement>('#overlay')
const status = document.querySelector<HTMLElement>('#status')

if (!video || !canvas || !status || !streamUrl.startsWith('ws://')) {
  throw new Error('invalid_media_fixture_parameters')
}

const context = canvas.getContext('2d', { willReadFrequently: true })
if (!context) throw new Error('canvas_2d_unavailable')
const segmentation = new SegmentationWorkerClient()
const result: MediaFixtureResult = {
  passed: false,
  directWsFlv: false,
  h264VideoNonblank: false,
  seiArrived: false,
  overlayCoordinates: false,
  segmentationRendered: false,
  diagnostics: [],
  videoFrameCount: 0,
  seiCount: 0,
  latencySamplesMs: [],
  overlaySkewFrames: [],
  reconnectDurationMs: 0,
  maxQueueDepth: 1,
  maxQueueAgeMs: 0,
}
window.__RKNODE_MEDIA_RESULT__ = result

let firstVideoMediaTime: number | null = null
let firstSeiPts: number | null = null
let sourceClockAtFirstSei: number | null = null
let latestEnvelope: InferenceEnvelope | null = null

const probe = document.createElement('canvas')
probe.width = 64
probe.height = 36
const probeContext = probe.getContext('2d', { willReadFrequently: true })
if (!probeContext) throw new Error('video_probe_unavailable')

function overlayChecks(): void {
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data
  let colored = 0
  let greenBoundary = 0
  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      const offset = (y * canvas.width + x) * 4
      if (pixels[offset + 3] > 0) colored += 1
      if (x >= 62 && x <= 68 && y >= 34 && y <= 130
          && pixels[offset + 1] > pixels[offset] + 25
          && pixels[offset + 1] > pixels[offset + 2] + 25) greenBoundary += 1
    }
  }
  result.overlayCoordinates = greenBoundary >= 20
  result.segmentationRendered = colored > 10_000
  updateResult()
}

function videoCheck(mediaTime: number): void {
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return
  probeContext.drawImage(video, 0, 0, probe.width, probe.height)
  const pixels = probeContext.getImageData(0, 0, probe.width, probe.height).data
  let minimum = 255
  let maximum = 0
  for (let offset = 0; offset < pixels.length; offset += 4) {
    const luminance = (pixels[offset] + pixels[offset + 1] + pixels[offset + 2]) / 3
    minimum = Math.min(minimum, luminance)
    maximum = Math.max(maximum, luminance)
  }
  result.videoFrameCount += 1
  result.h264VideoNonblank = maximum - minimum > 20
  if (firstVideoMediaTime === null) firstVideoMediaTime = mediaTime
  if (latestEnvelope !== null && firstSeiPts !== null) {
    const skewFrames = Math.abs(
      ((mediaTime - firstVideoMediaTime) - (latestEnvelope.ptsSeconds - firstSeiPts)) * 30,
    )
    result.overlaySkewFrames.push(skewFrames)
  }
  updateResult()
}

function updateResult(): void {
  result.directWsFlv = streamUrl.startsWith('ws://') && streamUrl.includes('.live.flv')
  result.passed = result.directWsFlv && result.h264VideoNonblank && result.seiArrived
    && result.overlayCoordinates && result.segmentationRendered
  status.textContent = result.passed ? 'passed' : 'receiving'
}

async function render(envelope: InferenceEnvelope): Promise<void> {
  const mask = envelope.primaryResult?.type === 'segmentation'
    ? await segmentation.decode(envelope.primaryResult)
    : null
  renderOverlay(context, envelope, { segmentation: mask })
  overlayChecks()
}

const adapter = createMpegtsAdapter(streamUrl)
adapter.onSei((payload) => {
  result.seiCount += 1
  const decoded = decodeInferenceSei(payload, taskId, revision)
  if (!decoded.ok) {
    result.diagnostics.push(decoded.code)
    return
  }
  result.seiArrived = true
  latestEnvelope = decoded.value
  firstSeiPts ??= decoded.value.ptsSeconds
  sourceClockAtFirstSei ??= performance.now() / 1000 - decoded.value.ptsSeconds
  if (sourceClockAtFirstSei !== null) {
    result.latencySamplesMs.push(Math.max(
      0,
      (performance.now() / 1000 - (sourceClockAtFirstSei + decoded.value.ptsSeconds)) * 1000,
    ))
  }
  void render(decoded.value).catch(() => result.diagnostics.push('render_failed'))
})
adapter.onError((error) => result.diagnostics.push(`${error.kind}_${error.code}`))
adapter.attach(video)
adapter.load()
void adapter.play().catch(() => result.diagnostics.push('play_failed'))

const scheduleFrame = () => {
  video.requestVideoFrameCallback((_now, metadata) => {
    videoCheck(metadata.mediaTime)
    scheduleFrame()
  })
}
scheduleFrame()

window.addEventListener('beforeunload', () => {
  adapter.destroy()
  segmentation.destroy()
})
