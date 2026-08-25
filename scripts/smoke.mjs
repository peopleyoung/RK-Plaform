import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { chromium } from 'playwright'

const baseUrl = process.env.PROTOTYPE_URL ?? 'http://127.0.0.1:5173'
const screenshotDir = process.env.SMOKE_SCREENSHOT_DIR
if (screenshotDir) await mkdir(screenshotDir, { recursive: true })
const browser = await chromium.launch({ headless: true })
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
page.addInitScript(() => sessionStorage.setItem('rknode.adminToken', 'admin'))
const errors = []

page.on('console', (message) => {
  if (message.type() === 'error') errors.push(message.text())
})
page.on('pageerror', (error) => errors.push(error.message))

const fixtureTime = new Date().toISOString()
const smokeDataset = {
  id: 'dataset_ui_pending', name: '界面测试数据集', description: '', version: 'v1', taskType: 'object_detection', datasetFormat: 'yolo', classes: [],
  status: 'ready', filename: 'dataset.zip', sizeBytes: 10, sha256: '0'.repeat(64), createdAt: fixtureTime, updatedAt: fixtureTime, errorMessage: null,
}
const smokeManifest = {
  profileId: 'yolo-detect', variant: 'yolov8n', taskType: 'object_detection', resolution: { width: 640, height: 640 },
  input: { shape: [1, 3, 640, 640], layout: 'NCHW', name: 'images' }, supportedPrecisions: ['int8', 'fp16'], outputContract: 'rknn_yolo_dfl_split_heads_v1',
}
const smokeTrainingJob = {
  id: 'train_ui_fixture', type: 'training', name: '界面训练任务', status: 'succeeded', profileId: 'yolo-detect', datasetId: smokeDataset.id,
  workerId: null, progress: 100, stage: 'completed', spec: { variant: 'yolov8n', accelerator: 'cpu', dataset: { name: smokeDataset.name }, resolution: { width: 640, height: 640 } }, result: {},
  retryCount: 0, maxRetries: 1, errorCode: null, errorMessage: null, createdAt: fixtureTime, updatedAt: fixtureTime, startedAt: fixtureTime, completedAt: fixtureTime,
}
const smokeFailedTrainingJob = {
  ...smokeTrainingJob,
  id: 'train_ui_failed', name: '界面失败训练任务', status: 'failed', progress: 38, stage: 'failed', result: null,
  errorCode: 'training_failed', errorMessage: '训练进程退出', completedAt: fixtureTime,
}
const smokeSourceArtifact = {
  id: 'artifact_ui_source', jobId: smokeTrainingJob.id, kind: 'onnx', filename: 'yolov8n-640x640.onnx', mediaType: 'application/octet-stream', sizeBytes: 10,
  sha256: '1'.repeat(64), manifest: smokeManifest, createdAt: fixtureTime,
}
const smokeConversionJob = {
  id: 'convert_ui_fixture', type: 'conversion', name: '界面转换任务', status: 'succeeded', profileId: 'yolo-detect', datasetId: smokeDataset.id,
  workerId: null, progress: 100, stage: 'completed', spec: { precision: 'fp16', sourceArtifact: { id: smokeSourceArtifact.id, filename: smokeSourceArtifact.filename }, manifest: smokeManifest }, result: { validation: { deploymentReady: true, performanceReady: true, toolkitVersion: '2.3.2', outputShapes: [[1, 84, 8400]], benchmark: { averageMs: 12.3, fps: 81.3 }, performance: { cpuFallbackDetected: false } } },
  retryCount: 0, maxRetries: 1, errorCode: null, errorMessage: null, createdAt: fixtureTime, updatedAt: fixtureTime, startedAt: fixtureTime, completedAt: fixtureTime,
}
const smokeFailedConversionJob = {
  ...smokeConversionJob,
  id: 'convert_ui_failed', name: '界面失败转换任务', status: 'failed', progress: 42, stage: 'failed', result: null,
  errorCode: 'rknn_build_failed', errorMessage: 'RKNN 构建失败', completedAt: fixtureTime,
}
const smokeOutputArtifact = {
  id: 'artifact_ui_rknn', jobId: smokeConversionJob.id, kind: 'rknn', filename: 'yolov8n-640x640.rknn', mediaType: 'application/octet-stream', sizeBytes: 10,
  sha256: '2'.repeat(64), manifest: null, createdAt: fixtureTime,
}
const smokeYoloRelease = {
  id: 'release_ui_yolo_primary', name: '界面 YOLO 主模型', version: 'v1.0.0', description: '', status: 'published',
  profileId: 'yolo-detect', variant: 'yolov8n', taskType: 'object_detection', precision: 'int8', adapter: 'yolo_dfl_split_v1',
  rknnArtifactId: smokeOutputArtifact.id, validationArtifactId: null, sourceTrainingJobId: smokeTrainingJob.id,
  sourceConversionJobId: smokeConversionJob.id, datasetId: smokeDataset.id, manifest: smokeManifest,
  createdAt: fixtureTime, publishedAt: fixtureTime,
}
const smokeSecondaryRelease = {
  ...smokeYoloRelease,
  id: 'release_ui_yolo_secondary', name: '界面 YOLO 二级模型', version: 'v1.0.1',
}
const smokeMediaGateway = {
  id: 'gateway_ui_fixture', name: '界面在线媒体网关', builtin: false, enabled: true,
  publishHost: '127.0.0.1', rtspPort: 8554, playbackHost: '127.0.0.1', wsPort: 8080,
  apiHost: '127.0.0.1', apiPort: 8080, app: 'live', status: 'online',
  apiSecretConfigured: true, hookIdentityConfigured: true, lastProbeAt: fixtureTime,
  lastHookAt: fixtureTime, lastError: null, createdAt: fixtureTime, updatedAt: fixtureTime,
}
const graphOperator = (operatorId, runtimeNode, category, title, defaults, configurableFields, options = {}) => ({
  operatorId, runtimeNode, category, title, description: title,
  inputPorts: category === 'capture' ? [] : ['frame'], outputPorts: category === 'output' ? [] : ['frame'],
  minInstances: options.minInstances ?? 0, maxInstances: options.maxInstances ?? 1,
  defaults, dependencies: options.dependencies ?? [], supportedAdapters: options.supportedAdapters ?? [],
  requiredFeatures: options.requiredFeatures ?? [], configurableFields, readOnlyFields: [],
})
const smokeOperatorCatalog = {
  schemaVersion: 1,
  catalogVersion: '2026.08.25',
  operators: [
    graphOperator('capture.opencv', 'VideoCaptureNode', 'capture', '通用解码', { loop: true, reconnectMs: 1000 }, ['loop', 'reconnectMs'], { minInstances: 1 }),
    graphOperator('capture.rkmpp', 'RkMppCaptureNode', 'capture', 'MPP 硬解码', { loop: true, reconnectMs: 1000 }, ['loop', 'reconnectMs'], { minInstances: 1, requiredFeatures: ['rkmpp_decode'] }),
    graphOperator('inference.primary', 'InferNode', 'inference', '主推理', { releaseId: '', interval: 1, confidence: 0.4, nms: 0.5, contextCount: 1, workerCount: 1 }, ['releaseId', 'interval', 'confidence', 'nms', 'contextCount', 'workerCount'], { minInstances: 1, supportedAdapters: ['yolo_dfl_split_v1', 'deeplab_logits_v1', 'ppocr_db_det_v1', 'ppocr_ctc_rec_v1'] }),
    graphOperator('processing.bytetrack', 'ByteTrackNode', 'processing', 'ByteTrack', { trackBuffer: 30 }, ['trackBuffer'], { dependencies: ['inference.primary'], supportedAdapters: ['yolo_dfl_split_v1'], requiredFeatures: ['bytetrack'] }),
    graphOperator('inference.secondary', 'SecondaryInferNode', 'inference', '二级推理', { releaseId: '', confidence: 0.25, sourceClassIds: [], contextCount: 1, workerCount: 1 }, ['releaseId', 'confidence', 'sourceClassIds', 'contextCount', 'workerCount'], { maxInstances: 4, dependencies: ['inference.primary', 'processing.bytetrack'], supportedAdapters: ['yolo_dfl_split_v1'], requiredFeatures: ['secondary_infer'] }),
    graphOperator('processing.analytics', 'AnalyticsNode', 'processing', '区域/越线分析', { areas: [], lines: [], osd: { enabled: true, showLabels: true, showConfidence: true, showTrackId: true, showAreas: true, showLines: true } }, ['areas', 'lines', 'osd'], { dependencies: ['processing.bytetrack'], supportedAdapters: ['yolo_dfl_split_v1'], requiredFeatures: ['analytics_area', 'analytics_line'] }),
    graphOperator('processing.events', 'EventOutputNode', 'processing', '事件抓拍/录像', { enabled: true, snapshot: true, record: false, preSeconds: 3, postSeconds: 5, retentionDays: 30 }, ['enabled', 'snapshot', 'record', 'preSeconds', 'postSeconds', 'retentionDays'], { dependencies: ['processing.analytics'], requiredFeatures: ['event_snapshot', 'event_record'] }),
    graphOperator('output.json', 'JsonOutputNode', 'output', 'JSONL/HTTP 输出', { type: 'jsonl', url: '', authorizationEnv: '', connectTimeoutMs: 1000, requestTimeoutMs: 3000 }, ['type', 'url', 'authorizationEnv', 'connectTimeoutMs', 'requestTimeoutMs']),
    graphOperator('output.kafka', 'KafkaOutputNode', 'output', 'Kafka 输出', { brokers: '', topic: 'sei_msg', key: '', queueMessages: 10000, messageTimeoutMs: 3000 }, ['brokers', 'topic', 'key', 'queueMessages', 'messageTimeoutMs'], { requiredFeatures: ['kafka'] }),
    graphOperator('output.zlm_sei', 'ZlmSeiOutputNode', 'output', 'ZLM SEI 输出', { gatewayId: '', streamName: '', reconnectMs: 1000 }, ['gatewayId', 'streamName', 'reconnectMs'], { dependencies: ['capture.rkmpp'], requiredFeatures: ['zlm_sei'] }),
  ],
}
const inferenceGraph = (releaseId, { zlm = true, streamName = 'ui-restart' } = {}) => {
  const nodes = [
    { id: 'capture', operator: zlm ? 'capture.rkmpp' : 'capture.opencv', config: { loop: true, reconnectMs: 1000 } },
    { id: 'primary', operator: 'inference.primary', config: { releaseId, interval: 1, confidence: 0.4, nms: 0.5, contextCount: 1, workerCount: 1 } },
    { id: 'json-output', operator: 'output.json', config: { type: 'jsonl', url: '', authorizationEnv: '', connectTimeoutMs: 1000, requestTimeoutMs: 3000 } },
  ]
  if (zlm) nodes.push({ id: 'zlm-output', operator: 'output.zlm_sei', config: { gatewayId: smokeMediaGateway.id, streamName, reconnectMs: 1000 } })
  return {
    schemaVersion: 1, catalogVersion: '2026.08.25', nodes,
    edges: [
      { source: 'capture', sourcePort: 'frame', target: 'primary', targetPort: 'frame' },
      { source: 'primary', sourcePort: 'frame', target: 'json-output', targetPort: 'frame' },
      ...(zlm ? [{ source: 'primary', sourcePort: 'frame', target: 'zlm-output', targetPort: 'frame' }] : []),
    ],
  }
}
const smokeTrainingEvents = [
  { id: 91001, type: 'log', level: 'info', message: 'Loading training dataset', data: { stage: 'train' }, createdAt: fixtureTime },
  { id: 91002, type: 'metric', level: 'info', message: 'epoch=1/2 train_loss=0.72 val_loss=0.65', data: { stage: 'train', epoch: 1, totalEpochs: 2, metrics: { train_loss: 0.72, val_loss: 0.65 } }, createdAt: fixtureTime },
  { id: 91003, type: 'metric', level: 'info', message: 'epoch=2/2 train_loss=0.48 val_loss=0.51', data: { stage: 'train', epoch: 2, totalEpochs: 2, metrics: { train_loss: 0.48, val_loss: 0.51 } }, createdAt: fixtureTime },
]
const smokeConversionEvents = [
  { id: 92001, type: 'progress', level: 'info', message: 'Validating ONNX graph contract', data: { stage: 'validate_onnx', progress: 20, metrics: {} }, createdAt: fixtureTime },
  { id: 92002, type: 'progress', level: 'info', message: 'Building RKNN model', data: { stage: 'build_rknn', progress: 50, metrics: {} }, createdAt: fixtureTime },
  { id: 92003, type: 'progress', level: 'info', message: 'Uploading RKNN artifact', data: { stage: 'upload_rknn', progress: 92, metrics: {} }, createdAt: fixtureTime },
]
const smokeInferenceTask = {
  id: 'itask_ui_stopped', name: '界面停止推理任务', status: 'stopped', nodeId: 'inode_ui_fixture', groupId: null,
  inputUri: 'rtsp://camera/ui-restart', graph: inferenceGraph(smokeYoloRelease.id), layout: { positions: {} }, graphRevisionId: 'graphrev_ui_stopped', graphHash: 'a'.repeat(64),
  npuCoreMask: 'auto', npuCorePolicy: 'shared', configRevision: 1, previewCapability: { state: 'available', reason: null },
  errorMessage: null, createdAt: fixtureTime, updatedAt: fixtureTime,
}
const smokeRunningInferenceTask = {
  ...smokeInferenceTask,
  id: 'itask_ui_running', name: '界面实时预览任务', status: 'running', inputUri: 'rtsp://camera/ui-preview', configRevision: 2,
}
let smokeCreatedInferenceTask = {
  ...smokeInferenceTask,
  id: 'itask_ui_created', name: '界面图编排草稿', status: 'draft', graph: inferenceGraph(smokeYoloRelease.id, { streamName: 'ui-created-stream' }), graphRevisionId: 'graphrev_ui_created', configRevision: 0,
}
const smokeRestartDeployment = {
  ...smokeInferenceTask, status: 'deploying', configRevision: 2,
}
const smokeRetiredInferenceNode = {
  id: 'inode_ui_retired', name: '界面退役测试板卡', groupId: null, labels: ['ui-fixture'], lifecycle: 'retired',
  connectivity: 'offline', health: 'unknown', deploymentStatus: 'idle', maxModelInstances: 1, hardwareId: 'ui-retired-board',
  runtimeVersion: 'rknn-runtime-2.3.2', driverVersion: 'fixture', pipelineVersion: 'fixture', adapters: ['deeplab_logits_v1'], metadata: {},
  desiredRevision: 0, actualRevision: 0, selfTestPassed: false, lastSeenAt: fixtureTime, createdAt: fixtureTime, updatedAt: fixtureTime,
}
const smokeActiveInferenceNode = {
  ...smokeRetiredInferenceNode,
  id: 'inode_ui_fixture', name: '界面健康测试板卡', lifecycle: 'active', connectivity: 'online', health: 'healthy', selfTestPassed: true,
  adapters: ['yolo_dfl_split_v1'], metadata: { features: ['rkmpp_decode', 'bytetrack', 'kafka', 'zlm_sei', 'analytics_area', 'analytics_line', 'event_snapshot', 'event_record', 'secondary_infer'] },
}
const smokeCompletedDeployment = {
  id: 'deployment_ui_completed', name: '界面已完成部署批次', status: 'succeeded',
  strategy: 'all_at_once', batchSize: 1, createdAt: fixtureTime, updatedAt: fixtureTime, completedAt: fixtureTime,
  targets: [{
    id: 'dtarget_ui_completed', deploymentId: 'deployment_ui_completed', nodeId: smokeRetiredInferenceNode.id, taskId: smokeInferenceTask.id,
    graphRevisionId: smokeInferenceTask.graphRevisionId, graphHash: smokeInferenceTask.graphHash, sequence: 0, desiredRevision: 1,
    state: 'healthy', progress: 100, stage: 'healthy', errorCode: null, errorMessage: null, startedAt: fixtureTime, completedAt: fixtureTime, updatedAt: fixtureTime,
  }],
}
let smokeInferenceRestarted = false
let smokeInferenceRestartCalls = 0
let smokeCreatedInference = false
let smokeCreatedInferenceCalls = 0
let smokeCreatedInferenceFirstPageReadAt = 0
let smokeCreatedInferencePayload = null
let smokePlaybackSessionCalls = 0
let smokeRetiredNodeDeleted = false
let smokeRetiredNodeDeleteCalls = 0
let smokeCompletedDeploymentDeleted = false
let smokeCompletedDeploymentDeleteCalls = 0
let smokeDeploymentFirstPageReadAt = 0

await page.route('**/api/v1/datasets', async (route) => {
  const request = route.request()
  if (request.method() !== 'GET' || new URL(request.url()).pathname !== '/api/v1/datasets') {
    await route.continue()
    return
  }
  const response = await route.fetch()
  const payload = await response.json()
  const datasets = Array.isArray(payload) ? payload : payload.items
  if (!Array.isArray(datasets)) throw new Error('datasets fixture expects an array or paginated items')
  if (!datasets.some((item) => item.id === smokeDataset.id)) datasets.push(smokeDataset)
  await route.fulfill({ response, json: Array.isArray(payload) ? datasets : { ...payload, items: datasets, total: datasets.length } })
})
await page.route('**/api/v1/jobs', async (route) => {
  const request = route.request()
  if (request.method() !== 'GET' || new URL(request.url()).pathname !== '/api/v1/jobs') {
    await route.continue()
    return
  }
  const response = await route.fetch()
  const payload = await response.json()
  const jobs = Array.isArray(payload) ? payload : payload.items
  if (!Array.isArray(jobs)) throw new Error('jobs fixture expects an array or paginated items')
  for (const job of [smokeTrainingJob, smokeFailedTrainingJob, smokeConversionJob, smokeFailedConversionJob]) {
    if (!jobs.some((item) => item.id === job.id)) jobs.push(job)
  }
  await route.fulfill({ response, json: Array.isArray(payload) ? jobs : { ...payload, items: jobs, total: jobs.length } })
})
await page.route('**/api/v1/jobs/train_ui_fixture/events*', async (route) => {
  const afterId = Number(new URL(route.request().url()).searchParams.get('afterId') ?? 0)
  await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(smokeTrainingEvents.filter((event) => event.id > afterId)) })
})
await page.route('**/api/v1/jobs/train_ui_fixture', (route) => route.fulfill({
  status: 200,
  contentType: 'application/json',
  body: JSON.stringify(smokeTrainingJob),
}))
await page.route('**/api/v1/jobs/convert_ui_fixture/events*', async (route) => {
  const afterId = Number(new URL(route.request().url()).searchParams.get('afterId') ?? 0)
  await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(smokeConversionEvents.filter((event) => event.id > afterId)) })
})
await page.route('**/api/v1/jobs/convert_ui_fixture', (route) => route.fulfill({
  status: 200,
  contentType: 'application/json',
  body: JSON.stringify(smokeConversionJob),
}))
await page.route('**/api/v1/jobs/train_ui_failed/retry', (route) => route.fulfill({
  status: 201,
  contentType: 'application/json',
  body: JSON.stringify({
    ...smokeFailedTrainingJob,
    id: 'train_ui_retry', status: 'queued', progress: 0, stage: 'queued', errorCode: null, errorMessage: null,
    spec: { ...smokeFailedTrainingJob.spec, retryOfJobId: smokeFailedTrainingJob.id }, completedAt: null,
  }),
}))
await page.route('**/api/v1/jobs/convert_ui_failed/retry', (route) => route.fulfill({
  status: 201,
  contentType: 'application/json',
  body: JSON.stringify({
    ...smokeFailedConversionJob,
    id: 'convert_ui_retry', status: 'queued', progress: 0, stage: 'queued', errorCode: null, errorMessage: null,
    spec: { ...smokeFailedConversionJob.spec, retryOfJobId: smokeFailedConversionJob.id }, completedAt: null,
  }),
}))
await page.route('**/api/v1/artifacts', async (route) => {
  const request = route.request()
  if (request.method() !== 'GET' || new URL(request.url()).pathname !== '/api/v1/artifacts') {
    await route.continue()
    return
  }
  const response = await route.fetch()
  const payload = await response.json()
  const artifacts = Array.isArray(payload) ? payload : payload.items
  if (!Array.isArray(artifacts)) throw new Error('artifacts fixture expects an array or paginated items')
  if (!artifacts.some((item) => item.id === smokeSourceArtifact.id)) artifacts.push(smokeSourceArtifact)
  if (!artifacts.some((item) => item.id === smokeOutputArtifact.id)) artifacts.push(smokeOutputArtifact)
  await route.fulfill({ response, json: Array.isArray(payload) ? artifacts : { ...payload, items: artifacts, total: artifacts.length } })
})
await page.route('**/api/v1/model-releases**', async (route) => {
  const request = route.request()
  const url = new URL(request.url())
  if (request.method() !== 'GET' || url.pathname !== '/api/v1/model-releases') {
    await route.continue()
    return
  }
  const pageNumber = Number(url.searchParams.get('page') ?? 1)
  const pageSize = Number(url.searchParams.get('pageSize') ?? 20)
  const items = pageNumber === 1 ? [smokeYoloRelease, smokeSecondaryRelease] : []
  await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ items, page: pageNumber, pageSize, total: 2 }) })
})
await page.route('**/api/v1/media-gateways', async (route) => {
  await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([smokeMediaGateway]) })
})
await page.route('**/api/v1/inference-operator-catalog', async (route) => {
  await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(smokeOperatorCatalog) })
})
await page.route('**/api/v1/inference-graphs/validate', async (route) => {
  const payload = JSON.parse(route.request().postData() ?? '{}')
  const graph = payload.graph
  const primary = graph.nodes.find((node) => node.operator === 'inference.primary')
  const requiredFeatures = graph.nodes.flatMap((node) => smokeOperatorCatalog.operators.find((operator) => operator.operatorId === node.operator)?.requiredFeatures ?? [])
  const requiredContexts = graph.nodes.filter((node) => ['inference.primary', 'inference.secondary'].includes(node.operator)).reduce((total, node) => total + Number(node.config.contextCount ?? 1), 0)
  await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
    valid: true, normalizedGraph: graph, graphHash: 'f'.repeat(64), issues: [],
    releaseIds: primary?.config.releaseId ? [primary.config.releaseId] : [], requiredFeatures,
    requiredAdapters: ['yolo_dfl_split_v1'], requiredContexts, compatibleNodeIds: [smokeActiveInferenceNode.id],
  }) })
})
await page.route('**/api/v1/inference-tasks**', async (route) => {
  const request = route.request()
  const url = new URL(request.url())
  if (request.method() === 'POST' && url.pathname === '/api/v1/inference-tasks') {
    smokeCreatedInference = true
    smokeCreatedInferenceCalls += 1
    smokeCreatedInferencePayload = JSON.parse(request.postData() ?? '{}')
    smokeCreatedInferenceTask = {
      ...smokeCreatedInferenceTask,
      name: smokeCreatedInferencePayload.name,
      nodeId: smokeCreatedInferencePayload.nodeId,
      inputUri: smokeCreatedInferencePayload.inputUri,
      graph: smokeCreatedInferencePayload.graph,
      layout: smokeCreatedInferencePayload.layout,
      npuCoreMask: smokeCreatedInferencePayload.npuCoreMask,
      npuCorePolicy: smokeCreatedInferencePayload.npuCorePolicy,
    }
    await route.fulfill({ status: 201, contentType: 'application/json', body: JSON.stringify(smokeCreatedInferenceTask) })
    return
  }
  if (request.method() === 'POST' && url.pathname.endsWith(`/${smokeInferenceTask.id}/restart`)) {
    smokeInferenceRestarted = true
    smokeInferenceRestartCalls += 1
    await route.fulfill({ status: 201, contentType: 'application/json', body: JSON.stringify(smokeRestartDeployment) })
    return
  }
  if (request.method() === 'POST' && url.pathname.endsWith(`/${smokeRunningInferenceTask.id}/playback-session`)) {
    smokePlaybackSessionCalls += 1
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        streamUrl: 'ws://127.0.0.1:8081/live/ui-preview.live.flv?playToken=fixture',
        expiresAt: new Date(Date.now() + 60_000).toISOString(), taskId: smokeRunningInferenceTask.id,
        revision: smokeRunningInferenceTask.configRevision, gatewayId: 'gateway_builtin', app: 'live', streamName: 'ui-preview', codec: 'h264', reconnectMs: 1000,
      }),
    })
    return
  }
  if (request.method() === 'GET' && url.pathname === '/api/v1/inference-tasks') {
    const pageNumber = Number(url.searchParams.get('page') ?? 1)
    const pageSize = Number(url.searchParams.get('pageSize') ?? 20)
    const items = []
    const fixture = { ...smokeInferenceTask, status: smokeInferenceRestarted ? 'deploying' : 'stopped' }
    if (pageNumber === 1) items.push(fixture, smokeRunningInferenceTask)
    if (pageNumber === 1 && smokeCreatedInference) {
      if (url.searchParams.get('pageSize') === '10' && !smokeCreatedInferenceFirstPageReadAt) smokeCreatedInferenceFirstPageReadAt = Date.now()
      items.push(smokeCreatedInferenceTask)
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ items, page: pageNumber, pageSize, total: smokeCreatedInference ? 3 : 2 }) })
    return
  }
  await route.continue()
})
await page.route('**/api/v1/inference-nodes**', async (route) => {
  const request = route.request()
  const url = new URL(request.url())
  if (request.method() === 'DELETE' && url.pathname.endsWith(`/${smokeRetiredInferenceNode.id}/record`)) {
    smokeRetiredNodeDeleted = true
    smokeRetiredNodeDeleteCalls += 1
    await route.fulfill({ status: 204, body: '' })
    return
  }
  if (request.method() !== 'GET' || url.pathname !== '/api/v1/inference-nodes') {
    await route.continue()
    return
  }
  const pageNumber = Number(url.searchParams.get('page') ?? 1)
  const pageSize = Number(url.searchParams.get('pageSize') ?? 20)
  const items = pageNumber === 1 ? [smokeActiveInferenceNode, ...(!smokeRetiredNodeDeleted ? [smokeRetiredInferenceNode] : [])] : []
  await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ items, page: pageNumber, pageSize, total: smokeRetiredNodeDeleted ? 1 : 2 }) })
})
await page.route('**/api/v1/deployments**', async (route) => {
  const request = route.request()
  const url = new URL(request.url())
  if (request.method() === 'DELETE' && url.pathname.endsWith(`/${smokeCompletedDeployment.id}`)) {
    smokeCompletedDeploymentDeleted = true
    smokeCompletedDeploymentDeleteCalls += 1
    await route.fulfill({ status: 204, body: '' })
    return
  }
  if (request.method() !== 'GET' || url.pathname !== '/api/v1/deployments') {
    await route.continue()
    return
  }
  const pageNumber = Number(url.searchParams.get('page') ?? 1)
  const pageSize = Number(url.searchParams.get('pageSize') ?? 20)
  const items = []
  if (pageNumber === 1 && !smokeCompletedDeploymentDeleted) {
    if (url.searchParams.get('pageSize') === '10' && !smokeDeploymentFirstPageReadAt) smokeDeploymentFirstPageReadAt = Date.now()
    const completed = Boolean(smokeDeploymentFirstPageReadAt && Date.now() - smokeDeploymentFirstPageReadAt >= 500)
    const deploymentFixture = {
      ...smokeCompletedDeployment,
      status: completed ? 'succeeded' : 'rolling',
      completedAt: completed ? fixtureTime : null,
      targets: smokeCompletedDeployment.targets.map((target) => ({
        ...target,
        state: completed ? 'healthy' : 'switching',
        progress: completed ? 100 : 70,
        stage: completed ? 'healthy' : 'switching',
        completedAt: completed ? fixtureTime : null,
      })),
    }
    items.push(deploymentFixture)
  }
  await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ items, page: pageNumber, pageSize, total: smokeCompletedDeploymentDeleted ? 0 : 1 }) })
})
const routes = {
  overview: '工作台',
  datasets: '数据集',
  training: '模型训练',
  conversion: '模型转换',
  nodes: '算力节点',
  inference: '推理下发',
  monitoring: '视频监控',
  settings: '系统设置',
}

for (const [route, heading] of Object.entries(routes)) {
  await page.goto(`${baseUrl}/#/${route}`)
  await page.getByRole('heading', { name: heading, exact: true, level: 1 }).waitFor()
  const dimensions = await page.evaluate(() => ({ body: document.body.scrollWidth, viewport: window.innerWidth }))
  assert.ok(dimensions.body <= dimensions.viewport, `${route} overflows: ${JSON.stringify(dimensions)}`)
}

await page.goto(`${baseUrl}/#/nodes`)
const nodeSummary = page.getByRole('region', { name: '节点接入概况' })
await nodeSummary.getByText('训练节点', { exact: true }).waitFor()
await nodeSummary.getByText('转换节点', { exact: true }).waitFor()
await nodeSummary.getByText('推理节点', { exact: true }).waitFor()
assert.equal(await page.getByRole('tab', { name: '训练', exact: true }).count(), 1)
assert.equal(await page.getByRole('tab', { name: '转换', exact: true }).count(), 1)
assert.equal(await page.getByRole('tab', { name: '推理', exact: true }).count(), 1)

await page.goto(`${baseUrl}/#/inference`)
const retiredNodeRow = page.locator('tbody tr').filter({ hasText: smokeRetiredInferenceNode.name })
await retiredNodeRow.getByRole('button', { name: `永久删除退役板卡 ${smokeRetiredInferenceNode.name}` }).click()
await page.getByRole('dialog').getByRole('heading', { name: '永久删除退役板卡' }).waitFor()
await page.getByRole('dialog').getByRole('button', { name: '永久删除', exact: true }).click()
await retiredNodeRow.waitFor({ state: 'detached' })
assert.equal(smokeRetiredNodeDeleteCalls, 1, 'retired node delete should call the record endpoint once')
await page.getByRole('button', { name: '节点管理' }).click()
await page.getByRole('heading', { name: '系统设置', exact: true, level: 1 }).waitFor()
await page.goBack()
await page.getByRole('heading', { name: '推理下发', exact: true, level: 1 }).waitFor()
await page.getByRole('tab', { name: '模型版本' }).click()
await page.getByRole('button', { name: '登记模型版本' }).click()
await page.getByRole('dialog').getByRole('heading', { name: '登记模型版本' }).waitFor()
await page.getByRole('button', { name: '取消' }).click()
await page.getByRole('tab', { name: '推理任务' }).click()
await page.getByRole('button', { name: '新建推理任务' }).click()
const inferenceTaskDialog = page.getByRole('dialog')
await inferenceTaskDialog.getByRole('heading', { name: '新建推理任务' }).waitFor()
await inferenceTaskDialog.locator('.graph-node').filter({ hasText: '主推理' }).click()
await inferenceTaskDialog.getByLabel(/模型版本/).selectOption(smokeYoloRelease.id)
assert.equal(await inferenceTaskDialog.getByRole('alert').count(), 0)
assert.equal(await inferenceTaskDialog.getByLabel('使用核心').inputValue(), 'auto')
assert.equal(await inferenceTaskDialog.getByLabel('核心策略').inputValue(), 'shared')
await inferenceTaskDialog.getByLabel('核心策略').selectOption('exclusive')
await inferenceTaskDialog.getByText('独占策略必须选择明确核心，平台会在部署前检查冲突。').waitFor()
await inferenceTaskDialog.getByLabel('使用核心').selectOption('core1')
assert.equal(await inferenceTaskDialog.getByLabel('使用核心').inputValue(), 'core1')
await inferenceTaskDialog.getByLabel('核心策略').selectOption('shared')
await inferenceTaskDialog.getByLabel('使用核心').selectOption('auto')
await inferenceTaskDialog.locator('.graph-node').filter({ hasText: 'JSONL/HTTP 输出' }).click()
await inferenceTaskDialog.getByLabel('输出方式').selectOption('http')
await inferenceTaskDialog.getByLabel('url').fill('https://consumer.example/results')
await inferenceTaskDialog.getByLabel('authorizationEnv').fill('RKNODE_RESULT_SINK_TOKEN')
assert.equal(await inferenceTaskDialog.getByLabel('connectTimeoutMs').inputValue(), '1000')
assert.equal(await inferenceTaskDialog.getByLabel('requestTimeoutMs').inputValue(), '3000')
await inferenceTaskDialog.getByLabel('输出方式').selectOption('jsonl')
await inferenceTaskDialog.getByRole('button', { name: /MPP 硬解码/ }).click()
await inferenceTaskDialog.getByRole('button', { name: /ZLM SEI 输出/ }).click()
await inferenceTaskDialog.getByLabel('媒体网关').selectOption(smokeMediaGateway.id)
await inferenceTaskDialog.getByLabel('streamName').fill('ui-created-stream')
await inferenceTaskDialog.getByLabel('任务名称').fill(smokeCreatedInferenceTask.name)
if (screenshotDir) await inferenceTaskDialog.screenshot({ path: `${screenshotDir}/inference-graph-desktop.png` })
await inferenceTaskDialog.getByRole('button', { name: '保存草稿' }).click()
await inferenceTaskDialog.waitFor({ state: 'detached' })
assert.equal(smokeCreatedInferenceCalls, 1, 'creating an inference task should call the create endpoint once')
assert.equal(smokeCreatedInferencePayload?.releaseId, undefined, 'new graph tasks must not submit top-level releaseId')
assert.equal(smokeCreatedInferencePayload?.media, undefined, 'new graph tasks must not submit top-level media')
const createdCapture = smokeCreatedInferencePayload?.graph?.nodes?.find((node) => node.operator === 'capture.rkmpp')
const createdZlm = smokeCreatedInferencePayload?.graph?.nodes?.find((node) => node.operator === 'output.zlm_sei')
assert.ok(createdCapture, `graph should contain RKMPP capture: ${JSON.stringify(smokeCreatedInferencePayload?.graph)}`)
assert.deepEqual(createdZlm?.config, {
  gatewayId: smokeMediaGateway.id,
  streamName: 'ui-created-stream',
  reconnectMs: 1000,
})
assert.deepEqual(smokeCreatedInferencePayload?.layout && Object.keys(smokeCreatedInferencePayload.layout), ['positions'])
const createdInferenceRow = page.locator('tbody tr').filter({ hasText: smokeCreatedInferenceTask.name })
await createdInferenceRow.getByText('草稿', { exact: true }).waitFor()
assert.equal(await createdInferenceRow.getByRole('button', { name: '重启' }).count(), 0, 'draft task should require an explicit deployment')
const runningInferenceRow = page.locator('tbody tr').filter({ hasText: smokeRunningInferenceTask.name })
await runningInferenceRow.getByRole('button', { name: `查看实时预览 ${smokeRunningInferenceTask.name}` }).click()
const previewDialog = page.getByRole('dialog')
await previewDialog.getByRole('heading', { name: `实时预览 · ${smokeRunningInferenceTask.name}` }).waitFor()
await previewDialog.locator('.inference-stream-surface').waitFor()
await previewDialog.locator('.stream-state-overlay strong').getByText('正在连接').waitFor()
assert.ok(smokePlaybackSessionCalls >= 1, 'playback should request a scoped session')
await previewDialog.getByRole('button', { name: '关闭' }).click()
const stoppedInferenceRow = page.locator('tbody tr').filter({ hasText: smokeInferenceTask.name })
await stoppedInferenceRow.getByText('自动').waitFor()
await stoppedInferenceRow.getByText('共享').waitFor()
await stoppedInferenceRow.getByRole('button', { name: '重启' }).click()
await stoppedInferenceRow.getByText('部署中').waitFor()
assert.equal(smokeInferenceRestartCalls, 1, 'restart button should call the inference task restart endpoint once')
await page.getByRole('tab', { name: '部署批次' }).click()
await page.getByRole('button', { name: '创建部署批次' }).click()
await page.getByRole('dialog').getByRole('heading', { name: '创建部署批次' }).waitFor()
await page.getByRole('button', { name: '取消' }).click()
const completedDeploymentRow = page.locator('tbody tr').filter({ hasText: smokeCompletedDeployment.name })
await completedDeploymentRow.getByText('滚动中', { exact: true }).waitFor()
await completedDeploymentRow.getByText('已完成', { exact: true }).waitFor({ timeout: 7000 })
await completedDeploymentRow.getByRole('button', { name: `删除部署批次 ${smokeCompletedDeployment.name}` }).click()
await page.getByRole('dialog').getByRole('heading', { name: '删除部署批次' }).waitFor()
await page.getByRole('dialog').getByRole('button', { name: '删除部署批次', exact: true }).click()
await completedDeploymentRow.waitFor({ state: 'detached' })
assert.equal(smokeCompletedDeploymentDeleteCalls, 1, 'deployment delete should call the endpoint once')

await page.evaluate(() => window.localStorage.removeItem('rknode.videoWall.v1'))
await page.goto(`${baseUrl}/#/monitoring`)
await page.getByRole('heading', { name: '视频监控', exact: true, level: 2 }).waitFor()
assert.equal(await page.locator('.video-wall-tile').count(), 4, 'monitoring should start with four tiles')
await page.getByRole('button', { name: '选择画面 2' }).click()
await page.getByRole('button', { name: `将 ${smokeRunningInferenceTask.name} 播放到画面 2` }).click()
await page.locator('.video-wall-tile:nth-child(2) .inference-stream-surface').waitFor()
assert.equal(await page.locator('.video-wall-tile:nth-child(1) .inference-stream-surface').count(), 0, 'assigning tile 2 should not alter tile 1')
await page.getByRole('button', { name: '六宫格' }).click()
assert.equal(await page.locator('.video-wall-tile').count(), 6, 'six-tile layout should render six stable tiles')
await page.getByRole('button', { name: '选择画面 6' }).click()
await page.getByRole('button', { name: `将 ${smokeRunningInferenceTask.name} 播放到画面 6` }).click()
await page.locator('.video-wall-tile:nth-child(6) .inference-stream-surface').waitFor()
await page.reload()
await page.getByRole('heading', { name: '视频监控', exact: true, level: 2 }).waitFor()
assert.equal(await page.locator('.video-wall-tile').count(), 6, 'monitoring layout should survive a reload')
await page.locator('.video-wall-tile:nth-child(2) .inference-stream-surface').waitFor()
await page.locator('.video-wall-tile:nth-child(6) .inference-stream-surface').waitFor()
const monitoringDimensions = await page.evaluate(() => ({ body: document.body.scrollWidth, viewport: window.innerWidth }))
assert.ok(monitoringDimensions.body <= monitoringDimensions.viewport, `monitoring overflows: ${JSON.stringify(monitoringDimensions)}`)

await page.goto(`${baseUrl}/#/overview`)
const currentTasksPanel = page.getByRole('heading', { name: '当前任务' }).locator('xpath=ancestor::section[1]')
const recentStatusPanel = page.getByRole('heading', { name: '最近状态' }).locator('xpath=ancestor::section[1]')
const restartedOverviewRow = currentTasksPanel.locator('tbody tr').filter({ hasText: smokeInferenceTask.name })
await restartedOverviewRow.getByText('推理任务', { exact: true }).waitFor()
await restartedOverviewRow.getByText('部署中', { exact: true }).waitFor()
const runningOverviewRow = currentTasksPanel.locator('tbody tr').filter({ hasText: smokeRunningInferenceTask.name })
await runningOverviewRow.getByText('运行中', { exact: true }).waitFor()
const [currentTasksBox, recentStatusBox] = await Promise.all([currentTasksPanel.boundingBox(), recentStatusPanel.boundingBox()])
assert.ok(currentTasksBox && recentStatusBox, 'overview lower panels should be measurable')
const currentTasksBottom = currentTasksBox.y + currentTasksBox.height
const recentStatusBottom = recentStatusBox.y + recentStatusBox.height
assert.ok(Math.abs(currentTasksBottom - recentStatusBottom) <= 1, `overview lower panels should align: ${currentTasksBottom} vs ${recentStatusBottom}`)

await page.goto(`${baseUrl}/#/datasets`)
const pendingDatasetRow = page.locator('tbody tr').filter({ hasText: smokeDataset.name })
await pendingDatasetRow.getByText('待训练识别').waitFor()
assert.equal(await pendingDatasetRow.getByText(smokeDataset.id, { exact: true }).count(), 0)
assert.equal(await page.getByPlaceholder('搜索数据集名称').count(), 1)
await page.getByRole('button', { name: '上传数据集' }).click()
const datasetDialog = page.getByRole('dialog')
await datasetDialog.waitFor()
await datasetDialog.getByLabel(/数据集名称/).fill('界面冒烟测试')
assert.equal(await datasetDialog.getByLabel(/类别名称/).count(), 0)
await datasetDialog.getByLabel(/数据格式/).selectOption('coco_detection')
assert.equal(await datasetDialog.getByLabel(/数据格式/).inputValue(), 'coco_detection')
await datasetDialog.getByLabel(/任务类型/).selectOption('semantic_segmentation')
assert.equal(await datasetDialog.getByLabel(/数据格式/).inputValue(), 'auto')
assert.deepEqual(
  await datasetDialog.getByLabel(/数据格式/).locator('option').allTextContents(),
  ['自动识别', '图像/掩码配对', 'COCO 分割', 'Pascal VOC 分割'],
)
await page.getByRole('button', { name: '取消' }).click()
await page.getByLabel('每页数量').selectOption('20')
assert.equal(await page.getByLabel('每页数量').inputValue(), '20')
await page.getByRole('button', { name: /^删除数据集 / }).first().click()
await page.getByRole('dialog').waitFor()
await page.getByRole('button', { name: '取消' }).click()

await page.goto(`${baseUrl}/#/training`)
const trainingFixtureRow = page.locator('tbody tr').filter({ hasText: smokeTrainingJob.name })
await trainingFixtureRow.getByRole('cell', { name: 'yolov8n', exact: true }).waitFor()
await trainingFixtureRow.getByRole('cell', { name: smokeDataset.name, exact: true }).waitFor()
assert.equal(await trainingFixtureRow.getByText(smokeTrainingJob.id, { exact: true }).count(), 0)
await page.getByRole('button', { name: '新建训练任务' }).click()
await page.getByLabel('输入宽度').fill('640')
await page.getByLabel('输入高度').fill('384')
assert.equal(await page.getByLabel('输入宽度').inputValue(), '640')
assert.equal(await page.getByLabel('输入高度').inputValue(), '384')
await page.getByRole('button', { name: '取消' }).click()
await page.getByLabel('每页数量').selectOption('20')
await page.getByRole('button', { name: `查看训练监控 ${smokeTrainingJob.name}` }).click()
const monitorDialog = page.getByRole('dialog')
await monitorDialog.getByRole('heading', { name: `训练监控 · ${smokeTrainingJob.name}` }).waitFor()
await monitorDialog.getByRole('img', { name: '验证损失变化曲线' }).waitFor()
await monitorDialog.getByRole('log').getByText(/epoch=2\/2 train_loss=0.48/).waitFor()
await monitorDialog.locator('.modal-footer').getByRole('button', { name: '关闭' }).click()
await page.getByRole('button', { name: `重新训练 ${smokeFailedTrainingJob.name}` }).click()
await page.getByRole('status').filter({ hasText: '重新训练任务 train_ui_retry 已进入队列' }).waitFor()
const trainingDelete = trainingFixtureRow.getByRole('button', { name: `删除训练任务 ${smokeTrainingJob.name}` })
if (await trainingDelete.count()) {
  await trainingDelete.click()
  await page.getByRole('dialog').waitFor()
  await page.getByRole('button', { name: '取消' }).click()
}

await page.goto(`${baseUrl}/#/conversion`)
const conversionFixtureRow = page.locator('tbody tr').filter({ hasText: smokeConversionJob.name })
await conversionFixtureRow.getByRole('cell', { name: 'yolov8n', exact: true }).waitFor()
await conversionFixtureRow.getByText(smokeOutputArtifact.filename, { exact: true }).waitFor()
assert.equal(await conversionFixtureRow.getByText(smokeConversionJob.id, { exact: true }).count(), 0)
await page.getByRole('button', { name: '新建转换任务' }).click()
await page.getByRole('dialog').waitFor()
await page.getByRole('button', { name: '取消' }).click()
await page.getByLabel('每页数量').selectOption('50')
await conversionFixtureRow.getByRole('button', { name: `查看转换进度 ${smokeConversionJob.name}` }).click()
const conversionMonitorDialog = page.getByRole('dialog')
await conversionMonitorDialog.getByRole('heading', { name: `转换进度 · ${smokeConversionJob.name}` }).waitFor()
await conversionMonitorDialog.getByText('构建 RKNN 模型', { exact: true }).first().waitFor()
await conversionMonitorDialog.getByText('12.30 ms', { exact: false }).waitFor()
await conversionMonitorDialog.getByRole('log').getByText('Uploading RKNN artifact', { exact: true }).waitFor()
await conversionMonitorDialog.locator('.modal-footer').getByRole('button', { name: '关闭' }).click()
await page.getByRole('button', { name: `重新转换 ${smokeFailedConversionJob.name}` }).click()
await page.getByRole('status').filter({ hasText: '重新转换任务 convert_ui_retry 已进入队列' }).waitFor()
const conversionDelete = conversionFixtureRow.getByRole('button', { name: `删除转换任务 ${smokeConversionJob.name}` })
if (await conversionDelete.count()) {
  await conversionDelete.click()
  await page.getByRole('dialog').waitFor()
  await page.getByRole('button', { name: '取消' }).click()
}

await page.setViewportSize({ width: 390, height: 844 })
if (screenshotDir) {
  await page.goto(`${baseUrl}/#/inference`)
  await page.getByRole('tab', { name: '推理任务' }).click()
  await page.getByRole('button', { name: '新建推理任务' }).click()
  const mobileGraphDialog = page.getByRole('dialog')
  await mobileGraphDialog.getByRole('heading', { name: '新建推理任务' }).waitFor()
  await mobileGraphDialog.screenshot({ path: `${screenshotDir}/inference-graph-mobile.png` })
  await mobileGraphDialog.getByRole('button', { name: '取消' }).click()
}
await page.goto(`${baseUrl}/#/conversion`)
const mobile = await page.evaluate(() => ({ body: document.body.scrollWidth, viewport: window.innerWidth }))
assert.ok(mobile.body <= mobile.viewport, `mobile page overflows: ${JSON.stringify(mobile)}`)
await page.goto(`${baseUrl}/#/training`)
const mobileTraining = await page.evaluate(() => ({ body: document.body.scrollWidth, viewport: window.innerWidth }))
assert.ok(mobileTraining.body <= mobileTraining.viewport, `mobile training page overflows: ${JSON.stringify(mobileTraining)}`)
await page.goto(`${baseUrl}/#/monitoring`)
await page.getByRole('heading', { name: '视频监控', exact: true, level: 2 }).waitFor()
const mobileMonitoring = await page.evaluate(() => ({ body: document.body.scrollWidth, viewport: window.innerWidth }))
assert.ok(mobileMonitoring.body <= mobileMonitoring.viewport, `mobile monitoring page overflows: ${JSON.stringify(mobileMonitoring)}`)

const unexpectedErrors = errors.filter((error) => ![
  'Failed to load resource: the server responded with a status of 404 (Not Found)',
  '[IOController] > Loader error, code = undefined, msg = undefined',
  '[TransmuxingController] > IOException: type = Exception, code = undefined, msg = undefined',
].includes(error))
assert.deepEqual(unexpectedErrors, [])

const artifactPage = await browser.newPage({ viewport: { width: 1440, height: 900 } })
artifactPage.addInitScript(() => sessionStorage.setItem('rknode.adminToken', 'admin'))
let resolveArtifactJob
const artifactJobPromise = new Promise((resolve) => { resolveArtifactJob = resolve })
const sourceDataset = {
  id: 'dataset_ui_source', name: '转换来源测试数据集', description: '', version: 'v1', taskType: 'object_detection', datasetFormat: 'yolo', classes: ['target'],
  status: 'ready', filename: 'source-dataset.zip', sizeBytes: 10, sha256: '0'.repeat(64), createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), errorMessage: null,
}
await artifactPage.route('**/api/v1/jobs', async (route) => {
  const response = await route.fetch()
  const payload = await response.json()
  const jobs = Array.isArray(payload) ? payload : payload.items
  if (!Array.isArray(jobs)) throw new Error('jobs fixture expects an array or paginated items')
  let artifactJobIndex = jobs.findIndex((item) => item.type === 'training' && item.status === 'succeeded')
  if (artifactJobIndex < 0) {
    jobs.push(smokeTrainingJob)
    artifactJobIndex = jobs.length - 1
  }
  const artifactJob = { ...jobs[artifactJobIndex], datasetId: sourceDataset.id }
  jobs[artifactJobIndex] = artifactJob
  resolveArtifactJob(artifactJob)
  await route.fulfill({ response, json: Array.isArray(payload) ? jobs : { ...payload, items: jobs, total: jobs.length } })
})
await artifactPage.route('**/api/v1/datasets', async (route) => {
  const response = await route.fetch()
  const payload = await response.json()
  const datasets = Array.isArray(payload) ? payload : payload.items
  if (!Array.isArray(datasets)) throw new Error('datasets fixture expects an array or paginated items')
  datasets.push(sourceDataset)
  await route.fulfill({ response, json: Array.isArray(payload) ? datasets : { ...payload, items: datasets, total: datasets.length } })
})
await artifactPage.route('**/api/v1/artifacts/artifact_ui_checkpoint/download', (route) => route.fulfill({
  status: 200,
  contentType: 'application/octet-stream',
  body: 'checkpoint',
}))
await artifactPage.route('**/api/v1/artifacts', async (route) => {
  const [response, artifactJob] = await Promise.all([route.fetch(), artifactJobPromise])
  const payload = await response.json()
  const sourceArtifacts = Array.isArray(payload) ? payload : payload.items
  if (!Array.isArray(sourceArtifacts)) throw new Error('artifacts fixture expects an array or paginated items')
  const artifacts = sourceArtifacts.filter((item) => item.jobId !== artifactJob.id || !['onnx', 'training_checkpoint'].includes(item.kind))
  artifacts.push({ id: 'artifact_ui_onnx', jobId: artifactJob.id, kind: 'onnx', filename: 'yolov8n-640x640.onnx', mediaType: 'application/octet-stream', sizeBytes: 10, sha256: '0'.repeat(64), manifest: smokeManifest, createdAt: new Date().toISOString() })
  artifacts.push({ id: 'artifact_ui_checkpoint', jobId: artifactJob.id, kind: 'training_checkpoint', filename: 'yolov8n-640x640.pt', mediaType: 'application/octet-stream', sizeBytes: 10, sha256: '0'.repeat(64), manifest: null, createdAt: new Date().toISOString() })
  await route.fulfill({ response, json: Array.isArray(payload) ? artifacts : { ...payload, items: artifacts, total: artifacts.length } })
})
await artifactPage.goto(`${baseUrl}/#/training`)
const artifactJob = await artifactJobPromise
const artifactRow = artifactPage.locator('tbody tr').filter({ hasText: artifactJob.name })
await artifactRow.getByRole('button', { name: `下载 ONNX 模型 ${artifactJob.name}` }).waitFor()
const checkpointDownload = artifactRow.getByRole('button', { name: `下载训练权重 ${artifactJob.name}` })
assert.equal(await checkpointDownload.innerText(), 'PT')
await checkpointDownload.click()
await artifactPage.getByRole('status').filter({ hasText: '正在下载 yolov8n-640x640.pt' }).waitFor()
await artifactPage.goto(`${baseUrl}/#/conversion`)
await artifactPage.getByRole('button', { name: '新建转换任务' }).click()
const sourceSelect = artifactPage.getByLabel('训练 ONNX 产物')
const sourceOptions = await sourceSelect.locator('option').allTextContents()
const linkedDatasetName = sourceDataset.name
assert.ok(sourceOptions.some((text) => text.includes(artifactJob.name) && text.includes(linkedDatasetName)), 'conversion source options should identify the originating training job and dataset')
await sourceSelect.selectOption('artifact_ui_onnx')
const sourceDetails = artifactPage.getByLabel('源模型关联信息')
await sourceDetails.waitFor()
const sourceDetailsText = await sourceDetails.innerText()
for (const label of ['训练模型', '训练数据集', '训练任务', 'ONNX 文件']) assert.ok(sourceDetailsText.includes(label), `source identity should show ${label}`)
assert.ok(sourceDetailsText.includes(artifactJob.name) && sourceDetailsText.includes(linkedDatasetName), 'source identity should preserve the selected training relation')
await artifactPage.getByRole('button', { name: '取消' }).click()
await artifactPage.close()

const authPage = await browser.newPage({ viewport: { width: 1024, height: 768 } })
await authPage.route('**/api/v1/**', (route) => route.fulfill({
  status: 401,
  contentType: 'application/json',
  body: JSON.stringify({ error: { code: 'unauthorized', message: 'Unauthorized' } }),
}))
await authPage.goto(`${baseUrl}/#/datasets`)
const authDialog = authPage.getByRole('dialog')
await authDialog.getByRole('heading', { name: '需要管理员身份' }).waitFor()
assert.equal(await authDialog.getByRole('button', { name: '关闭' }).count(), 0)
assert.equal(await authDialog.getByRole('button', { name: '验证并继续' }).isDisabled(), true)
await authPage.close()

if (process.env.PROTOTYPE_EXPECT_PROXY_AUTH === '1') {
  const staleTokenPage = await browser.newPage({ viewport: { width: 1024, height: 768 } })
  await staleTokenPage.addInitScript(() => sessionStorage.setItem('rknode.adminToken', 'stale-token'))
  await staleTokenPage.goto(`${baseUrl}/#/datasets`)
  await staleTokenPage.getByRole('button', { name: '上传数据集' }).click()
  await staleTokenPage.getByRole('dialog').waitFor()
  await staleTokenPage.getByRole('button', { name: '取消' }).click()
  await staleTokenPage.close()
}

await browser.close()
console.log('UI smoke passed: API-backed actions, training/conversion monitors, four/six-tile live monitoring, persistence, artifact downloads, auth gate, dialogs, mobile width, no console errors.')
