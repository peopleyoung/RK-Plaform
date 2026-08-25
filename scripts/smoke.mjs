import assert from 'node:assert/strict'
import { chromium } from 'playwright'

const baseUrl = process.env.PROTOTYPE_URL ?? 'http://127.0.0.1:5173'
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
  id: 'itask_ui_stopped', name: '界面停止推理任务', status: 'stopped', releaseId: 'release_ui_fixture', nodeId: 'inode_ui_fixture', groupId: null,
  inputUri: 'rtsp://camera/ui-restart', interval: 0, thresholds: {}, output: { type: 'jsonl' }, npuCoreMask: 'auto', npuCorePolicy: 'shared', configRevision: 1,
  previewCapability: { state: 'available', reason: null }, media: { decoder: 'rkmpp', zlmSei: { enabled: true, gatewayId: 'gateway_builtin', streamName: 'ui-restart' } }, analytics: {},
  errorMessage: null, createdAt: fixtureTime, updatedAt: fixtureTime,
}
const smokeRunningInferenceTask = {
  ...smokeInferenceTask,
  id: 'itask_ui_running', name: '界面实时预览任务', status: 'running', inputUri: 'rtsp://camera/ui-preview', configRevision: 2,
}
const smokeCreatedInferenceTask = {
  ...smokeInferenceTask,
  id: 'itask_ui_created', name: '界面自动启动任务', status: 'stopped', releaseId: smokeYoloRelease.id, configRevision: 0,
}
const smokeCreatedStartedInferenceTask = {
  ...smokeCreatedInferenceTask,
  status: 'deploying', configRevision: 3,
}
const smokeRestartDeployment = {
  id: 'deployment_ui_restart', name: `Restart ${smokeInferenceTask.name}`, status: 'rolling', releaseId: smokeInferenceTask.releaseId,
  strategy: 'all_at_once', batchSize: 1, createdAt: fixtureTime, updatedAt: fixtureTime, completedAt: null,
  targets: [{
    id: 'dtarget_ui_restart', deploymentId: 'deployment_ui_restart', nodeId: smokeInferenceTask.nodeId, taskId: smokeInferenceTask.id,
    releaseId: smokeInferenceTask.releaseId, previousReleaseId: smokeInferenceTask.releaseId, sequence: 0, desiredRevision: 2,
    state: 'pending', progress: 0, stage: 'queued', errorCode: null, errorMessage: null, startedAt: null, completedAt: null, updatedAt: fixtureTime,
  }],
}
const smokeRetiredInferenceNode = {
  id: 'inode_ui_retired', name: '界面退役测试板卡', groupId: null, labels: ['ui-fixture'], lifecycle: 'retired',
  connectivity: 'offline', health: 'unknown', deploymentStatus: 'idle', maxModelInstances: 1, hardwareId: 'ui-retired-board',
  runtimeVersion: 'rknn-runtime-2.3.2', driverVersion: 'fixture', pipelineVersion: 'fixture', adapters: ['deeplab_logits_v1'], metadata: {},
  desiredRevision: 0, actualRevision: 0, selfTestPassed: false, lastSeenAt: fixtureTime, createdAt: fixtureTime, updatedAt: fixtureTime,
}
const smokeCompletedDeployment = {
  id: 'deployment_ui_completed', name: '界面已完成部署批次', status: 'succeeded', releaseId: smokeInferenceTask.releaseId,
  strategy: 'all_at_once', batchSize: 1, createdAt: fixtureTime, updatedAt: fixtureTime, completedAt: fixtureTime,
  targets: [{
    id: 'dtarget_ui_completed', deploymentId: 'deployment_ui_completed', nodeId: smokeRetiredInferenceNode.id, taskId: smokeInferenceTask.id,
    releaseId: smokeInferenceTask.releaseId, previousReleaseId: null, sequence: 0, desiredRevision: 1,
    state: 'healthy', progress: 100, stage: 'healthy', errorCode: null, errorMessage: null, startedAt: fixtureTime, completedAt: fixtureTime, updatedAt: fixtureTime,
  }],
}
let smokeInferenceRestarted = false
let smokeInferenceRestartCalls = 0
let smokeCreatedInference = false
let smokeCreatedInferenceCalls = 0
let smokeCreatedInferenceStartCalls = 0
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
  const response = await route.fetch()
  const payload = await response.json()
  if (Array.isArray(payload.items) && Number(url.searchParams.get('page') ?? 1) === 1) {
    for (const release of [smokeYoloRelease, smokeSecondaryRelease]) {
      if (!payload.items.some((item) => item.id === release.id)) payload.items.push(release)
    }
    payload.total += 2
  }
  await route.fulfill({ response, json: payload })
})
await page.route('**/api/v1/media-gateways', async (route) => {
  const response = await route.fetch()
  const payload = await response.json()
  if (!Array.isArray(payload)) throw new Error('media gateway fixture expects an array')
  if (!payload.some((item) => item.id === smokeMediaGateway.id)) payload.push(smokeMediaGateway)
  await route.fulfill({ response, json: payload })
})
await page.route('**/api/v1/inference-tasks**', async (route) => {
  const request = route.request()
  const url = new URL(request.url())
  if (request.method() === 'POST' && url.pathname === '/api/v1/inference-tasks') {
    smokeCreatedInference = true
    smokeCreatedInferenceCalls += 1
    smokeCreatedInferencePayload = JSON.parse(request.postData() ?? '{}')
    await route.fulfill({ status: 201, contentType: 'application/json', body: JSON.stringify(smokeCreatedInferenceTask) })
    return
  }
  if (request.method() === 'POST' && url.pathname.endsWith(`/${smokeCreatedInferenceTask.id}/restart`)) {
    smokeCreatedInferenceStartCalls += 1
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(smokeCreatedStartedInferenceTask) })
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
  const response = await route.fetch()
  const payload = await response.json()
  if (Array.isArray(payload.items) && Number(url.searchParams.get('page') ?? 1) === 1) {
    const fixture = { ...smokeInferenceTask, status: smokeInferenceRestarted ? 'deploying' : 'stopped' }
    if (!payload.items.some((item) => item.id === fixture.id)) payload.items.push(fixture)
    if (!payload.items.some((item) => item.id === smokeRunningInferenceTask.id)) payload.items.push(smokeRunningInferenceTask)
    if (smokeCreatedInference && !payload.items.some((item) => item.id === smokeCreatedInferenceTask.id)) {
      if (url.searchParams.get('pageSize') === '10' && !smokeCreatedInferenceFirstPageReadAt) smokeCreatedInferenceFirstPageReadAt = Date.now()
      const createdTask = smokeCreatedInferenceStartCalls
        ? { ...smokeCreatedStartedInferenceTask, status: smokeCreatedInferenceFirstPageReadAt && Date.now() - smokeCreatedInferenceFirstPageReadAt >= 500 ? 'running' : 'deploying' }
        : smokeCreatedInferenceTask
      payload.items.push(createdTask)
    }
    payload.total += smokeCreatedInference ? 3 : 2
  }
  await route.fulfill({ response, json: payload })
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
  const response = await route.fetch()
  const payload = await response.json()
  if (Array.isArray(payload.items) && Number(url.searchParams.get('page') ?? 1) === 1 && !smokeRetiredNodeDeleted) {
    if (!payload.items.some((item) => item.id === smokeRetiredInferenceNode.id)) payload.items.push(smokeRetiredInferenceNode)
    payload.total += 1
  }
  await route.fulfill({ response, json: payload })
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
  const response = await route.fetch()
  const payload = await response.json()
  if (Array.isArray(payload.items) && Number(url.searchParams.get('page') ?? 1) === 1 && !smokeCompletedDeploymentDeleted) {
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
    if (!payload.items.some((item) => item.id === deploymentFixture.id)) payload.items.push(deploymentFixture)
    payload.total += 1
  }
  await route.fulfill({ response, json: payload })
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
await inferenceTaskDialog.getByLabel('输出方式').selectOption('http')
await inferenceTaskDialog.getByLabel(/业务接口 URL/).fill('https://consumer.example/results')
await inferenceTaskDialog.getByLabel('Bearer 令牌环境变量').fill('RKNODE_RESULT_SINK_TOKEN')
const inferenceTimeouts = inferenceTaskDialog.locator('.inline-fields input')
assert.deepEqual(await inferenceTimeouts.evaluateAll((inputs) => inputs.map((input) => input.value)), ['1000', '3000'])
await inferenceTaskDialog.getByLabel('输出方式').selectOption('jsonl')
assert.equal(await inferenceTaskDialog.getByLabel(/业务接口 URL/).count(), 0)
const rtspSeiToggle = inferenceTaskDialog.getByRole('checkbox', { name: /RTSP \+ SEI 实时预览/ })
await rtspSeiToggle.check()
assert.equal(await inferenceTaskDialog.getByLabel('解码方式').inputValue(), 'rkmpp')
assert.equal(await inferenceTaskDialog.getByLabel('解码方式').isDisabled(), true)
await inferenceTaskDialog.getByLabel(/^媒体网关/).selectOption(smokeMediaGateway.id)
await inferenceTaskDialog.getByLabel(/发布流名称/).fill('ui-created-stream')
await inferenceTaskDialog.getByLabel('任务名称').fill(smokeCreatedInferenceTask.name)
await inferenceTaskDialog.getByRole('button', { name: '创建任务' }).click()
await inferenceTaskDialog.waitFor({ state: 'detached' })
assert.equal(smokeCreatedInferenceCalls, 1, 'creating an inference task should call the create endpoint once')
assert.equal(smokeCreatedInferenceStartCalls, 1, 'creating an inference task should start it immediately')
assert.deepEqual(smokeCreatedInferencePayload?.media?.zlmSei, {
  enabled: true,
  gatewayId: smokeMediaGateway.id,
  streamName: 'ui-created-stream',
  reconnectMs: 1000,
})
assert.equal(smokeCreatedInferencePayload?.media?.decoder, 'rkmpp')
const createdInferenceRow = page.locator('tbody tr').filter({ hasText: smokeCreatedInferenceTask.name })
await createdInferenceRow.getByText('部署中', { exact: true }).waitFor()
assert.equal(await createdInferenceRow.getByRole('button', { name: '重启' }).count(), 0, 'newly created task should not show restart')
await createdInferenceRow.getByText('运行中', { exact: true }).waitFor({ timeout: 7000 })
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
